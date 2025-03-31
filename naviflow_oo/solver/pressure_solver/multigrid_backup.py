"""
Matrix-free geometric multigrid solver for pressure correction equation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product
from .helpers.multigrid_helpers import restrict, interpolate
from .jacobi import JacobiSolver
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from scipy import sparse
from .helpers.spectral_radius_damping import find_optimal_damping, estimate_optimal_damping

class MultiGridSolver(PressureSolver):
    """
    Matrix-free geometric multigrid solver for pressure correction equation.
    
    This solver uses a geometric multigrid approach with V-cycles to solve
    the pressure correction equation without explicitly forming matrices.
    It requires grid sizes to be 2^k-1 (e.g., 3, 7, 15, 31, 63, 127, etc.)
    to ensure proper coarsening down to a 1x1 grid.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=1000, 
                 pre_smoothing=3, post_smoothing=3,
                 smoother_omega=None,
                 smoother=None,
                 max_diagnostic_history=10,  # Limit diagnostic history
                 store_vcycle_data=False):  # Option to disable v-cycle data storage
        """
        Initialize the multigrid solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the overall solver
        max_iterations : int, optional
            Maximum number of iterations for the overall solver
        pre_smoothing : int, optional
            Number of pre-smoothing steps
        post_smoothing : int, optional
            Number of post-smoothing steps
        smoother_omega : float, optional
            Relaxation factor for the smoother (if None and auto_omega=True, will be computed)
        smoother : PressureSolver, optional
            External smoother to use (if None, will use internal Jacobi smoother)
        max_diagnostic_history : int, optional
            Maximum number of diagnostic entries to store per grid level (to limit memory usage)
        store_vcycle_data : bool, optional
            Whether to store V-cycle data for visualization (uses more memory)
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.pre_smoothing = pre_smoothing
        self.post_smoothing = post_smoothing
        self.smoother_omega = smoother_omega
        self.residual_history = []
        self.vcycle_data = []  # Store V-cycle data
        self.presmooth_diagnostics = {}  # Dictionary to store presmooth diagnostics across different grid levels
        self.current_iteration = 0  # Track current iteration
        self.max_diagnostic_history = max_diagnostic_history
        self.store_vcycle_data = store_vcycle_data
        
        # Initialize smoother (will be updated in solve if auto_omega is True)
        self.smoother = smoother if smoother is not None else JacobiSolver(
            tolerance=1e-12,
            omega=smoother_omega if smoother_omega is not None else 0.8
        )
        
        # Pre-allocate mesh cache for different grid levels
        self.mesh_cache = {}
        
         
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using the matrix-free multigrid method.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        u_star, v_star : ndarray
            Intermediate velocity fields
        d_u, d_v : ndarray
            Momentum equation coefficients
        p_star : ndarray
            Current pressure field
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()

        rho = 1.0  # This should come from fluid properties
        
        # Reset residual history and diagnostics - clean up memory
        self.residual_history = []
        self.current_iteration = 0
        
        # Get right-hand side of pressure correction equation
        b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Initial guess - zero pressure correction field
        x = np.zeros_like(b)
         
        
        # Pre-allocate arrays for residual calculation
        Ax = np.zeros_like(b)
        r = np.zeros_like(b)
        
        # Perform multiple V-cycles until convergence or max iterations
        for k in range(self.max_iterations):
            # Update current iteration
            self.current_iteration = k + 1
            
            # Ensure B.C. are applied
            x = x.reshape((nx, ny), order='F')
            x = self.apply_pressure_boundary_conditions(x)
            x = x.flatten('F')

            grid_calculations = []  # Track grid sizes used
            
            # Run one V-cycle
            x_new = self._v_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                             self.pre_smoothing, self.post_smoothing, grid_calculations)
            
            # Apply boundary conditions
            x_new = x_new.reshape((nx, ny), order='F')
            x_new = self.apply_pressure_boundary_conditions(x_new)
            
            # Compute current residual: r = b - Ax
            Ax = compute_Ap_product(x_new.flatten('F'), nx, ny, dx, dy, rho, d_u, d_v)
            r = b - Ax
            
            # Calculate residual norm
            r_norm = np.linalg.norm(r,2)
            b_norm = np.linalg.norm(b,2)
            res_norm = r_norm / b_norm

            
            # Store scaled residual in history
            self.residual_history.append(res_norm)
            
            # Check convergence
            if r_norm < self.tolerance:
                print(f"Converged in {k+1} iterations, multigrid residual: {r_norm:.6e}")
                break
                
            print(f"Residual: {res_norm:.6e}")
            
            # Update solution for next iteration
            x = x_new.flatten('F')
    
        # Calculate overall convergence rate
        if len(self.residual_history) > 1:
            # Use geometric mean of convergence rates for overall rate
            conv_rates = [self.residual_history[i] / self.residual_history[i-1] 
                          for i in range(1, len(self.residual_history)) 
                          if self.residual_history[i-1] != 0]
            
            if conv_rates:
                overall_conv_rate = np.power(np.prod(conv_rates), 1.0 / len(conv_rates))
                print(f"Overall convergence rate: {overall_conv_rate:.4f}")
        
        
        # Reshape to 2D with Fortran ordering
        p_prime = x.reshape((nx, ny), order='F')
        
        return p_prime
    
    def _solve_residual_direct(self, mesh, residual, d_u, d_v, rho=1.0):
        """
        Solve the residual equation directly using sparse matrix methods.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        residual : ndarray
            The residual vector to solve for
        d_u, d_v : ndarray
            Momentum equation coefficients
        rho : float, optional
            Fluid density
            
        Returns:
        --------
        p_prime : ndarray
            Solution of the residual equation
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        # Get coefficient matrix
        from .helpers.coeff_matrix import get_coeff_mat
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Fix reference pressure at P(1,1) - which is (0,0) in 0-based indexing
        # Set the first row of the matrix to enforce P'(1,1) = 0
        A_lil = A.tolil()
        A_lil[0, :] = 0
        A_lil[0, 0] = 1
        A = A_lil.tocsr()
        
        # Add small regularization to improve conditioning
        eps = 1e-10
        diag_indices = sparse.find(A.diagonal())[0]
        A = A.tolil()
        for i in diag_indices:
            A[i, i] += eps
        A = A.tocsr()
        
        # Solve the system
        from scipy.sparse.linalg import spsolve
        p_prime_flat = spsolve(A, residual)
        
        # Reshape to 2D with Fortran ordering
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        p_prime = self.apply_pressure_boundary_conditions(p_prime)
        
        return p_prime


    def _v_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                grid_calculations, level=0):
        """
        Performs one V-cycle of the multigrid method.
        
        Parameters:
        -----------
        u : ndarray
            Current solution (flattened)
        f : ndarray
            Right-hand side (flattened)
        mesh : StructuredMesh
            Current mesh
        rho : float
            Fluid density
        d_u, d_v : ndarray
            Momentum equation coefficients
        omega : float
            Relaxation factor
        pre_smoothing : int
            Number of pre-smoothing steps
        post_smoothing : int
            Number of post-smoothing steps
        grid_calculations : list
            List to track grid sizes used
        level : int
            Current recursion level
            
        Returns:
        --------
        u : ndarray
            Updated solution (flattened)
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # Skip smoother for level=0 (initial call) to avoid unnecessary work on initial guess
        if level > 0 and pre_smoothing > 0:
            # Pre-smoothing steps
            u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                  d_u=d_u, d_v=d_v, rho=rho, 
                                  num_iterations=pre_smoothing, track_residuals=False)
        
        # If we're at the coarsest grid (31x31 or smaller), solve directly
        if nx <= 7:
            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            return u
            
        # Compute residual: r = f - Au (reshape for operations)
        u_2d = u.reshape((nx, ny), order='F')
        u_2d = self.apply_pressure_boundary_conditions(u_2d)
        u = u_2d.flatten('F')
        
        # Calculate residual
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        r_2d = r.reshape((nx, ny), order='F')
        
        # Restrict the residual to coarse grid
        r_coarse = restrict(r_2d)
        
        # Get coarse grid dimensions
        coarse_nx, coarse_ny = r_coarse.shape
        
        # Create coarse grid mesh
        mesh_coarse = StructuredMesh(nx=coarse_nx, ny=coarse_ny, 
                                    length=mesh.length, height=mesh.height)
        
        # Get coarse grid cell sizes
        dx_coarse, dy_coarse = mesh_coarse.get_cell_sizes()
        
        # Create coefficient arrays for coarse grid
        d_u_coarse = np.zeros((coarse_nx+1, coarse_nx))
        d_v_coarse = np.zeros((coarse_nx, coarse_nx+1))
        
        # Fill with proper physics-based coefficients
        for i in range(1, coarse_nx+1):
            for j in range(coarse_nx):
                d_u_coarse[i, j] = 1.0 / (rho * dx_coarse)
                
        for i in range(coarse_nx):
            for j in range(1, coarse_nx+1):
                d_v_coarse[i, j] = 1.0 / (rho * dy_coarse)
        
        # Initialize error correction on coarse grid
        e_coarse = np.zeros(coarse_nx * coarse_ny)
        
        # Recursive call to solve the error equation on coarse grid
        e_coarse = self._v_cycle(u=e_coarse, f=r_coarse.flatten('F'), 
                               mesh=mesh_coarse, rho=rho, 
                               d_u=d_u_coarse, d_v=d_v_coarse, 
                               omega=omega, pre_smoothing=pre_smoothing, 
                               post_smoothing=post_smoothing, 
                               grid_calculations=grid_calculations, 
                               level=level+1)
        
        # Reshape error for interpolation
        e_coarse_2d = e_coarse.reshape((coarse_nx, coarse_ny), order='F')
        
        # Interpolate error to fine grid
        e_fine = interpolate(e_coarse_2d, nx)
        
        # Apply damping factor to ensure stability
        # 2D grid ratio factor - less aggressive than h^2
        
        # Apply correction to the solution
        u_2d = u.reshape((nx, ny), order='F')
        u_2d += e_fine  # Add correction
        
        # Apply boundary conditions
        u_2d = self.apply_pressure_boundary_conditions(u_2d)
        
        # Post-smoothing steps
        if post_smoothing > 0:
            u = u_2d.flatten('F')
            u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                  d_u=d_u, d_v=d_v, rho=rho, 
                                  num_iterations=post_smoothing, track_residuals=False)
        else:
            u = u_2d.flatten('F')
            
        return u
    
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics
        """
        # Get smoother info if available
        smoother_info = {}
        if hasattr(self.smoother, 'get_solver_info'):
            smoother_info = self.smoother.get_solver_info()
        
        info = {
            'name': 'MultiGridSolver',
            'inner_iterations_history': [],  # We don't track this directly
            'total_inner_iterations': 0,     # We don't track this directly
            'convergence_rate': None         # We don't track this directly
        }
        
        # Add solver-specific information
        info['solver_specific'] = {
            'pre_smoothing': self.pre_smoothing,
            'post_smoothing': self.post_smoothing,
            'smoother_type': smoother_info.get('name', 'Unknown'),
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations
        }
        
        return info
