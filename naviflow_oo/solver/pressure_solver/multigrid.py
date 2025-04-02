"""
Matrix-free geometric multigrid solver for pressure correction equation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product
from .helpers.multigrid_helpers import restrict, interpolate, restrict_coefficients
from .jacobi import JacobiSolver
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from scipy import sparse

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
                 cycle_type='v'):
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
        cycle_type : str, optional
            Type of multigrid cycle to use: 'v', 'w', or 'f'
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.pre_smoothing = pre_smoothing
        self.post_smoothing = post_smoothing
        self.smoother_omega = smoother_omega
        self.residual_history = []
        self.vcycle_data = []  # Store V-cycle data
        self.presmooth_diagnostics = {}  # Dictionary to store presmooth diagnostics across different grid levels
        self.current_iteration = 0  # Track current iteration
        
        # Validate cycle type
        if cycle_type.lower() not in ['v', 'w', 'f']:
            raise ValueError("cycle_type must be one of: 'v', 'w', 'f'")
        self.cycle_type = cycle_type.lower()
        
        self.smoother = smoother if smoother is not None else JacobiSolver(
            omega=smoother_omega if smoother_omega is not None else 0.8
        )
       
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
        
        # Reset residual history and diagnostics
        self.residual_history = []
        self.presmooth_diagnostics = {}
        self.current_iteration = 0
        
        # Get right-hand side of pressure correction equation
        b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Initial guess
        x = np.zeros_like(b)
        
        # Perform multigrid cycles based on the selected type
        grid_calculations = []  # Track grid sizes used
        

    
        # For V-cycle or W-cycle, use the standard approach
        for k in range(self.max_iterations):
            # Update current iteration
            self.current_iteration = k + 1
            
            # Apply the appropriate cycle
            x = self._v_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                self.pre_smoothing, self.post_smoothing, grid_calculations)
            
            # Compute residual: r = b - Ax
            Ax = compute_Ap_product(x, nx, ny, dx, dy, rho, d_u, d_v)
            r = b - Ax
            
            # Calculate residual norm
            r_norm = np.linalg.norm(r, 2)
            b_norm = np.linalg.norm(b, 2)
            
            res_norm = r_norm / b_norm
            
            self.residual_history.append(res_norm)
            
            # Check convergence
            if res_norm < self.tolerance:
                print(f"Converged in {k+1} iterations, multigrid residual: {res_norm:.6e}")
                break
    
        # Calculate overall convergence rate
        if len(self.residual_history) > 1:
            # Use geometric mean of convergence rates for overall rate
            try:
                conv_rates = []
                for i in range(1, len(self.residual_history)):
                    prev_res = self.residual_history[i-1]
                    if prev_res > 1e-12:  # Avoid division by very small values
                        rate = self.residual_history[i] / prev_res
                        # Filter out invalid or extreme values
                        if not np.isnan(rate) and not np.isinf(rate) and abs(rate) < 1.0:
                            conv_rates.append(rate)
                
                if conv_rates:
                    overall_conv_rate = np.power(np.prod(conv_rates), 1.0 / len(conv_rates))
                    print(f"Overall convergence rate: {overall_conv_rate:.4f}")
                else:
                    print("Could not calculate convergence rate - unstable values detected")
            except Exception as e:
                print(f"Error calculating convergence rate: {e}")
        
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
        
        # Set the RHS value at the reference point to enforce P'(1,1) = 0
        #residual[0] = 0.0
        
        # Solve the system
        from scipy.sparse.linalg import spsolve
        p_prime_flat = spsolve(A, residual)
        
        # Reshape to 2D with Fortran ordering
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        # Ensure reference pressure is exactly zero
        #p_prime[0, 0] = 0.0
        
        return p_prime


    def _v_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                grid_calculations, level=0):
        """
        Performs one V-cycle of the multigrid method.
        """
        
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # If we're at the coarsest grid, solve directly
        if nx <=7:
            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            return u#-1/16*f  # Return solution on coarsest grid
  
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        
      
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        r_reshaped = r.reshape((nx, ny), order='F')
        
        # Restrict residual to coarser grid
        r_coarse = restrict(r_reshaped) 
        
        
        # Size of coarse grid
        coarse_grid_size = r_coarse.shape[0]
        
        # Create coarse grid mesh
        mesh_coarse = StructuredMesh(nx=coarse_grid_size, ny=coarse_grid_size, 
                                   length=mesh.length, height=mesh.height)
        
        # Use the new function for better coefficient restriction
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u, d_v, 
            nx, ny, 
            coarse_grid_size, coarse_grid_size, 
            dx, dy
        )
        
        # Recursive V-cycle on coarse grid
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse = self._v_cycle(u=np.zeros_like(r_coarse_flat), f=r_coarse_flat, 
                                mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                omega=omega, pre_smoothing=pre_smoothing, 
                                post_smoothing=post_smoothing, grid_calculations=grid_calculations, 
                                level=level+1)
       
        # should already be at the right scale
        e_interpolated = interpolate(e_coarse, nx)
        
        # Apply correction - ensure both operands are 1D with the same shape
        e_interpolated_flat = e_interpolated.flatten('F')
        u += e_interpolated_flat.reshape((nx, ny), order='F') #* 1.2

        
        # Post-smoothing on fine grid
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
      
        u_reshaped = u.reshape((nx, ny), order='F')
        
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
            'max_iterations': self.max_iterations,
            'cycle_type': self.cycle_type
        }
        
        return info