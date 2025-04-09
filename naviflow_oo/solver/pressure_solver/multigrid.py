import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product
from .helpers.multigrid_helpers import restrict, interpolate, restrict_coefficients
from .jacobi import JacobiSolver
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .helpers.coeff_matrix import get_coeff_mat

class MultiGridSolver(PressureSolver):
    """
    Matrix-free geometric multigrid solver for pressure correction equation.
    
    This solver uses a geometric multigrid approach with V-cycles to solve
    the pressure correction equation without explicitly forming matrices.
    It requires grid sizes to be 2^k-1 (e.g., 3, 7, 15, 31, 63, 127, etc.)
    to ensure proper coarsening down to a 1x1 grid.
    
    Added debug functionality: when debug is True, the solver stores intermediate 
    arrays (after pre-smoothing, residual computation, restriction, interpolation, 
    correction, and post-smoothing) and outputs a multi-page PDF that plots these arrays 
    in chronological order.
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
            Convergence tolerance for the overall solver.
        max_iterations : int, optional
            Maximum number of iterations for the overall solver.
        pre_smoothing : int, optional
            Number of pre-smoothing steps.
        post_smoothing : int, optional
            Number of post-smoothing steps.
        smoother_omega : float, optional
            Relaxation factor for the smoother (if None and auto_omega=True, will be computed).
        smoother : PressureSolver, optional
            External smoother to use (if None, will use internal Jacobi smoother).
        cycle_type : str, optional
            Type of multigrid cycle to use: 'v', 'w', or 'f'.
        debug : bool, optional
            When True, stores intermediate debugging arrays and plots them into a PDF.
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.pre_smoothing = pre_smoothing
        self.post_smoothing = post_smoothing
        self.smoother_omega = smoother_omega
        self.residual_history = []
        self.vcycle_data = []  # Existing storage for V-cycle data (if needed)
        self.presmooth_diagnostics = {}  # Diagnostics across different grid levels
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
            The computational mesh.
        u_star, v_star : ndarray
            Intermediate velocity fields.
        d_u, d_v : ndarray
            Momentum equation coefficients.
        p_star : ndarray
            Current pressure field.
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field.
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
        
     

        # Apply the appropriate cycle based on cycle_type
        for k in range(self.max_iterations):
            self.current_iteration = k + 1
            
            # Apply the appropriate cycle based on cycle_type
            if self.cycle_type == 'v':
                x = self._v_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                self.pre_smoothing, self.post_smoothing, level=0)
            elif self.cycle_type == 'w':
                x = self._w_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                self.pre_smoothing, self.post_smoothing, level=0)
            elif self.cycle_type == 'f':
                # For F-cycle, we only do it once at the beginning
                if k == 0:
                    x = self._f_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                    self.pre_smoothing, self.post_smoothing, level=0)
                else:
                    # After the initial F-cycle, use V-cycles until convergence
                    x = self._w_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                    self.pre_smoothing, self.post_smoothing, level=0)
            
            # Compute residual: r = b - A*x
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
            #print(f"Multigrid iteration {k+1}, residual: {res_norm:.6e}")
    
        # Calculate overall convergence rate (if possible)
        if len(self.residual_history) > 1:
            try:
                conv_rates = []
                for i in range(1, len(self.residual_history)):
                    prev_res = self.residual_history[i-1]
                    if prev_res > 1e-12:  # Avoid division by very small values
                        rate = self.residual_history[i] / prev_res
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
            The computational mesh.
        residual : ndarray
            The residual vector to solve for.
        d_u, d_v : ndarray
            Momentum equation coefficients.
        rho : float, optional
            Fluid density.
            
        Returns:
        --------
        p_prime : ndarray
            Solution of the residual equation.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Fix reference pressure at P(1,1) - which is (0,0) in 0-based indexing
        # to lil
        A = A.tolil()
        A[0, :] = 0
        A[0, 0] = 1
        eps = 1e-10
        diag_indices = sparse.find(A.diagonal())[0]
        for i in diag_indices:
            A[i, i] += eps
        A = A.tocsr()
        
        # Solve the system
        p_prime_flat = spsolve(A, residual)
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        return p_prime

    def _v_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                 level=0):
        """
        Performs one V-cycle of the multigrid method.
        
        Parameters:
        -----------
        u : ndarray
            Current approximation (flattened).
        f : ndarray
            Right-hand side (flattened).
        mesh : StructuredMesh
            The current grid mesh.
        rho : float
            Fluid density.
        d_u, d_v : ndarray
            Momentum equation coefficients.
        omega : float
            Relaxation factor.
        pre_smoothing : int
            Number of pre-smoothing iterations.
        post_smoothing : int
            Number of post-smoothing iterations.
        level : int
            Current grid level of the multigrid hierarchy.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # If at the coarsest grid, solve directly (here, we simply return zeros)
        if nx <= 7:

            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            # use smoother
            #u = self.smoother.solve(mesh=mesh, p=u, b=f,
            #                    d_u=d_u, d_v=d_v, rho=rho, num_iterations=50, track_residuals=False)
            return u  
  
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
   

        
        # Compute the residual: r = f - A*u
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        # Reshape residual to 2D for restriction
        r_reshaped = r.reshape((nx, ny), order='F')
        
        # Restrict residual to coarser grid
        r_coarse = restrict(r_reshaped) 
        
        # Determine coarse grid size and create coarse grid mesh
        coarse_grid_size = r_coarse.shape[0]

        mesh_coarse = StructuredMesh(nx=coarse_grid_size, ny=coarse_grid_size, 
                                        length=mesh.length, height=mesh.height)
        
        # Restrict coefficients for the coarse grid
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
                                 post_smoothing=post_smoothing, 
                                 level=level+1)
        # Optionally store the coarse grid error solution
        e_coarse_2D = e_coarse.reshape((coarse_grid_size, coarse_grid_size), order='F')
       
        # Interpolate error correction to fine grid
        e_interpolated = interpolate(e_coarse, nx)
        
        # Apply the error correction
        u += e_interpolated.flatten('F').reshape((nx, ny), order='F') 
        
        # Post-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
        return u
    
    def _w_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                 level=0):
        """
        Performs one W-cycle of the multigrid method.
        
        Parameters:
        -----------
        u : ndarray
            Current approximation (flattened).
        f : ndarray
            Right-hand side (flattened).
        mesh : StructuredMesh
            The current grid mesh.
        rho : float
            Fluid density.
        d_u, d_v : ndarray
            Momentum equation coefficients.
        omega : float
            Relaxation factor.
        pre_smoothing : int
            Number of pre-smoothing iterations.
        post_smoothing : int
            Number of post-smoothing iterations.
        grid_calculations : list
            List to track grid sizes (not used for debugging here).
        level : int
            Current grid level of the multigrid hierarchy.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # If at the coarsest grid, solve directly
        if nx <= 7:
            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            #u = self.smoother.solve(mesh=mesh, p=u, b=f,
            #                    d_u=d_u, d_v=d_v, rho=rho, num_iterations=20, track_residuals=False)
            return u  
  
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        
        # Compute the residual: r = f - A*u
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        # Reshape residual to 2D for restriction
        r_reshaped = r.reshape((nx, ny), order='F')
        
        # Restrict residual to coarser grid
        r_coarse = restrict(r_reshaped) 
        
        # Determine coarse grid size and create coarse grid mesh
        coarse_grid_size = r_coarse.shape[0]
        mesh_coarse = StructuredMesh(nx=coarse_grid_size, ny=coarse_grid_size, 
                                        length=mesh.length, height=mesh.height)
        
        # Restrict coefficients for the coarse grid
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u, d_v, 
            nx, ny, 
            coarse_grid_size, coarse_grid_size, 
            dx, dy
        )
        
        # First recursive W-cycle on coarse grid
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse1 = self._w_cycle(u=np.zeros_like(r_coarse_flat), f=r_coarse_flat, 
                                 mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                 omega=omega, pre_smoothing=pre_smoothing, 
                                 post_smoothing=post_smoothing, 
                                 level=level+1)
        
        # Second recursive W-cycle on coarse grid (this is what makes it a W-cycle)
        e_coarse2 = self._w_cycle(u=e_coarse1, f=r_coarse_flat, 
                                 mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                 omega=omega, pre_smoothing=pre_smoothing, 
                                 post_smoothing=post_smoothing, 
                                 level=level+1)
        
        # Optionally store the coarse grid error solution
        e_coarse_2D = e_coarse2.reshape((coarse_grid_size, coarse_grid_size), order='F')
       
        # Interpolate error correction to fine grid
        e_interpolated = interpolate(e_coarse2, nx)
        
        # Apply the error correction
        u += e_interpolated.flatten('F').reshape((nx, ny), order='F') 
        
        # Post-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
        return u
    
    def _f_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                 level=0):
        """
        Performs one F-cycle of the multigrid method.
        An F-cycle starts from the coarsest grid and moves up, then performs V-cycles at the finest level.
        
        Parameters:
        -----------
        u : ndarray
            Current approximation (flattened).
        f : ndarray
            Right-hand side (flattened).
        mesh : StructuredMesh
            The current grid mesh.
        rho : float
            Fluid density.
        d_u, d_v : ndarray
            Momentum equation coefficients.
        omega : float
            Relaxation factor.
        pre_smoothing : int
            Number of pre-smoothing iterations.
        post_smoothing : int
            Number of post-smoothing iterations.
        level : int
            Current grid level of the multigrid hierarchy.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # If at the coarsest grid, solve directly
        if nx <= 7:
            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            return u
        
        # For F-cycle, we need to track the finest level to know when to switch to V-cycles
        if level == 0:
            # At the finest level, we'll do a V-cycle
            return self._v_cycle(u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                                level)
        
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        
        # Compute the residual: r = f - A*u
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        # Reshape residual to 2D for restriction
        r_reshaped = r.reshape((nx, ny), order='F')
        
        # Restrict residual to coarser grid
        r_coarse = restrict(r_reshaped) 
        
        # Determine coarse grid size and create coarse grid mesh
        coarse_grid_size = r_coarse.shape[0]
        mesh_coarse = StructuredMesh(nx=coarse_grid_size, ny=coarse_grid_size, 
                                        length=mesh.length, height=mesh.height)
        
        # Restrict coefficients for the coarse grid
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u, d_v, 
            nx, ny, 
            coarse_grid_size, coarse_grid_size, 
            dx, dy
        )
        
        # Recursive F-cycle on coarse grid
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse = self._f_cycle(u=np.zeros_like(r_coarse_flat), f=r_coarse_flat, 
                                 mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                 omega=omega, pre_smoothing=pre_smoothing, 
                                 post_smoothing=post_smoothing, 
                                 level=level+1)
        
        # Optionally store the coarse grid error solution
        e_coarse_2D = e_coarse.reshape((coarse_grid_size, coarse_grid_size), order='F')
       
        # Interpolate error correction to fine grid
        e_interpolated = interpolate(e_coarse, nx)
        
        # Apply the error correction
        u += e_interpolated.flatten('F').reshape((nx, ny), order='F') 
        
        # Post-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
        return u
    
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics.
        """
        smoother_info = {}
        if hasattr(self.smoother, 'get_solver_info'):
            smoother_info = self.smoother.get_solver_info()
        
        info = {
            'name': 'MultiGridSolver',
            'inner_iterations_history': [],  # Not tracked directly
            'total_inner_iterations': 0,       # Not tracked directly
            'convergence_rate': None           # Not tracked directly
        }
        
        info['solver_specific'] = {
            'pre_smoothing': self.pre_smoothing,
            'post_smoothing': self.post_smoothing,
            'smoother_type': smoother_info.get('name', 'Unknown'),
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations,
            'cycle_type': self.cycle_type,
        }
        
        return info
