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
        
        if self.cycle_type == 'f':
            # For F-cycle, perform Full Multigrid (coarse-to-fine)
            x = self._perform_f_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega,
                                    self.pre_smoothing, self.post_smoothing, grid_calculations)
                
            # After F-cycle, use V-cycles until convergence
            for k in range(self.max_iterations):
                # Update current iteration
                self.current_iteration = k + 1
                
                # Apply V-cycle
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
                
        else:
            # For V-cycle or W-cycle, use the standard approach
            for k in range(self.max_iterations):
                # Update current iteration
                self.current_iteration = k + 1
                
                # Apply the appropriate cycle
                if self.cycle_type == 'v':
                    x = self._v_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                    self.pre_smoothing, self.post_smoothing, grid_calculations)
                elif self.cycle_type == 'w':
                    x = self._w_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
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
        if nx <= 7:
            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            return u  # Return solution on coarsest grid
  
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
        
        
        
        # d_u is staggered in x-direction: (coarse_grid_size+1, coarse_grid_size)
        d_u_coarse = np.zeros((coarse_grid_size+1, coarse_grid_size))
        # d_v is staggered in y-direction: (coarse_grid_size, coarse_grid_size+1)
        d_v_coarse = np.zeros((coarse_grid_size, coarse_grid_size+1))
        
        # Calculate scaling factor for proper grid-dependent coefficient scaling
        # For a Poisson equation, we need to scale by h^2 ratio
        dx_h = dx
        dy_h = dy
        dx_2h = dx_h * 2  # Coarse grid spacing (double the fine grid)
        h_ratio = dx_2h / dx_h  # Ratio of coarse to fine grid spacing
        coeff_scale_factor = h_ratio * h_ratio  # (dx_2h/dx_h)^2
        
        # Fill d_u_coarse with proper grid-dependent scaling
        # Create masks for valid regions
        i_indices = np.arange(coarse_grid_size + 1)[:, np.newaxis]
        j_indices = np.arange(coarse_grid_size)[np.newaxis, :]
        d_u_valid = (i_indices > 0) & (i_indices <= nx) & (j_indices < ny)

        # Fill d_u_coarse using vectorized operations
        d_u_coarse = np.where(d_u_valid,
                             d_u[:coarse_grid_size+1, :coarse_grid_size],
                             1.0 / (rho * (1.0/dx_h)))
        d_u_coarse /= coeff_scale_factor

        # Create masks for d_v
        i_indices = np.arange(coarse_grid_size)[:, np.newaxis]
        j_indices = np.arange(coarse_grid_size + 1)[np.newaxis, :]
        d_v_valid = (i_indices < nx) & (j_indices > 0) & (j_indices <= ny)

        # Fill d_v_coarse using vectorized operations
        d_v_coarse = np.where(d_v_valid,
                             d_v[:coarse_grid_size, :coarse_grid_size+1],
                             1.0 / (rho * (1.0/dy_h)))
        d_v_coarse /= coeff_scale_factor
        
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
        u += e_interpolated_flat.reshape((nx, ny), order='F')

        
        # Post-smoothing on fine grid
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
      
        u_reshaped = u.reshape((nx, ny), order='F')
        
        return u
    
    def _w_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                grid_calculations, level=0):
        """
        Performs one W-cycle of the multigrid method.
        
        W-cycle recursively visits the coarse grid twice before interpolating
        back to the finer grid.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # If we're at the coarsest grid, solve directly
        if nx <= 7:
            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            return u  # Return solution on coarsest grid
     
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        
       
        # Compute residual: r = f - Au
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        # Restrict residual to coarser grid
        r_reshaped = r.reshape((nx, ny), order='F')
        r_coarse = restrict(r_reshaped) 
        
        # Size of coarse grid
        coarse_grid_size = r_coarse.shape[0]
        
        # Create coarse grid mesh
        mesh_coarse = StructuredMesh(nx=coarse_grid_size, ny=coarse_grid_size, 
                                   length=mesh.length, height=mesh.height)
        
        # Get cell sizes for coarse grid
        dx_coarse, dy_coarse = mesh_coarse.get_cell_sizes()
        
        # Create d_u and d_v coefficients for coarse grid with appropriate scaling
        # The d_u and d_v coefficients should scale properly with grid spacing
        
        # d_u is staggered in x-direction: (coarse_grid_size+1, coarse_grid_size)
        d_u_coarse = np.zeros((coarse_grid_size+1, coarse_grid_size))
        # d_v is staggered in y-direction: (coarse_grid_size, coarse_grid_size+1)
        d_v_coarse = np.zeros((coarse_grid_size, coarse_grid_size+1))
        
        # Calculate scaling factor for proper grid-dependent coefficient scaling
        # For a Poisson equation, we need to scale by h^2 ratio
        dx_h = dx
        dy_h = dy
        dx_2h = dx_h * 2  # Coarse grid spacing (double the fine grid)
        h_ratio = dx_2h / dx_h  # Ratio of coarse to fine grid spacing
        coeff_scale_factor = h_ratio * h_ratio  # (dx_2h/dx_h)^2
        
        # Fill d_u_coarse with proper grid-dependent scaling
        # Create masks for valid regions
        i_indices = np.arange(coarse_grid_size + 1)[:, np.newaxis]
        j_indices = np.arange(coarse_grid_size)[np.newaxis, :]
        d_u_valid = (i_indices > 0) & (i_indices <= nx) & (j_indices < ny)

        # Fill d_u_coarse using vectorized operations
        d_u_coarse = np.where(d_u_valid,
                             d_u[:coarse_grid_size+1, :coarse_grid_size],
                             1.0 / (rho * (1.0/dx_h)))
        d_u_coarse /= coeff_scale_factor

        # Create masks for d_v
        i_indices = np.arange(coarse_grid_size)[:, np.newaxis]
        j_indices = np.arange(coarse_grid_size + 1)[np.newaxis, :]
        d_v_valid = (i_indices < nx) & (j_indices > 0) & (j_indices <= ny)

        # Fill d_v_coarse using vectorized operations
        d_v_coarse = np.where(d_v_valid,
                             d_v[:coarse_grid_size, :coarse_grid_size+1],
                             1.0 / (rho * (1.0/dy_h)))
        d_v_coarse /= coeff_scale_factor
        
        # First W-cycle recursive call on coarse grid
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse = self._w_cycle(u=np.zeros_like(r_coarse_flat), f=r_coarse_flat, 
                                mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                omega=omega, pre_smoothing=pre_smoothing, 
                                post_smoothing=post_smoothing, grid_calculations=grid_calculations, 
                                level=level+1)
        
        # Ensure e_coarse is flattened
        if e_coarse.ndim == 2:
            e_coarse = e_coarse.flatten('F')
                                
        # Second W-cycle recursive call on coarse grid
        # Compute new residual on coarse grid
        Ae_coarse = compute_Ap_product(e_coarse, coarse_grid_size, coarse_grid_size, 
                                      dx_coarse, dy_coarse, rho, d_u_coarse, d_v_coarse)
        r_coarse_new = r_coarse_flat - Ae_coarse
        
        # Apply second W-cycle
        e_coarse_correction = self._w_cycle(u=np.zeros_like(r_coarse_new), f=r_coarse_new, 
                                          mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                          omega=omega, pre_smoothing=pre_smoothing, 
                                          post_smoothing=post_smoothing, grid_calculations=grid_calculations, 
                                          level=level+1)
                     
        # Combine corrections
        e_coarse += e_coarse_correction.flatten('F')
        
        # Reshape to 2D for storage and interpolation
        e_coarse_2d = e_coarse.reshape((coarse_grid_size, coarse_grid_size), order='F')
            
        
        # Interpolate error to fine grid
        e_interpolated = interpolate(e_coarse_2d, nx)
        
        
        # Apply correction
        e_interpolated_flat = e_interpolated.flatten('F')
        u += e_interpolated_flat.reshape((nx, ny), order='F')

        
        # Post-smoothing on fine grid
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
  
        return u
        
    def _perform_f_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                        grid_calculations, level=0):
        """
        Performs the Full Multigrid (F-cycle) algorithm starting from the coarsest grid.
        
        Args:
            u (ndarray): Initial guess (for the finest grid)
            f (ndarray): Right-hand side of the equation (for the finest grid)
            mesh (StructuredMesh): The computational mesh
            rho (float): Density
            d_u, d_v (ndarray): Momentum equation coefficients
            omega (float): Relaxation parameter
            pre_smoothing (int): Number of pre-smoothing steps
            post_smoothing (int): Number of post-smoothing steps
            grid_calculations (list): List to store grid computation data
            level (int): Current level in the multigrid hierarchy
            
        Returns:
            ndarray: Solution on the finest grid
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        # Determine all grid levels from coarsest to finest
        # For multigrid to work properly, the coarsest grid should be small (e.g., 3x3, 7x7)
        min_grid_size = 7  # Smallest grid size to solve directly
        
        # Calculate how many levels we need
        n = nx  # Assuming nx = ny for simplicity
        num_levels = 0
        temp_n = n
        
        # Count levels from finest to coarsest
        while temp_n > min_grid_size:
            num_levels += 1
            temp_n = (temp_n + 1) // 2 - 1  # Grid coarsening
        
        # Add the coarsest level
        num_levels += 1
        
        # Now we have the number of levels, work from coarsest to finest
        # First, get the coarsest grid size
        coarsest_n = n
        for _ in range(num_levels - 1):
            coarsest_n = (coarsest_n + 1) // 2 - 1
        
        # Create a list of grid sizes from coarsest to finest
        grid_sizes = [coarsest_n]
        for i in range(num_levels - 2, -1, -1):
            grid_sizes.append((grid_sizes[-1] + 1) * 2 - 1)
        
        # Initialize solution on the coarsest grid
        current_n = grid_sizes[0]
        coarsest_mesh = StructuredMesh(nx=current_n, ny=current_n, 
                                      length=mesh.length, height=mesh.height)
        
        # Create appropriately sized coefficients for the coarsest grid
        d_u_coarse = np.ones((current_n+1, current_n)) / (rho * (coarsest_mesh.dx))
        d_v_coarse = np.ones((current_n, current_n+1)) / (rho * (coarsest_mesh.dy))
        
        # Get right-hand side for the coarsest grid
        f_restricted = f.reshape((nx, ny), order='F')
        for i in range(num_levels - 1):
            f_restricted = restrict(f_restricted)
        
        # Solve exactly on the coarsest grid
        u_current = self._solve_residual_direct(
            coarsest_mesh, f_restricted.flatten('F'), d_u_coarse, d_v_coarse, rho)
        
        # Now work our way up from coarse to fine grids
        for i in range(1, len(grid_sizes)):
            # Get the next finer grid size
            next_n = grid_sizes[i]
            
            # Create the next finer mesh
            next_mesh = StructuredMesh(nx=next_n, ny=next_n, 
                                      length=mesh.length, height=mesh.height)
            
            # Create coefficients for the next finer grid
            d_u_next = np.ones((next_n+1, next_n)) / (rho * (next_mesh.dx))
            d_v_next = np.ones((next_n, next_n+1)) / (rho * (next_mesh.dy))
            
            # Interpolate the solution to the next finer grid
            u_current_2d = u_current.reshape((current_n, current_n), order='F') if u_current.ndim == 1 else u_current
            u_interpolated = interpolate(u_current_2d, next_n)
            u_interpolated_flat = u_interpolated.flatten('F')
            
            # Get right-hand side for the next finer grid
            if i == len(grid_sizes) - 1:
                # For the finest grid, use the original RHS
                f_next = f
            else:
                # For intermediate grids, restrict from the original RHS
                f_restricted = f.reshape((nx, ny), order='F')
                for j in range(num_levels - 1 - i):
                    f_restricted = restrict(f_restricted)
                f_next = f_restricted.flatten('F')
            
            # Perform V-cycles to improve the solution on this grid level
            for _ in range(1):  # Usually just one V-cycle is enough
                u_interpolated_flat = self._v_cycle(
                    u_interpolated_flat, f_next, next_mesh, rho, d_u_next, d_v_next, 
                    omega, pre_smoothing, post_smoothing, grid_calculations)
            
            # Update for the next iteration
            u_current = u_interpolated_flat
            current_n = next_n
        
        # Return the solution on the finest grid
        return u_current
    
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