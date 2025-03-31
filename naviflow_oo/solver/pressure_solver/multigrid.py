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
                 smoother=None):
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
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.pre_smoothing = pre_smoothing
        self.post_smoothing = post_smoothing
        self.smoother_omega = smoother_omega
        self.residual_history = []
        self.vcycle_data = []  # Store V-cycle data
        self.presmooth_diagnostics = {}  # Dictionary to store presmooth diagnostics across different grid levels
        self.current_iteration = 0  # Track current iteration
        
        # Initialize smoother (will be updated in solve if auto_omega is True)
        self.smoother = smoother if smoother is not None else JacobiSolver(
            tolerance=1e-12,
            omega=smoother_omega if smoother_omega is not None else 0.8
        )
        
    def _store_vcycle_data(self, level, step, data):
        """Store data from V-cycle for later analysis."""
        shape = data.shape
        self.vcycle_data.append({
            'level': level,
            'step': step,
            'shape': shape,
            'min': np.min(data),
            'max': np.max(data),
            'mean': np.mean(data),
            'data': data.copy()  # Store the actual data for plotting
        })

        # Collect extra diagnostics for the presmooth step
        if step == 'after_presmooth':
            norm_val = float(np.linalg.norm(data))
            
            # Store diagnostics in dictionary by grid size
            key = f"grid_{shape[0]}x{shape[1]}"
            if key not in self.presmooth_diagnostics:
                self.presmooth_diagnostics[key] = []
                
            self.presmooth_diagnostics[key].append({
                'level': level,
                'iteration': self.current_iteration,
                'min': float(np.min(data)),
                'max': float(np.max(data)), 
                'mean': float(np.mean(data)),
                'norm': norm_val
            })
            
            # Print diagnostics immediately

    def plot_vcycle_results(self, output_path='debug_output/vcycle_analysis.pdf'):
        """Plot the V-cycle results from stored data."""
        # Convert data to DataFrame for easier manipulation
        df = pd.DataFrame(self.vcycle_data)
        
        # Define the order of steps we want to show
        step_order = [
            'initial_solution',
            'after_presmooth',
            'residual',
            'restricted_residual',
            'coarse_solution',
            'coarse_correction',
            'interpolated_correction',
            'before_correction',
            'after_correction',
            'after_postsmooth'
        ]
        
        # Get unique levels
        levels = sorted(df['level'].unique())
        
        # Create subplots for each level
        n_levels = len(levels)
        n_steps = len(step_order)
        fig, axes = plt.subplots(n_levels, n_steps, figsize=(5*n_steps, 5*n_levels))
        
        if n_levels == 1:
            axes = axes.reshape(1, -1)
        
        # Add a main title to the figure
        fig.suptitle('V-cycle Analysis - Steps in Chronological Order', fontsize=16, y=1.02)
        
        for i, level in enumerate(levels):
            for j, step in enumerate(step_order):
                ax = axes[i, j]
                
                # Try to get data for this level and step
                data_filter = df[(df['level'] == level) & (df['step'] == step)]
                if not data_filter.empty:
                    data = data_filter.iloc[0]
                    
                    # Plot the data using matshow for correct mathematical orientation
                    # (origin at bottom left, increasing values go up and right)
                    im = ax.matshow(data['data'], cmap='viridis')
                    ax.set_title(f'Level {level}\n{step}\nShape: {data["shape"]}', fontsize=10)
                    plt.colorbar(im, ax=ax)
                    
                    # Add statistics
                    stats_text = f'min: {data["min"]:.2e}\nmax: {data["max"]:.2e}\nmean: {data["mean"]:.2e}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No data', 
                           horizontalalignment='center', 
                           verticalalignment='center',
                           transform=ax.transAxes)
                    ax.set_title(f'Level {level}\n{step}')
                
                # Remove axis ticks for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
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
        
        # Perform multiple V-cycles until convergence or max iterations
        for k in range(self.max_iterations):
            # Update current iteration
            self.current_iteration = k + 1
            
            # Apply V-cycle
            grid_calculations = []  # Track grid sizes used
            x = self._v_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                             self.pre_smoothing, self.post_smoothing, grid_calculations)
                
            # Compute residual: r = b - Ax
            Ax = compute_Ap_product(x, nx, ny, dx, dy, rho, d_u, d_v)
            r = b - Ax
            
            # Calculate residual norm (excluding reference point)
            r_norm = np.linalg.norm(r, 2)
            b_norm = np.linalg.norm(b, 2)
            
            res_norm = r_norm / b_norm
            
            self.residual_history.append(res_norm)
            
            # Check convergence
            if res_norm < self.tolerance:
                print(f"Converged in {k+1} iterations, multigrid residual: {res_norm:.6e}")
                break
            
            #print(f"Iteration {k+1}, multigrid residual: {res_norm:.6e}")
            
            # Calculate and print convergence rate if we have at least two iterations
            if k > 0:
                conv_rate = self.residual_history[k] / self.residual_history[k-1] if self.residual_history[k-1] != 0 else 1.0
                #print(f"Convergence rate: {conv_rate:.4f}")
    
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
            u_reshaped = u.reshape((nx, ny), order='F') if u.ndim == 1 else u.copy()
            self._store_vcycle_data(level, 'coarse_solution', u_reshaped)
            return u  # Return solution on coarsest grid
            
        # Ensure u is flattened for consistent handling
        if u.ndim == 2:
            u = u.flatten('F')
            
        # Store initial solution    
        u_reshaped = u.reshape((nx, ny), order='F')
        self._store_vcycle_data(level, 'initial_solution', u_reshaped)
        
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        
        # Ensure u is flattened for consistent handling
        if u.ndim == 2:
            u = u.flatten('F')
            
        u_reshaped = u.reshape((nx, ny), order='F')
        self._store_vcycle_data(level, 'after_presmooth', u_reshaped)
        
        # Compute residual: r = f - Au
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        r_reshaped = r.reshape((nx, ny), order='F')
        self._store_vcycle_data(level, 'residual', r_reshaped)
        
        # Restrict residual to coarser grid
        r_coarse = restrict(r_reshaped) * 2
        
        self._store_vcycle_data(level, 'restricted_residual', r_coarse)
        
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
        
        # Print diagnostics about original vs coarse coefficients
        if level == 0:  # Only print for the first level transition
            d_u_fine_mean = np.mean(d_u[d_u > 0])
            d_v_fine_mean = np.mean(d_v[d_v > 0])
            d_u_coarse_mean = np.mean(d_u_coarse[d_u_coarse > 0])
            d_v_coarse_mean = np.mean(d_v_coarse[d_v_coarse > 0])
            ratio_u = d_u_coarse_mean / d_u_fine_mean
            ratio_v = d_v_coarse_mean / d_v_fine_mean

        # Recursive V-cycle on coarse grid
        # Make sure to flatten the coarse grid data
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse = self._v_cycle(u=np.zeros_like(r_coarse_flat), f=r_coarse_flat, 
                                mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                omega=omega, pre_smoothing=pre_smoothing, 
                                post_smoothing=post_smoothing, grid_calculations=grid_calculations, 
                                level=level+1)
        
        # Ensure e_coarse is in the right shape for storing data
        if e_coarse.ndim == 1:
            e_coarse_2d = e_coarse.reshape((coarse_grid_size, coarse_grid_size), order='F')
        else:
            e_coarse_2d = e_coarse.copy()
            
        self._store_vcycle_data(level, 'coarse_correction', e_coarse_2d)
        
        # Interpolate error to fine grid
        # We need to keep the solution at the correct scale, so no need to scale the interpolation
        # Since we already scaled the residual and coefficients, the solution on the coarse grid
        # should already be at the right scale
        e_interpolated = interpolate(e_coarse_2d, nx)
        
        self._store_vcycle_data(level, 'interpolated_correction', e_interpolated)
        
        # Store solution before correction
        u_reshaped = u.reshape((nx, ny), order='F')
        self._store_vcycle_data(level, 'before_correction', u_reshaped)
        
        # Apply correction - ensure both operands are 1D with the same shape
        e_interpolated_flat = e_interpolated.flatten('F')
        u += e_interpolated_flat

        # Show the current state
        u_reshaped = u.reshape((nx, ny), order='F')
        self._store_vcycle_data(level, 'after_correction', u_reshaped)
        
        # Post-smoothing on fine grid
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
        # Ensure u is flattened for consistent handling
        if u.ndim == 2:
            u = u.flatten('F')
            
        u_reshaped = u.reshape((nx, ny), order='F')
        self._store_vcycle_data(level, 'after_postsmooth', u_reshaped)
        
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