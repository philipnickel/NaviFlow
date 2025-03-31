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
        
    def _store_vcycle_data(self, level, step, data):
        """Store data from V-cycle for later analysis."""
        # Skip data storage if disabled
        if not self.store_vcycle_data:
            return
            
        shape = data.shape
        
        # Only store statistics by default, not the full data
        vcycle_entry = {
            'level': level,
            'step': step,
            'shape': shape,
            'min': np.min(data),
            'max': np.max(data),
            'mean': np.mean(data)
        }
        
        # Only store actual data if needed for visualization
        if self.store_vcycle_data:
            vcycle_entry['data'] = data.copy()  # Store the actual data for plotting
            
        self.vcycle_data.append(vcycle_entry)

        # Collect extra diagnostics for the presmooth step
        if step == 'after_presmooth':
            norm_val = float(np.linalg.norm(data))
            
            # Store diagnostics in dictionary by grid size
            key = f"grid_{shape[0]}x{shape[1]}"
            if key not in self.presmooth_diagnostics:
                self.presmooth_diagnostics[key] = []
                
            # Limit the number of entries stored
            if len(self.presmooth_diagnostics[key]) >= self.max_diagnostic_history:
                # Remove oldest entry
                self.presmooth_diagnostics[key].pop(0)
                
            # Store basic statistics only (no data copy)
            self.presmooth_diagnostics[key].append({
                'level': level,
                'iteration': self.current_iteration,
                'min': float(np.min(data)),
                'max': float(np.max(data)), 
                'mean': float(np.mean(data)),
                'norm': norm_val
            })

    def cleanup_memory(self):
        """Clear stored data to free memory"""
        self.vcycle_data.clear()
        self.presmooth_diagnostics.clear()
        self.residual_history.clear()

    def plot_vcycle_results(self, output_path='debug_output/vcycle_analysis.pdf'):
        """Plot the V-cycle results from stored data."""
        if not self.store_vcycle_data or not self.vcycle_data:
            print("No V-cycle data available for plotting")
            return
            
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
                    if 'data' in data:
                        im = ax.matshow(data['data'], cmap='viridis')
                        ax.set_title(f'Level {level}\n{step}\nShape: {data["shape"]}', fontsize=10)
                        plt.colorbar(im, ax=ax)
                        
                        # Add statistics
                        stats_text = f'min: {data["min"]:.2e}\nmax: {data["max"]:.2e}\nmean: {data["mean"]:.2e}'
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    else:
                        ax.text(0.5, 0.5, 'Stats only (no data stored)', 
                               horizontalalignment='center', 
                               verticalalignment='center',
                               transform=ax.transAxes)
                        ax.set_title(f'Level {level}\n{step}')
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
        plt.close(fig)  # Explicitly close the figure to release memory
        
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
        self.cleanup_memory()
        self.current_iteration = 0
        
        # Get right-hand side of pressure correction equation
        b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Initial guess
        x = np.zeros_like(b)
        
        # Pre-allocate arrays for residual calculation to avoid creating new ones
        Ax = np.zeros_like(b)
        r = np.zeros_like(b)
        
        # Perform multiple V-cycles until convergence or max iterations
        for k in range(self.max_iterations):
            # Update current iteration
            self.current_iteration = k + 1
            
            # ensure B.C. are applied
            x = x.reshape((nx, ny), order='F')
            #x = self.apply_pressure_boundary_conditions(x)
            x = x.flatten('F')

            grid_calculations = []  # Track grid sizes used
            x = self._v_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                             self.pre_smoothing, self.post_smoothing, grid_calculations)
                             
            # Compute residual: r = b - Ax
            compute_Ap_product(x, nx, ny, dx, dy, rho, d_u, d_v, out=Ax)  # Use pre-allocated array
            np.subtract(b, Ax, out=r)  # In-place subtraction
            
            # Calculate residual norm (excluding reference point)
            r_norm = np.linalg.norm(r, 2)
            b_norm = np.linalg.norm(b, 2)
            
            res_norm = r_norm / b_norm
            
            self.residual_history.append(res_norm)
            
            # Check convergence
            if res_norm < self.tolerance:
                print(f"Converged in {k+1} iterations, multigrid residual: {res_norm:.6e}")
                break
            print(f"Residual: {res_norm:.6e}")
            
            # Calculate and print convergence rate if we have at least two iterations
            if k > 0:
                conv_rate = self.residual_history[k] / self.residual_history[k-1] if self.residual_history[k-1] != 0 else 1.0
    
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
        
        # Clear any remaining vcycle data to free memory
        if not self.store_vcycle_data:
            self.cleanup_memory()
            
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
        
    def _get_cached_mesh(self, nx, ny, length, height):
        """Get a cached mesh or create a new one if not in cache"""
        key = f"{nx}x{ny}"
        if key not in self.mesh_cache:
            self.mesh_cache[key] = StructuredMesh(nx=nx, ny=ny, length=length, height=height)
        return self.mesh_cache[key]

    def _v_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                grid_calculations, level=0):
        """
        Performs one V-cycle of the multigrid method.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # If we're at the coarsest grid, solve directly
        if nx <= 3:
            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            self._store_vcycle_data(level, 'coarse_solution', u.reshape((nx, ny), order='F'))
            return u  # Return solution on coarsest grid
            
        # Store initial solution
        self._store_vcycle_data(level, 'initial_solution', u.reshape((nx, ny), order='F'))

        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        self._store_vcycle_data(level, 'after_presmooth', u.reshape((nx, ny), order='F'))
        
        # Compute residual: r = f - Au
        # Pre-allocate residual arrays
        r_flat = np.zeros_like(f)
        Au = np.zeros_like(f)
        
        # Compute residual: r = f - Au using in-place operations
        compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v, out=Au)
        np.subtract(f, Au, out=r_flat)  # In-place subtraction
        
        r = r_flat.reshape((nx, ny), order='F')
        self._store_vcycle_data(level, 'residual', r)
        
        # Calculate grid properties
        dx_h = dx
        dy_h = dy
        dx_2h = dx_h * 2  # Coarse grid spacing (double the fine grid)
        h_ratio = dx_2h / dx_h  # Ratio of coarse to fine grid spacing
        
        # Calculate scaling factor for proper grid-dependent coefficient scaling
        # For a Poisson equation, we need to scale by h^2 ratio
        coeff_scale_factor = h_ratio * h_ratio  # (dx_2h/dx_h)^2
        
        # Restrict the residual without scaling - scaling will be handled in the coefficients
        r_coarse = restrict(r) #/ 4
        #r_coarse = self.apply_pressure_boundary_conditions(r_coarse)

        self._store_vcycle_data(level, 'restricted_residual', r_coarse)
        
        # Size of coarse grid
        coarse_grid_size = r_coarse.shape[0]
        
        # Get cached mesh or create new one
        mesh_coarse = self._get_cached_mesh(
            nx=coarse_grid_size, 
            ny=coarse_grid_size, 
            length=mesh.length, 
            height=mesh.height
        )
        
        # Get cell sizes for coarse grid
        dx_coarse, dy_coarse = mesh_coarse.get_cell_sizes()
        
        # Pre-allocate d_u and d_v arrays for coarse grid
        d_u_coarse = np.zeros((coarse_grid_size+1, coarse_grid_size))
        d_v_coarse = np.zeros((coarse_grid_size, coarse_grid_size+1))
        
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
        # Pre-allocate zero array for e_coarse
        e_coarse = np.zeros_like(r_coarse.flatten('F'))
        e_coarse = self._v_cycle(u=e_coarse, f=r_coarse.flatten('F'), 
                                mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                omega=omega, pre_smoothing=pre_smoothing, 
                                post_smoothing=post_smoothing, grid_calculations=grid_calculations, 
                                level=level+1)
        
        e_coarse_2d = e_coarse.reshape((coarse_grid_size, coarse_grid_size), order='F')
        self._store_vcycle_data(level, 'coarse_correction', e_coarse_2d)
        
        # Interpolate error to fine grid
        e_interpolated = interpolate(e_coarse_2d, nx)
        #e_interpolated = self.apply_pressure_boundary_conditions(e_interpolated)

        self._store_vcycle_data(level, 'interpolated_correction', e_interpolated.reshape((nx, ny), order='F'))
        
        # Store solution before correction
        self._store_vcycle_data(level, 'before_correction', u.reshape((nx, ny), order='F'))
        
        # Apply correction in-place
        u_reshaped = u.reshape((nx, ny), order='F')
        u_reshaped += e_interpolated
        u = u_reshaped.flatten('F')

        self._store_vcycle_data(level, 'after_correction', u.reshape((nx, ny), order='F'))

        # Post-smoothing on fine grid
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)

        self._store_vcycle_data(level, 'after_postsmooth', u.reshape((nx, ny), order='F'))

        # Apply boundary conditions to the solution 
        u = u.reshape((nx, ny), order='F')
        #u = self.apply_pressure_boundary_conditions(u)
        u = u.flatten('F')
        
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
