"""
Matrix-free geometric multigrid solver for pressure correction equation.
"""

import numpy as np
import math
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product
from .jacobi import JacobiSolver

class MultiGridSolver(PressureSolver):
    """
    Matrix-free geometric multigrid solver for pressure correction equation.
    
    This solver uses a geometric multigrid approach with V-cycles to solve
    the pressure correction equation without explicitly forming matrices.
    It requires grid sizes to be 2^k-1 (e.g., 3, 7, 15, 31, 63, 127, etc.)
    to ensure proper coarsening down to a 1x1 grid.
    
    The solver supports both V-cycles and F-cycles. F-cycles provide better
    convergence in many cases by combining aspects of V-cycles and W-cycles.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=100, 
                 pre_smoothing=3, post_smoothing=3,
                 smoother_iterations=2, smoother_omega=0.8,
                 smoother=None, cycle_type='v'):
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
        smoother_iterations : int, optional
            Number of iterations for the smoother
        smoother_omega : float, optional
            Relaxation factor for the smoother (0.8 is often good for Jacobi)
        smoother : PressureSolver, optional
            External smoother to use (if None, will use internal Jacobi smoother)
        cycle_type : str, optional
            Type of multigrid cycle to use ('v' for V-cycle, 'f' for F-cycle)
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.pre_smoothing = pre_smoothing
        self.post_smoothing = post_smoothing
        self.smoother_iterations = smoother_iterations
        self.smoother_omega = smoother_omega
        self.residual_history = []
        self.cycle_type = cycle_type.lower()
        
        # Initialize smoother
        self.smoother = smoother if smoother is not None else JacobiSolver(
            tolerance=1e-4,  # Not used for fixed iterations
            max_iterations=1000,  # Not used for fixed iterations
            omega=smoother_omega
        )
        
    def _is_valid_grid_size(self, n):
        """
        Check if the grid size is valid (2^k-1).
        
        Parameters:
        -----------
        n : int
            Grid size to check
            
        Returns:
        --------
        bool
            True if the grid size is valid, False otherwise
        """
        # Check if n is of the form 2^k-1
        k = math.log2(n + 1)
        return k.is_integer() and k >= 1
        
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
        
        # Check if grid sizes are valid (2^k-1)
        if not self._is_valid_grid_size(nx) or not self._is_valid_grid_size(ny):
            raise ValueError(f"Grid size must be 2^k-1 (e.g., 3, 7, 15, 31, 63, 127, etc.). Got {nx}x{ny}.")
        
        # Only square grids are supported for now
        if nx != ny:
            raise ValueError(f"Only square grids are supported. Got {nx}x{ny}.")
        
        
        dx, dy = mesh.get_cell_sizes()
        rho = 1.0  # This should come from fluid properties
        
        # Reset residual history
        self.residual_history = []
        
        # Get right-hand side of pressure correction equation
        b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Initial guess
        x = np.zeros_like(b)
        
        # Store parameters needed for matrix-free operations
        params = {
            'nx': nx,
            'ny': ny,
            'dx': dx,
            'dy': dy,
            'rho': rho,
            'd_u': d_u,
            'd_v': d_v
        }
        
        # If using F-cycle, start with one F-cycle
        if self.cycle_type == 'f':
            grid_calculations = []  # Track grid sizes used
            x = self._f_cycle(x, b, params, grid_calculations)
            
            # Compute initial residual after F-cycle
            Ax = compute_Ap_product(x, nx, ny, dx, dy, rho, d_u, d_v)
            r = b - Ax
            r_norm = np.linalg.norm(r[1:])
            b_norm = np.linalg.norm(b[1:])
            
            # Avoid division by zero
            if b_norm < 1e-15:
                res_norm = r_norm
            else:
                res_norm = r_norm / b_norm
                
            self.residual_history.append(res_norm)
            
            # Check if we've already converged after the F-cycle
            if res_norm < self.tolerance:
                # Reshape to 2D
                p_prime = x.reshape((nx, ny), order='F')
                return p_prime
        
        # Perform multiple V-cycles until convergence or max iterations
        for k in range(self.max_iterations):
            # Apply V-cycle
            grid_calculations = []  # Track grid sizes used
            x = self._v_cycle(x, b, params, grid_calculations)
            
            # Compute residual: r = b - Ax
            Ax = compute_Ap_product(x, nx, ny, dx, dy, rho, d_u, d_v)
            r = b - Ax
            
            # Calculate residual norm (excluding reference point)
            r_norm = np.linalg.norm(r[1:])
            b_norm = np.linalg.norm(b[1:])
            
            # Avoid division by zero
            if b_norm < 1e-15:
                res_norm = r_norm
            else:
                res_norm = r_norm / b_norm
                
            self.residual_history.append(res_norm)
            
            # Check convergence
            if res_norm < self.tolerance:
                break
        
        # Reshape to 2D
        p_prime = x.reshape((nx, ny), order='F')
        return p_prime
    
    def _f_cycle(self, u, f, params, grid_calculations):
        """
        Performs one F-cycle of the multigrid method.
        
        An F-cycle is a more powerful cycle that combines aspects of V-cycles and W-cycles.
        It starts by recursively going to the coarsest grid, then on the way up, it performs
        a V-cycle at each level before continuing to the next finer level.
        
        Parameters:
        -----------
        u : ndarray
            Initial guess for the solution (1D)
        f : ndarray
            Right-hand side (1D)
        params : dict
            Parameters for matrix-free operations
        grid_calculations : list
            List to track grid sizes used
            
        Returns:
        --------
        u : ndarray
            Updated solution after one F-cycle (1D)
        """
        nx, ny = params['nx'], params['ny']
        dx, dy = params['dx'], params['dy']
        rho = params['rho']
        d_u, d_v = params['d_u'], params['d_v']
        
        # Track grid size
        grid_calculations.append(nx)
        
        # Base case: 1x1 grid (coarsest level)
        if nx == 1 and ny == 1:
            return self._solve_directly(f, params)
        
        # 1. Pre-smoothing
        u = self._smooth(u, f, params, num_iterations=self.pre_smoothing)
        
        # 2. Compute residual: r = f - Au
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        # 3. Restrict residual to coarser grid using full weighting
        r_2d = r.reshape((nx, ny), order='F')
        r_coarse = self._restrict(r_2d)
        mc = r_coarse.shape[0]  # Size of coarse grid
        
        # 4. Create new parameters for coarse grid
        coarse_params = params.copy()
        coarse_params['nx'] = mc
        coarse_params['ny'] = mc
        coarse_params['dx'] = dx * 2
        coarse_params['dy'] = dy * 2
        
        # 5. Restrict d_u and d_v to coarser grid using improved method
        d_u_coarse, d_v_coarse = self._restrict_coefficients(d_u, d_v, nx, ny)
        coarse_params['d_u'] = d_u_coarse
        coarse_params['d_v'] = d_v_coarse
        
        # 6. Recursive call to solve on coarser grid (F-cycle)
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse = np.zeros_like(r_coarse_flat)
        e_coarse = self._f_cycle(e_coarse, r_coarse_flat, coarse_params, grid_calculations)
        
        # 7. Perform a V-cycle at this level before going up
        e_coarse = self._v_cycle(e_coarse, r_coarse_flat, coarse_params, grid_calculations)
        
        # 8. Prolongate error to fine grid
        e_coarse_2d = e_coarse.reshape((mc, mc), order='F')
        e_2d = self._interpolate(e_coarse_2d, nx)
        e = e_2d.flatten('F')
        
        # 9. Update solution
        u = u + e
        
        # 10. Post-smoothing
        u = self._smooth(u, f, params, num_iterations=self.post_smoothing)
        
        return u
    
    def _restrict(self, fine_grid):
        """
        Restricts a fine grid to a coarse grid using full weighting.
        
        Parameters:
        -----------
        fine_grid : ndarray
            The input fine grid to be restricted (2D)
            
        Returns:
        --------
        ndarray
            The restricted grid (2D)
        """
        # Reshape to 2D if needed
        if fine_grid.ndim == 1:
            m = int(np.sqrt(fine_grid.size))
            fine_grid = fine_grid.reshape((m, m), order='F')
            
        m = fine_grid.shape[0]
        mc = (m + 1) // 2 - 1  # Size of coarse grid
        
        # Create coarse grid
        coarse_grid = np.zeros((mc, mc))
        
        # Vectorized full weighting restriction
        # Create indices for the fine grid points that correspond to coarse grid points
        i_fine = np.arange(1, m, 2)[:mc]
        j_fine = np.arange(1, m, 2)[:mc]
        I_fine, J_fine = np.meshgrid(i_fine, j_fine, indexing='ij')
        
        # Center points (weight 1/4)
        center_points = fine_grid[I_fine, J_fine]
        
        # Adjacent points (weight 1/8)
        # Create masks for valid adjacent indices
        valid_im1 = I_fine > 0
        valid_ip1 = I_fine < m-1
        valid_jm1 = J_fine > 0
        valid_jp1 = J_fine < m-1
        
        # Initialize adjacent sum with zeros
        adjacent_sum = np.zeros((mc, mc))
        
        # Add valid adjacent points
        adjacent_count = np.zeros((mc, mc))
        
        # Left points
        mask = valid_im1
        adjacent_sum[mask] += fine_grid[I_fine[mask]-1, J_fine[mask]]
        adjacent_count[mask] += 1
        
        # Right points
        mask = valid_ip1
        adjacent_sum[mask] += fine_grid[I_fine[mask]+1, J_fine[mask]]
        adjacent_count[mask] += 1
        
        # Bottom points
        mask = valid_jm1
        adjacent_sum[mask] += fine_grid[I_fine[mask], J_fine[mask]-1]
        adjacent_count[mask] += 1
        
        # Top points
        mask = valid_jp1
        adjacent_sum[mask] += fine_grid[I_fine[mask], J_fine[mask]+1]
        adjacent_count[mask] += 1
        
        # Diagonal points (weight 1/16)
        # Initialize diagonal sum with zeros
        diagonal_sum = np.zeros((mc, mc))
        diagonal_count = np.zeros((mc, mc))
        
        # Bottom-left points
        mask = np.logical_and(valid_im1, valid_jm1)
        diagonal_sum[mask] += fine_grid[I_fine[mask]-1, J_fine[mask]-1]
        diagonal_count[mask] += 1
        
        # Bottom-right points
        mask = np.logical_and(valid_ip1, valid_jm1)
        diagonal_sum[mask] += fine_grid[I_fine[mask]+1, J_fine[mask]-1]
        diagonal_count[mask] += 1
        
        # Top-left points
        mask = np.logical_and(valid_im1, valid_jp1)
        diagonal_sum[mask] += fine_grid[I_fine[mask]-1, J_fine[mask]+1]
        diagonal_count[mask] += 1
        
        # Top-right points
        mask = np.logical_and(valid_ip1, valid_jp1)
        diagonal_sum[mask] += fine_grid[I_fine[mask]+1, J_fine[mask]+1]
        diagonal_count[mask] += 1
        
        # Avoid division by zero
        adjacent_count[adjacent_count == 0] = 1
        diagonal_count[diagonal_count == 0] = 1
        
        # Combine with weights
        coarse_grid = 0.25 * center_points + 0.125 * (adjacent_sum / adjacent_count) + 0.0625 * (diagonal_sum / diagonal_count)
        
        return coarse_grid
    
    def _interpolate(self, coarse_grid, m):
        """
        Interpolates a coarse grid to a fine grid using bilinear interpolation.
        
        Parameters:
        -----------
        coarse_grid : ndarray
            The input coarse grid to be interpolated (2D)
        m : int
            Size of the target fine grid (m x m)
            
        Returns:
        --------
        ndarray
            The interpolated fine grid (2D)
        """
        # Reshape to 2D if needed
        if coarse_grid.ndim == 1:
            mc = int(np.sqrt(coarse_grid.size))
            coarse_grid = coarse_grid.reshape((mc, mc), order='F')
            
        # Get coarse grid dimensions
        mc = coarse_grid.shape[0]
        
        # Create fine grid
        fine_grid = np.zeros((m, m))
        
        # Handle edge cases for small grids
        if m <= 3:
            # Direct injection for coincident points
            i_coarse = np.arange(mc)
            j_coarse = np.arange(mc)
            I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
            
            # Calculate fine grid indices
            I_fine = 2 * I_coarse + 1
            J_fine = 2 * J_coarse + 1
            
            # Filter valid indices
            mask = np.logical_and(I_fine < m, J_fine < m)
            fine_grid[I_fine[mask], J_fine[mask]] = coarse_grid[I_coarse[mask], J_coarse[mask]]
            
            return fine_grid
        
        # Direct injection for coincident points
        i_coarse = np.arange(mc)
        j_coarse = np.arange(mc)
        I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
        
        # Calculate fine grid indices
        I_fine = 2 * I_coarse + 1
        J_fine = 2 * J_coarse + 1
        
        # Filter valid indices
        mask = np.logical_and(I_fine < m, J_fine < m)
        fine_grid[I_fine[mask], J_fine[mask]] = coarse_grid[I_coarse[mask], J_coarse[mask]]
        
        # Horizontal interpolation (odd rows, even columns)
        i_coarse = np.arange(mc)
        j_coarse = np.arange(mc-1)
        I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
        
        I_fine = 2 * I_coarse + 1
        J_fine = 2 * J_coarse + 2
        
        mask = np.logical_and(I_fine < m, J_fine < m)
        fine_grid[I_fine[mask], J_fine[mask]] = 0.5 * (
            coarse_grid[I_coarse[mask], J_coarse[mask]] + 
            coarse_grid[I_coarse[mask], J_coarse[mask]+1]
        )
        
        # Vertical interpolation (even rows, odd columns)
        i_coarse = np.arange(mc-1)
        j_coarse = np.arange(mc)
        I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
        
        I_fine = 2 * I_coarse + 2
        J_fine = 2 * J_coarse + 1
        
        mask = np.logical_and(I_fine < m, J_fine < m)
        fine_grid[I_fine[mask], J_fine[mask]] = 0.5 * (
            coarse_grid[I_coarse[mask], J_coarse[mask]] + 
            coarse_grid[I_coarse[mask]+1, J_coarse[mask]]
        )
        
        # Diagonal interpolation (even rows, even columns)
        i_coarse = np.arange(mc-1)
        j_coarse = np.arange(mc-1)
        I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
        
        I_fine = 2 * I_coarse + 2
        J_fine = 2 * J_coarse + 2
        
        mask = np.logical_and(I_fine < m, J_fine < m)
        fine_grid[I_fine[mask], J_fine[mask]] = 0.25 * (
            coarse_grid[I_coarse[mask], J_coarse[mask]] + 
            coarse_grid[I_coarse[mask], J_coarse[mask]+1] + 
            coarse_grid[I_coarse[mask]+1, J_coarse[mask]] + 
            coarse_grid[I_coarse[mask]+1, J_coarse[mask]+1]
        )
        
        return fine_grid
    
    def _smooth(self, u, f, params, num_iterations=None):
        """
        Apply smoother for a specified number of iterations.
        
        Parameters:
        -----------
        u : ndarray
            Current solution (1D)
        f : ndarray
            Right-hand side (1D)
        params : dict
            Parameters for matrix-free operations
        num_iterations : int, optional
            Number of smoothing iterations
            
        Returns:
        --------
        u : ndarray
            Smoothed solution (1D)
        """
        if num_iterations is None:
            num_iterations = self.smoother_iterations
            
        nx, ny = params['nx'], params['ny']
        dx, dy = params['dx'], params['dy']
        rho = params['rho']
        d_u, d_v = params['d_u'], params['d_v']
        
        return self.smoother.perform_iteration(u, f, nx, ny, dx, dy, rho, d_u, d_v, num_iterations)
  
    def _solve_directly(self, f, params):
        """
        Solve the 1x1 system directly.
        
        Parameters:
        -----------
        f : ndarray
            Right-hand side (1D)
        params : dict
            Parameters for matrix-free operations
            
        Returns:
        --------
        u : ndarray
            Solution (1D)
        """
        # For a 1x1 grid, the solution is simply f/aP
        # But since we're enforcing the reference pressure point to be 0,
        # the solution is just 0
        return np.zeros_like(f)
    
    def _restrict_coefficients(self, d_u, d_v, nx, ny):
        """
        Restrict the momentum equation coefficients to a coarser grid.
        
        Parameters:
        -----------
        d_u, d_v : ndarray
            Momentum equation coefficients on the fine grid
        nx, ny : int
            Dimensions of the fine grid
            
        Returns:
        --------
        d_u_coarse, d_v_coarse : ndarray
            Momentum equation coefficients on the coarse grid
        """
        # Calculate coarse grid dimensions
        nx_coarse = (nx + 1) // 2 - 1
        ny_coarse = (ny + 1) // 2 - 1
        
        # Initialize coarse grid coefficients
        d_u_coarse = np.zeros((nx_coarse + 1, ny_coarse))
        d_v_coarse = np.zeros((nx_coarse, ny_coarse + 1))
        
        # Vectorized restriction for d_u (staggered in x-direction)
        # Create indices for the fine grid points
        i_fine = np.arange(0, min(2*nx_coarse+1, d_u.shape[0]), 2)
        j_fine = np.arange(1, min(2*ny_coarse+1, d_u.shape[1]), 2)
        I_fine, J_fine = np.meshgrid(i_fine, j_fine, indexing='ij')
        
        # Create masks for valid adjacent indices
        valid_jm1 = J_fine > 0
        valid_jp1 = J_fine < d_u.shape[1]-1
        
        # Initialize sum and count arrays
        d_u_sum = np.zeros((nx_coarse + 1, ny_coarse))
        d_u_count = np.ones((nx_coarse + 1, ny_coarse))  # Start with 1 to avoid division by zero
        
        # Add center points
        d_u_sum += d_u[I_fine[:nx_coarse+1, :ny_coarse], J_fine[:nx_coarse+1, :ny_coarse]]
        
        # Add points below if available
        mask = valid_jm1[:nx_coarse+1, :ny_coarse]
        if np.any(mask):
            d_u_sum[mask] += d_u[I_fine[:nx_coarse+1, :ny_coarse][mask], J_fine[:nx_coarse+1, :ny_coarse][mask]-1]
            d_u_count[mask] += 1
        
        # Add points above if available
        mask = valid_jp1[:nx_coarse+1, :ny_coarse]
        if np.any(mask):
            d_u_sum[mask] += d_u[I_fine[:nx_coarse+1, :ny_coarse][mask], J_fine[:nx_coarse+1, :ny_coarse][mask]+1]
            d_u_count[mask] += 1
        
        # Average
        d_u_coarse = d_u_sum / d_u_count
        
        # Vectorized restriction for d_v (staggered in y-direction)
        # Create indices for the fine grid points
        i_fine = np.arange(1, min(2*nx_coarse+1, d_v.shape[0]), 2)
        j_fine = np.arange(0, min(2*ny_coarse+2, d_v.shape[1]), 2)
        I_fine, J_fine = np.meshgrid(i_fine, j_fine, indexing='ij')
        
        # Create masks for valid adjacent indices
        valid_im1 = I_fine > 0
        valid_ip1 = I_fine < d_v.shape[0]-1
        
        # Initialize sum and count arrays
        d_v_sum = np.zeros((nx_coarse, ny_coarse + 1))
        d_v_count = np.ones((nx_coarse, ny_coarse + 1))  # Start with 1 to avoid division by zero
        
        # Add center points
        d_v_sum += d_v[I_fine[:nx_coarse, :ny_coarse+1], J_fine[:nx_coarse, :ny_coarse+1]]
        
        # Add points to the left if available
        mask = valid_im1[:nx_coarse, :ny_coarse+1]
        if np.any(mask):
            d_v_sum[mask] += d_v[I_fine[:nx_coarse, :ny_coarse+1][mask]-1, J_fine[:nx_coarse, :ny_coarse+1][mask]]
            d_v_count[mask] += 1
        
        # Add points to the right if available
        mask = valid_ip1[:nx_coarse, :ny_coarse+1]
        if np.any(mask):
            d_v_sum[mask] += d_v[I_fine[:nx_coarse, :ny_coarse+1][mask]+1, J_fine[:nx_coarse, :ny_coarse+1][mask]]
            d_v_count[mask] += 1
        
        # Average
        d_v_coarse = d_v_sum / d_v_count
        
        return d_u_coarse, d_v_coarse
    
    def _v_cycle(self, u, f, params, grid_calculations):
        """
        Performs one V-cycle of the multigrid method.
        
        Parameters:
        -----------
        u : ndarray
            Initial guess for the solution (1D)
        f : ndarray
            Right-hand side (1D)
        params : dict
            Parameters for matrix-free operations
        grid_calculations : list
            List to track grid sizes used
            
        Returns:
        --------
        u : ndarray
            Updated solution after one V-cycle (1D)
        """
        nx, ny = params['nx'], params['ny']
        dx, dy = params['dx'], params['dy']
        rho = params['rho']
        d_u, d_v = params['d_u'], params['d_v']
        
        # Track grid size
        grid_calculations.append(nx)
        
        # Base case: 1x1 grid (coarsest level)
        if nx == 1 and ny == 1:
            return self._solve_directly(f, params)
        
        # 1. Pre-smoothing
        u = self._smooth(u, f, params, num_iterations=self.pre_smoothing)
        
        # 2. Compute residual: r = f - Au
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        # 3. Restrict residual to coarser grid using full weighting
        r_2d = r.reshape((nx, ny), order='F')
        r_coarse = self._restrict(r_2d)
        mc = r_coarse.shape[0]  # Size of coarse grid
        
        # 4. Create new parameters for coarse grid
        coarse_params = params.copy()
        coarse_params['nx'] = mc
        coarse_params['ny'] = mc
        coarse_params['dx'] = dx * 2
        coarse_params['dy'] = dy * 2
        
        # 5. Restrict d_u and d_v to coarser grid using improved method
        d_u_coarse, d_v_coarse = self._restrict_coefficients(d_u, d_v, nx, ny)
        coarse_params['d_u'] = d_u_coarse
        coarse_params['d_v'] = d_v_coarse
        
        # 6. Recursive call to solve on coarser grid
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse = np.zeros_like(r_coarse_flat)
        e_coarse = self._v_cycle(e_coarse, r_coarse_flat, coarse_params, grid_calculations)
        
        # 7. Prolongate error to fine grid
        e_coarse_2d = e_coarse.reshape((mc, mc), order='F')
        e_2d = self._interpolate(e_coarse_2d, nx)
        e = e_2d.flatten('F')
        
        # 8. Update solution
        u = u + e
        
        # 9. Post-smoothing
        u = self._smooth(u, f, params, num_iterations=self.post_smoothing)
        
        return u
