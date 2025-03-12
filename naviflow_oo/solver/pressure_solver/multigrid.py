"""
Geometric multigrid solver for pressure correction equation.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.coeff_matrix import get_coeff_mat
from .helpers.matrix_free import compute_Ap_product

class MultiGridSolver(PressureSolver):
    """
    Geometric multigrid solver for pressure correction equation.
    
    This solver uses a geometric multigrid approach with V-cycles to solve
    the pressure correction equation. It uses the Jacobi method as a smoother.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=100, num_levels=3, 
                 pre_smoothing=2, post_smoothing=2, coarsest_size=5,
                 smoother_iterations=5, smoother_omega=0.8):
        """
        Initialize the multigrid solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance
        max_iterations : int, optional
            Maximum number of V-cycles
        num_levels : int, optional
            Number of grid levels (automatically determined if None)
        pre_smoothing : int, optional
            Number of pre-smoothing steps
        post_smoothing : int, optional
            Number of post-smoothing steps
        coarsest_size : int, optional
            Minimum size of the coarsest grid
        smoother_iterations : int, optional
            Number of iterations for the smoother
        smoother_omega : float, optional
            Relaxation factor for the smoother (0.8 is often good for Jacobi)
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.num_levels = num_levels
        self.pre_smoothing = pre_smoothing
        self.post_smoothing = post_smoothing
        self.coarsest_size = coarsest_size
        self.smoother_iterations = smoother_iterations
        self.smoother_omega = smoother_omega
        self.residual_history = []
        
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using the multigrid method.
        
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
        
        # Reset residual history
        self.residual_history = []
        
        # Get right-hand side of pressure correction equation
        rhs = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Determine number of levels if not specified
        if self.num_levels is None:
            self.num_levels = self._determine_num_levels(nx, ny)
        
        # Initial guess
        x = np.zeros((nx, ny), order='F')
        x_flat = x.reshape(-1, order='F')
        
        # V-cycle iteration
        for k in range(self.max_iterations):
            # Compute residual using matrix-free approach
            r_flat = rhs - compute_Ap_product(x_flat, nx, ny, dx, dy, rho, d_u, d_v)
            res_norm = np.linalg.norm(r_flat) / np.linalg.norm(rhs)
            self.residual_history.append(res_norm)
            
            # Check convergence
            if res_norm < self.tolerance:
                print(f"Multigrid converged in {k+1} iterations, residual: {res_norm:.6e}")
                break
            
            # Reshape residual to 2D
            r = r_flat.reshape((nx, ny), order='F')
            
            # Apply V-cycle
            correction = self._v_cycle(x, r, nx, ny, dx, dy, rho, d_u, d_v)
            
            # Update solution
            x += correction
            x_flat = x.reshape(-1, order='F')
        
        else:
            print(f"Multigrid did not converge in {self.max_iterations} iterations, "
                 f"final residual: {res_norm:.6e}")
        
        return x
    
    def _determine_num_levels(self, nx, ny):
        """Determine the optimal number of levels based on grid size."""
        min_dim = min(nx, ny)
        num_levels = 1
        while min_dim > self.coarsest_size:
            min_dim = min_dim // 2
            num_levels += 1
        return num_levels
    
    def _v_cycle(self, x, r, nx, ny, dx, dy, rho, d_u, d_v):
        """
        Perform one V-cycle of the multigrid method.
        
        Parameters:
        -----------
        x : ndarray
            Current solution (2D)
        r : ndarray
            Current residual (2D)
        nx, ny : int
            Grid dimensions
        dx, dy : float
            Grid spacing
        rho : float
            Density
        d_u, d_v : ndarray
            Momentum equation coefficients
            
        Returns:
        --------
        correction : ndarray
            Correction to the solution (2D)
        """
        # Base case: coarsest grid
        if nx <= self.coarsest_size or ny <= self.coarsest_size:
            # Solve directly on coarsest grid
            A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
            r_flat = r.reshape(-1, order='F')
            correction_flat = spsolve(A, r_flat)
            return correction_flat.reshape((nx, ny), order='F')
        
        # Pre-smoothing
        x = self._smooth(x, r, nx, ny, dx, dy, rho, d_u, d_v, self.pre_smoothing)
        
        # Compute residual using matrix-free approach
        x_flat = x.reshape(-1, order='F')
        r_flat = r.reshape(-1, order='F')
        new_r_flat = r_flat - compute_Ap_product(x_flat, nx, ny, dx, dy, rho, d_u, d_v)
        new_r = new_r_flat.reshape((nx, ny), order='F')
        
        # Restrict residual to coarser grid
        nx_coarse = nx // 2
        ny_coarse = ny // 2
        dx_coarse = dx * 2
        dy_coarse = dy * 2
        
        # Create coarse grid residual using vectorized restriction
        r_coarse = self._restrict(new_r, nx, ny, nx_coarse, ny_coarse)
        
        # Create coarse grid d_u and d_v using vectorized restriction
        d_u_coarse = self._restrict_d_u(d_u, nx, ny, nx_coarse, ny_coarse)
        d_v_coarse = self._restrict_d_v(d_v, nx, ny, nx_coarse, ny_coarse)
        
        # Solve on coarser grid
        x_coarse = np.zeros((nx_coarse, ny_coarse), order='F')
        correction_coarse = self._v_cycle(x_coarse, r_coarse, nx_coarse, ny_coarse, 
                                         dx_coarse, dy_coarse, rho, d_u_coarse, d_v_coarse)
        
        # Prolongate correction to finer grid
        correction = self._prolongate(correction_coarse, nx_coarse, ny_coarse, nx, ny)
        
        # Post-smoothing
        correction = self._smooth(correction, r, nx, ny, dx, dy, rho, d_u, d_v, self.post_smoothing)
        
        return correction
    
    def _restrict(self, fine_grid, nx_fine, ny_fine, nx_coarse, ny_coarse):
        """
        Restrict a fine grid to a coarse grid using full weighting.
        
        Parameters:
        -----------
        fine_grid : ndarray
            Fine grid values (2D)
        nx_fine, ny_fine : int
            Fine grid dimensions
        nx_coarse, ny_coarse : int
            Coarse grid dimensions
            
        Returns:
        --------
        coarse_grid : ndarray
            Coarse grid values (2D)
        """
        coarse_grid = np.zeros((nx_coarse, ny_coarse), order='F')
        
        # Create indices for coarse grid points
        i_coarse = np.arange(nx_coarse)
        j_coarse = np.arange(ny_coarse)
        i_coarse_grid, j_coarse_grid = np.meshgrid(i_coarse, j_coarse, indexing='ij')
        
        # Map to fine grid
        i_fine = i_coarse_grid * 2
        j_fine = j_coarse_grid * 2
        
        # Center points (weight 0.25)
        mask = (i_fine < nx_fine) & (j_fine < ny_fine)
        coarse_grid[mask] += 0.25 * fine_grid[i_fine[mask], j_fine[mask]]
        
        # Horizontal neighbors (weight 0.125)
        # Left
        i_left = i_fine - 1
        mask = (i_left >= 0) & (j_fine < ny_fine)
        coarse_grid[mask] += 0.125 * fine_grid[i_left[mask], j_fine[mask]]
        
        # Right
        i_right = i_fine + 1
        mask = (i_right < nx_fine) & (j_fine < ny_fine)
        coarse_grid[mask] += 0.125 * fine_grid[i_right[mask], j_fine[mask]]
        
        # Vertical neighbors (weight 0.125)
        # Bottom
        j_bottom = j_fine - 1
        mask = (i_fine < nx_fine) & (j_bottom >= 0)
        coarse_grid[mask] += 0.125 * fine_grid[i_fine[mask], j_bottom[mask]]
        
        # Top
        j_top = j_fine + 1
        mask = (i_fine < nx_fine) & (j_top < ny_fine)
        coarse_grid[mask] += 0.125 * fine_grid[i_fine[mask], j_top[mask]]
        
        # Diagonal neighbors (weight 0.0625)
        # Bottom-left
        mask = (i_left >= 0) & (j_bottom >= 0)
        coarse_grid[mask] += 0.0625 * fine_grid[i_left[mask], j_bottom[mask]]
        
        # Bottom-right
        mask = (i_right < nx_fine) & (j_bottom >= 0)
        coarse_grid[mask] += 0.0625 * fine_grid[i_right[mask], j_bottom[mask]]
        
        # Top-left
        mask = (i_left >= 0) & (j_top < ny_fine)
        coarse_grid[mask] += 0.0625 * fine_grid[i_left[mask], j_top[mask]]
        
        # Top-right
        mask = (i_right < nx_fine) & (j_top < ny_fine)
        coarse_grid[mask] += 0.0625 * fine_grid[i_right[mask], j_top[mask]]
        
        return coarse_grid
    
    def _restrict_d_u(self, d_u, nx, ny, nx_coarse, ny_coarse):
        """
        Restrict d_u from fine grid to coarse grid.
        
        Parameters:
        -----------
        d_u : ndarray
            d_u values on fine grid (nx+1, ny)
        nx, ny : int
            Fine grid dimensions
        nx_coarse, ny_coarse : int
            Coarse grid dimensions
            
        Returns:
        --------
        d_u_coarse : ndarray
            d_u values on coarse grid (nx_coarse+1, ny_coarse)
        """
        d_u_coarse = np.zeros((nx_coarse+1, ny_coarse), order='F')
        
        # Create indices for coarse grid points
        i_coarse = np.arange(nx_coarse+1)
        j_coarse = np.arange(ny_coarse)
        i_coarse_grid, j_coarse_grid = np.meshgrid(i_coarse, j_coarse, indexing='ij')
        
        # Map to fine grid
        i_fine = i_coarse_grid * 2
        j_fine = j_coarse_grid * 2
        
        # Direct injection for d_u
        mask = (i_fine < nx+1) & (j_fine < ny)
        d_u_coarse[mask] = d_u[i_fine[mask], j_fine[mask]]
        
        return d_u_coarse
    
    def _restrict_d_v(self, d_v, nx, ny, nx_coarse, ny_coarse):
        """
        Restrict d_v from fine grid to coarse grid.
        
        Parameters:
        -----------
        d_v : ndarray
            d_v values on fine grid (nx, ny+1)
        nx, ny : int
            Fine grid dimensions
        nx_coarse, ny_coarse : int
            Coarse grid dimensions
            
        Returns:
        --------
        d_v_coarse : ndarray
            d_v values on coarse grid (nx_coarse, ny_coarse+1)
        """
        d_v_coarse = np.zeros((nx_coarse, ny_coarse+1), order='F')
        
        # Create indices for coarse grid points
        i_coarse = np.arange(nx_coarse)
        j_coarse = np.arange(ny_coarse+1)
        i_coarse_grid, j_coarse_grid = np.meshgrid(i_coarse, j_coarse, indexing='ij')
        
        # Map to fine grid
        i_fine = i_coarse_grid * 2
        j_fine = j_coarse_grid * 2
        
        # Direct injection for d_v
        mask = (i_fine < nx) & (j_fine < ny+1)
        d_v_coarse[mask] = d_v[i_fine[mask], j_fine[mask]]
        
        return d_v_coarse
    
    def _prolongate(self, coarse_grid, nx_coarse, ny_coarse, nx_fine, ny_fine):
        """
        Prolongate a coarse grid to a fine grid using bilinear interpolation.
        
        Parameters:
        -----------
        coarse_grid : ndarray
            Coarse grid values (2D)
        nx_coarse, ny_coarse : int
            Coarse grid dimensions
        nx_fine, ny_fine : int
            Fine grid dimensions
            
        Returns:
        --------
        fine_grid : ndarray
            Fine grid values (2D)
        """
        fine_grid = np.zeros((nx_fine, ny_fine), order='F')
        
        # Create indices for fine grid points
        i_fine = np.arange(nx_fine)
        j_fine = np.arange(ny_fine)
        i_fine_grid, j_fine_grid = np.meshgrid(i_fine, j_fine, indexing='ij')
        
        # Compute coarse grid indices for each fine grid point
        i_coarse = np.minimum(i_fine_grid // 2, nx_coarse - 1)
        j_coarse = np.minimum(j_fine_grid // 2, ny_coarse - 1)
        
        # Direct injection for coincident points
        mask_direct = (i_fine_grid % 2 == 0) & (j_fine_grid % 2 == 0)
        fine_grid[mask_direct] = coarse_grid[i_coarse[mask_direct], j_coarse[mask_direct]]
        
        # Horizontal interpolation
        mask_h = (i_fine_grid % 2 == 1) & (j_fine_grid % 2 == 0)
        mask_h_interior = mask_h & (i_coarse < nx_coarse - 1)
        fine_grid[mask_h_interior] = 0.5 * (
            coarse_grid[i_coarse[mask_h_interior], j_coarse[mask_h_interior]] + 
            coarse_grid[i_coarse[mask_h_interior] + 1, j_coarse[mask_h_interior]]
        )
        
        mask_h_boundary = mask_h & (i_coarse >= nx_coarse - 1)
        fine_grid[mask_h_boundary] = coarse_grid[i_coarse[mask_h_boundary], j_coarse[mask_h_boundary]]
        
        # Vertical interpolation
        mask_v = (i_fine_grid % 2 == 0) & (j_fine_grid % 2 == 1)
        mask_v_interior = mask_v & (j_coarse < ny_coarse - 1)
        fine_grid[mask_v_interior] = 0.5 * (
            coarse_grid[i_coarse[mask_v_interior], j_coarse[mask_v_interior]] + 
            coarse_grid[i_coarse[mask_v_interior], j_coarse[mask_v_interior] + 1]
        )
        
        mask_v_boundary = mask_v & (j_coarse >= ny_coarse - 1)
        fine_grid[mask_v_boundary] = coarse_grid[i_coarse[mask_v_boundary], j_coarse[mask_v_boundary]]
        
        # Diagonal interpolation
        mask_d = (i_fine_grid % 2 == 1) & (j_fine_grid % 2 == 1)
        
        # Full interior points
        mask_d_interior = mask_d & (i_coarse < nx_coarse - 1) & (j_coarse < ny_coarse - 1)
        fine_grid[mask_d_interior] = 0.25 * (
            coarse_grid[i_coarse[mask_d_interior], j_coarse[mask_d_interior]] + 
            coarse_grid[i_coarse[mask_d_interior] + 1, j_coarse[mask_d_interior]] + 
            coarse_grid[i_coarse[mask_d_interior], j_coarse[mask_d_interior] + 1] + 
            coarse_grid[i_coarse[mask_d_interior] + 1, j_coarse[mask_d_interior] + 1]
        )
        
        # Right boundary
        mask_d_right = mask_d & (i_coarse >= nx_coarse - 1) & (j_coarse < ny_coarse - 1)
        fine_grid[mask_d_right] = 0.5 * (
            coarse_grid[i_coarse[mask_d_right], j_coarse[mask_d_right]] + 
            coarse_grid[i_coarse[mask_d_right], j_coarse[mask_d_right] + 1]
        )
        
        # Top boundary
        mask_d_top = mask_d & (i_coarse < nx_coarse - 1) & (j_coarse >= ny_coarse - 1)
        fine_grid[mask_d_top] = 0.5 * (
            coarse_grid[i_coarse[mask_d_top], j_coarse[mask_d_top]] + 
            coarse_grid[i_coarse[mask_d_top] + 1, j_coarse[mask_d_top]]
        )
        
        # Top-right corner
        mask_d_corner = mask_d & (i_coarse >= nx_coarse - 1) & (j_coarse >= ny_coarse - 1)
        fine_grid[mask_d_corner] = coarse_grid[i_coarse[mask_d_corner], j_coarse[mask_d_corner]]
        
        return fine_grid
    
    def _smooth(self, x, r, nx, ny, dx, dy, rho, d_u, d_v, num_iterations):
        """
        Apply smoother for a specified number of iterations.
        
        Parameters:
        -----------
        x : ndarray
            Current solution (2D)
        r : ndarray
            Current residual (2D)
        nx, ny : int
            Grid dimensions
        dx, dy : float
            Grid spacing
        rho : float
            Density
        d_u, d_v : ndarray
            Momentum equation coefficients
        num_iterations : int
            Number of smoothing iterations
            
        Returns:
        --------
        x : ndarray
            Smoothed solution (2D)
        """
        # Create coefficient matrix for diagonal extraction only
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        D = A.diagonal()
        D_inv = 1.0 / D
        
        # Reshape to 1D for matrix-free operations
        x_flat = x.reshape(-1, order='F')
        r_flat = r.reshape(-1, order='F')
        
        # Jacobi iteration using matrix-free approach
        for _ in range(num_iterations):
            # Compute Ax using matrix-free approach
            Ax = compute_Ap_product(x_flat, nx, ny, dx, dy, rho, d_u, d_v)
            
            # Extract L+U part: Ax - Dx
            L_plus_U_x = Ax - D * x_flat
            
            # Update x using Jacobi formula: x = (1-ω)x + ω D⁻¹(b - (L+U)x)
            x_flat = (1 - self.smoother_omega) * x_flat + self.smoother_omega * D_inv * (r_flat - L_plus_U_x)
        
        # Reshape back to 2D
        return x_flat.reshape((nx, ny), order='F') 