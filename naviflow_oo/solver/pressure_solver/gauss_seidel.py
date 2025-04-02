"""
Matrix-free Gauss-Seidel solver for pressure correction equation.
"""

import numpy as np
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product

class GaussSeidelSolver(PressureSolver):
    """
    Matrix-free Gauss-Seidel solver for pressure correction equation.
    
    This solver uses the Gauss-Seidel iterative method without explicitly forming
    the coefficient matrix, which can be more memory-efficient for large problems.
    Gauss-Seidel differs from Jacobi by immediately using updated values during each iteration,
    which generally leads to faster convergence than Jacobi.
    
    This implementation supports both standard (lexicographic) Gauss-Seidel iteration
    and Red-Black Gauss-Seidel (RBGS) which allows for better vectorization.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=1000, omega=1.0, use_red_black=True):
        """
        Initialize the Gauss-Seidel solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the Gauss-Seidel method
        max_iterations : int, optional
            Maximum number of iterations for the Gauss-Seidel method
        omega : float, optional
            Relaxation factor for SOR (Successive Over-Relaxation)
            omega=1.0 for standard Gauss-Seidel, 1.0 < omega < 2.0 for SOR
        use_red_black : bool, optional
            Whether to use Red-Black Gauss-Seidel for better vectorization (True)
            or standard lexicographic Gauss-Seidel (False)
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.omega = omega
        self.use_red_black = use_red_black
        self.residual_history = []
        self.inner_iterations_history = []
        self.total_inner_iterations = 0
        self.convergence_rates = []
    
    def solve(self, mesh=None, u_star=None, v_star=None, d_u=None, d_v=None, p_star=None, 
              p=None, b=None, nx=None, ny=None, dx=None, dy=None, rho=1.0, num_iterations=None, 
              track_residuals=True):
        """
        Solve the pressure correction equation using the Gauss-Seidel method.
        
        Parameters:
        -----------
        mesh : StructuredMesh, optional
            The computational mesh
        u_star, v_star : ndarray, optional
            Intermediate velocity fields
        d_u, d_v : ndarray
            Momentum equation coefficients
        p_star : ndarray, optional
            Current pressure field (not used)
        p : ndarray, optional
            Initial pressure field
        b : ndarray, optional
            Right-hand side
        nx, ny : int, optional
            Grid dimensions
        dx, dy : float, optional
            Grid spacing
        rho : float, optional
            Fluid density (default: 1.0)
        num_iterations : int, optional
            Number of iterations to perform
        track_residuals : bool, optional
            Whether to track residuals (default: True)
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        """
        # Get grid dimensions and spacing if mesh is provided
        if mesh is not None:
            nx, ny = mesh.get_dimensions()
            dx, dy = mesh.get_cell_sizes()
            
        # Set number of iterations
        if num_iterations is None:
            num_iterations = self.max_iterations
            
        # Get right-hand side
        if b is None:
            b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
            
        # Initial guess
        if p is None:
            p = np.zeros_like(b)
            
        # Track inner iterations and reset residuals if needed
        inner_iterations = 0
        if track_residuals:
            self.residual_history = []
        
        # Determine output shape based on p_star
        output_shape = None
        if p_star is not None:
            output_shape = p_star.shape
        
        # Convert to 2D arrays for computation
        if p.ndim == 1:
            p_2d = p.reshape((nx, ny), order='F')
        else:
            p_2d = p.copy()
            
        if b.ndim == 1:
            b_2d = b.reshape((nx, ny), order='F')
        else:
            b_2d = b.copy()
        
        # Pre-compute coefficient arrays
        # East coefficients
        aE = np.zeros((nx, ny))
        aE[:-1, :] = rho * d_u[1:nx, :] * dy
        
        # West coefficients
        aW = np.zeros((nx, ny))
        aW[1:, :] = rho * d_u[1:nx, :] * dy
        
        # North coefficients
        aN = np.zeros((nx, ny))
        aN[:, :-1] = rho * d_v[:, 1:ny] * dx
        
        # South coefficients
        aS = np.zeros((nx, ny))
        aS[:, 1:] = rho * d_v[:, 1:ny] * dx
        
        # Diagonal coefficients
        aP = aE + aW + aN + aS
        
        # Fix reference pressure at (0,0)
        aP[0, 0] = 1.0
        aE[0, 0] = aW[0, 0] = aN[0, 0] = aS[0, 0] = 0.0
        b_2d[0, 0] = 0.0
        
        # Avoid division by zero
        aP[aP == 0] = 1.0
        
        # Inverse of aP for efficiency
        inv_aP = 1.0 / aP
        
        # Fix reference pressure at (0,0)
        p_2d[0, 0] = 0.0
        
        # Main iteration loop
        for k in range(num_iterations):
            # Ensure reference point stays fixed
            p_2d[0, 0] = 0.0
            
            # Loop over all points in lexicographic order
            for i in range(nx):
                for j in range(ny):
                    # Skip reference point
                    if i == 0 and j == 0:
                        continue
                    
                    # Get contributions from neighbors (always using latest values)
                    east = aE[i, j] * (p_2d[i+1, j] if i < nx-1 else 0)
                    west = aW[i, j] * (p_2d[i-1, j] if i > 0 else 0)
                    north = aN[i, j] * (p_2d[i, j+1] if j < ny-1 else 0)
                    south = aS[i, j] * (p_2d[i, j-1] if j > 0 else 0)
                    
                    # Compute new value
                    p_new = (b_2d[i, j] + east + west + north + south) * inv_aP[i, j]
                    
                    # SOR update
                    p_2d[i, j] = p_2d[i, j] + self.omega * (p_new - p_2d[i, j])
            
            # Track convergence if needed
            if track_residuals:
                # Calculate residual using Ax - b
                p_flat = p_2d.flatten('F')
                r = b.ravel() - compute_Ap_product(p_flat, nx, ny, dx, dy, rho, d_u, d_v)
                res_norm = np.linalg.norm(r)
                self.residual_history.append(res_norm)
                
                # Check solution change
                if k > 0:
                    change = np.linalg.norm(p_2d - p_old) / np.linalg.norm(p_2d)
                    if change < self.tolerance * 0.1:
                        print(f"Gauss-Seidel converged in {k+1} iterations, solution change: {change:.6e}")
                        break
                
                # Store current solution for next iteration
                p_old = p_2d.copy()
                
                # Check residual-based convergence
                if res_norm < self.tolerance:
                    print(f"Gauss-Seidel converged in {k+1} iterations, residual: {res_norm:.6e}")
                    break
                #print(f" GSResidual: {res_norm:.6e}")

        
        
        return p_2d  
    
    def _standard_gauss_seidel_step(self, p, b, aE, aW, aN, aS, inv_aP, nx, ny):
        """
        Perform one standard (lexicographic) Gauss-Seidel iteration with SOR.
        
        This is the standard implementation that updates points in lexicographic order (i,j).
        It's not very vectorizable but preserves the exact Gauss-Seidel update pattern.
        """
        # Vectorize preparation of shifted arrays for neighbors
        # These will be updated during the loop
        p_east = np.zeros_like(p)
        p_east[:-1, :] = p[1:, :]
        
        p_west = np.zeros_like(p)
        p_west[1:, :] = p[:-1, :]
        
        p_north = np.zeros_like(p)
        p_north[:, :-1] = p[:, 1:]
        
        p_south = np.zeros_like(p)
        p_south[:, 1:] = p[:, :-1]
        
        # Loop over all points in lexicographic order
        for i in range(nx):
            for j in range(ny):
                # Skip reference point
                if i == 0 and j == 0:
                    continue
                
                # Get contributions from neighbors (always using latest values)
                east = aE[i, j] * (p[i+1, j] if i < nx-1 else 0)
                west = aW[i, j] * (p[i-1, j] if i > 0 else 0)
                north = aN[i, j] * (p[i, j+1] if j < ny-1 else 0)
                south = aS[i, j] * (p[i, j-1] if j > 0 else 0)
                
                # Compute new value
                p_new = (b[i, j] + east + west + north + south) * inv_aP[i, j]
                
                # SOR update
                p[i, j] = p[i, j] + self.omega * (p_new - p[i, j])
                
                # Update neighbor arrays for next iterations
                if i < nx-1:
                    p_west[i+1, j] = p[i, j]
                if i > 0:
                    p_east[i-1, j] = p[i, j]
                if j < ny-1:
                    p_south[i, j+1] = p[i, j]
                if j > 0:
                    p_north[i, j-1] = p[i, j]
    
    def _rb_gauss_seidel_step(self, p, b, aE, aW, aN, aS, inv_aP, nx, ny):
        """
        Perform one Red-Black Gauss-Seidel iteration with SOR.
        
        This implementation updates all "red" points first, then all "black" points.
        Red points are those where i+j is even, black points where i+j is odd.
        This approach allows for better vectorization while preserving the Gauss-Seidel property
        that updates depend on the latest values of neighbors.
        """
        # Create red and black masks (vectorized)
        i_indices, j_indices = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        red_mask = ((i_indices + j_indices) % 2 == 0)
        
        # Ensure reference point is properly handled
        red_mask[0, 0] = False  # Exclude reference point from updates
        black_mask = ~red_mask
        
        # Update RED points first (vectorized)
        # Get east, west, north, south neighbors
        east = np.zeros_like(p)
        west = np.zeros_like(p)
        north = np.zeros_like(p)
        south = np.zeros_like(p)
        
        # Safe array indexing with proper bounds checking
        east[:-1, :] = aE[:-1, :] * p[1:, :]
        west[1:, :] = aW[1:, :] * p[:-1, :]
        north[:, :-1] = aN[:, :-1] * p[:, 1:]
        south[:, 1:] = aS[:, 1:] * p[:, :-1]
        
        # Calculate new values for red points
        p_new_red = (b + east + west + north + south) * inv_aP
        p_new_red = p + self.omega * (p_new_red - p)
        
        # Apply only to red points
        p[red_mask] = p_new_red[red_mask]
        
        # Update BLACK points next (vectorized)
        # Recalculate neighbors with updated red values
        east[:-1, :] = aE[:-1, :] * p[1:, :]
        west[1:, :] = aW[1:, :] * p[:-1, :]
        north[:, :-1] = aN[:, :-1] * p[:, 1:]
        south[:, 1:] = aS[:, 1:] * p[:, :-1]
        
        # Calculate new values for black points
        p_new_black = (b + east + west + north + south) * inv_aP
        p_new_black = p + self.omega * (p_new_black - p)
        
        # Apply only to black points
        p[black_mask] = p_new_black[black_mask]
    
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics
        """
        # Calculate convergence rate if available
        if len(self.convergence_rates) > 0:
            # Use the average of the last few iterations for stability
            last_rates = self.convergence_rates[-min(10, len(self.convergence_rates)):]
            avg_rate = sum(last_rates) / len(last_rates)
        else:
            avg_rate = None
            
        return {
            'name': 'GaussSeidelSolver',
            'inner_iterations_history': self.inner_iterations_history,
            'total_inner_iterations': self.total_inner_iterations,
            'convergence_rate': avg_rate,
            'omega': self.omega,
            'method': 'Red-Black' if self.use_red_black else 'Standard'
        } 