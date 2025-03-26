"""
Gauss-Seidel solver for pressure correction equation.
"""

import numpy as np
from naviflow_oo.solver.pressure_solver.base_pressure_solver import PressureSolver
from naviflow_oo.solver.pressure_solver.helpers.rhs_construction import get_rhs

class GaussSeidelSolver(PressureSolver):
    """
    Gauss-Seidel solver for pressure correction equation in structured grids.
    
    This implements both standard Gauss-Seidel and Red-Black Gauss-Seidel (RBGS)
    with optional SOR (Successive Over-Relaxation) to accelerate convergence.
    """
    
    def __init__(self, omega=1.0, tolerance=1e-6, max_iterations=1000, use_red_black=False):
        """
        Initialize the Gauss-Seidel solver.
        
        Parameters:
        -----------
        omega : float, optional
            Relaxation parameter (omega=1.0 is standard Gauss-Seidel,
            omega>1.0 is over-relaxation, omega<1.0 is under-relaxation)
        tolerance : float, optional
            Convergence tolerance
        max_iterations : int, optional
            Maximum number of iterations
        use_red_black : bool, optional
            Whether to use Red-Black Gauss-Seidel (True) or standard Gauss-Seidel (False)
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.omega = omega
        self.use_red_black = use_red_black
        self.residual_history = []
    
    def solve(self, mesh=None, u_star=None, v_star=None, d_u=None, d_v=None, p_star=None, 
              p=None, b=None, nx=None, ny=None, dx=None, dy=None, rho=1.0, num_iterations=None, 
              track_residuals=True):
        """
        Solve the pressure correction equation using Gauss-Seidel method.
        
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
        # Get grid dimensions and spacing
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
            
        # Reset residual history if tracking
        if track_residuals:
            self.residual_history = []
        
        # Check if p_star is 2D to determine the expected output shape
        p_star_is_2d = p_star is not None and p_star.ndim == 2
            
        # Reshape to 2D if needed
        p_is_1d = p.ndim == 1
        if p_is_1d:
            p_2d = p.reshape((nx, ny), order='F')
        else:
            p_2d = p.copy()  # We need a copy since Gauss-Seidel updates in-place
            
        if b.ndim == 1:
            b_2d = b.reshape((nx, ny), order='F')
        else:
            b_2d = b.copy()
        
        # Pre-compute coefficient arrays
        # East coefficients (aE)
        aE = np.zeros((nx, ny))
        aE[:-1, :] = rho * d_u[1:nx, :] * dy
        
        # West coefficients (aW)
        aW = np.zeros((nx, ny))
        aW[1:, :] = rho * d_u[1:nx, :] * dy
        
        # North coefficients (aN)
        aN = np.zeros((nx, ny))
        aN[:, :-1] = rho * d_v[:, 1:ny] * dx
        
        # South coefficients (aS)
        aS = np.zeros((nx, ny))
        aS[:, 1:] = rho * d_v[:, 1:ny] * dx
        
        # Diagonal coefficients (aP)
        aP = aE + aW + aN + aS
        
        # Ensure reference point has proper coefficient
        aP[0, 0] = 1.0
        aE[0, 0] = 0.0
        aN[0, 0] = 0.0
        aW[0, 0] = 0.0
        aS[0, 0] = 0.0
        b_2d[0, 0] = 0.0
        
        # Avoid division by zero
        aP[aP == 0] = 1.0
        
        # Perform iterations
        for k in range(num_iterations):
            # Keep a copy for convergence check
            p_old = p_2d.copy()
            
            if self.use_red_black:
                # Red-Black Gauss-Seidel
                self._rbgs_iteration(p_2d, b_2d, aE, aW, aN, aS, aP, nx, ny)
            else:
                # Standard Gauss-Seidel
                self._gs_iteration(p_2d, b_2d, aE, aW, aN, aS, aP, nx, ny)
            
            # Ensure reference pressure point remains zero
            p_2d[0, 0] = 0.0
            
            # Check convergence if tracking residuals
            if track_residuals:
                res = np.linalg.norm(p_2d - p_old)
                self.residual_history.append(res)
                
                # Check convergence
                if res < self.tolerance:
                    break
            print(f"Iteration {k+1}, Residual: {res:.6e}")
                
        # Return in the expected format
        if p_is_1d or (p_star is not None and p_star.ndim == 1):
            return p_2d.flatten(order='F')
        else:
            return p_2d
    
    def _gs_iteration(self, p, b, aE, aW, aN, aS, aP, nx, ny):
        """
        Perform one standard Gauss-Seidel iteration.
        """
        # Loop over all interior points in a specific order
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Skip the reference pressure point
                if i == 0 and j == 0:
                    continue
                    
                # Sum of neighbor contributions
                neighbor_sum = 0.0
                
                # East neighbor
                if i < nx-1:
                    neighbor_sum += aE[i, j] * p[i+1, j]
                    
                # West neighbor
                if i > 0:
                    neighbor_sum += aW[i, j] * p[i-1, j]
                    
                # North neighbor
                if j < ny-1:
                    neighbor_sum += aN[i, j] * p[i, j+1]
                    
                # South neighbor
                if j > 0:
                    neighbor_sum += aS[i, j] * p[i, j-1]
                
                # Update with relaxation
                p_new = (b[i, j] + neighbor_sum) / aP[i, j]
                p[i, j] = (1.0 - self.omega) * p[i, j] + self.omega * p_new
    
    def _rbgs_iteration(self, p, b, aE, aW, aN, aS, aP, nx, ny):
        """
        Perform one Red-Black Gauss-Seidel iteration.
        """
        # Process red points (i+j even)
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if (i + j) % 2 == 0:
                    # Skip the reference pressure point
                    if i == 0 and j == 0:
                        continue
                        
                    # Sum of neighbor contributions
                    neighbor_sum = 0.0
                    
                    # East neighbor
                    if i < nx-1:
                        neighbor_sum += aE[i, j] * p[i+1, j]
                        
                    # West neighbor
                    if i > 0:
                        neighbor_sum += aW[i, j] * p[i-1, j]
                        
                    # North neighbor
                    if j < ny-1:
                        neighbor_sum += aN[i, j] * p[i, j+1]
                        
                    # South neighbor
                    if j > 0:
                        neighbor_sum += aS[i, j] * p[i, j-1]
                    
                    # Update with relaxation
                    p_new = (b[i, j] + neighbor_sum) / aP[i, j]
                    p[i, j] = (1.0 - self.omega) * p[i, j] + self.omega * p_new
                    
        # Process black points (i+j odd)
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if (i + j) % 2 == 1:
                    # Skip the reference pressure point
                    if i == 0 and j == 0:
                        continue
                        
                    # Sum of neighbor contributions
                    neighbor_sum = 0.0
                    
                    # East neighbor
                    if i < nx-1:
                        neighbor_sum += aE[i, j] * p[i+1, j]
                        
                    # West neighbor
                    if i > 0:
                        neighbor_sum += aW[i, j] * p[i-1, j]
                        
                    # North neighbor
                    if j < ny-1:
                        neighbor_sum += aN[i, j] * p[i, j+1]
                        
                    # South neighbor
                    if j > 0:
                        neighbor_sum += aS[i, j] * p[i, j-1]
                    
                    # Update with relaxation
                    p_new = (b[i, j] + neighbor_sum) / aP[i, j]
                    p[i, j] = (1.0 - self.omega) * p[i, j] + self.omega * p_new 