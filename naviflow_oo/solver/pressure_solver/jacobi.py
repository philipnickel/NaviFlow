"""
Matrix-free Jacobi iterative solver for pressure correction equation.
"""

import numpy as np
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs

class JacobiSolver(PressureSolver):
    """
    Matrix-free Jacobi iterative solver for pressure correction equation.
    
    This solver uses the Jacobi iterative method to solve the pressure
    correction equation without explicitly forming the coefficient matrix.
    It is simple but may converge slowly for ill-conditioned problems.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=1000, omega=1.0):
        """
        Initialize the Jacobi solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance
        max_iterations : int, optional
            Maximum number of iterations
        omega : float, optional
            Relaxation factor (1.0 for standard Jacobi, 0-2 for weighted Jacobi)
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.omega = omega
        self.residual_history = []
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using the matrix-free Jacobi method.
        
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
        b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        b_2d = b.reshape((nx, ny), order='F')
        
        # Initial guess
        p = np.zeros((nx, ny))
        
        # Set reference pressure point
        p[0, 0] = 0.0
        
        # Pre-compute coefficient arrays for vectorized operations
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
        
        # Avoid division by zero
        aP[aP == 0] = 1.0
        
        # Jacobi iteration
        for k in range(self.max_iterations):
            # Create shifted arrays for neighbor values
            p_east = np.zeros_like(p)
            p_west = np.zeros_like(p)
            p_north = np.zeros_like(p)
            p_south = np.zeros_like(p)
            
            p_east[:-1, :] = p[1:, :]
            p_west[1:, :] = p[:-1, :]
            p_north[:, :-1] = p[:, 1:]
            p_south[:, 1:] = p[:, :-1]
            
            # Vectorized Jacobi update
            p_new = (1 - self.omega) * p + self.omega * (
                (aE * p_east + aW * p_west + aN * p_north + aS * p_south - b_2d) / aP
            )
            
            # Ensure reference pressure point remains zero
            p_new[0, 0] = 0.0
            
            # Calculate residual (excluding reference point)
            mask = np.ones_like(p, dtype=bool)
            mask[0, 0] = False
            res = np.sum((p_new[mask] - p[mask])**2)
            res_norm = np.sqrt(res) / ((nx * ny) - 1)
            self.residual_history.append(res_norm)
            
            # Check convergence
            if res_norm < self.tolerance:
                print(f"Jacobi converged in {k+1} iterations, residual: {res_norm:.6e}")
                break
            
            # Update solution
            p = p_new
        
        else:
            print(f"Jacobi did not converge in {self.max_iterations} iterations, "
                 f"final residual: {res_norm:.6e}")
        
        return p 