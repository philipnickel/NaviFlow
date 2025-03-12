"""
Matrix-free Jacobi iterative solver for pressure correction equation.
"""

import numpy as np
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product

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
        aW[0, 0] = 0.0
        aS[0, 0] = 0.0
        b_2d[0, 0] = 0.0
        
        # Avoid division by zero
        aP[aP == 0] = 1.0
        
        # Pre-compute 1/aP for efficiency
        inv_aP = 1.0 / aP
        
        # Pre-allocate arrays for neighbor values
        p_east = np.zeros_like(p)
        p_west = np.zeros_like(p)
        p_north = np.zeros_like(p)
        p_south = np.zeros_like(p)
        
        # Jacobi iteration
        for k in range(self.max_iterations):
            # Update shifted arrays for neighbor values
            p_east[:-1, :] = p[1:, :]
            p_west[1:, :] = p[:-1, :]
            p_north[:, :-1] = p[:, 1:]
            p_south[:, 1:] = p[:, :-1]
            
            # Vectorized Jacobi update - CORRECTED FORMULA
            # For Ap = b, the Jacobi update is p_new = (b - (A-D)p)/D
            # where D is the diagonal of A
            # This simplifies to p_new = (b + sum(a_nb * p_nb))/a_P
            p_new = (1 - self.omega) * p + self.omega * (
                (b_2d + aE * p_east + aW * p_west + aN * p_north + aS * p_south) * inv_aP
            )
            
            # Ensure reference pressure point remains zero
            p_new[0, 0] = 0.0
            
            # Calculate residual using true residual: r = b - Ap
            r = b - compute_Ap_product(p_new.flatten('F'), nx, ny, dx, dy, rho, d_u, d_v)
            r_norm = np.linalg.norm(r[1:])  # Exclude reference point
            b_norm = np.linalg.norm(b[1:])  # Exclude reference point
            
            # Normalize residual by RHS norm to get relative residual
            if b_norm > 1e-15:
                res_norm = r_norm / b_norm
            else:
                res_norm = r_norm
                
            self.residual_history.append(res_norm)
            
            # Also check relative change in solution
            if k > 0:
                change = np.linalg.norm(p_new - p) / (np.linalg.norm(p_new) + 1e-15)
                if change < self.tolerance * 0.1:  # Tighter tolerance for solution change
                    print(f"Jacobi converged in {k+1} iterations, solution change: {change:.6e}")
                    p = p_new
                    break
            
            # Check convergence based on residual
            if res_norm < self.tolerance:
                print(f"Jacobi converged in {k+1} iterations, residual: {res_norm:.6e}")
                p = p_new
                break
            
            # Update solution
            p = p_new
        
        return p
    
    def perform_iteration(self, p, b, nx, ny, dx, dy, rho, d_u, d_v, num_iterations=1):
        """
        Perform a specified number of Jacobi iterations.
        
        This method can be used by other solvers (like multigrid) to perform
        Jacobi smoothing steps.
        
        Parameters:
        -----------
        p : ndarray
            Current pressure field (1D or 2D)
        b : ndarray
            Right-hand side (1D or 2D)
        nx, ny : int
            Grid dimensions
        dx, dy : float
            Grid spacing
        rho : float
            Fluid density
        d_u, d_v : ndarray
            Momentum equation coefficients
        num_iterations : int, optional
            Number of iterations to perform
            
        Returns:
        --------
        p_new : ndarray
            Updated pressure field (same shape as input p)
        """
        # Reshape to 2D if needed
        p_is_1d = p.ndim == 1
        if p_is_1d:
            p_2d = p.reshape((nx, ny), order='F')
        else:
            p_2d = p
            
        if b.ndim == 1:
            b_2d = b.reshape((nx, ny), order='F')
        else:
            b_2d = b
        
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
        aW[0, 0] = 0.0
        aS[0, 0] = 0.0
        b_2d[0, 0] = 0.0
        
        # Avoid division by zero
        aP[aP == 0] = 1.0
        
        # Pre-compute 1/aP for efficiency
        inv_aP = 1.0 / aP
        
        # Pre-allocate arrays for neighbor values
        p_east = np.zeros_like(p_2d)
        p_west = np.zeros_like(p_2d)
        p_north = np.zeros_like(p_2d)
        p_south = np.zeros_like(p_2d)
        
        # Perform iterations
        for _ in range(num_iterations):
            # Update shifted arrays for neighbor values
            p_east[:-1, :] = p_2d[1:, :]
            p_west[1:, :] = p_2d[:-1, :]
            p_north[:, :-1] = p_2d[:, 1:]
            p_south[:, 1:] = p_2d[:, :-1]
            
            # Vectorized Jacobi update
            p_2d = (1 - self.omega) * p_2d + self.omega * (
                (b_2d + aE * p_east + aW * p_west + aN * p_north + aS * p_south) * inv_aP
            )
            
            # Ensure reference pressure point remains zero
            p_2d[0, 0] = 0.0
        
        # Return in the same format as input
        if p_is_1d:
            return p_2d.flatten('F')
        else:
            return p_2d 