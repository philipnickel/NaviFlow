"""
Matrix-free Jacobi solver for pressure correction equation.
"""

import numpy as np
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product

class JacobiSolver(PressureSolver):
    """
    Matrix-free Jacobi solver for pressure correction equation.
    
    This solver uses the Jacobi iterative method without explicitly forming
    the coefficient matrix, which can be more memory-efficient for large problems.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=1000, omega=1.0):
        """
        Initialize the Jacobi solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the Jacobi method
        max_iterations : int, optional
            Maximum number of iterations for the Jacobi method
        omega : float, optional
            Relaxation factor for weighted Jacobi (1.0 for standard Jacobi)
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.omega = omega
        self.residual_history = []
        self.inner_iterations_history = []
        self.total_inner_iterations = 0
        self.convergence_rates = []
    
    def solve(self, mesh=None, u_star=None, v_star=None, d_u=None, d_v=None, p_star=None, 
              p=None, b=None, nx=None, ny=None, dx=None, dy=None, rho=1.0, num_iterations=None, 
              track_residuals=True):
        """
        Solve the pressure correction equation using the matrix-free Jacobi method.
        
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
        
        # Track inner iterations for this solve
        inner_iterations = 0
        
        # Check if p_star is 2D to determine the expected output shape
        p_star_is_2d = p_star is not None and p_star.ndim == 2
            
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
        for k in range(num_iterations):
            # Update shifted arrays for neighbor values
            p_east[:-1, :] = p_2d[1:, :]
            p_west[1:, :] = p_2d[:-1, :]
            p_north[:, :-1] = p_2d[:, 1:]
            p_south[:, 1:] = p_2d[:, :-1]
            
            # Vectorized Jacobi update
            p_new = (1 - self.omega) * p_2d + self.omega * (
                (b_2d + aE * p_east + aW * p_west + aN * p_north + aS * p_south) * inv_aP
            )
            
            # Ensure reference pressure point remains zero
            p_new[0, 0] = 0.0
            
            # Increment inner iteration counter
            inner_iterations += 1
            
            # Check convergence if tracking residuals
            if track_residuals:
                # Calculate residual using true residual: r = b - Ap
                r = b - compute_Ap_product(
                    p_new.flatten('F') if p_is_1d else p_new, 
                    nx, ny, dx, dy, rho, d_u, d_v
                )
                    
                r_norm = np.linalg.norm(r[1:])  # Exclude reference point
                b_norm = np.linalg.norm(b[1:])  # Exclude reference point
                
                # Normalize residual by RHS norm to get relative residual
                if b_norm > 1e-15:
                    res_norm = r_norm / b_norm
                else:
                    res_norm = r_norm
                self.residual_history.append(res_norm)
                #print(f"Residual Jacobi: {res_norm}")
                
                # Calculate convergence rate if we have enough iterations
                if k >= 2 and self.residual_history[-2] > 1e-15:
                    conv_rate = self.residual_history[-1] / self.residual_history[-2]
                    self.convergence_rates.append(conv_rate)
                
                # Also check relative change in solution
                if k > 0:
                    change = np.linalg.norm(p_new - p_2d) / (np.linalg.norm(p_new) + 1e-15)
                    if change < self.tolerance * 0.1:  # Tighter tolerance for solution change
                        print(f"Jacobi converged in {k+1} iterations, solution change: {change:.6e}")
                        p_2d = p_new
                        break
                
                # Check convergence based on residual
                if res_norm < self.tolerance:
                    print(f"Jacobi converged in {k+1} iterations, residual: {res_norm:.6e}")
                    p_2d = p_new
                    break
            
            # Update solution
            p_2d = p_new
        
        # Store inner iterations for this solve
        self.inner_iterations_history.append(inner_iterations)
        self.total_inner_iterations += inner_iterations
        
        # Return in the same format as input or as expected by the caller
        if p_star_is_2d:
            return p_2d
        elif p_is_1d:
            return p_2d.flatten('F')
        else:
            return p_2d 
    
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics
        """
        info = {
            'name': 'JacobiSolver',
            'inner_iterations_history': self.inner_iterations_history,
            'total_inner_iterations': self.total_inner_iterations
        }
        
        # Calculate average convergence rate if available
        if self.convergence_rates:
            avg_conv_rate = sum(self.convergence_rates) / len(self.convergence_rates)
            info['convergence_rate'] = avg_conv_rate
        
        # Add solver-specific information
        info['solver_specific'] = {
            'omega': self.omega,
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations
        }
        
        return info 