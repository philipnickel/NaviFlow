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
    
    def _get_diagonal_elements(self, nx, ny, dx, dy, rho, d_u, d_v):
        """
        Calculate the diagonal elements of the pressure matrix.
        This is needed for the Jacobi method since we need the inverse of diagonal elements.
        
        Returns:
        --------
        ndarray
            Diagonal elements of the pressure matrix
        """
        # Create diagonal array
        diag = np.zeros((nx, ny))
        
        # East and West contributions
        diag[:-1, :] += rho * d_u[1:nx, :] * dy  # East neighbors
        diag[1:, :] += rho * d_u[1:nx, :] * dy   # West neighbors
        
        # North and South contributions
        diag[:, :-1] += rho * d_v[:, 1:ny] * dx  # North neighbors
        diag[:, 1:] += rho * d_v[:, 1:ny] * dx   # South neighbors
        
        # Apply boundary conditions 
        # West boundary (i=0)
        diag[0, :] += diag[0, :]
        
        # East boundary (i=nx-1)
        diag[nx-1, :] += diag[nx-1, :]
        
        # South boundary (j=0)
        diag[:, 0] += diag[:, 0]
        
        # North boundary (j=ny-1)
        diag[:, ny-1] += diag[:, ny-1]
        
        # Ensure diagonal elements are non-zero
        diag[diag < 1e-15] = 1.0
        
        # Fix reference point
        diag[0, 0] = 1.0
        
        return diag

    def solve(self, mesh=None, u_star=None, v_star=None, d_u=None, d_v=None, p_star=None, 
              p=None, b=None, nx=None, ny=None, dx=None, dy=None, rho=1.0, num_iterations=None, 
              track_residuals=True, return_dict=False):
        """
        Solve the pressure correction equation using the Jacobi method.
        
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
        return_dict : bool, optional
            Whether to return a dictionary of results (default: False)
            
        Returns:
        --------
        p_prime : ndarray or tuple
            Pressure correction field, or tuple of (p_prime, info_dict) if return_dict=True
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
        
        # Check input formats and reshape as needed
        p_star_is_2d = p_star is not None and p_star.ndim == 2
        p_is_1d = p.ndim == 1
        b_is_1d = b.ndim == 1
        
        # Convert to 2D arrays for computation
        if p_is_1d:
            p_2d = p.reshape((nx, ny), order='F')
        else:
            p_2d = p.copy()
            
        if b_is_1d:
            b_2d = b.reshape((nx, ny), order='F')
        else:
            b_2d = b.copy()
        
        # Get diagonal elements for Jacobi method
        diag = self._get_diagonal_elements(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Set reference pressure point
        p_2d[0, 0] = 0.0
        b_2d[0, 0] = 0.0
        
        # Main iteration loop
        for k in range(num_iterations):
            # Make sure the reference pressure point stays at zero
            p_2d[0, 0] = 0.0
            
            # Compute Ap product using the matrix-free function
            p_flat = p_2d.flatten('F')
            Ap = compute_Ap_product(p_flat, nx, ny, dx, dy, rho, d_u, d_v)
            Ap = Ap.reshape((nx, ny), order='F')
            
            # Jacobi iteration formula: p_new = p + omega * (b - Ap) / diag
            # This computes p_new = p + omega * D⁻¹(b - Ap)
            p_new = p_2d + self.omega * (b_2d - Ap) / diag
            
            # Fix reference point
            p_new[0, 0] = 0.0
            
            # Track iterations
            inner_iterations += 1
            
            # Check convergence if needed
            if track_residuals:
                # Calculate residual using Ax - b
                r = b.ravel() - compute_Ap_product(p_new.flatten('F'), nx, ny, dx, dy, rho, d_u, d_v)
                res_norm = np.linalg.norm(r, ord=2)
                rel_norm = res_norm / np.linalg.norm(b.ravel(), ord=2)
                self.residual_history.append(res_norm)
                
                # Calculate convergence rate if we have enough iterations
                if k >= 2 and self.residual_history[-2] > 1e-15:
                    conv_rate = self.residual_history[-1] / self.residual_history[-2]
                    self.convergence_rates.append(conv_rate)
                 
                
                # Check residual-based convergence
                if rel_norm < self.tolerance:
                    print(f"Jacobi converged in {k+1} iterations, residual: {res_norm:.6e}")
                    p_2d = p_new
                    break
            
            p_2d = p_new
        
        # Store inner iterations
        self.inner_iterations_history.append(inner_iterations)
        self.total_inner_iterations += inner_iterations
        
        # If the caller wants a dictionary of results
        if return_dict:
            # Create dictionary with result information
            result_info = {
                'rel_norm': 1.0 if not self.residual_history else self.residual_history[-1] / max(self.residual_history[0], 1e-10),
                'abs_norm': self.residual_history[-1] if self.residual_history else 1.0,
                'iterations': inner_iterations,
                'field': r.reshape((nx, ny), order='F')  # Return a copy of the residual field for visualization
            }
            return p_2d, result_info
        
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