"""
Jacobi iterative solver for pressure correction equation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from .base_pressure_solver import PressureSolver
from .helpers.coeff_matrix import get_coeff_mat
from .helpers.rhs_construction import get_rhs

class JacobiSolver(PressureSolver):
    """
    Jacobi iterative solver for pressure correction equation.
    
    This solver uses the Jacobi iterative method to solve the pressure
    correction equation. It is simple but may converge slowly for
    ill-conditioned problems.
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
        Solve the pressure correction equation using the Jacobi method.
        
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
        
        # Get coefficient matrix
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Extract diagonal and off-diagonal parts
        D = A.diagonal()
        D_inv = 1.0 / D
        L_plus_U = A - sp.diags(D)
        
        # Initial guess
        p_prime_flat = np.zeros_like(rhs)
        
        # Jacobi iteration
        for k in range(self.max_iterations):
            # Compute residual
            r = rhs - A @ p_prime_flat
            res_norm = np.linalg.norm(r) / np.linalg.norm(rhs)
            self.residual_history.append(res_norm)
            
            # Check convergence
            if res_norm < self.tolerance:
                print(f"Jacobi converged in {k+1} iterations, residual: {res_norm:.6e}")
                break
            
            # Jacobi update: x_{k+1} = (1-omega)*x_k + omega*D^{-1}*(b - (L+U)*x_k)
            p_prime_flat_new = (1 - self.omega) * p_prime_flat + \
                              self.omega * D_inv * (rhs - L_plus_U @ p_prime_flat)
            
            # Update solution
            p_prime_flat = p_prime_flat_new
        
        else:
            print(f"Jacobi did not converge in {self.max_iterations} iterations, "
                 f"final residual: {res_norm:.6e}")
        
        # Reshape to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        return p_prime 