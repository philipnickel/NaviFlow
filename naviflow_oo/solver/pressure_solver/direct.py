"""
Direct solver for pressure correction equation.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from ..pressure_solver.base_pressure_solver import PressureSolver
from ..pressure_solver.helpers.coeff_matrix import get_coeff_mat
from ..pressure_solver.helpers.rhs_construction import get_rhs
from ..pressure_solver.helpers.pressure_corrections import pres_correct

class DirectPressureSolver(PressureSolver):
    """
    Direct solver for pressure correction equation using sparse matrix methods.
    
    This solver uses a direct method (scipy.sparse.linalg.spsolve) to solve
    the pressure correction equation, so it doesn't require iterations or
    convergence tolerance.
    """
    
    def __init__(self):
        """
        Initialize the direct pressure solver.
        """
        # No need for tolerance or max_iterations for a direct solver
        super().__init__()
        self.residual_history = []
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using a direct method.
        
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
        
        # Get right-hand side of pressure correction equation
        rhs = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Get coefficient matrix
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Explicitly fix reference pressure at bottom-left corner (0,0)
        # This makes the system non-singular
        row_idx = 0  # Index for the (0,0) cell
        A[row_idx, :] = 0  # Zero out the row
        A[row_idx, row_idx] = 1  # Set diagonal to 1
        rhs[row_idx] = 0  # Set RHS to 0
        
        # Add small regularization to improve conditioning
        # This helps with near-singular matrices
        eps = 1e-10
        diag_indices = sparse.find(A.diagonal())[0]
        A = A.tolil()
        for i in diag_indices:
            A[i, i] += eps
        A = A.tocsr()
        
        # Solve the system
        p_prime_flat = spsolve(A, rhs)
        
        # Calculate residual for tracking
        residual = np.linalg.norm(A.dot(p_prime_flat) - rhs)
        self.residual_history.append(residual)
        
        # Reshape to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        # Ensure reference pressure is exactly zero
        p_prime[0, 0] = 0.0
        
        return p_prime
        
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics
        """
        info = {
            'name': 'DirectPressureSolver',
            'inner_iterations_history': [1] * len(self.residual_history),  # Direct solver uses 1 iteration per solve
            'total_inner_iterations': len(self.residual_history),
            'convergence_rate': None  # Not applicable for direct solver
        }
        
        # Add solver-specific information
        info['solver_specific'] = {
            'method': 'direct',
            'library': 'scipy.sparse.linalg.spsolve'
        }
        
        return info

def penta_diag_solve(solver_params):
    """Solve the pentadiagonal system Ax = b."""
    # Extract needed parameters
    A = solver_params['A']
    b = solver_params['b']
    
    # Use a more robust solver like UMFPACK
    #x = spsolve_triangular(A, b, lower=False)
    
    # Add a small value to the diagonal to improve conditioning
    diag = A.diagonal()
    diag_plus_eps = diag + 1e-10
    A.setdiag(diag_plus_eps)
    
    x = spsolve(A, b)
    return x
 