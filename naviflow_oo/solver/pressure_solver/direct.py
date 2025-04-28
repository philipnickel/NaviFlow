"""
Direct solver for pressure correction equation.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from ..pressure_solver.base_pressure_solver import PressureSolver
from ..pressure_solver.helpers.coeff_matrix import get_coeff_mat, get_coeff_mat_structured
from ..pressure_solver.helpers.rhs_construction import get_rhs, get_rhs_structured
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
        self.mesh = None
        self.bc_manager = None
        self.p = None

        # Initialize coefficient matrices for pressure correction equation
        self.p_a_e = None
        self.p_a_w = None
        self.p_a_n = None
        self.p_a_s = None
        self.p_a_p = None
        self.p_source = None
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star, return_dict=True):
        """
        Solve the pressure correction equation in a mesh-agnostic way.
        
        Parameters:
        -----------
        mesh : Mesh
            The computational mesh
        u_star, v_star : ndarray
            Intermediate velocity fields
        d_u, d_v : ndarray
            Momentum equation coefficients (face-based)
        p_star : ndarray
            Current pressure field
        return_dict : bool
            Return detailed info if True (default)
        
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        residual_info : dict
            Residual info (optional)
        """

        # Store the mesh for use in boundary conditions
        self.mesh = mesh
        self.p = p_star
        self.bc_manager = getattr(mesh, 'bc_manager', None)
        
        rho = 1.0  # Assuming constant density (could generalize later)

        # Build RHS
        rhs = get_rhs(mesh, rho, u_star, v_star)

        # Build Coefficient Matrix
        A = get_coeff_mat(mesh, rho, d_u, d_v)

        # Solve sparse linear system
        p_prime_flat = spsolve(A, rhs)

        # Track residual
        residual_vector = rhs - A.dot(p_prime_flat)
        residual_norm = np.linalg.norm(residual_vector)

        # Normalize by RHS norm (for monitoring convergence)
        if not hasattr(self, 'p_max_l2'):
            self.p_max_l2 = residual_norm
        else:
            self.p_max_l2 = max(self.p_max_l2, residual_norm)

        rel_norm = residual_norm / np.linalg.norm(rhs)

        # Track history
        self.residual_history.append(rel_norm)

        # Reshape if needed
        if hasattr(p_star, 'shape'):
            try:
                # Try to reshape to match p_star
                p_prime = p_prime_flat.reshape(p_star.shape)
            except ValueError:
                # If reshaping fails, keep it flat
                p_prime = p_prime_flat
        else:
            p_prime = p_prime_flat

        # Package residual info
        residual_info = {
            'rel_norm': rel_norm,
            'field': residual_vector
        }

        if return_dict:
            return p_prime, residual_info
        else:
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
    #diag = A.diagonal()
    #diag_plus_eps = diag + 1e-10
    #A.setdiag(diag_plus_eps)
    
    x = spsolve(A, b)
    return x
 