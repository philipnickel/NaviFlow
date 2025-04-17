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
        self.mesh = None
        self.bc_manager = None
        self.p = None
    
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
        
        # Store the mesh for use in boundary conditions
        self.mesh = mesh
        self.p = p_star
        self.bc_manager = getattr(mesh, 'bc_manager', None)
        
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho = 1.0  # This should come from fluid properties
        
        # Get right-hand side of pressure correction equation
        rhs = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Get coefficient matrix with boundary conditions already integrated
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Fix reference pressure at (0,0) - first index in flattened system
        # Convert to LIL format for efficient single element modifications
        #A = A.tolil()
        #A[0, :] = 0  # Zero out entire row for reference point
        #A[0, 0] = 1  # Set diagonal to 1 for reference equation
        #rhs[0] = 0  # Set right-hand side to zero at reference point
        
        # Solve the system
        p_prime_flat = spsolve(A, rhs)
        
        # Calculate residual for tracking
        residual = np.linalg.norm(A.dot(p_prime_flat) - rhs)
        self.residual_history.append(residual)
        
        # Reshape to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        # Enforce pressure boundary conditions
        #self._enforce_pressure_boundary_conditions(p_prime, nx, ny)
        
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
 