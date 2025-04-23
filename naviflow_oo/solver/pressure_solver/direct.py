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

        # Initialize coefficient matrices for pressure correction equation
        self.p_a_e = None
        self.p_a_w = None
        self.p_a_n = None
        self.p_a_s = None
        self.p_a_p = None
        self.p_source = None
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star, return_dict=True):
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
        return_dict : bool, optional
            If True, returns a dictionary with complete residual information (default)
            If False, returns separate residual values (deprecated)
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        residual_info : dict
            Dictionary with residual information: 
            - 'rel_norm': l2(r)/max(l2(r))
            - 'field': residual field
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
        
        # Solve the system
        p_prime_flat = spsolve(A, rhs)
        
        # Apply zero gradient boundary conditions to the direct solution
        # Calculate residual for tracking
        Ax = A.dot(p_prime_flat)
        r = rhs - Ax # This is the residual field (1D)
        
        # Reshape residual to 2D for interior point extraction
        r_field_full = r.reshape((nx, ny), order='F')  # Reshape residual field 
        
        # Extract interior points only (1:nx-1, 1:ny-1) for pressure
        r_interior = r_field_full[1:nx-1, 1:ny-1]
        
        # Calculate L2 norm on interior points only - use more stable calculation
        p_current_l2 = np.sqrt(np.sum(r_interior**2))
        
        # Keep track of the maximum L2 norm for relative scaling
        if not hasattr(self, 'p_max_l2'):
            self.p_max_l2 = p_current_l2
        else:
            self.p_max_l2 = max(self.p_max_l2, p_current_l2)
        
        # Calculate relative norm as l2(r)/max(l2(r))
        p_rel_norm = p_current_l2 / self.p_max_l2 if self.p_max_l2 > 0 else 1.0
        
        # Track history of normalized residual
        self.residual_history.append(p_rel_norm)
        
        # Reshape solution to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        # Create the minimal residual information dictionary
        residual_info = {
            'rel_norm': p_rel_norm,  # l2(r)/max(l2(r))
            'field': r_field_full     # Absolute residual field
        }
        
        # Return the solution and residual info
        return p_prime, residual_info
 
        
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
 