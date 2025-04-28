"""
Matrix-free algebraic multigrid solver for pressure correction equation using PyAMG.
"""

import numpy as np
from scipy.sparse import diags, csr_matrix, lil_matrix
from scipy.sparse.linalg import LinearOperator
import pyamg
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product
from .helpers.coeff_matrix import get_coeff_mat

class PyAMGSolver(PressureSolver):
    """
    Matrix-free algebraic multigrid solver for pressure correction equation using PyAMG.
    
    This solver leverages the PyAMG library to efficiently solve the pressure correction
    equation using algebraic multigrid methods. It can operate in either matrix-free mode
    or with an explicit matrix construction.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=100, matrix_free=True, 
                 smoother='gauss_seidel', presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
                 postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
                 cycle_type='V'):
        """
        Initialize the PyAMG solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the solver
        max_iterations : int, optional
            Maximum number of iterations for the solver
        matrix_free : bool, optional
            Whether to use matrix-free operations (True) or explicit matrix construction (False)
        smoother : str, optional
            Type of smoother to use ('gauss_seidel', 'jacobi', etc.)
        presmoother : tuple, optional
            Presmoother configuration (type, options)
        postsmoother : tuple, optional
            Postsmoother configuration (type, options)
        cycle_type : str, optional
            Type of multigrid cycle to use ('V', 'W', 'F')
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.matrix_free = matrix_free
        self.smoother = smoother
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.cycle_type = cycle_type
        self.residual_history = []
        self.inner_iterations = []
        
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star, return_dict=True):
        """
        Solve the pressure correction equation using PyAMG.
        
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
             If True, returns residual_info dict.
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        residual_info : dict, optional
            Dictionary with residual information if return_dict is True.
        """
        # Apply boundary conditions
        #p_star = self.apply_pressure_boundary_conditions(p_star)
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho = 1.0  # This should come from fluid properties
        
        # Reset residual history
        self.residual_history = []
        self.inner_iterations = []
        
        # Get right-hand side of pressure correction equation
        # Ensure inputs are flattened if necessary for get_rhs (assuming it handles 1D/2D)
        u_star_flat = u_star.flatten() if u_star.ndim > 1 else u_star
        v_star_flat = v_star.flatten() if v_star.ndim > 1 else v_star
        b = get_rhs(mesh, rho, u_star_flat, v_star_flat)
        b_flat = b # get_rhs should return 1D
        
        # Initial guess (flattened)
        x0 = np.zeros_like(b_flat)
            
        # Construct the coefficient matrix explicitly
        # Ensure inputs are flattened if necessary for get_coeff_mat
        d_u_flat = d_u.flatten() if d_u.ndim > 1 else d_u
        d_v_flat = d_v.flatten() if d_v.ndim > 1 else d_v
        A = get_coeff_mat(mesh, rho, d_u_flat, d_v_flat)
                
        # Setup PyAMG solver
        ml = pyamg.smoothed_aggregation_solver(
            A, 
            presmoother=self.presmoother,
            postsmoother=self.postsmoother
        )
        
        # Solve using PyAMG
        # PyAMG solve returns flat array
        x = ml.solve(b_flat, x0=x0, tol=self.tolerance, maxiter=self.max_iterations, 
                        residuals=self.residual_history, accel='cg')
        #print(f"Residual history: {self.residual_history}")
        self.inner_iterations.append(len(self.residual_history))
    
        # Reshape solution to 2D (original expected output shape?)
        # Or should it return flat like momentum solver?
        # Let's return flat for consistency with mesh-agnostic approach
        p_prime = x # Keep flat
        # p_prime = x.reshape((nx, ny), order='F') # Old 2D return

        # Enforce boundary conditions?
        # self._enforce_pressure_boundary_conditions(p_prime, nx, ny)
        
        if return_dict:
            # Calculate residual info
            r = b_flat - A.dot(x) # Use flat x (solution) and flat b (rhs)
            r_norm = np.linalg.norm(r)
            b_norm = np.linalg.norm(b_flat)
            rel_norm = r_norm / max(b_norm, 1e-10)
            residual_info = {
                'rel_norm': rel_norm,
                'field': r # Return 1D residual field
            }
            return p_prime, residual_info # Return dict
        else:
             # For backward compatibility if needed
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
            'name': 'PyAMGSolver',
            'inner_iterations_history': self.inner_iterations,
            'total_inner_iterations': sum(self.inner_iterations),
            'convergence_rate': None  # Could calculate this if needed
        }
        
        # Add solver-specific information
        info['solver_specific'] = {
            'method': 'algebraic_multigrid',
            'library': 'pyamg',
            'matrix_free': self.matrix_free,
            'smoother': self.smoother,
            'cycle_type': self.cycle_type
        }
        
        return info 