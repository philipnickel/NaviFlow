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
        
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
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
        self.inner_iterations = []
        
        # Get right-hand side of pressure correction equation
        b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Initial guess
        x0 = np.zeros_like(b)
        
        # Fix reference pressure at (0,0)
        b[0] = 0.0
    
        # Construct the coefficient matrix explicitly
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Fix reference pressure at (0,0) - use lil_matrix for efficient modification
        A_lil = A.tolil()
        A_lil[0, :] = 0
        A_lil[0, 0] = 1
        A = A_lil.tocsr()
        
        # Setup PyAMG solver
        ml = pyamg.smoothed_aggregation_solver(
            A, 
            presmoother=self.presmoother,
            postsmoother=self.postsmoother
        )
        
        # Solve using PyAMG
        x = ml.solve(b, x0=x0, tol=self.tolerance, maxiter=self.max_iterations, 
                        residuals=self.residual_history, accel='cg')
        print(f"Residual history: {self.residual_history}")
        self.inner_iterations.append(len(self.residual_history))
    
        # Reshape to 2D
        p_prime = x.reshape((nx, ny), order='F')
        
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