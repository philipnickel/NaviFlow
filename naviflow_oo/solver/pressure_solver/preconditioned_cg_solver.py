"""
Preconditioned conjugate gradient solver for pressure correction equation.

This solver uses PyAMG as a preconditioner for the conjugate gradient method,
which can significantly improve convergence rates compared to standard CG.
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.linalg import bicgstab

import pyamg
from .base_pressure_solver import PressureSolver
from .helpers.matrix_free import compute_Ap_product
from .helpers.rhs_construction import get_rhs
from .helpers.coeff_matrix import get_coeff_mat

class PreconditionedCGSolver(PressureSolver):
    """
    Preconditioned conjugate gradient solver for pressure correction equation.
    
    This solver uses the conjugate gradient method with PyAMG as a preconditioner,
    which can significantly improve convergence rates for difficult problems.
    """
    
    def __init__(self, tolerance=1e-7, max_iterations=1000,
                 smoother='gauss_seidel', 
                 presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                 postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                 cycle_type='V'):
        """
        Initialize the preconditioned conjugate gradient solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the conjugate gradient method
        max_iterations : int, optional
            Maximum number of iterations for the conjugate gradient method
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
        self.smoother = smoother
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.cycle_type = cycle_type
        self.residual_history = []
        self.inner_iterations = []
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using the preconditioned conjugate gradient method.
        
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
        
        # Setup PyAMG solver as preconditioner
        ml = pyamg.smoothed_aggregation_solver(
            A,
            B=None,
            strength='evolution',  # More robust strength measure
            aggregate='standard',
            smooth=('jacobi', {'omega': 4.0/3.0}),
            presmoother=('gauss_seidel', {'sweep': 'forward'}),
            postsmoother=('gauss_seidel', {'sweep': 'backward'}),
            max_levels=10,
            max_coarse=10,
            keep=False
        )
        
        # Create a preconditioner function using PyAMG's multigrid cycle
        M_x = lambda x: ml.solve(x, x0=np.zeros_like(x), tol=self.tolerance*0.1, 
                                 maxiter=3, cycle='V')
        
        # Create a LinearOperator to represent our preconditioner
        M = LinearOperator((len(b), len(b)), matvec=M_x)
        
        # Create a callback function to track convergence
        def callback(xk):
            # Compute residual: ||Ax - b||
            r = A.dot(xk) - b
            res_norm = np.linalg.norm(r)
            self.residual_history.append(res_norm)
            return False  # Continue iteration
        
        # Use conjugate gradient with preconditioner
        p_prime_flat, info = cg(
            A,
            b,
            x0=x0,
            M=M, 
            atol=self.tolerance,
            maxiter=self.max_iterations,
            callback=callback
        )
        
        self.inner_iterations.append(len(self.residual_history))
        
        if info != 0:
            print(f"Warning: Preconditioned CG did not converge, info={info}")
        
        # Reshape to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        
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
            'name': 'PreconditionedCGSolver',
            'inner_iterations_history': self.inner_iterations,
            'total_inner_iterations': sum(self.inner_iterations),
            'convergence_rate': None  # Could calculate this if needed
        }
        
        # Add solver-specific information
        info['solver_specific'] = {
            'method': 'preconditioned_conjugate_gradient',
            'preconditioner': 'pyamg',
            'smoother': self.smoother,
            'cycle_type': self.cycle_type
        }
        
        return info 