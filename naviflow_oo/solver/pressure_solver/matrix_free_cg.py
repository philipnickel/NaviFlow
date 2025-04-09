"""
Matrix-free conjugate gradient solver for pressure correction equation.
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from .base_pressure_solver import PressureSolver
from .helpers.matrix_free import compute_Ap_product
from .helpers.rhs_construction import get_rhs

class MatrixFreeCGSolver(PressureSolver):
    """
    Matrix-free conjugate gradient solver for pressure correction equation.
    
    This solver uses the conjugate gradient method without explicitly forming
    the coefficient matrix, which can be more memory-efficient for large problems.
    """
    
    def __init__(self, tolerance=1e-7, max_iterations=1000):
        """
        Initialize the matrix-free conjugate gradient solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the conjugate gradient method
        max_iterations : int, optional
            Maximum number of iterations for the conjugate gradient method
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.residual_history = []
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using the matrix-free conjugate gradient method.
        
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
        
        # Initial guess
        x0 = np.zeros_like(rhs)
        
        # Create a callback function to track convergence
        def callback(xk):
            # Compute residual: ||Ax - b||
            r = compute_Ap_product(xk, nx, ny, dx, dy, rho, d_u, d_v) - rhs
            res_norm = np.linalg.norm(r)
            self.residual_history.append(res_norm)
            return False  # Continue iteration
        
        # Create a lambda function for the matrix-vector product
        mv_product = lambda v: compute_Ap_product(
            v, nx, ny, dx, dy, rho, d_u, d_v
        )
        
        # Create a LinearOperator to represent our matrix operation
        A_op = LinearOperator((len(rhs), len(rhs)), matvec=mv_product)
        
        # Use conjugate gradient to solve system
        p_prime_flat, info = cg(
            A_op, 
            rhs, 
            x0=x0, 
            atol=self.tolerance, 
            maxiter=self.max_iterations,
            callback=callback
        )
        
        if info != 0:
            print(f"Warning: CG did not converge, info={info}")
        
        # Reshape to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        return p_prime 