"""
Conjugate gradient solver for pressure correction equation.

This solver uses the conjugate gradient method to solve the pressure correction equation.
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator, spilu

from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.coeff_matrix import get_coeff_mat

class ConjugateGradientSolver(PressureSolver):
    """
    Conjugate gradient solver for pressure correction equation.
    
    This solver uses the conjugate gradient method to solve the pressure correction equation.
    """
    
    def __init__(self, tolerance=1e-7, max_iterations=1000, use_preconditioner=False):
        """
        Initialize the conjugate gradient solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the conjugate gradient method
        max_iterations : int, optional
            Maximum number of iterations for the conjugate gradient method
        use_preconditioner : bool, optional
            Whether to use ILU preconditioning for faster convergence
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.use_preconditioner = use_preconditioner
        self.residual_history = []
        self.inner_iterations = []
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using the conjugate gradient method.
        
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
        
        # Construct the coefficient matrix explicitly
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        
    
        #Create a callback function to track convergence
        def callback(xk):
            # Compute residual: ||Ax - b||
            r = A.dot(xk) - b
            res_norm = np.linalg.norm(r)
            self.residual_history.append(res_norm)
            return False  # Continue iteration
        
        # Use preconditioner if requested
        M = None
        if self.use_preconditioner:
            ilu = spilu(A.tocsc())
            M = LinearOperator(A.shape, ilu.solve)
        
        # Use conjugate gradient
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
            print(f"Warning: Conjugate Gradient did not converge, info={info}")
        
        # Reshape to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        # Enforce boundary conditions
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
            'name': 'ConjugateGradientSolver',
            'inner_iterations_history': self.inner_iterations,
            'total_inner_iterations': sum(self.inner_iterations),
            'convergence_rate': None  # Could calculate this if needed
        }
        
        # Add solver-specific information
        preconditioner_type = 'ilu' if self.use_preconditioner else 'none'
        info['solver_specific'] = {
            'method': 'conjugate_gradient',
            'preconditioner': preconditioner_type
        }
        
        return info 