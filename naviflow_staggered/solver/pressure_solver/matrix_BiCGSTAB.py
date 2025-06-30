"""
Biconjugate Gradient Stabilized solver for pressure correction equation.

This solver uses the BiCGSTAB method to solve the pressure correction equation.
"""

import numpy as np
from scipy.sparse.linalg import bicgstab, LinearOperator, spilu

from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.coeff_matrix import get_coeff_mat

class BiCGSTABSolver(PressureSolver):
    """
    Biconjugate Gradient Stabilized solver for pressure correction equation.
    
    This solver uses the BiCGSTAB method to solve the pressure correction equation.
    """
    
    def __init__(self, tolerance=1e-7, max_iterations=1000, use_preconditioner=False):
        """
        Initialize the BiCGSTAB solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the BiCGSTAB method
        max_iterations : int, optional
            Maximum number of iterations for the BiCGSTAB method
        use_preconditioner : bool, optional
            Whether to use ILU preconditioning for faster convergence
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.use_preconditioner = use_preconditioner
        self.residual_history = []
        self.inner_iterations = []
        self.p_max_l2 = 1.0  # Initialize max L2 norm for relative scaling
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star, return_dict=False):
        """
        Solve the pressure correction equation using the BiCGSTAB method.
        
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
            If True, returns a dictionary with complete residual information
            If False, returns only the pressure correction field
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        residual_info : dict (only if return_dict=True)
            Dictionary with residual information: 
            - 'rel_norm': l2(r)/max(l2(r))
            - 'field': residual field
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
        
        # Use BiCGSTAB
        p_prime_flat, info = bicgstab(
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
            print(f"Warning: BiCGSTAB did not converge, info={info}")
        
        # Reshape to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        # If return_dict is True, calculate residual information
        if return_dict:
            # Calculate residual for tracking
            r = b - A.dot(p_prime_flat)  # This is the residual field (1D)
            r_field_full = r.reshape((nx, ny), order='F')  # Reshape residual field
            
            # Calculate L2 norm on interior points only
            r_interior = r_field_full[1:nx-1, 1:ny-1]
            p_current_l2 = np.linalg.norm(r_interior, 2)
            
            # Keep track of the maximum L2 norm for relative scaling
            self.p_max_l2 = max(self.p_max_l2, p_current_l2)
            
            # Calculate relative norm as l2(r)/max(l2(r))
            p_rel_norm = p_current_l2 / self.p_max_l2 if self.p_max_l2 > 0 else 1.0
            
            # Create the residual information dictionary
            residual_info = {
                'rel_norm': p_rel_norm,  # l2(r)/max(l2(r))
                'field': r_field_full     # Absolute residual field
            }
            
            return p_prime, residual_info
        
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
            'name': 'BiCGSTABSolver',
            'inner_iterations_history': self.inner_iterations,
            'total_inner_iterations': sum(self.inner_iterations),
            'convergence_rate': None  # Could calculate this if needed
        }
        
        # Add solver-specific information
        preconditioner_type = 'ilu' if self.use_preconditioner else 'none'
        info['solver_specific'] = {
            'method': 'bicgstab',
            'preconditioner': preconditioner_type
        }
        
        return info 