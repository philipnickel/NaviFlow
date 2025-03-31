from abc import ABC, abstractmethod

class PressureSolver(ABC):
    """
    Base class for pressure solvers.
    """
    def __init__(self, tolerance=1e-6, max_iterations=1000):
        """
        Initialize the pressure solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance (for iterative solvers)
        max_iterations : int, optional
            Maximum number of iterations (for iterative solvers)
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def apply_pressure_boundary_conditions(self, p_prime):
        """
        Apply pressure boundary conditions to the pressure correction field.
        
        Parameters:
        -----------
        p_prime : ndarray
            Pressure correction field
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field with boundary conditions applied
        """
        # Apply zero gradient boundary conditions at walls
        # Left wall
        p_prime[0, :] = p_prime[1, :]
        # Right wall
        p_prime[-1, :] = p_prime[-2, :]
        # Bottom wall
        p_prime[:, 0] = p_prime[:, 1]
        # Top wall
        p_prime[:, -1] = p_prime[:, -2]
        
        # Fix reference pressure at bottom-left corner
        #p_prime[0, 0] = 0.0
        
        return p_prime

    @abstractmethod
    def solve(self, mesh, rhs, d_u, d_v, p_star, alpha):
        """
        Solve the pressure correction equation.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        rhs : ndarray
            Right-hand side of the equation
        d_u, d_v : ndarray
            Momentum equation coefficients
        p_star : ndarray
            Current pressure field
        alpha : float
            Relaxation factor
            
        Returns:
        --------
        p, p_prime : ndarray
            Updated pressure field and pressure correction
        """
        pass 