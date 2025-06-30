"""
Abstract base class for velocity solvers.
"""

from abc import ABC, abstractmethod

class VelocityUpdater(ABC):
    """
    Base class for velocity updaters.
    """
    def __init__(self):
        """Initialize the velocity updater."""
        pass
    
    @abstractmethod
    def update_velocity(self, mesh, u_star, v_star, p_prime, d_u, d_v, boundary_conditions):
        """
        Update velocities based on pressure correction.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        u_star, v_star : ndarray
            Intermediate velocity fields
        p_prime : ndarray
            Pressure correction field
        d_u, d_v : ndarray
            Momentum equation coefficients
        boundary_conditions : dict
            Boundary conditions
            
        Returns:
        --------
        u, v : ndarray
            Updated velocity fields
        """
        pass 