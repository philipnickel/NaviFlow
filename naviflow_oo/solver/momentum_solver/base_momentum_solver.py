"""
Abstract base class for momentum solvers.
"""

from abc import ABC, abstractmethod

class MomentumSolver(ABC):
    """
    Base class for momentum solvers.
    """
    def __init__(self):
        """
        Initialize the momentum solver.
        """
        pass
    
    @abstractmethod
    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7):
        """
        Solve the u-momentum equation.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        u, v : ndarray
            Current velocity fields
        p : ndarray
            Current pressure field
        relaxation_factor : float, optional
            Relaxation factor for the momentum equation
            
        Returns:
        --------
        u_star, d_u : ndarray
            Intermediate velocity field and momentum equation coefficient
        """
        pass
    
    @abstractmethod
    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7):
        """
        Solve the v-momentum equation.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        u, v : ndarray
            Current velocity fields
        p : ndarray
            Current pressure field
        relaxation_factor : float, optional
            Relaxation factor for the momentum equation
            
        Returns:
        --------
        v_star, d_v : ndarray
            Intermediate velocity field and momentum equation coefficient
        """
        pass 