"""
Base algorithm for CFD solvers.

This module provides the base class for all CFD algorithms in the NaviFlow framework.
"""

from abc import ABC, abstractmethod
import numpy as np
from ...constructor.boundary_conditions import BoundaryConditionManager

class BaseAlgorithm(ABC):
    """
    Base class for CFD algorithms.
    
    This class provides common functionality for all CFD algorithms, including:
    - Field initialization
    - Boundary condition management
    - Common utility methods
    
    All specific algorithms (SIMPLE, SIMPLER, PISO, etc.) should inherit from this class.
    """
    def __init__(self, mesh, fluid, pressure_solver=None, momentum_solver=None, 
                 velocity_updater=None, boundary_conditions=None):
        """
        Initialize the algorithm.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        pressure_solver : PressureSolver, optional
            Solver for pressure equation
        momentum_solver : MomentumSolver, optional
            Solver for momentum equations
        velocity_updater : VelocityUpdater, optional
            Method to update velocities
        boundary_conditions : dict or BoundaryConditionManager, optional
            Boundary conditions
        """
        self.mesh = mesh
        self.fluid = fluid
        self.pressure_solver = pressure_solver
        self.momentum_solver = momentum_solver
        self.velocity_updater = velocity_updater
        
        # Initialize boundary condition manager
        if isinstance(boundary_conditions, BoundaryConditionManager):
            self.bc_manager = boundary_conditions
        else:
            self.bc_manager = BoundaryConditionManager()
            
        # For backward compatibility
        self.boundary_conditions = self.bc_manager.to_dict()
        
        # Initialize fields
        self.initialize_fields()
        
        # Residual history for convergence tracking
        self.residual_history = []
    
    def initialize_fields(self):
        """Initialize velocity and pressure fields."""
        nx, ny = self.mesh.get_dimensions()
        
        # Initialize pressure field
        self.p = np.zeros((nx, ny))
        
        # Initialize velocity fields (staggered grid)
        self.u = np.zeros((nx+1, ny))
        self.v = np.zeros((nx, ny+1))
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
    
    def apply_boundary_conditions(self):
        """Apply boundary conditions to the fields."""
        nx, ny = self.mesh.get_dimensions()
        
        # Use the boundary condition manager to apply velocity boundary conditions
        self.u, self.v = self.bc_manager.apply_velocity_boundary_conditions(
            self.u, self.v, nx, ny
        )
    
    def set_boundary_condition(self, boundary, condition_type, values=None):
        """
        Set boundary conditions for the simulation.
        
        Parameters:
        -----------
        boundary : str
            Boundary name ('top', 'bottom', 'left', 'right')
        condition_type : str
            Type of boundary condition ('velocity', 'pressure', 'wall')
        values : dict, optional
            Values for the boundary condition
        """
        # Set the condition in the manager
        self.bc_manager.set_condition(boundary, condition_type, values)
        
        # Update the boundary_conditions dictionary for backward compatibility
        self.boundary_conditions = self.bc_manager.to_dict()
        
        # Apply the boundary condition
        self.apply_boundary_conditions()
    
    def calculate_divergence(self):
        """
        Calculate the divergence of the velocity field.
        
        Returns:
        --------
        ndarray
            Divergence field
        """
        from ...postprocessing.validation.cavity_flow import calculate_divergence
        dx, dy = self.mesh.get_cell_sizes()
        return calculate_divergence(self.u, self.v, dx, dy)
    
    def get_max_divergence(self):
        """
        Return the maximum absolute divergence in the interior of the domain.
        
        Returns:
        --------
        float
            Maximum absolute divergence
        """
        divergence = self.calculate_divergence()
        
        # Get dimensions
        nx, ny = self.mesh.get_dimensions()
        
        # Create a mask to exclude boundary cells (one cell in from each boundary)
        mask = np.ones_like(divergence, dtype=bool)
        mask[0, :] = False  # Left boundary
        mask[-1, :] = False  # Right boundary
        mask[:, 0] = False  # Bottom boundary
        mask[:, -1] = False  # Top boundary
        
        # Calculate maximum divergence in the interior
        interior_divergence = divergence[mask]
        max_div = np.max(np.abs(interior_divergence))
        
        return max_div
    
    @abstractmethod
    def solve(self, max_iterations=1000, tolerance=1e-6):
        """
        Solve the flow field.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        SimulationResult
            Object containing the solution fields and convergence history
        """
        pass 