# Algorithms module initialization 

from abc import ABC, abstractmethod
import numpy as np
from ...constructor.boundary_conditions import BoundaryConditionManager

class Algorithm(ABC):
    """
    Base class for CFD algorithms.
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