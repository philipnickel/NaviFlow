"""
Base algorithm for CFD solvers.

This module provides the base class for all CFD algorithms in the NaviFlow framework.
"""

from abc import ABC, abstractmethod
import numpy as np
import os
from ...constructor.boundary_conditions import BoundaryConditionManager
from ...utils.profiler import Profiler

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
        
        # Initialize profiler
        self.profiler = Profiler(self.__class__.__name__, mesh, fluid, algorithm=self)
        
        # Initialize fields
        self.initialize_fields()
        
        # Residual history for convergence tracking
        self.residual_history = []
    
    def initialize_fields(self):
        """Initialize velocity and pressure fields."""
        u_shape, v_shape, p_shape = self.mesh.get_field_shapes()
        
        # Initialize fields with proper shapes for collocated grid
        self.u = np.zeros(u_shape)
        self.v = np.zeros(v_shape)
        self.p = np.zeros(p_shape)
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
    
    def apply_boundary_conditions(self):
        """Apply boundary conditions to the fields."""
        self.profiler.start_section()
        
        nx, ny = self.mesh.get_dimensions()
        
        # Use the boundary condition manager to apply velocity boundary conditions
        self.u, self.v = self.bc_manager.apply_velocity_boundary_conditions(
            self.u, self.v, nx, ny
        )
        
        self.profiler.end_section('boundary_condition_time')
    
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
        self.profiler.start_section()
        
        # Set the condition in the manager
        self.bc_manager.set_condition(boundary, condition_type, values)
        
        # Update the boundary_conditions dictionary for backward compatibility
        self.boundary_conditions = self.bc_manager.to_dict()
        
        # Apply the boundary condition
        self.apply_boundary_conditions()
        
        self.profiler.end_section('boundary_condition_time')
    
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
    
    def _enforce_pressure_boundary_conditions(self):
        """
        Apply appropriate pressure boundary conditions based on boundary types.
        
        This method:
        1. Extracts the boundary types from the boundary condition manager
        2. Applies appropriate pressure boundary conditions for each boundary
        3. Sets a reference pressure point to prevent a floating pressure field
        
        By default, it applies Neumann (zero gradient) conditions for all boundaries,
        which is appropriate for most incompressible flow problems.
        
        Note for derived classes:
        This method should be called after any pressure field updates in the solver
        algorithm implementation to ensure proper pressure boundary treatment.
        Typical places to call this method include:
        - After updating pressure with pressure corrections
        - After solving intermediate pressure fields
        - Before using pressure fields to calculate velocity fields
        """
        nx, ny = self.mesh.get_dimensions()
        boundary_types = self.bc_manager.get_boundary_types()
        
        # Apply boundary conditions based on boundary type and location
        for boundary, bc_type in boundary_types.items():
            if boundary == 'left':
                # Left boundary (i=0): Apply zero gradient
                self.p[0, :] = self.p[1, :]
            elif boundary == 'right':
                # Right boundary (i=nx-1): Apply zero gradient
                self.p[nx-1, :] = self.p[nx-2, :]
            elif boundary == 'bottom':
                # Bottom boundary (j=0): Apply zero gradient
                self.p[:, 0] = self.p[:, 1]
            elif boundary == 'top':
                # Top boundary (j=ny-1): Apply zero gradient
                self.p[:, ny-1] = self.p[:, ny-2]
        
    
    def save_profiling_data(self, filename=None, profile_dir='results/profiles'):
        """
        Save profiling data to a file.
        
        Parameters:
        -----------
        filename : str, optional
            Name of the file to save the data to. If None, a default name is generated.
        profile_dir : str, optional
            Directory to save profiling data
            
        Returns:
        --------
        str
            Path to the saved file
        """
        return self.profiler.save(filename, profile_dir)
    
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