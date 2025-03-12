"""
Case definition and setup.
Handles the overall configuration of simulation cases.
"""

import numpy as np
from ..preprocessing.mesh.structured import StructuredMesh
from ..constructor.properties.fluid import FluidProperties
from ..solver.Algorithms.simple import SimpleSolver

class CFDSolver:
    """
    Main CFD solver class.
    """
    def __init__(self, mesh=None, fluid=None, algorithm=None):
        """
        Initialize the CFD solver.
        
        Parameters:
        -----------
        mesh : StructuredMesh, optional
            The computational mesh
        fluid : FluidProperties, optional
            Fluid properties
        algorithm : Algorithm, optional
            The solution algorithm (SIMPLE, PISO, etc.)
        """
        self.mesh = mesh
        self.fluid = fluid
        self.algorithm = algorithm
    
    @classmethod
    def create_lid_driven_cavity(cls, nx=129, ny=129, reynolds=100, 
                                 alpha_p=0.3, alpha_u=0.7):
        """
        Factory method to create a lid-driven cavity simulation.
        
        Parameters:
        -----------
        nx, ny : int
            Grid dimensions
        reynolds : float
            Reynolds number
        alpha_p, alpha_u : float
            Relaxation factors
            
        Returns:
        --------
        CFDSolver
            Configured solver for lid-driven cavity
        """
        # Create mesh
        mesh = StructuredMesh(nx, ny)
        
        # Create fluid properties
        fluid = FluidProperties(density=1.0, reynolds_number=reynolds)
        
        # Create algorithm
        algorithm = SimpleSolver(mesh, fluid, alpha_p=alpha_p, alpha_u=alpha_u)
        
        # Set boundary conditions
        algorithm.set_boundary_condition('top', 'velocity', {'u': 1.0})
        
        # Create and return solver
        return cls(mesh, fluid, algorithm)
    
    def solve(self, max_iterations=1000, tolerance=1e-6):
        """
        Solve the CFD problem.
        
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
        if self.algorithm is None:
            raise ValueError("No algorithm specified")
            
        return self.algorithm.solve(max_iterations, tolerance) 