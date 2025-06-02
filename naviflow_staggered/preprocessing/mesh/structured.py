"""
Structured mesh generation utilities.
"""

import numpy as np

class StructuredMesh:
    """
    Structured mesh for CFD simulations.
    """
    def __init__(self, nx, ny, length=1.0, height=1.0):
        """
        Create a structured mesh for CFD simulations.
        
        Parameters:
        -----------
        nx, ny : int
            Number of cells in x and y directions
        length, height : float
            Physical dimensions of the domain
        """
        self.nx = nx
        self.ny = ny
        self.length = length
        self.height = height
        
        # Calculate cell sizes
        self.dx = length / (nx - 1)
        self.dy = height / (ny - 1)
        
        # Create cell centers
        self.x = np.linspace(self.dx/2, length-self.dx/2, nx)
        self.y = np.linspace(self.dy/2, height-self.dy/2, ny)
        
        # Create meshgrid for plotting
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
    
    def get_dimensions(self):
        """Return the dimensions of the mesh."""
        return self.nx, self.ny
    
    def get_cell_sizes(self):
        """Return the cell sizes."""
        return self.dx, self.dy 