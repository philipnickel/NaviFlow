"""
Structured mesh generation utilities.
"""

import numpy as np
from ..mesh.structured import StructuredUniform, StructuredNonUniform

class StructuredMeshGenerator:
    """
    Generator for structured meshes.
    """
    
    @staticmethod
    def generate_uniform(xmin, xmax, ymin, ymax, nx, ny, zmin=0.0, zmax=0.0, nz=1):
        """
        Generate a uniform 2D structured mesh.
        
        Parameters:
        -----------
        xmin, xmax : float
            Domain limits in the x direction
        ymin, ymax : float
            Domain limits in the y direction
        nx, ny : int
            Number of nodes in x and y directions
        zmin, zmax : float, optional
            Domain limits in the z direction (for 3D meshes)
        nz : int, optional
            Number of nodes in z direction (for 3D meshes)
            
        Returns:
        --------
        mesh : StructuredUniform
            The generated uniform structured mesh
        """
        # Create evenly spaced nodes
        x_nodes = np.linspace(xmin, xmax, nx)
        y_nodes = np.linspace(ymin, ymax, ny)
        
        return StructuredUniform(x_nodes, y_nodes)
    
    @staticmethod
    def generate_nonuniform(x_coords, y_coords, z_coords=None):
        """
        Generate a non-uniform 2D structured mesh from given x and y coordinates.
        
        Parameters:
        -----------
        x_coords : array-like
            X-coordinates of the grid lines
        y_coords : array-like
            Y-coordinates of the grid lines
        z_coords : array-like, optional
            Z-coordinates of the grid lines (for 3D meshes)
            
        Returns:
        --------
        mesh : StructuredNonUniform
            The generated non-uniform structured mesh
        """
        return StructuredNonUniform(x_coords, y_coords) 