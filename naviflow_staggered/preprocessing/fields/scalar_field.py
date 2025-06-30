"""
Scalar field implementation for pressure, temperature, etc.
"""

import numpy as np

class ScalarField:
    """
    Scalar field for CFD simulations (e.g., pressure).
    """
    def __init__(self, mesh, initial_value=0.0, name="scalar"):
        """
        Initialize a scalar field.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        initial_value : float or ndarray
            Initial value(s) for the field
        name : str
            Name of the field
        """
        self.mesh = mesh
        self.name = name
        self.nx, self.ny = mesh.get_dimensions()
        
        # Initialize the field
        if isinstance(initial_value, (int, float)):
            self.data = np.full((self.nx, self.ny), initial_value)
        else:
            self.data = np.array(initial_value)
            
        # Ensure correct shape
        if self.data.shape != (self.nx, self.ny):
            raise ValueError(f"Field shape {self.data.shape} does not match mesh dimensions ({self.nx}, {self.ny})")
    
    def get_data(self):
        """Return the field data."""
        return self.data
    
    def set_data(self, data):
        """Set the field data."""
        if data.shape != (self.nx, self.ny):
            raise ValueError(f"Data shape {data.shape} does not match field dimensions ({self.nx}, {self.ny})")
        self.data = data
    
    def set_boundary_value(self, boundary, value):
        """
        Set boundary values for the field.
        
        Parameters:
        -----------
        boundary : str
            Boundary name ('top', 'bottom', 'left', 'right')
        value : float or ndarray
            Value(s) to set at the boundary
        """
        if boundary == 'top':
            self.data[:, -1] = value
        elif boundary == 'bottom':
            self.data[:, 0] = value
        elif boundary == 'left':
            self.data[0, :] = value
        elif boundary == 'right':
            self.data[-1, :] = value
        else:
            raise ValueError(f"Unknown boundary: {boundary}") 