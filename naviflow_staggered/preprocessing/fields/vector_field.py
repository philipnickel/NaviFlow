"""
Vector field implementation for velocity, gradients, etc.
"""

import numpy as np

class VectorField:
    """
    Vector field for CFD simulations (e.g., velocity).
    """
    def __init__(self, mesh, staggered=True, initial_value=(0.0, 0.0), name="vector"):
        """
        Initialize a vector field.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        staggered : bool
            Whether to use a staggered grid
        initial_value : tuple or (ndarray, ndarray)
            Initial values for the field components
        name : str
            Name of the field
        """
        self.mesh = mesh
        self.name = name
        self.staggered = staggered
        self.nx, self.ny = mesh.get_dimensions()
        
        # Initialize the field components
        if staggered:
            # Staggered grid: u at i+1/2,j and v at i,j+1/2
            if isinstance(initial_value[0], (int, float)):
                self.u = np.full((self.nx+1, self.ny), initial_value[0])
            else:
                self.u = np.array(initial_value[0])
                
            if isinstance(initial_value[1], (int, float)):
                self.v = np.full((self.nx, self.ny+1), initial_value[1])
            else:
                self.v = np.array(initial_value[1])
                
            # Ensure correct shapes
            if self.u.shape != (self.nx+1, self.ny):
                raise ValueError(f"U component shape {self.u.shape} does not match staggered mesh dimensions ({self.nx+1}, {self.ny})")
            if self.v.shape != (self.nx, self.ny+1):
                raise ValueError(f"V component shape {self.v.shape} does not match staggered mesh dimensions ({self.nx}, {self.ny+1})")
        else:
            # Collocated grid: u and v at cell centers
            if isinstance(initial_value[0], (int, float)):
                self.u = np.full((self.nx, self.ny), initial_value[0])
            else:
                self.u = np.array(initial_value[0])
                
            if isinstance(initial_value[1], (int, float)):
                self.v = np.full((self.nx, self.ny), initial_value[1])
            else:
                self.v = np.array(initial_value[1])
                
            # Ensure correct shapes
            if self.u.shape != (self.nx, self.ny):
                raise ValueError(f"U component shape {self.u.shape} does not match mesh dimensions ({self.nx}, {self.ny})")
            if self.v.shape != (self.nx, self.ny):
                raise ValueError(f"V component shape {self.v.shape} does not match mesh dimensions ({self.nx}, {self.ny})")
    
    def get_components(self):
        """Return the field components."""
        return self.u, self.v
    
    def set_components(self, u, v):
        """Set the field components."""
        if self.staggered:
            if u.shape != (self.nx+1, self.ny):
                raise ValueError(f"U component shape {u.shape} does not match staggered mesh dimensions ({self.nx+1}, {self.ny})")
            if v.shape != (self.nx, self.ny+1):
                raise ValueError(f"V component shape {v.shape} does not match staggered mesh dimensions ({self.nx}, {self.ny+1})")
        else:
            if u.shape != (self.nx, self.ny):
                raise ValueError(f"U component shape {u.shape} does not match mesh dimensions ({self.nx}, {self.ny})")
            if v.shape != (self.nx, self.ny):
                raise ValueError(f"V component shape {v.shape} does not match mesh dimensions ({self.nx}, {self.ny})")
        
        self.u = u
        self.v = v
    
    def set_boundary_value(self, boundary, u_value=0.0, v_value=0.0):
        """
        Set boundary values for the field.
        
        Parameters:
        -----------
        boundary : str
            Boundary name ('top', 'bottom', 'left', 'right')
        u_value, v_value : float or ndarray
            Values to set at the boundary
        """
        if self.staggered:
            if boundary == 'top':
                self.u[:, self.ny-1] = u_value
                # v is staggered, so we need to set the value at the boundary and one cell in
                self.v[:, self.ny] = -self.v[:, self.ny-1]
            elif boundary == 'bottom':
                self.u[:, 0] = u_value
                self.v[:, 0] = -self.v[:, 1]
            elif boundary == 'left':
                self.u[0, :] = -self.u[1, :]
                self.v[0, :] = v_value
            elif boundary == 'right':
                self.u[self.nx, :] = -self.u[self.nx-1, :]
                self.v[self.nx-1, :] = v_value
            else:
                raise ValueError(f"Unknown boundary: {boundary}")
        else:
            # Implementation for collocated grid
            pass 