"""
Boundary conditions management for naviflow_oo.

This module provides classes and utilities for managing boundary conditions
in a consistent way across the simulation.
"""

import numpy as np
from enum import Enum, auto


class BoundaryType(Enum):
    """Enumeration of boundary condition types."""
    WALL = auto()
    VELOCITY = auto()
    PRESSURE = auto()
    INFLOW = auto()
    OUTFLOW = auto()
    SYMMETRY = auto()


class BoundaryLocation(Enum):
    """Enumeration of boundary locations."""
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()


class BoundaryCondition:
    """
    Class representing a single boundary condition.
    
    Attributes:
    -----------
    location : BoundaryLocation
        Location of the boundary
    type : BoundaryType
        Type of boundary condition
    values : dict
        Values for the boundary condition (e.g., velocity components)
    """
    
    def __init__(self, location, bc_type, values=None):
        """
        Initialize a boundary condition.
        
        Parameters:
        -----------
        location : BoundaryLocation or str
            Location of the boundary
        bc_type : BoundaryType or str
            Type of boundary condition
        values : dict, optional
            Values for the boundary condition
        """
        # Convert string to enum if needed
        if isinstance(location, str):
            location = location.upper()
            try:
                self.location = BoundaryLocation[location]
            except KeyError:
                raise ValueError(f"Unknown boundary location: {location}")
        else:
            self.location = location
            
        # Convert string to enum if needed
        if isinstance(bc_type, str):
            bc_type = bc_type.upper()
            try:
                self.type = BoundaryType[bc_type]
            except KeyError:
                raise ValueError(f"Unknown boundary type: {bc_type}")
        else:
            self.type = bc_type
            
        self.values = values or {}
    
    def get_value(self, key, default=0.0):
        """Get a value from the boundary condition."""
        return self.values.get(key, default)


class BoundaryConditionManager:
    """
    Manager for all boundary conditions in a simulation.
    
    This class provides methods to set, get, and apply boundary conditions
    to various fields in the simulation.
    """
    
    def __init__(self):
        """Initialize an empty boundary condition manager."""
        self.conditions = {}
        
    def set_condition(self, location, bc_type, values=None):
        """
        Set a boundary condition.
        
        Parameters:
        -----------
        location : BoundaryLocation or str
            Location of the boundary
        bc_type : BoundaryType or str
            Type of boundary condition
        values : dict, optional
            Values for the boundary condition
        """
        # Convert to string for dictionary key if needed
        if isinstance(location, BoundaryLocation):
            location_key = location.name.lower()
        else:
            location_key = location.lower()
            
        # Create the boundary condition
        bc = BoundaryCondition(location, bc_type, values)
        
        # Store in dictionary
        if location_key not in self.conditions:
            self.conditions[location_key] = {}
            
        if isinstance(bc_type, BoundaryType):
            bc_type_key = bc_type.name.lower()
        else:
            bc_type_key = bc_type.lower()
            
        self.conditions[location_key][bc_type_key] = bc.values or {}
    
    def get_condition(self, location, bc_type=None):
        """
        Get a boundary condition.
        
        Parameters:
        -----------
        location : BoundaryLocation or str
            Location of the boundary
        bc_type : BoundaryType or str, optional
            Type of boundary condition
            
        Returns:
        --------
        dict or None
            Boundary condition values or None if not found
        """
        # Convert to string for dictionary key if needed
        if isinstance(location, BoundaryLocation):
            location_key = location.name.lower()
        else:
            location_key = location.lower()
            
        if location_key not in self.conditions:
            return None
            
        if bc_type is None:
            return self.conditions[location_key]
            
        if isinstance(bc_type, BoundaryType):
            bc_type_key = bc_type.name.lower()
        else:
            bc_type_key = bc_type.lower()
            
        return self.conditions[location_key].get(bc_type_key)
    
    def apply_velocity_boundary_conditions(self, u, v, nx, ny):
        """
        Apply velocity boundary conditions to u and v fields.
        
        Parameters:
        -----------
        u, v : ndarray
            Velocity fields
        nx, ny : int
            Grid dimensions
            
        Returns:
        --------
        u, v : ndarray
            Updated velocity fields
        """
        # Initialize all boundaries to zero (wall condition)
        u[0, :] = 0.0                      # left wall
        # Right u wall: Check shape before indexing nx
        if u.shape[0] == nx + 1:
            u[nx, :] = 0.0
        elif u.shape[0] == nx and nx > 0: # Check nx > 0 to avoid index -1
             u[nx - 1, :] = 0.0
        
        u[:, 0] = 0.0                      # bottom wall
        # Top u wall: Check shape before indexing ny-1
        if u.shape[1] > ny - 1 and ny > 0: # Check ny > 0 to avoid index -1
             u[:, ny - 1] = 0.0

        v[0, :] = 0.0                      # left wall
        # Right v wall: Check shape before indexing nx-1
        if v.shape[0] > nx - 1 and nx > 0: # Check nx > 0 to avoid index -1
             v[nx - 1, :] = 0.0

        v[:, 0] = 0.0                      # bottom wall
        # Top v wall: Check shape before indexing ny
        if v.shape[1] == ny + 1:
            v[:, ny] = 0.0
        elif v.shape[1] == ny and ny > 0: # Check ny > 0 to avoid index -1
            v[:, ny - 1] = 0.0
        
        # Apply specific boundary conditions
        for location, conditions in self.conditions.items():
            for bc_type, values in conditions.items():
                if bc_type == 'velocity':
                    if location == 'top':
                        # Top u velocity: Check shape before indexing ny-1
                        if u.shape[1] > ny - 1 and ny > 0:
                            u[:, ny - 1] = values.get('u', 0.0)
                        # Top v velocity: Check shape before indexing ny
                        if v.shape[1] == ny + 1:
                            v[:, ny] = values.get('v', 0.0)
                        elif v.shape[1] == ny and ny > 0:
                            v[:, ny - 1] = values.get('v', 0.0)
                    elif location == 'bottom':
                        u[:, 0] = values.get('u', 0.0)     # bottom wall
                        v[:, 0] = values.get('v', 0.0)     # bottom wall
                    elif location == 'left':
                        u[0, :] = values.get('u', 0.0)     # left wall
                        v[0, :] = values.get('v', 0.0)     # left wall
                    elif location == 'right':
                         # Right u velocity: Check shape before indexing nx
                        if u.shape[0] == nx + 1:
                             u[nx, :] = values.get('u', 0.0)
                        elif u.shape[0] == nx and nx > 0:
                             u[nx - 1, :] = values.get('u', 0.0)
                         # Right v velocity: Check shape before indexing nx-1
                        if v.shape[0] > nx - 1 and nx > 0:
                             v[nx - 1, :] = values.get('v', 0.0)
                elif bc_type == 'wall':
                    # Wall boundary condition (zero velocity)
                    if location == 'top':
                        # Top u wall: Check shape before indexing ny-1
                        if u.shape[1] > ny - 1 and ny > 0:
                            u[:, ny - 1] = 0.0
                        # Top v wall: Check shape before indexing ny
                        if v.shape[1] == ny + 1:
                             v[:, ny] = 0.0
                        elif v.shape[1] == ny and ny > 0:
                             v[:, ny - 1] = 0.0
                    elif location == 'bottom':
                        u[:, 0] = 0.0
                        v[:, 0] = 0.0
                    elif location == 'left':
                        u[0, :] = 0.0
                        v[0, :] = 0.0
                    elif location == 'right':
                         # Right u wall: Check shape before indexing nx
                        if u.shape[0] == nx + 1:
                             u[nx, :] = 0.0
                        elif u.shape[0] == nx and nx > 0:
                             u[nx - 1, :] = 0.0
                         # Right v wall: Check shape before indexing nx-1
                        if v.shape[0] > nx - 1 and nx > 0:
                             v[nx - 1, :] = 0.0
        
        return u, v
    
    def to_dict(self):
        """Convert boundary conditions to a dictionary format."""
        return self.conditions 
        
    def get_boundary_types(self):
        """
        Get a dictionary mapping boundary locations to their boundary types.
        
        Returns:
        --------
        dict
            Dictionary with boundary locations as keys and their types as values.
            For boundaries with multiple types, returns the first type.
        """
        boundary_types = {}
        for location, conditions in self.conditions.items():
            # Use the first boundary type defined for each location
            if conditions:
                first_type = next(iter(conditions.keys()))
                boundary_types[location] = first_type
        
        # Ensure all four boundaries are included
        for boundary in ['top', 'bottom', 'left', 'right']:
            if boundary not in boundary_types:
                boundary_types[boundary] = 'wall'  # Default to wall
                
        return boundary_types 