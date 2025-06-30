"""
Helper functions for enforcing boundary conditions in pressure solvers.
"""

import numpy as np

def enforce_zero_gradient_bc(p):
    """
    Enforce zero gradient (Neumann) boundary conditions on a pressure field.
    
    Parameters:
    -----------
    p : ndarray
        2D array representing the pressure field
        
    Returns:
    --------
    ndarray
        Pressure field with zero gradient boundary conditions applied
    """
    if p.ndim != 2:
        raise ValueError("Input pressure field must be a 2D array")
    
    nx, ny = p.shape
    
    # West boundary (i = 0)
    p[0, :] = p[1, :]
    
    # East boundary (i = nx-1)
    p[nx-1, :] = p[nx-2, :]
    
    # South boundary (j = 0)
    p[:, 0] = p[:, 1]
    
    # North boundary (j = ny-1)
    p[:, ny-1] = p[:, ny-2]
    
    return p

def enforce_zero_pressure_bc(p, boundaries=None):
    """
    Enforce zero pressure (Dirichlet) boundary conditions on a pressure field.
    
    Parameters:
    -----------
    p : ndarray
        2D array representing the pressure field
    boundaries : list or None, optional
        Specifies which boundaries to apply zero pressure BC.
        Can include 'west', 'east', 'south', 'north', or 'all'.
        If None, applies to all boundaries.
        
    Returns:
    --------
    ndarray
        Pressure field with zero pressure boundary conditions applied
    """
    if p.ndim != 2:
        raise ValueError("Input pressure field must be a 2D array")
    
    nx, ny = p.shape
    
    # Default to all boundaries if not specified
    if boundaries is None:
        boundaries = ['west', 'east', 'south', 'north']
    elif boundaries == 'all':
        boundaries = ['west', 'east', 'south', 'north']
    
    # Apply zero pressure conditions to specified boundaries
    if 'west' in boundaries:
        p[0, :] = 0.0
    
    if 'east' in boundaries:
        p[nx-1, :] = 0.0
    
    if 'south' in boundaries:
        p[:, 0] = 0.0
    
    if 'north' in boundaries:
        p[:, ny-1] = 0.0
    
    return p 