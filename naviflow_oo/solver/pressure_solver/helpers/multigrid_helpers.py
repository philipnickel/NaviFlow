"""
Helper functions for the multigrid solver.
"""

import numpy as np

def restrict(fine_grid: np.ndarray) -> np.ndarray:
    """
    Reduces a fine grid to a coarse grid by taking every other point.
    Uses direct injection at odd indices (1, 3, 5, ...) for consistent
    behavior with Fortran ordering.
    
    Parameters:
        fine_grid (np.ndarray): The input fine grid to be coarsened
        
    Returns:
        np.ndarray: The coarsened grid
    """
    if fine_grid.ndim == 1:
        # For 1D arrays, reshape to 2D with Fortran ordering
        size = int(np.sqrt(fine_grid.size))
        grid_2d = fine_grid.reshape((size, size), order='F')
        
        # Apply restriction in 2D
        coarse_grid = grid_2d[1::2, 1::2]
        
        # Return as 1D array with Fortran ordering
        return coarse_grid.flatten(order='F')
    else:
        # For 2D arrays, apply direct injection
        return fine_grid[1::2, 1::2]


def interpolate(coarse_grid, m):
    """
    Interpolates a coarse grid to a fine grid using bilinear interpolation.
    Maintains Fortran ordering consistency.
    
    Parameters:
        coarse_grid (np.ndarray): The input coarse grid to be interpolated
        m (int): Size of the target fine grid
        
    Returns:
        np.ndarray: The interpolated fine grid
    """
    # Reshape to 2D if needed, using Fortran ordering
    if coarse_grid.ndim == 1:
        mc = int(np.sqrt(coarse_grid.size))
        coarse_grid = coarse_grid.reshape((mc, mc), order='F')
    else:
        # If already 2D, get the coarse grid dimensions
        mc = coarse_grid.shape[0]
    
    # Create fine grid
    fine_grid = np.zeros((m, m))
    
    # Handle edge cases for small grids
    if m <= 3:
        # Direct injection for coincident points
        i_coarse = np.arange(mc)
        j_coarse = np.arange(mc)
        I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
        
        # Calculate fine grid indices
        I_fine = 2 * I_coarse + 1
        J_fine = 2 * J_coarse + 1
        
        # Filter valid indices
        mask = np.logical_and(I_fine < m, J_fine < m)
        fine_grid[I_fine[mask], J_fine[mask]] = coarse_grid[I_coarse[mask], J_coarse[mask]]
        
        # Return with proper ordering if input was 1D
        if coarse_grid.ndim == 1:
            return fine_grid.flatten(order='F')
        return fine_grid
    
    # Direct injection for coincident points
    i_coarse = np.arange(mc)
    j_coarse = np.arange(mc)
    I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
    
    # Calculate fine grid indices
    I_fine = 2 * I_coarse + 1
    J_fine = 2 * J_coarse + 1
    
    # Filter valid indices
    mask = np.logical_and(I_fine < m, J_fine < m)
    fine_grid[I_fine[mask], J_fine[mask]] = coarse_grid[I_coarse[mask], J_coarse[mask]]
    
    # Horizontal interpolation (odd rows, even columns)
    i_coarse = np.arange(mc)
    j_coarse = np.arange(mc-1)
    I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
    
    I_fine = 2 * I_coarse + 1
    J_fine = 2 * J_coarse + 2
    
    mask = np.logical_and(I_fine < m, J_fine < m)
    fine_grid[I_fine[mask], J_fine[mask]] = 0.5 * (
        coarse_grid[I_coarse[mask], J_coarse[mask]] + 
        coarse_grid[I_coarse[mask], J_coarse[mask]+1]
    )
    
    # Vertical interpolation (even rows, odd columns)
    i_coarse = np.arange(mc-1)
    j_coarse = np.arange(mc)
    I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
    
    I_fine = 2 * I_coarse + 2
    J_fine = 2 * J_coarse + 1
    
    mask = np.logical_and(I_fine < m, J_fine < m)
    fine_grid[I_fine[mask], J_fine[mask]] = 0.5 * (
        coarse_grid[I_coarse[mask], J_coarse[mask]] + 
        coarse_grid[I_coarse[mask]+1, J_coarse[mask]]
    )
    
    # Diagonal interpolation (even rows, even columns)
    i_coarse = np.arange(mc-1)
    j_coarse = np.arange(mc-1)
    I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
    
    I_fine = 2 * I_coarse + 2
    J_fine = 2 * J_coarse + 2
    
    mask = np.logical_and(I_fine < m, J_fine < m)
    fine_grid[I_fine[mask], J_fine[mask]] = 0.25 * (
        coarse_grid[I_coarse[mask], J_coarse[mask]] + 
        coarse_grid[I_coarse[mask]+1, J_coarse[mask]] +
        coarse_grid[I_coarse[mask], J_coarse[mask]+1] +
        coarse_grid[I_coarse[mask]+1, J_coarse[mask]+1]
    )
    
    # Handle boundary values
    # Left boundary
    fine_grid[1:-1, 0] = fine_grid[1:-1, 1]
    # Right boundary
    fine_grid[1:-1, -1] = fine_grid[1:-1, -2]
    # Bottom boundary
    fine_grid[0, 1:-1] = fine_grid[1, 1:-1]
    # Top boundary
    fine_grid[-1, 1:-1] = fine_grid[-2, 1:-1]
    # Corners
    fine_grid[0, 0] = fine_grid[1, 1]
    fine_grid[0, -1] = fine_grid[1, -2]
    fine_grid[-1, 0] = fine_grid[-2, 1]
    fine_grid[-1, -1] = fine_grid[-2, -2]
    
    # If input was 1D, return 1D with consistent ordering
    if coarse_grid.ndim == 1:
        return fine_grid.flatten(order='F')
    
    return fine_grid
