"""
Helper functions for the multigrid solver.
"""

import numpy as np

def restrict(fine_grid: np.ndarray, out=None) -> np.ndarray:
    """
    Reduces a fine grid to a coarse grid by taking every other point.
    Uses direct injection at odd indices (1, 3, 5, ...) for consistent
    behavior with Fortran ordering.
    
    Parameters:
        fine_grid (np.ndarray): The input fine grid to be coarsened
        out (np.ndarray, optional): Optional output array for the result
        
    Returns:
        np.ndarray: The coarsened grid
    """
    # Calculate size of coarse grid
    coarse_shape = ((fine_grid.shape[0] - 1) // 2, (fine_grid.shape[1] - 1) // 2)
    
    # Create or reuse output array
    if out is None:
        coarse_grid = np.zeros(coarse_shape)
    else:
        coarse_grid = out
        if coarse_grid.shape != coarse_shape:
            raise ValueError(f"Output array has wrong shape: {coarse_grid.shape}, expected {coarse_shape}")
        # Clear the array if reusing
        coarse_grid.fill(0.0)
    
    # Perform restriction by direct injection
    coarse_grid[:, :] = fine_grid[1::2, 1::2]
    
    return coarse_grid


def interpolate(coarse_grid, m, out=None):
    """
    Interpolates a coarse grid to a fine grid using bilinear interpolation.
    Maintains Fortran ordering consistency.
    
    Parameters:
        coarse_grid (np.ndarray): The input coarse grid to be interpolated
        m (int): Size of the target fine grid
        out (np.ndarray, optional): Optional output array for the result
        
    Returns:
        np.ndarray: The interpolated fine grid
    """
    # Reshape to 2D if needed, using Fortran ordering
    is_1d_input = False
    if coarse_grid.ndim == 1:
        is_1d_input = True
        mc = int(np.sqrt(coarse_grid.size))
        coarse_grid = coarse_grid.reshape((mc, mc), order='F')
    else:
        # If already 2D, get the coarse grid dimensions
        mc = coarse_grid.shape[0]
    
    # Create or reuse fine grid
    if out is None:
        fine_grid = np.zeros((m, m))
    else:
        if is_1d_input:
            if out.size != m*m:
                raise ValueError(f"Output array has wrong size: {out.size}, expected {m*m}")
            fine_grid = out.reshape((m, m), order='F')
        else:
            if out.shape != (m, m):
                raise ValueError(f"Output array has wrong shape: {out.shape}, expected ({m}, {m})")
            fine_grid = out
        # Clear the array if reusing
        fine_grid.fill(0.0)
    
    # Handle edge cases for small grids
    if m <= 3:
        # Direct injection for coincident points
        if mc > 0:
            fine_grid[1:m:2, 1:m:2] = coarse_grid[:min(mc, (m-1)//2+1), :min(mc, (m-1)//2+1)]
        
        # Return with proper ordering if input was 1D
        if is_1d_input:
            if out is None:
                return fine_grid.flatten(order='F')
            else:
                # If out was provided, it's already the right shape
                return out
        return fine_grid
    
    # Direct injection for coincident points
    fine_grid[1:m:2, 1:m:2] = coarse_grid[:min(mc, (m-1)//2+1), :min(mc, (m-1)//2+1)]
    
    # Horizontal interpolation (odd rows, even columns)
    for i in range(mc):
        i_f = 2 * i + 1  # Fine grid row index
        if i_f >= m:
            break
            
        for j in range(mc-1):
            j_f = 2 * j + 2  # Fine grid column index
            if j_f >= m:
                break
                
            fine_grid[i_f, j_f] = 0.5 * (coarse_grid[i, j] + coarse_grid[i, j+1])
    
    # Vertical interpolation (even rows, odd columns)
    for i in range(mc-1):
        i_f = 2 * i + 2  # Fine grid row index
        if i_f >= m:
            break
            
        for j in range(mc):
            j_f = 2 * j + 1  # Fine grid column index
            if j_f >= m:
                break
                
            fine_grid[i_f, j_f] = 0.5 * (coarse_grid[i, j] + coarse_grid[i+1, j])
    
    # Diagonal interpolation (even rows, even columns)
    for i in range(mc-1):
        i_f = 2 * i + 2  # Fine grid row index
        if i_f >= m:
            break
            
        for j in range(mc-1):
            j_f = 2 * j + 2  # Fine grid column index
            if j_f >= m:
                break
                
            fine_grid[i_f, j_f] = 0.25 * (
                coarse_grid[i, j] + 
                coarse_grid[i+1, j] +
                coarse_grid[i, j+1] +
                coarse_grid[i+1, j+1]
            )
    
    # Handle boundary values
    # Left boundary
    if m > 2:
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
    
    # If input was 1D and no output array was provided, return 1D with consistent ordering
    if is_1d_input:
        if out is None:
            return fine_grid.flatten(order='F')
        else:
            # If out was provided, it's already the right shape
            return out
    
    return fine_grid

# Remove the unused interpolate2 function to save memory
"""
def interpolate2(coarse_grid: np.ndarray, m: int) -> np.ndarray:
    # This function is unused and has been removed to save memory
    pass
"""
