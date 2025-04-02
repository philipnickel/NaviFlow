"""
Helper functions for the multigrid solver.
"""

import numpy as np

def restrict_coefficients(d_u, d_v, nx_fine, ny_fine, nx_coarse, ny_coarse, dx_fine, dy_fine):
    """
    Properly restrict coefficients from fine to coarse grid using harmonic averaging.
    
    Parameters:
    -----------
    d_u, d_v : ndarray
        Momentum equation coefficients on fine grid
    nx_fine, ny_fine : int
        Dimensions of fine grid
    nx_coarse, ny_coarse : int
        Dimensions of coarse grid
    dx_fine, dy_fine : float
        Cell sizes on fine grid
    
    Returns:
    --------
    d_u_coarse, d_v_coarse : ndarray
        Properly restricted coefficients for coarse grid
    """
    # Initialize coarse grid coefficients
    d_u_coarse = np.zeros((nx_coarse+1, ny_coarse))
    d_v_coarse = np.zeros((nx_coarse, ny_coarse+1))
    
    # Scale factor for Poisson equation
    dx_coarse = dx_fine * 2
    dy_coarse = dy_fine * 2
    scale_factor = (dx_fine/dx_coarse)**2
    
    # FULLY VECTORIZED IMPLEMENTATION
    
    # --- For d_u coefficients (interior points) ---
    # Create meshgrid of coarse grid indices
    i_coarse = np.arange(1, nx_coarse)
    j_coarse = np.arange(ny_coarse)
    I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
    
    # Map to fine grid indices (multiply by 2)
    I_fine = 2 * I_coarse
    J_fine = 2 * J_coarse
    
    # Create mask for valid indices (within fine grid bounds)
    valid_mask = (I_fine < nx_fine) & (J_fine < ny_fine)
    
    # Extract values from fine grid only where valid
    # Need to use advanced indexing, so we flatten the indices
    valid_indices = np.where(valid_mask)
    i_fine_valid = I_fine[valid_indices]
    j_fine_valid = J_fine[valid_indices]
    
    # Get the actual values
    d1 = d_u[i_fine_valid, j_fine_valid]
    d2 = d_u[i_fine_valid + 1, j_fine_valid]
    
    # Compute harmonic mean where both values are positive
    harmonic_mask = (d1 > 0) & (d2 > 0)
    
    # Initialize with arithmetic mean
    result = 0.5 * (d1 + d2)
    
    # Apply harmonic mean where applicable
    if np.any(harmonic_mask):
        result[harmonic_mask] = 2.0 / (1.0 / d1[harmonic_mask] + 1.0 / d2[harmonic_mask])
    
    # Assign results back to coarse grid
    i_coarse_valid = I_coarse[valid_indices]
    j_coarse_valid = J_coarse[valid_indices]
    d_u_coarse[i_coarse_valid, j_coarse_valid] = result
    
    # --- For d_v coefficients (interior points) ---
    # Create meshgrid of coarse grid indices
    i_coarse = np.arange(nx_coarse)
    j_coarse = np.arange(1, ny_coarse)
    I_coarse, J_coarse = np.meshgrid(i_coarse, j_coarse, indexing='ij')
    
    # Map to fine grid indices (multiply by 2)
    I_fine = 2 * I_coarse
    J_fine = 2 * J_coarse
    
    # Create mask for valid indices (within fine grid bounds)
    valid_mask = (I_fine < nx_fine) & (J_fine < ny_fine)
    
    # Extract values from fine grid only where valid
    valid_indices = np.where(valid_mask)
    i_fine_valid = I_fine[valid_indices]
    j_fine_valid = J_fine[valid_indices]
    
    # Get the actual values
    d1 = d_v[i_fine_valid, j_fine_valid]
    d2 = d_v[i_fine_valid, j_fine_valid + 1]
    
    # Compute harmonic mean where both values are positive
    harmonic_mask = (d1 > 0) & (d2 > 0)
    
    # Initialize with arithmetic mean
    result = 0.5 * (d1 + d2)
    
    # Apply harmonic mean where applicable
    if np.any(harmonic_mask):
        result[harmonic_mask] = 2.0 / (1.0 / d1[harmonic_mask] + 1.0 / d2[harmonic_mask])
    
    # Assign results back to coarse grid
    i_coarse_valid = I_coarse[valid_indices]
    j_coarse_valid = J_coarse[valid_indices]
    d_v_coarse[i_coarse_valid, j_coarse_valid] = result
    
    # --- Handle boundary coefficients using vectorized operations ---
    
    # Left boundary (i=0) for d_u
    j_indices = np.arange(ny_coarse)
    j_fine_indices = 2 * j_indices
    valid_j = j_fine_indices < ny_fine
    d_u_coarse[0, valid_j] = d_u[0, j_fine_indices[valid_j]]
    
    # Right boundary (i=nx_coarse) for d_u
    d_u_coarse[nx_coarse, valid_j] = d_u[nx_fine, j_fine_indices[valid_j]]
    
    # Bottom boundary (j=0) for d_v
    i_indices = np.arange(nx_coarse)
    i_fine_indices = 2 * i_indices
    valid_i = i_fine_indices < nx_fine
    d_v_coarse[valid_i, 0] = d_v[i_fine_indices[valid_i], 0]
    
    # Top boundary (j=ny_coarse) for d_v
    d_v_coarse[valid_i, ny_coarse] = d_v[i_fine_indices[valid_i], ny_fine]
    
    # Apply scaling based on the PDE and grid coarsening
    d_u_coarse *= scale_factor
    d_v_coarse *= scale_factor
    
    return d_u_coarse, d_v_coarse

def restrict(fine_grid, out=None):
    """
    Full weighting restriction from fine to coarse grid.
    Standard 9-point stencil with weights:
    1/16 * [1 2 1]
           [2 4 2]
           [1 2 1]
    """
    # Calculate size of coarse grid
    coarse_shape = ((fine_grid.shape[0] - 1) // 2, (fine_grid.shape[1] - 1) // 2)
    
    # Create output array if needed
    if out is None:
        coarse_grid = np.zeros(coarse_shape)
    else:
        coarse_grid = out
        coarse_grid.fill(0.0)
    
    # Apply full weighting using slicing with consistent indices
    # Center points (weight 4/16)
    coarse_grid += 4 * fine_grid[1::2, 1::2][:coarse_shape[0], :coarse_shape[1]]
    
    # Edge points (weight 2/16)
    coarse_grid += 2 * fine_grid[0::2, 1::2][:coarse_shape[0], :coarse_shape[1]]  # Top
    coarse_grid += 2 * fine_grid[2::2, 1::2][:coarse_shape[0], :coarse_shape[1]]  # Bottom
    coarse_grid += 2 * fine_grid[1::2, 0::2][:coarse_shape[0], :coarse_shape[1]]  # Left
    coarse_grid += 2 * fine_grid[1::2, 2::2][:coarse_shape[0], :coarse_shape[1]]  # Right
    
    # Corner points (weight 1/16)
    coarse_grid += fine_grid[0::2, 0::2][:coarse_shape[0], :coarse_shape[1]]  # Top-left
    coarse_grid += fine_grid[0::2, 2::2][:coarse_shape[0], :coarse_shape[1]]  # Top-right
    coarse_grid += fine_grid[2::2, 0::2][:coarse_shape[0], :coarse_shape[1]]  # Bottom-left
    coarse_grid += fine_grid[2::2, 2::2][:coarse_shape[0], :coarse_shape[1]]  # Bottom-right
    
    # Apply scaling
    coarse_grid /= 16.0
    
    return coarse_grid

def interpolate(coarse_grid, m, out=None):
    """
    Fully vectorized bilinear interpolation from coarse to fine grid.
    Handles both 1D and 2D input arrays.
    
    Parameters:
    -----------
    coarse_grid : ndarray
        The coarse grid values to interpolate from (1D or 2D)
    m : int
        Size of the fine grid (both dimensions will be m x m)
    out : ndarray, optional
        Output array to store results (created if not provided)
        
    Returns:
    --------
    fine_grid : ndarray
        The interpolated fine grid (m x m)
    """
    # Ensure coarse_grid is 2D
    if coarse_grid.ndim == 1:
        # If 1D, reshape to 2D square grid, assuming it's a flattened square grid
        mc = int(np.sqrt(coarse_grid.size))
        coarse_grid_2d = coarse_grid.reshape((mc, mc), order='F')
    else:
        coarse_grid_2d = coarse_grid
        
    # Get dimensions of coarse grid
    mc, nc = coarse_grid_2d.shape
    
    # Create output array if needed (use m x m as requested by caller)
    if out is None:
        fine_grid = np.zeros((m, m))
    else:
        fine_grid = out
        fine_grid.fill(0.0)
    
    # Calculate fine grid size that would result from coarse grid
    # The interpolated region will be (2*mc-1) x (2*nc-1), but
    # we'll fill it into the requested m x m grid
    mf = min(m, 2 * mc - 1)
    nf = min(m, 2 * nc - 1)
    
    # VECTORIZED IMPLEMENTATION
    
    # 1. Direct injection for coincident points (even-even indices)
    i_coarse = np.arange(mc)
    j_coarse = np.arange(nc)
    
    # Map to fine grid (multiply by 2)
    i_fine = 2 * i_coarse
    j_fine = 2 * j_coarse
    
    # Filter to ensure within bounds
    valid_i = i_fine < mf
    valid_j = j_fine < nf
    
    if np.any(valid_i) and np.any(valid_j):
        # Create meshgrid for valid indices
        I_coarse, J_coarse = np.meshgrid(i_coarse[valid_i], j_coarse[valid_j], indexing='ij')
        I_fine, J_fine = np.meshgrid(i_fine[valid_i], j_fine[valid_j], indexing='ij')
        
        # Direct injection (even-even)
        fine_grid[I_fine, J_fine] = coarse_grid_2d[I_coarse, J_coarse]
    
    # 2. Horizontal interpolation (even-odd indices)
    if nc > 1:  # Only if we have at least 2 columns
        j_coarse_h = np.arange(nc-1)  # One less for horizontal interpolation
        j_fine_h = 2 * j_coarse_h + 1  # Odd j indices
        
        # Filter to ensure within bounds
        valid_j_h = j_fine_h < nf
        
        if np.any(valid_i) and np.any(valid_j_h):
            # Create meshgrid for valid indices
            I_coarse_h, J_coarse_h = np.meshgrid(i_coarse[valid_i], j_coarse_h[valid_j_h], indexing='ij')
            I_fine_h, J_fine_h = np.meshgrid(i_fine[valid_i], j_fine_h[valid_j_h], indexing='ij')
            
            # Horizontal interpolation: average between left and right points
            fine_grid[I_fine_h, J_fine_h] = 0.5 * (
                coarse_grid_2d[I_coarse_h, J_coarse_h] + 
                coarse_grid_2d[I_coarse_h, J_coarse_h + 1]
            )
    
    # 3. Vertical interpolation (odd-even indices)
    if mc > 1:  # Only if we have at least 2 rows
        i_coarse_v = np.arange(mc-1)  # One less for vertical interpolation
        i_fine_v = 2 * i_coarse_v + 1  # Odd i indices
        
        # Filter to ensure within bounds
        valid_i_v = i_fine_v < mf
        
        if np.any(valid_i_v) and np.any(valid_j):
            # Create meshgrid for valid indices
            I_coarse_v, J_coarse_v = np.meshgrid(i_coarse_v[valid_i_v], j_coarse[valid_j], indexing='ij')
            I_fine_v, J_fine_v = np.meshgrid(i_fine_v[valid_i_v], j_fine[valid_j], indexing='ij')
            
            # Vertical interpolation: average between top and bottom points
            fine_grid[I_fine_v, J_fine_v] = 0.5 * (
                coarse_grid_2d[I_coarse_v, J_coarse_v] + 
                coarse_grid_2d[I_coarse_v + 1, J_coarse_v]
            )
    
    # 4. Diagonal interpolation (odd-odd indices)
    if mc > 1 and nc > 1:  # Only if we have at least 2 rows and 2 columns
        i_coarse_v = np.arange(mc-1)  # One less for vertical interpolation
        i_fine_v = 2 * i_coarse_v + 1  # Odd i indices
        j_coarse_h = np.arange(nc-1)  # One less for horizontal interpolation
        j_fine_h = 2 * j_coarse_h + 1  # Odd j indices
        
        # Filter to ensure within bounds
        valid_i_v = i_fine_v < mf
        valid_j_h = j_fine_h < nf
        
        if np.any(valid_i_v) and np.any(valid_j_h):
            # Create meshgrid for valid indices
            I_coarse_d, J_coarse_d = np.meshgrid(i_coarse_v[valid_i_v], j_coarse_h[valid_j_h], indexing='ij')
            I_fine_d, J_fine_d = np.meshgrid(i_fine_v[valid_i_v], j_fine_h[valid_j_h], indexing='ij')
            
            # Diagonal interpolation: average between all four corners
            fine_grid[I_fine_d, J_fine_d] = 0.25 * (
                coarse_grid_2d[I_coarse_d, J_coarse_d] + 
                coarse_grid_2d[I_coarse_d, J_coarse_d + 1] + 
                coarse_grid_2d[I_coarse_d + 1, J_coarse_d] + 
                coarse_grid_2d[I_coarse_d + 1, J_coarse_d + 1]
            )
    
    return fine_grid

# Remove the unused interpolate2 function to save memory
"""
def interpolate2(coarse_grid: np.ndarray, m: int) -> np.ndarray:
    # This function is unused and has been removed to save memory
    pass
"""
