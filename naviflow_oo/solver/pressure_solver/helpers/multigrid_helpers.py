"""
Helper functions for the multigrid solver.
"""

import numpy as np

def restrict(fine_grid):
    """
    Restricts a fine grid to a coarse grid using injection with averaging.
    Includes proper scaling to maintain magnitudes between levels.
    
    Parameters:
    -----------
    fine_grid : ndarray
        The input fine grid to be restricted (2D)
        
    Returns:
    --------
    ndarray
        The restricted grid (2D)
    """
    # Reshape to 2D if needed
    if fine_grid.ndim == 1:
        m = int(np.sqrt(fine_grid.size))
        fine_grid = fine_grid.reshape((m, m), order='F')
        
    m = fine_grid.shape[0]
    mc = (m + 1) // 2 - 1  # Size of coarse grid
    
    # Print initial grid statistics
    print(f"\nFine grid ({m}x{m}) statistics:")
    print(f"Min: {np.min(fine_grid):.6f}, Max: {np.max(fine_grid):.6f}, Mean: {np.mean(fine_grid):.6f}")
    print(f"Number of negative values: {np.sum(fine_grid < 0)}")
    
    # Create coarse grid
    coarse_grid = np.zeros((mc, mc))
    
    # Vectorized restriction with simple averaging
    # Create indices for the fine grid points that correspond to coarse grid points
    i_fine = np.arange(1, m, 2)[:mc]
    j_fine = np.arange(1, m, 2)[:mc]
    I_fine, J_fine = np.meshgrid(i_fine, j_fine, indexing='ij')
    
    # Center points (weight 1/2)
    center_points = fine_grid[I_fine, J_fine]
    
    # Print center points statistics
    print(f"\nCenter points statistics:")
    print(f"Min: {np.min(center_points):.6f}, Max: {np.max(center_points):.6f}, Mean: {np.mean(center_points):.6f}")
    print(f"Number of negative values: {np.sum(center_points < 0)}")
    
    # Adjacent points (weight 1/8)
    # Create masks for valid adjacent indices
    valid_im1 = I_fine > 0
    valid_ip1 = I_fine < m-1
    valid_jm1 = J_fine > 0
    valid_jp1 = J_fine < m-1
    
    # Initialize adjacent sum with zeros
    adjacent_sum = np.zeros((mc, mc))
    
    # Add valid adjacent points
    adjacent_count = np.zeros((mc, mc))
    
    # Left points
    mask = valid_im1
    adjacent_sum[mask] += fine_grid[I_fine[mask]-1, J_fine[mask]]
    adjacent_count[mask] += 1
    
    # Right points
    mask = valid_ip1
    adjacent_sum[mask] += fine_grid[I_fine[mask]+1, J_fine[mask]]
    adjacent_count[mask] += 1
    
    # Bottom points
    mask = valid_jm1
    adjacent_sum[mask] += fine_grid[I_fine[mask], J_fine[mask]-1]
    adjacent_count[mask] += 1
    
    # Top points
    mask = valid_jp1
    adjacent_sum[mask] += fine_grid[I_fine[mask], J_fine[mask]+1]
    adjacent_count[mask] += 1
    
    # Print adjacent points statistics
    print(f"\nAdjacent points statistics:")
    print(f"Min: {np.min(adjacent_sum):.6f}, Max: {np.max(adjacent_sum):.6f}, Mean: {np.mean(adjacent_sum):.6f}")
    print(f"Number of negative values: {np.sum(adjacent_sum < 0)}")
    
    # Avoid division by zero
    adjacent_count[adjacent_count == 0] = 1
    
    # Normalize adjacent points first
    adjacent_avg = adjacent_sum / adjacent_count
    
    # Print normalized adjacent points statistics
    print(f"\nNormalized adjacent points statistics:")
    print(f"Min: {np.min(adjacent_avg):.6f}, Max: {np.max(adjacent_avg):.6f}, Mean: {np.mean(adjacent_avg):.6f}")
    print(f"Number of negative values: {np.sum(adjacent_avg < 0)}")
    
    # Combine with weights that preserve magnitude
    coarse_grid = 0.5 * center_points + 0.125 * adjacent_avg
    
    # Print pre-scaling statistics
    print(f"\nPre-scaling statistics:")
    print(f"Min: {np.min(coarse_grid):.6f}, Max: {np.max(coarse_grid):.6f}, Mean: {np.mean(coarse_grid):.6f}")
    print(f"Number of negative values: {np.sum(coarse_grid < 0)}")
    
    # Scale to maintain proper magnitude between levels
    # This accounts for the grid size change
    scale_factor = (m + 1) / (mc + 1)
    coarse_grid *= scale_factor
    
    # Print final statistics
    print(f"\nFinal coarse grid ({mc}x{mc}) statistics:")
    print(f"Scale factor: {scale_factor:.6f}")
    print(f"Min: {np.min(coarse_grid):.6f}, Max: {np.max(coarse_grid):.6f}, Mean: {np.mean(coarse_grid):.6f}")
    print(f"Number of negative values: {np.sum(coarse_grid < 0)}")
    
    return coarse_grid

def interpolate(coarse_grid, m):
    """
    Interpolates a coarse grid to a fine grid using bilinear interpolation.
    
    Parameters:
    -----------
    coarse_grid : ndarray
        The input coarse grid to be interpolated (2D)
    m : int
        Size of the target fine grid (m x m)
        
    Returns:
    --------
    ndarray
        The interpolated fine grid (2D)
    """
    # Reshape to 2D if needed
    if coarse_grid.ndim == 1:
        mc = int(np.sqrt(coarse_grid.size))
        coarse_grid = coarse_grid.reshape((mc, mc), order='F')
        
    # Get coarse grid dimensions
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
    
    return fine_grid

def restrict_coefficients(d_u, d_v, nx, ny):
    """
    Restrict the momentum equation coefficients to a coarser grid using arithmetic averaging.
    This maintains better magnitude consistency between levels compared to harmonic averaging.
    
    Parameters:
    -----------
    d_u, d_v : ndarray
        Momentum equation coefficients on the fine grid
    nx, ny : int
        Dimensions of the fine grid
        
    Returns:
    --------
    d_u_coarse, d_v_coarse : ndarray
        Momentum equation coefficients on the coarse grid
    """
    # Calculate coarse grid dimensions
    nx_coarse = (nx + 1) // 2 - 1
    ny_coarse = (ny + 1) // 2 - 1
    
    # Initialize coarse grid coefficients
    d_u_coarse = np.zeros((nx_coarse + 1, ny_coarse))
    d_v_coarse = np.zeros((nx_coarse, ny_coarse + 1))
    
    # Vectorized restriction for d_u (staggered in x-direction)
    # Create indices for the fine grid points
    i_fine = np.arange(0, min(2*nx_coarse+1, d_u.shape[0]), 2)
    j_fine = np.arange(1, min(2*ny_coarse+1, d_u.shape[1]), 2)
    I_fine, J_fine = np.meshgrid(i_fine, j_fine, indexing='ij')
    
    # Create masks for valid adjacent indices
    valid_jm1 = J_fine > 0
    valid_jp1 = J_fine < d_u.shape[1]-1
    
    # Initialize sum and count arrays for arithmetic averaging
    d_u_sum = np.zeros((nx_coarse + 1, ny_coarse))
    d_u_count = np.ones((nx_coarse + 1, ny_coarse))
    
    # Add center points
    d_u_sum += d_u[I_fine[:nx_coarse+1, :ny_coarse], J_fine[:nx_coarse+1, :ny_coarse]]
    
    # Add points below if available
    mask = valid_jm1[:nx_coarse+1, :ny_coarse]
    if np.any(mask):
        d_u_sum[mask] += d_u[I_fine[:nx_coarse+1, :ny_coarse][mask], J_fine[:nx_coarse+1, :ny_coarse][mask]-1]
        d_u_count[mask] += 1
    
    # Add points above if available
    mask = valid_jp1[:nx_coarse+1, :ny_coarse]
    if np.any(mask):
        d_u_sum[mask] += d_u[I_fine[:nx_coarse+1, :ny_coarse][mask], J_fine[:nx_coarse+1, :ny_coarse][mask]+1]
        d_u_count[mask] += 1
    
    # Arithmetic average
    d_u_coarse = d_u_sum / d_u_count
    
    # Vectorized restriction for d_v (staggered in y-direction)
    # Create indices for the fine grid points
    i_fine = np.arange(1, min(2*nx_coarse+1, d_v.shape[0]), 2)
    j_fine = np.arange(0, min(2*ny_coarse+2, d_v.shape[1]), 2)
    I_fine, J_fine = np.meshgrid(i_fine, j_fine, indexing='ij')
    
    # Create masks for valid adjacent indices
    valid_im1 = I_fine > 0
    valid_ip1 = I_fine < d_v.shape[0]-1
    
    # Initialize sum and count arrays for arithmetic averaging
    d_v_sum = np.zeros((nx_coarse, ny_coarse + 1))
    d_v_count = np.ones((nx_coarse, ny_coarse + 1))
    
    # Add center points
    d_v_sum += d_v[I_fine[:nx_coarse, :ny_coarse+1], J_fine[:nx_coarse, :ny_coarse+1]]
    
    # Add points to the left if available
    mask = valid_im1[:nx_coarse, :ny_coarse+1]
    if np.any(mask):
        d_v_sum[mask] += d_v[I_fine[:nx_coarse, :ny_coarse+1][mask]-1, J_fine[:nx_coarse, :ny_coarse+1][mask]]
        d_v_count[mask] += 1
    
    # Add points to the right if available
    mask = valid_ip1[:nx_coarse, :ny_coarse+1]
    if np.any(mask):
        d_v_sum[mask] += d_v[I_fine[:nx_coarse, :ny_coarse+1][mask]+1, J_fine[:nx_coarse, :ny_coarse+1][mask]]
        d_v_count[mask] += 1
    
    # Arithmetic average
    d_v_coarse = d_v_sum / d_v_count
    
    return d_u_coarse, d_v_coarse
