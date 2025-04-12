"""
Helper functions for the multigrid solver.
"""

import numpy as np

def restrict_inject(fine_grid: np.ndarray) -> np.ndarray:
    """
    Reduces a fine grid to a coarse grid by taking every other point.
    Uses direct injection at odd indices (1, 3, 5, ...) for consistent
    behavior with Fortran ordering.
    
    Parameters:
        fine_grid (np.ndarray): The input fine grid to be coarsened
        
    Returns:
        np.ndarray: The coarsened grid
    """
    coarse_grid = fine_grid[1::2, 1::2]
    return coarse_grid

def restrict_full_weighting(fine_grid: np.ndarray) -> np.ndarray:
    """
    Reduces a fine grid to a coarse grid using full weighting.
    
    Full weighting restriction uses a weighted average of neighboring points:
    - Corner points get weight 1/16
    - Edge points get weight 1/8  
    - Center point gets weight 1/4
    
    Parameters:
        fine_grid (np.ndarray): The input fine grid to be coarsened
        
    Returns:
        np.ndarray: The coarsened grid with full weighting applied
    """
    # Get dimensions of fine grid
    nf = fine_grid.shape[0]
    # Size of coarse grid 
    nc = (nf - 1) // 2
    
    # Extract points using array slicing
    # Center points
    centers = fine_grid[1:-1:2, 1:-1:2]
    
    # Edge points (north, south, east, west)
    north = fine_grid[1:-1:2, 2::2]
    south = fine_grid[1:-1:2, :-2:2]
    east = fine_grid[2::2, 1:-1:2]
    west = fine_grid[:-2:2, 1:-1:2]
    
    # Corner points (northeast, northwest, southeast, southwest)
    ne = fine_grid[2::2, 2::2]
    nw = fine_grid[:-2:2, 2::2]
    se = fine_grid[2::2, :-2:2]
    sw = fine_grid[:-2:2, :-2:2]
    
    # Combine with appropriate weights
    coarse_grid = (
        centers / 4.0 +                    # Center weight 1/4
        (north + south + east + west) / 8.0 +  # Edge weights 1/8
        (ne + nw + se + sw) / 16.0            # Corner weights 1/16
    )
    
    return coarse_grid

def restrict_hybrid(fine_grid: np.ndarray) -> np.ndarray:
    """
    Hybrid restriction: 
    - Use injection at the boundary
    - Use full-weighting in the interior
    """
    nf = fine_grid.shape[0]
    nc = (nf - 1) // 2
    coarse_grid = np.zeros((nc, nc))
    
    # Injection for boundaries
    coarse_grid[:, :] = fine_grid[1::2, 1::2]

    # Full-weighting in the interior (excluding edges)
    if nc > 2:
        # Index ranges for the interior coarse grid
        i = slice(1, -1)
        j = slice(1, -1)

        centers = fine_grid[3:-2:2, 3:-2:2]
        north   = fine_grid[3:-2:2, 4:-1:2]
        south   = fine_grid[3:-2:2, 2:-4:2]
        east    = fine_grid[4:-1:2, 3:-2:2]
        west    = fine_grid[2:-4:2, 3:-2:2]
        ne      = fine_grid[4:-1:2, 4:-1:2]
        nw      = fine_grid[2:-4:2, 4:-1:2]
        se      = fine_grid[4:-1:2, 2:-4:2]
        sw      = fine_grid[2:-4:2, 2:-4:2]

        # Apply full-weighting in the interior
        coarse_grid[i, j] = (
            centers / 4.0 +
            (north + south + east + west) / 8.0 +
            (ne + nw + se + sw) / 16.0
        )
    
    return coarse_grid

def restrict_half_weighting(fine_grid: np.ndarray) -> np.ndarray:
    """
    Half-weighting restriction:
    - Center point: 1/2
    - Edge points (N, S, E, W): 1/8 each
    """
    nf = fine_grid.shape[0]
    nc = (nf - 1) // 2

    centers = fine_grid[1:-1:2, 1:-1:2]
    north   = fine_grid[1:-1:2, 2::2]
    south   = fine_grid[1:-1:2, :-2:2]
    east    = fine_grid[2::2, 1:-1:2]
    west    = fine_grid[:-2:2, 1:-1:2]

    coarse_grid = (
        0.5 * centers +
        0.125 * (north + south + east + west)
    )

    return coarse_grid


def interpolate_linear(coarse_grid, m):
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

def interpolate_cubic(coarse_grid, m):
    """
    Performs cubic interpolation from coarse to fine grid.
    This provides higher-order accuracy than bilinear interpolation.
    
    Parameters:
        coarse_grid (np.ndarray): The input coarse grid to be interpolated
        m (int): Size of the target fine grid
        
    Returns:
        np.ndarray: The interpolated fine grid with cubic interpolation
    """
    # Reshape to 2D if needed
    if coarse_grid.ndim == 1:
        mc = int(np.sqrt(coarse_grid.size))
        coarse_grid = coarse_grid.reshape((mc, mc))
    else:
        mc = coarse_grid.shape[0]
    
    # For cubic interpolation, we'll use a 2D interpolation approach
    # First, create coordinate grids
    x_coarse = np.linspace(0, 1, mc)
    y_coarse = np.linspace(0, 1, mc)
    x_fine = np.linspace(0, 1, m)
    y_fine = np.linspace(0, 1, m)
    
    # Use numpy's vectorized approach to build interpolation arrays
    from scipy import interpolate as sp_interp
    
    # Create 2D cubic spline interpolation
    # For cubic splines, we need at least 4 points in each dimension
    if mc >= 4:
        # Full cubic spline interpolation
        interp_func = sp_interp.RectBivariateSpline(x_coarse, y_coarse, coarse_grid)
        fine_grid = interp_func(x_fine, y_fine)
    else:
        # Fall back to quadratic or linear for small grids
        if mc == 3:
            # Quadratic interpolation
            kind = 'quadratic'
        else:
            # Linear interpolation for very small grids
            kind = 'linear'
            
        # Use regular grid interpolator for small grids
        coords_coarse = np.array(np.meshgrid(x_coarse, y_coarse, indexing='ij')).T.reshape(-1, 2)
        values = coarse_grid.flatten()
        interp_func = sp_interp.RegularGridInterpolator((x_coarse, y_coarse), coarse_grid, 
                                                       method=kind, bounds_error=False, 
                                                       fill_value=None)
        
        # Create fine grid coordinates
        XX, YY = np.meshgrid(x_fine, y_fine, indexing='ij')
        points = np.vstack([XX.ravel(), YY.ravel()]).T
        
        # Interpolate and reshape
        fine_values = interp_func(points)
        fine_grid = fine_values.reshape((m, m))
    
    return fine_grid