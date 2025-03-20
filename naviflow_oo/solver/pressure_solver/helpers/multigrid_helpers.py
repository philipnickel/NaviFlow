"""
Helper functions for multigrid operations (restriction and prolongation).
"""

import numpy as np


def restrict(fine_array, nx, ny, dx, dy):
    """
    Standard full-weighting restriction from an nx x ny grid 
    down to roughly half in each dimension.
    
    Parameters:
    -----------
    fine_array : ndarray
        Fine grid array to restrict
    nx, ny : int
        Fine grid dimensions
    dx, dy : float
        Fine grid cell sizes
        
    Returns:
    --------
    coarse_array : ndarray
        Restricted array (flattened)
    nx_c, ny_c : int
        Coarse grid dimensions
    dx_c, dy_c : float
        Coarse grid cell sizes
    """
    # Convert to 2D
    fine_2d = fine_array.reshape((nx, ny), order='F')
    
    # Next coarser level sizes - modify to use 2^k-1 coarsening pattern
    nx_c = max(1, (nx + 1)//2 - 1)  # For 2^k-1 grid sizes
    ny_c = max(1, (ny + 1)//2 - 1)
    dx_c = dx * 2
    dy_c = dy * 2
    
    coarse_2d = np.zeros((nx_c, ny_c), dtype=fine_2d.dtype)
    
    # Full weighting standard stencil (in index terms):
    #   coarse(i,j) = 1/16 [  fine(2i, 2j)*4 
    #                       + fine(2i±1,2j)*2 
    #                       + fine(2i,2j±1)*2
    #                       + fine(2i±1,2j±1)*1  ]
    #
    # We'll clamp indices to be within range.

    for i_c in range(nx_c):
        i_f = 2 * i_c
        for j_c in range(ny_c):
            j_f = 2 * j_c

            w_sum = 0.0
            w_fac = 0.0

            # Center (4 weight)
            if i_f < nx and j_f < ny:
                w_sum += 4.0 * fine_2d[i_f, j_f]
                w_fac += 4.0

            # Offsets ±1 in i, j => must check boundary
            for di in [-1, 1]:
                i_n = i_f + di
                if 0 <= i_n < nx and j_f < ny:
                    w_sum += 2.0 * fine_2d[i_n, j_f]
                    w_fac += 2.0

            for dj in [-1, 1]:
                j_n = j_f + dj
                if i_f < nx and 0 <= j_n < ny:
                    w_sum += 2.0 * fine_2d[i_f, j_n]
                    w_fac += 2.0

            # Diagonals
            for di in [-1, 1]:
                for dj in [-1, 1]:
                    i_n = i_f + di
                    j_n = j_f + dj
                    if 0 <= i_n < nx and 0 <= j_n < ny:
                        w_sum += fine_2d[i_n, j_n]
                        w_fac += 1.0

            # Normalize by actual weights used (handles boundaries properly)
            if w_fac > 0:
                coarse_2d[i_c, j_c] = w_sum / w_fac
            else:
                coarse_2d[i_c, j_c] = 0.0

    return coarse_2d.flatten('F'), nx_c, ny_c, dx_c, dy_c


def restrict_coefficient(fine_array, nx_f, ny_f, nx_c, ny_c):
    """
    Simple restriction for coefficients like d_u and d_v.
    Takes every other point from the fine grid, following 2^k-1 grid pattern.
    
    Parameters:
    -----------
    fine_array : ndarray
        Fine grid coefficient array
    nx_f, ny_f : int
        Fine grid dimensions
    nx_c, ny_c : int
        Coarse grid dimensions
        
    Returns:
    --------
    coarse_array : ndarray
        Restricted coefficient array
    """
    if fine_array is None:
        return None
        
    # Ensure the array has the right shape
    fine_array_2d = fine_array
    if fine_array.ndim == 1:
        fine_array_2d = fine_array.reshape((nx_f, ny_f), order='F')
    
    # Initialize the coarse array
    coarse_array = np.zeros((nx_c, ny_c), dtype=fine_array_2d.dtype)
    
    # Compute proper ratio for 2^k-1 grid pattern
    ratio_x = (nx_f - 1) / (nx_c - 1) if nx_c > 1 else nx_f
    ratio_y = (ny_f - 1) / (ny_c - 1) if ny_c > 1 else ny_f
    
    # Take points according to the 2^k-1 pattern
    for i_c in range(nx_c):
        # Calculate fine grid index - evenly distribute points
        i_f = min(int(i_c * ratio_x), nx_f-1)
        
        for j_c in range(ny_c):
            # Calculate fine grid index - evenly distribute points
            j_f = min(int(j_c * ratio_y), ny_f-1)
            
            coarse_array[i_c, j_c] = fine_array_2d[i_f, j_f]
            
    return coarse_array


def interpolate(coarse_array, nx_c, ny_c, nx_f, ny_f):
    """
    Bilinear interpolation from (nx_c x ny_c) to (nx_f x ny_f),
    accounting for 2^k-1 grid pattern.
    
    Parameters:
    -----------
    coarse_array : ndarray
        Coarse grid array to interpolate
    nx_c, ny_c : int
        Coarse grid dimensions
    nx_f, ny_f : int
        Fine grid dimensions
        
    Returns:
    --------
    fine_array : ndarray
        Interpolated array on the fine grid (flattened)
    """
    coarse_2d = coarse_array.reshape((nx_c, ny_c), order='F')
    fine_2d = np.zeros((nx_f, ny_f), dtype=coarse_2d.dtype)

    # Compute proper mapping ratios for 2^k-1 grid pattern
    ratio_x = (nx_f - 1) / max(1, nx_c - 1)
    ratio_y = (ny_f - 1) / max(1, ny_c - 1)
    
    # For each point in the fine grid
    for i_f in range(nx_f):
        # Find its fractional position in coarse grid coordinates
        x_c = i_f / ratio_x if nx_c > 1 else 0
        i_c_low = min(int(x_c), nx_c-1)
        i_c_high = min(i_c_low + 1, nx_c-1)
        wx = x_c - i_c_low if i_c_high > i_c_low else 0.0
        
        for j_f in range(ny_f):
            y_c = j_f / ratio_y if ny_c > 1 else 0
            j_c_low = min(int(y_c), ny_c-1)
            j_c_high = min(j_c_low + 1, ny_c-1)
            wy = y_c - j_c_low if j_c_high > j_c_low else 0.0
            
            # Bilinear interpolation weights
            w00 = (1.0 - wx) * (1.0 - wy)
            w10 = wx * (1.0 - wy)
            w01 = (1.0 - wx) * wy
            w11 = wx * wy
            
            # Use only valid weights based on available points
            if i_c_high == i_c_low and j_c_high == j_c_low:
                # Only one point available
                fine_2d[i_f, j_f] = coarse_2d[i_c_low, j_c_low]
            elif i_c_high == i_c_low:
                # Only j-direction interpolation
                fine_2d[i_f, j_f] = (1.0 - wy) * coarse_2d[i_c_low, j_c_low] + wy * coarse_2d[i_c_low, j_c_high]
            elif j_c_high == j_c_low:
                # Only i-direction interpolation
                fine_2d[i_f, j_f] = (1.0 - wx) * coarse_2d[i_c_low, j_c_low] + wx * coarse_2d[i_c_high, j_c_low]
            else:
                # Full bilinear interpolation
                fine_2d[i_f, j_f] = (w00 * coarse_2d[i_c_low, j_c_low] +
                                     w10 * coarse_2d[i_c_high, j_c_low] +
                                     w01 * coarse_2d[i_c_low, j_c_high] +
                                     w11 * coarse_2d[i_c_high, j_c_high])
    
    return fine_2d.flatten('F')


def solve_directly(f):
    """
    Direct solve on the coarsest grid (typically 1x1).
    
    On a 1x1 grid with a pinned reference pressure, we typically set p'=0.
    Alternatively for a direct solve for p'[0], one could use:
        return f / aP
    for some appropriate aP value.
    
    Parameters:
    -----------
    f : ndarray
        Right-hand side on the coarsest grid
        
    Returns:
    --------
    solution : ndarray
        Solution on the coarsest grid
    """
    return np.zeros_like(f) 