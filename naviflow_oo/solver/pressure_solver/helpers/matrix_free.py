"""
Matrix-free implementation methods.
"""
import numpy as np

def compute_Ap_product(p_flat, imax, jmax, dx, dy, rho, d_u, d_v, out=None):
    """
    Compute matrix-vector product Ap without forming the matrix explicitly.
    
    Parameters:
    -----------
    p_flat : ndarray
        Flattened pressure array
    imax, jmax : int
        Grid dimensions
    dx, dy : float
        Cell sizes
    rho : float
        Fluid density
    d_u, d_v : ndarray
        Momentum equation coefficients
    out : ndarray, optional
        Output array for result (must be same shape as p_flat)
        
    Returns:
    --------
    result : ndarray
        The result of the matrix-vector product Ap
    """
    # Reshape p_flat to 2D for easier manipulation
    p = p_flat.reshape((imax, jmax), order='F')
    
    # Initialize result array
    if out is None:
        result = np.zeros_like(p_flat)
    else:
        # Use the provided output array
        result = out
        
    result_2d = result.reshape((imax, jmax), order='F')
    
    # Clear the result array if reusing
    if out is not None:
        result_2d.fill(0.0)
    
    # Handle reference pressure point (0,0) separately
    result_2d[0, 0] = p[0, 0]
    
    # Create mask for all non-reference points
    mask = np.ones((imax, jmax), dtype=bool)
    mask[0, 0] = False
    
    # Ensure d_u and d_v have the correct shapes for this grid
    d_u_padded = np.zeros((imax+1, jmax))
    d_v_padded = np.zeros((imax, jmax+1))
    
    # Copy available values from d_u and d_v
    d_u_i_max = min(d_u.shape[0], imax+1)
    d_u_j_max = min(d_u.shape[1], jmax)
    d_v_i_max = min(d_v.shape[0], imax)
    d_v_j_max = min(d_v.shape[1], jmax+1)
    
    d_u_padded[:d_u_i_max, :d_u_j_max] = d_u[:d_u_i_max, :d_u_j_max]
    d_v_padded[:d_v_i_max, :d_v_j_max] = d_v[:d_v_i_max, :d_v_j_max]
    
    # Compute coefficients for all points at once
    # East coefficients
    aE = np.zeros_like(p)
    if imax > 1:
        aE[:-1, :] = rho * d_u_padded[1:-1, :] * dy
    
    # West coefficients
    aW = np.zeros_like(p)
    if imax > 1:
        aW[1:, :] = rho * d_u_padded[1:-1, :] * dy
    
    # North coefficients
    aN = np.zeros_like(p)
    if jmax > 1:
        aN[:, :-1] = rho * d_v_padded[:, 1:-1] * dx
    
    # South coefficients
    aS = np.zeros_like(p)
    if jmax > 1:
        aS[:, 1:] = rho * d_v_padded[:, 1:-1] * dx
    
    # Sum all coefficients to get diagonal (aP)
    aP = aE + aW + aN + aS
    
    # Prepare shifted arrays for each direction
    p_east = np.zeros_like(p)
    p_west = np.zeros_like(p)
    p_north = np.zeros_like(p)
    p_south = np.zeros_like(p)
    
    # Fill shifted arrays safely
    if imax > 1:
        p_east[:-1, :] = p[1:, :]
        p_west[1:, :] = p[:-1, :]
    
    if jmax > 1:
        p_north[:, :-1] = p[:, 1:]
        p_south[:, 1:] = p[:, :-1]
    
    # Compute result: diagonal term - neighbor terms
    result_2d[mask] = (aP * p - aE * p_east - aW * p_west - aN * p_north - aS * p_south)[mask]
    
    # Ensure result is flattened before returning
    result = result_2d.flatten('F')
    return result


def get_coeff_mat_matrix_free(imax, jmax, dx, dy, rho, d_u, d_v):
    """Instead of forming coefficient matrix, return parameters needed for matrix-free operations."""
    # Return a dictionary with parameters needed for matrix-vector product
    return {
        'imax': imax,
        'jmax': jmax,
        'dx': dx,
        'dy': dy,
        'rho': rho,
        'd_u': d_u,
        'd_v': d_v
    }
 