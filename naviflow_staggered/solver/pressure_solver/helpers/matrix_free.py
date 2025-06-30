"""
Matrix-free implementation methods.
"""
import numpy as np

def compute_Ap_product(p_flat, imax, jmax, dx, dy, rho, d_u, d_v, out=None, pin_pressure=True):
    """
    Compute matrix-vector product Ap without forming the matrix explicitly.
    This implementation follows the same coefficient calculation as get_coeff_mat
    in coeff_matrix.py for consistency, but uses vectorized operations for efficiency.
    
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
    pin_pressure : bool, optional
        Whether to pin pressure at bottom-left corner (default: True)
        
    Returns:
    --------
    result : ndarray
        The result of the matrix-vector product Ap
    """
    p = p_flat.reshape((imax, jmax), order='F')
    
    if out is None:
        result = np.zeros_like(p_flat)
    else:
        result = out
        result.fill(0.0)
        
    result_2d = result.reshape((imax, jmax), order='F')
    
    # Initialize coefficient arrays
    east = np.zeros((imax, jmax))
    west = np.zeros((imax, jmax))
    north = np.zeros((imax, jmax))
    south = np.zeros((imax, jmax))
    diag = np.zeros((imax, jmax))
    
    # East coefficients (aE): apply to cells with i < imax-1
    east[:-1, :] = rho * d_u[1:imax, :] * dy
    
    # West coefficients (aW): apply to cells with i > 0
    west[1:, :]  = rho * d_u[1:imax, :] * dy
    
    # North coefficients (aN): apply to cells with j < jmax-1
    north[:, :-1] = rho * d_v[:, 1:jmax] * dx
    
    # South coefficients (aS): apply to cells with j > 0
    south[:, 1:]  = rho * d_v[:, 1:jmax] * dx
    
    # Apply zero gradient boundary conditions by modifying the coefficients:
    
    # West boundary (i=0): add east coefficient to diagonal, zero out west
    diag[0, :] += east[0, :]
    
    # East boundary (i=imax-1): add west coefficient to diagonal, zero out east
    diag[imax-1, :] += west[imax-1, :]
    
    # South boundary (j=0): add north coefficient to diagonal, zero out south
    diag[:, 0] += north[:, 0]
    
    # North boundary (j=jmax-1): add south coefficient to diagonal, zero out north
    diag[:, jmax-1] += south[:, jmax-1]
    
    # Set boundary coefficients to zero after adding to diagonal
    east[0, :] = 0
    west[imax-1, :] = 0
    north[:, 0] = 0
    south[:, jmax-1] = 0
    
    # Calculate diagonal coefficients (sum of off-diagonal coefficients)
    diag += east + west + north + south
    
    if pin_pressure:
        # For pinned pressure point, set diagonal to 1 and compute Ax = x
        if imax > 0 and jmax > 0:
            # Create a mask for all non-pinned cells
            mask = np.ones((imax, jmax), dtype=bool)
            mask[0, 0] = False
            
            # Set the pinned pressure result directly
            result_2d[0, 0] = p[0, 0]  # Ax = x for pinned node
            
            # Apply the matrix-vector product for all other cells
            result_2d[mask] = diag[mask] * p[mask]
            
            # Subtract neighbor contributions for all cells except the pinned one
            if imax > 1:
                # Contribution from east neighbors
                east_contrib = east[:-1, :] * p[1:, :]
                # Only apply where mask is True (non-pinned cells)
                east_mask = mask[:-1, :]
                result_2d[:-1, :][east_mask] -= east_contrib[east_mask]
                
                # Contribution from west neighbors
                west_contrib = west[1:, :] * p[:-1, :]
                west_mask = mask[1:, :]
                result_2d[1:, :][west_mask] -= west_contrib[west_mask]
            
            if jmax > 1:
                # Contribution from north neighbors
                north_contrib = north[:, :-1] * p[:, 1:]
                north_mask = mask[:, :-1]
                result_2d[:, :-1][north_mask] -= north_contrib[north_mask]
                
                # Contribution from south neighbors
                south_contrib = south[:, 1:] * p[:, :-1]
                south_mask = mask[:, 1:]
                result_2d[:, 1:][south_mask] -= south_contrib[south_mask]
                
    else:
        # No pinned pressure - apply matrix-vector product to all cells
        result_2d[:] = diag * p
        
        if imax > 1:
            result_2d[:-1, :] -= east[:-1, :] * p[1:, :]
            result_2d[1:, :]  -= west[1:, :] * p[:-1, :]
        
        if jmax > 1:
            result_2d[:, :-1] -= north[:, :-1] * p[:, 1:]
            result_2d[:, 1:]  -= south[:, 1:] * p[:, :-1]
    
    return result.reshape(p_flat.shape, order='F')


def get_coeff_mat_matrix_free(imax, jmax, dx, dy, rho, d_u, d_v, pin_pressure=True):
    """Instead of forming coefficient matrix, return parameters needed for matrix-free operations.
    
    Parameters:
    -----------
    imax, jmax : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    rho : float
        Fluid density
    d_u, d_v : ndarray
        Momentum equation coefficients
    pin_pressure : bool, optional
        Whether to pin pressure at a point to avoid singularity (default: True)
        
    Returns:
    --------
    dict
        Dictionary with parameters needed for matrix-vector product
    """
    # Return a dictionary with parameters needed for matrix-vector product
    return {
        'imax': imax,
        'jmax': jmax,
        'dx': dx,
        'dy': dy,
        'rho': rho,
        'd_u': d_u,
        'd_v': d_v,
        'pin_pressure': pin_pressure
    }
 