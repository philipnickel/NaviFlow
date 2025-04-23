import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


def get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v, pin_pressure=True):
    """
    Construct the coefficient matrix for the pressure correction equation.
    
    Parameters:
    -----------
    nx, ny : int
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
    sparse.csr_matrix
        Coefficient matrix in CSR format
    """
    # Total number of cells
    n_cells = nx * ny
    
    # Allocate arrays for diagonal and off-diagonal entries
    diag = np.zeros(n_cells)
    east = np.zeros(n_cells)
    west = np.zeros(n_cells)
    north = np.zeros(n_cells)
    south = np.zeros(n_cells)
    
    # Create indices for vectorized operations
    i_indices, j_indices = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    i_indices = i_indices.flatten()
    j_indices = j_indices.flatten()
    linear_indices = i_indices + j_indices * nx  # Fortran ordering
    
    # East coefficients (aE)
    # For interior cells: i < nx-1
    interior_east = i_indices < nx-1
    east[interior_east] = rho * d_u[i_indices[interior_east]+1, j_indices[interior_east]] * dy
    
    # West coefficients (aW)
    # For interior cells: i > 0
    interior_west = i_indices > 0
    west[interior_west] = rho * d_u[i_indices[interior_west], j_indices[interior_west]] * dy
    
    # North coefficients (aN)
    # For interior cells: j < ny-1
    interior_north = j_indices < ny-1
    north[interior_north] = rho * d_v[i_indices[interior_north], j_indices[interior_north]+1] * dx
    
    # South coefficients (aS)
    # For interior cells: j > 0
    interior_south = j_indices > 0
    south[interior_south] = rho * d_v[i_indices[interior_south], j_indices[interior_south]] * dx
    
    # Apply zero gradient boundary conditions by modifying the coefficients:
    
    # West boundary (i=0): add east coefficient to diagonal, zero out west
    west_boundary = i_indices == 0
    diag[west_boundary] += east[west_boundary]
    east[west_boundary] = 0
    
    # East boundary (i=nx-1): add west coefficient to diagonal, zero out east
    east_boundary = i_indices == nx-1
    diag[east_boundary] += west[east_boundary]
    west[east_boundary] = 0
    
    # South boundary (j=0): add north coefficient to diagonal, zero out south
    south_boundary = j_indices == 0
    diag[south_boundary] += north[south_boundary]
    north[south_boundary] = 0
    
    # North boundary (j=ny-1): add south coefficient to diagonal, zero out north
    north_boundary = j_indices == ny-1
    diag[north_boundary] += south[north_boundary]
    south[north_boundary] = 0
    
    # Diagonal coefficients (aP)
    # For interior cells: sum of off-diagonal coefficients
    diag += east + west + north + south
    
    # Create row and column indices for sparse matrix
    i_indices_e = linear_indices[interior_east]
    j_indices_e = linear_indices[interior_east] + 1  # East neighbor
    
    i_indices_w = linear_indices[interior_west]
    j_indices_w = linear_indices[interior_west] - 1  # West neighbor
    
    i_indices_n = linear_indices[interior_north]
    j_indices_n = linear_indices[interior_north] + nx  # North neighbor (adding nx moves down one row in Fortran ordering)
    
    i_indices_s = linear_indices[interior_south]
    j_indices_s = linear_indices[interior_south] - nx  # South neighbor
    
    # Concatenate indices
    rows = np.concatenate([linear_indices, i_indices_e, i_indices_w, i_indices_n, i_indices_s])
    cols = np.concatenate([linear_indices, j_indices_e, j_indices_w, j_indices_n, j_indices_s])
    data = np.concatenate([diag, -east[interior_east], -west[interior_west], 
                          -north[interior_north], -south[interior_south]])
    
    # Create sparse matrix
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    A = A.tocsr()
    
    # Pin pressure at a reference point (bottom-left corner) to avoid singularity
    if pin_pressure:

        # Fix pressure at bottom-left (i=0, j=0)
        pin_index = 0
        A[pin_index, :] = 0
        A[pin_index, pin_index] = 1
    
    return A


