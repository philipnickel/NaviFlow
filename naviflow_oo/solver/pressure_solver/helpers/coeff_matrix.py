import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


def get_coeff_mat(imax, jmax, dx, dy, rho, d_u, d_v):
    """Form the coefficient matrix for the pressure correction equation."""
    # Create sparse matrix in COO format
    row_indices = []
    col_indices = []
    values = []
    
    # Interior points
    for i in range(imax):
        for j in range(jmax):
            # Current cell index in the flattened array
            idx = i + j * imax
            
            # Diagonal coefficient
            aP = 0
            
            # East neighbor
            if i < imax-1:
                aE = rho * d_u[i+1, j] * dy
                aP += aE
                row_indices.append(idx)
                col_indices.append(idx + 1)
                values.append(-aE)
            
            # West neighbor
            if i > 0:
                aW = rho * d_u[i, j] * dy
                aP += aW
                row_indices.append(idx)
                col_indices.append(idx - 1)
                values.append(-aW)
            
            # North neighbor
            if j < jmax-1:
                aN = rho * d_v[i, j+1] * dx
                aP += aN
                row_indices.append(idx)
                col_indices.append(idx + imax)
                values.append(-aN)
            
            # South neighbor
            if j > 0:
                aS = rho * d_v[i, j] * dx
                aP += aS
                row_indices.append(idx)
                col_indices.append(idx - imax)
                values.append(-aS)
            
            # Diagonal term
            row_indices.append(idx)
            col_indices.append(idx)
            values.append(aP)
    
    # Create sparse matrix
    A = sparse.coo_matrix((values, (row_indices, col_indices)), shape=(imax*jmax, imax*jmax))
    
    # Convert to CSR format for efficient matrix operations
    return A.tocsr()

def get_coeff_mat_vec(imax, jmax, dx, dy, rho, d_u, d_v):
    """
    Form the coefficient matrix for the pressure correction equation in a vectorized way.

    Parameters:
      imax, jmax  : integers defining the grid dimensions.
      dx, dy      : grid spacing in x and y directions.
      rho         : density (or analogous scaling factor).
      d_u, d_v    : arrays of diffusivity for u and v velocities,
                    assumed to be of shape (imax, jmax)
                    and indexed as d_u[i, j] (and similarly for d_v).
    
    Returns:
      A           : CSR sparse matrix of shape (imax*jmax, imax*jmax)
    """
    # Total number of grid points
    n = imax * jmax

    # Create grid of flattened indices in Fortran order (so that idx = i + j*imax)
    grid = np.reshape(np.arange(n), (jmax, imax), order='F')
    
    # Get arrays of i and j coordinates for all grid points.
    # Note: grid[j, i] = i + j*imax.
    j_indices, i_indices = np.indices((jmax, imax))
    
    # ---- Off-diagonal neighbor contributions ----
    # East neighbors: valid for i < imax - 1
    mask_east = i_indices < (imax - 1)
    current_east = grid[mask_east]
    east_neighbor = grid[j_indices[mask_east], i_indices[mask_east] + 1]
    # aE = rho * d_u[i+1, j] * dy
    aE = rho * d_u[i_indices[mask_east] + 1, j_indices[mask_east]] * dy

    # West neighbors: valid for i > 0
    mask_west = i_indices > 0
    current_west = grid[mask_west]
    west_neighbor = grid[j_indices[mask_west], i_indices[mask_west] - 1]
    # aW = rho * d_u[i, j] * dy
    aW = rho * d_u[i_indices[mask_west], j_indices[mask_west]] * dy

    # North neighbors: valid for j < jmax - 1
    mask_north = j_indices < (jmax - 1)
    current_north = grid[mask_north]
    north_neighbor = grid[j_indices[mask_north] + 1, i_indices[mask_north]]
    # aN = rho * d_v[i, j+1] * dx
    aN = rho * d_v[i_indices[mask_north], j_indices[mask_north] + 1] * dx

    # South neighbors: valid for j > 0
    mask_south = j_indices > 0
    current_south = grid[mask_south]
    south_neighbor = grid[j_indices[mask_south] - 1, i_indices[mask_south]]
    # aS = rho * d_v[i, j] * dx
    aS = rho * d_v[i_indices[mask_south], j_indices[mask_south]] * dx

    # ---- Diagonal term (aP) computation ----
    # a cell's diagonal coefficient is the sum of its neighbor contributions.
    # We initialize an array for the diagonal terms for every grid cell.
    aP = np.zeros((jmax, imax))
    # For cells with east neighbors:
    aP[mask_east] += rho * d_u[i_indices[mask_east] + 1, j_indices[mask_east]] * dy
    # For cells with west neighbors:
    aP[mask_west] += rho * d_u[i_indices[mask_west], j_indices[mask_west]] * dy
    # For cells with north neighbors:
    aP[mask_north] += rho * d_v[i_indices[mask_north], j_indices[mask_north] + 1] * dx
    # For cells with south neighbors:
    aP[mask_south] += rho * d_v[i_indices[mask_south], j_indices[mask_south]] * dx
    # Flatten the diagonal in the same order as the grid:
    diag_vals = aP.ravel(order='F')
    
    # ---- Assemble the sparse matrix data ----
    # Concatenate the row, column, and data arrays from all contributions.
    row_indices = np.concatenate([
        current_east.ravel(),
        current_west.ravel(),
        current_north.ravel(),
        current_south.ravel(),
        np.arange(n)
    ])
    
    col_indices = np.concatenate([
        east_neighbor.ravel(),
        west_neighbor.ravel(),
        north_neighbor.ravel(),
        south_neighbor.ravel(),
        np.arange(n)
    ])
    
    # For off-diagonals, the contribution is negative; the diagonal uses the summed value.
    data_vals = np.concatenate([
        (-aE).ravel(),
        (-aW).ravel(),
        (-aN).ravel(),
        (-aS).ravel(),
        diag_vals
    ])
    
    # Create the sparse matrix in COO format and then convert to CSR format.
    A = sparse.coo_matrix((data_vals, (row_indices, col_indices)), shape=(n, n)).tocsr()
    """ 
    # Calculate and print condition number
    # Get largest and smallest singular values
    largest_sv = svds(A, k=1, which='LM', return_singular_vectors=False)[0]
    smallest_sv = svds(A, k=1, which='SM', return_singular_vectors=False)[0]
    cond_num = largest_sv / smallest_sv
    print(f"Condition number of coefficient matrix: {cond_num:.2e}")
    """ 
    return A