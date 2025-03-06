import numpy as np
from scipy import sparse


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

