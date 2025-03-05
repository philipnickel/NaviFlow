import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def get_rhs(imax, jmax, dx, dy, rho, u_star, v_star):
    """Calculate RHS vector of the pressure correction equation."""
    # Vectorized implementation
    bp = np.zeros(imax*jmax)
    
    # Create 2D matrix first - easier to work with
    bp_2d = np.zeros((imax, jmax))
    
    # Compute entire array at once
    bp_2d = rho * (u_star[:-1, :] * dy - u_star[1:, :] * dy + 
                   v_star[:, :-1] * dx - v_star[:, 1:] * dx)
    
    # Flatten to 1D array in correct order
    bp = bp_2d.flatten('F')  # Fortran-style order (column-major)
    
    # Modify for p_prime(0,0) - pressure at first node is fixed
    bp[0] = 0
    
    return bp




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

def penta_diag_solve(A, b):
    """Solve the pentadiagonal system Ax = b."""
    # Use scipy's sparse solver
    x = spsolve(A, b)
    return x

def pres_correct(imax, jmax, rhsp, Ap, p, alpha):
    """Solve for pressure correction and update pressure."""
    # Solve for pressure correction
    p_prime_interior = penta_diag_solve(Ap, rhsp)
    
    # Reshape to 2D array using vectorized operation
    p_prime = p_prime_interior.reshape((imax, jmax), order='F')
    
    # Vectorized pressure update
    pressure = p + alpha * p_prime
    
    # Fix pressure at a reference point
    pressure[0, 0] = 0.0
    
    return pressure, p_prime

