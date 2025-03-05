
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def compute_Ap_product(p_flat, imax, jmax, dx, dy, rho, d_u, d_v):
    """Compute matrix-vector product Ap without forming the matrix explicitly."""
    # Reshape p_flat to 2D for easier manipulation
    p = p_flat.reshape((imax, jmax), order='F')
    
    # Initialize result array
    result_2d = np.zeros_like(p)
    
    # Handle reference pressure point (0,0) separately
    result_2d[0, 0] = p[0, 0]
    
    # Create mask for all non-reference points
    mask = np.ones((imax, jmax), dtype=bool)
    mask[0, 0] = False
    
    # Prepare shifted arrays for each direction
    # For east connection (i+1,j)
    p_east = np.zeros_like(p)
    p_east[:-1, :] = p[1:, :]
    
    # For west connection (i-1,j)
    p_west = np.zeros_like(p)
    p_west[1:, :] = p[:-1, :]
    
    # For north connection (i,j+1)
    p_north = np.zeros_like(p)
    p_north[:, :-1] = p[:, 1:]
    
    # For south connection (i,j-1)
    p_south = np.zeros_like(p)
    p_south[:, 1:] = p[:, :-1]
    
    # Compute coefficients for all points at once
    # East coefficients
    aE = np.zeros_like(p)
    aE[:-1, :] = rho * d_u[1:-1, :] * dy
    
    # West coefficients
    aW = np.zeros_like(p)
    aW[1:, :] = rho * d_u[1:-1, :] * dy
    
    # North coefficients
    aN = np.zeros_like(p)
    aN[:, :-1] = rho * d_v[:, 1:-1] * dx
    
    # South coefficients
    aS = np.zeros_like(p)
    aS[:, 1:] = rho * d_v[:, 1:-1] * dx
    
    # Sum all coefficients to get diagonal (aP)
    aP = aE + aW + aN + aS
    
    # Compute result: diagonal term - neighbor terms
    result_2d[mask] = (aP * p - aE * p_east - aW * p_west - aN * p_north - aS * p_south)[mask]
    
    # Flatten result back to 1D
    result = result_2d.flatten('F')
    return result

def penta_diag_solve_matrix_free(A, b):
    """Solve the pentadiagonal system Ax = b using iterative method."""
    from scipy.sparse.linalg import cg, LinearOperator
    
    # Initial guess
    x0 = np.zeros_like(b)
    
    # Get problem size
    N = len(b)
    imax = int(np.sqrt(N))
    jmax = N // imax
    
    # Define matrix-vector product function
    def mv_product(v):
        return compute_Ap_product(v, imax, jmax, dx_global, dy_global, rho_global, d_u_global, d_v_global)
    
    # Create a LinearOperator to represent our matrix operation
    A_op = LinearOperator((N, N), matvec=mv_product)
    
    # Use conjugate gradient to solve system
    x, info = cg(A_op, b, x0=x0, atol=1e-6, maxiter=1000)
    
    if info != 0:
        print(f"Warning: CG did not converge, info={info}")
    
    return x

def pres_correct_matrix_free(imax, jmax, rhsp, Ap, p, alpha):
    """Solve for pressure correction and update pressure."""
    # Set global variables for the mv_product closure
    global d_u_global, d_v_global, dx_global, dy_global, rho_global
    d_u_global = Ap['d_u']
    d_v_global = Ap['d_v']
    dx_global = Ap['dx']
    dy_global = Ap['dy']
    rho_global = Ap['rho']
    
    # Solve for pressure correction using matrix-free approach
    p_prime_interior = penta_diag_solve_matrix_free(None, rhsp)
    
    # Reshape to 2D array using vectorized operation
    p_prime = p_prime_interior.reshape((imax, jmax), order='F')
    
    # Vectorized pressure update
    pressure = p + alpha * p_prime
    
    # Fix pressure at a reference point
    pressure[0, 0] = 0.0
    
    return pressure, p_prime

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
