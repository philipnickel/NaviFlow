import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from .helpers.matrix_free import compute_Ap_product

def cg_matrix_free(solver_params):
    """Solve the pentadiagonal system Ax = b using iterative method."""
    # Extract needed parameters
    b = solver_params['b']
    params = solver_params['params']
    use_numba = solver_params.get('use_numba', False)
    
    # Initial guess
    x0 = np.zeros_like(b)
    
    # Get problem size
    N = len(b)
    imax = solver_params['imax']
    jmax = solver_params['jmax']
    
    # Extract parameters
    dx = params['dx']
    dy = params['dy']
    rho = params['rho']
    d_u = params['d_u']
    d_v = params['d_v']
    
    # Create a lambda function for the matrix-vector product
    mv_product = lambda v: compute_Ap_product(v, imax, jmax, dx, dy, rho, d_u, d_v, use_numba)
    
    # Create a LinearOperator to represent our matrix operation
    A_op = LinearOperator((N, N), matvec=mv_product)
    
    # Use conjugate gradient to solve system
    x, info = cg(A_op, b, x0=x0, atol=1e-7, maxiter=1000)
    
    if info != 0:
        print(f"Warning: CG did not converge, info={info}")
    
    return x