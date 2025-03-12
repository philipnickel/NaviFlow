import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from .helpers.rhs import get_rhs
from .helpers.coeff_mat import get_coeff_mat
from scipy.sparse.linalg import spsolve

def penta_diag_solve(solver_params):
    """Solve the pentadiagonal system Ax = b."""
    # Extract needed parameters
    A = solver_params['A']
    b = solver_params['b']
    
    # Use a more robust solver like UMFPACK
    #x = spsolve_triangular(A, b, lower=False)
    
    # Add a small value to the diagonal to improve conditioning
    diag = A.diagonal()
    diag_plus_eps = diag + 1e-10
    A.setdiag(diag_plus_eps)
    
    x = spsolve(A, b)
    return x
