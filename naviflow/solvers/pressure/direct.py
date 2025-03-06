import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .helpers.rhs import get_rhs
from .helpers.coeff_mat import get_coeff_mat

def penta_diag_solve(solver_params):
    """Solve the pentadiagonal system Ax = b."""
    # Extract needed parameters
    A = solver_params['A']
    b = solver_params['b']
    
    # Use scipy's sparse solver
    x = spsolve(A, b)
    return x
