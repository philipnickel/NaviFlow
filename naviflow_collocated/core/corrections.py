import numpy as np
from numba import njit, prange

@njit(parallel=True)
def velocity_correction(mesh,grad_p_prime_cell, bold_D_cell):
    """
    Apply velocity correction: U' = -D_U * grad(p')
    """
    n_cells = mesh.cell_centers.shape[0]
    U_prime = np.zeros((n_cells, 2))
    for i in prange(n_cells):
        U_prime[i] = -bold_D_cell[i] * grad_p_prime_cell[i]


    return U_prime
