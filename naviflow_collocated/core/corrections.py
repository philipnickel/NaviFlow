import numpy as np
from numba import njit, prange

@njit(parallel=True)
def velocity_correction(mesh, grad_p_prime_cell, Ap_u_cell, Ap_v_cell):
    """
    Apply velocity correction u' = - (1 / a_P) * grad(p')
    using physical (unrelaxed) momentum matrix diagonals.
    """
    n_cells = mesh.cell_centers.shape[0]
    uv_field_corr = np.zeros((n_cells, 2))
    _SMALL = 1.0e-12

    for i in prange(n_cells):
        D_u = mesh.cell_volumes[i] / (Ap_u_cell[i] )
        D_v = mesh.cell_volumes[i] / (Ap_v_cell[i] )

        uv_field_corr[i, 0] = -D_u * grad_p_prime_cell[i, 0]
        uv_field_corr[i, 1] = -D_v * grad_p_prime_cell[i, 1]

    return uv_field_corr
