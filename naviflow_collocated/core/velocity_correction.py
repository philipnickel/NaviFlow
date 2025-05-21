import numpy as np
from numba import njit, prange

# It's assumed that Ap_u_cell and Ap_v_cell are the diagonal coefficients
# from the discretized momentum equations (a_P terms).


@njit(parallel=True)
def correct_cell_velocities(
    u_star: np.ndarray,
    v_star: np.ndarray,
    p_prime: np.ndarray,  # Solved pressure correction field
    grad_p_prime_cell: np.ndarray,  # Cell-centered gradient of p_prime [n_cells, 2]
    Ap_u_cell: np.ndarray,  # Diagonal coefficient from U-momentum matrix (a_P for u)
    Ap_v_cell: np.ndarray,  # Diagonal coefficient from V-momentum matrix (a_P for v)
    cell_volumes: np.ndarray,  # Mesh cell volumes
    # alpha_uv_relax is the under-relaxation factor typically applied AFTER solving momentum for u*,v*.
    # The correction itself u = u* + u_corr is usually not relaxed again, but can be if needed.
    # Moukalled refers to d_u = Vol/Ap_u. The velocity correction is u_corr = -d_u * grad_p_prime_x
):
    """
    Corrects cell-centered velocities using the pressure correction field.

    u_new = u_star - (Volume / Ap_u_cell) * (dp'/dx)_cell
    v_new = v_star - (Volume / Ap_v_cell) * (dp'/dy)_cell

    Parameters:
    - u_star, v_star: Provisional cell-centered velocities.
    - p_prime: Solved pressure correction field at cell centers.
    - grad_p_prime_cell: Cell-centered gradient of p_prime (dp'/dx, dp'/dy).
    - Ap_u_cell, Ap_v_cell: Diagonal coefficients from the U and V momentum equations.
    - cell_volumes: Array of cell volumes.

    Returns:
    - u_new: Corrected cell-centered u-velocity.
    - v_new: Corrected cell-centered v-velocity.
    - u_corr_val: The calculated u-velocity correction values (- (Vol/Ap_u) * dp'/dx).
    - v_corr_val: The calculated v-velocity correction values (- (Vol/Ap_v) * dp'/dy).
    """
    n_cells = u_star.shape[0]
    u_new = np.empty_like(u_star)
    v_new = np.empty_like(v_star)
    u_corr_val = np.empty_like(u_star)
    v_corr_val = np.empty_like(v_star)

    _SMALL = 1.0e-12  # To prevent division by zero

    for i in prange(n_cells):
        # d_u = Volume / Ap_u for the cell
        d_u_i = cell_volumes[i] / (Ap_u_cell[i] + _SMALL)
        d_v_i = cell_volumes[i] / (Ap_v_cell[i] + _SMALL)

        # Velocity correction components
        u_corr_i = -d_u_i * grad_p_prime_cell[i, 0]  # dp'/dx at cell i
        v_corr_i = -d_v_i * grad_p_prime_cell[i, 1]  # dp'/dy at cell i

        u_corr_val[i] = u_corr_i
        v_corr_val[i] = v_corr_i

        # Corrected velocities
        u_new[i] = u_star[i] + u_corr_i
        v_new[i] = v_star[i] + v_corr_i

    return u_new, v_new, u_corr_val, v_corr_val
