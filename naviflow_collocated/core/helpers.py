import numpy as np
from numba import njit, prange

from numba import njit, prange
import numpy as np

from numba import njit, prange
import numpy as np

from numba import njit
import numpy as np

@njit(parallel=False)
def relax_momentum_equation(rhs, A_diag, phi, alpha):
    """
    In-place Patankar-style under-relaxation of a momentum equation system.
    Modifies `rhs` in-place, writes the relaxed diagonal into `A_diag`.
    """
    inv_alpha = 1.0 / alpha
    scale = (1.0 - alpha) / alpha
    n = rhs.shape[0]
    relaxed_diagonal = np.zeros(n, dtype=np.float64)
    relaxed_rhs = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        a = A_diag[i]
        a_relaxed = a * inv_alpha
        relaxed_diagonal[i] = a_relaxed
        relaxed_rhs[i] = rhs[i] + scale * a * phi[i]  

    return relaxed_diagonal, relaxed_rhs


from numba import njit, prange
import numpy as np

@njit(parallel=False)
def compute_residual(data, indices, indptr, x, b):
    """
    Compute residual field and relative L2 norm: r = b - A @ x.

    Parameters
    ----------
    data, indices, indptr : CSR matrix format (A)
    x : ndarray, solution vector
    b : ndarray, right-hand side vector

    Returns
    -------
    L2_norm : float
        Relative L2 norm of the residual ||r|| / ||b||
    r : ndarray
        Residual vector: r = b - A @ x
    """
    n = b.shape[0]
    res_field = np.zeros(n, dtype=np.float64)
    res_sq = 0.0
    b_sq = 0.0

    for i in prange(n):
        Ax_i = 0.0
        for j in range(indptr[i], indptr[i+1]):
            Ax_i += data[j] * x[indices[j]]
        r_i = b[i] - Ax_i
        res_field[i] = r_i

    for i in prange(n):
        res_sq += res_field[i] * res_field[i]
        b_sq += b[i] * b[i]

    L2_res = np.sqrt(res_sq)

    return L2_res, res_field

@njit(parallel=False)
def interpolate_to_face(mesh, quantity):
    """
    interpolate quantity to faces using face_interp_factors
    Quantity may be vector or scalar
    """
    n_faces = mesh.face_areas.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    interpolated_quantity = np.zeros((n_faces, quantity.shape[1]), dtype=np.float64)

    for i in prange(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]
        gf = mesh.face_interp_factors[f]
        interpolated_quantity[f] = gf * quantity[N] + (1.0 - gf) * quantity[P]

    for i in prange(n_boundary):
        f = mesh.boundary_faces[i]
        P = mesh.owner_cells[f]
        interpolated_quantity[f] = quantity[P]

    return interpolated_quantity


@njit(parallel=False)
def bold_Dv_calculation(mesh, A_u_diag, A_v_diag):
    n_cells = mesh.cell_volumes.shape[0]
    bold_Dv = np.zeros((n_cells, 2), dtype=np.float64)

    for i in prange(n_cells):
        bold_Dv[i, 0] = mesh.cell_volumes[i] / (A_u_diag[i] + 1e-14)  # D_u
        bold_Dv[i, 1] = mesh.cell_volumes[i] / (A_v_diag[i] + 1e-14)  # D_v

    return bold_Dv

