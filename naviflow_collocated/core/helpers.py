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
def compute_residual(data, indices, indptr, x, b, max_residual=None):
    """
    Compute residual field and relative L2 norm: r = b - A @ x.
    If max_residual is provided, compute relative to that instead of ||b||.

    Parameters
    ----------
    data, indices, indptr : CSR matrix format (A)
    x : ndarray, solution vector
    b : ndarray, right-hand side vector
    max_residual : float, optional
        Maximum residual to use for relative calculation. If None, uses ||b||

    Returns
    -------
    rel_res : float
        Relative L2 norm of the residual ||r|| / max_residual if provided, else ||r|| / ||b||
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
    L2_b = np.sqrt(b_sq)
    
    # Use max_residual if provided, otherwise use L2_b
    denominator = max_residual if max_residual is not None else L2_b
    
    # Return 1.0 if denominator is zero, otherwise return relative residual
    #rel_res = 1.0 if denominator == 0.0 else L2_res / denominator

    return L2_res, res_field

@njit(parallel=True)
def interpolate_to_face(mesh, quantity):
    """
    Interpolate quantity to faces using face_interp_factors
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

from scipy.sparse import csr_matrix, vstack, hstack
import numpy as np

def apply_mean_zero_constraint(A: csr_matrix, b: np.ndarray, volumes: np.ndarray):
    """
    Augments system A x = b with a zero-mean constraint: sum(volumes * x) = 0.

    Parameters
    ----------
    A : csr_matrix
        Original system matrix (n x n).
    b : np.ndarray
        Original right-hand side vector (n,).
    volumes : np.ndarray
        Cell volumes or integration weights (n,).

    Returns
    -------
    A_aug : csr_matrix
        Augmented system matrix (n+1 x n+1).
    b_aug : np.ndarray
        Augmented RHS (n+1).
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1] == b.shape[0] == volumes.shape[0], "Incompatible dimensions"

    # Row and column for constraint
    v_row = csr_matrix(volumes.reshape(1, -1))  # shape (1, n)
    v_col = csr_matrix(volumes.reshape(-1, 1))  # shape (n, 1)
    zero_scalar = csr_matrix((1, 1))            # shape (1, 1)

    # Assemble augmented system
    A_aug = vstack([
        hstack([A, v_col]),         # (n, n+1)
        hstack([v_row, zero_scalar])  # (1, n+1)
    ], format="csr")

    b_aug = np.concatenate([b, [0.0]])

    return A_aug, b_aug
BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 4
BC_NEUMANN = 3

def set_pressure_boundaries(mesh, p): 
    n_boundary = mesh.boundary_faces.shape[0]
    for i in prange(n_boundary):
        f = mesh.boundary_faces[i]
        if mesh.boundary_types[f,0] == BC_OUTLET:
            p[f] = p[mesh.owner_cells[f]]
        #elif mesh.boundary_types[f,0] == BC_INLET:
            #p[f] = p[mesh.owner_cells[f]]
        #elif mesh.boundary_types[f,0] == BC_WALL:
            #p[f] = p[mesh.owner_cells[f]]
        #elif mesh.boundary_types[f,0] == BC_DIRICHLET:
            #p[f] = p[mesh.owner_cells[f]]
        #elif mesh.boundary_types[f,0] == BC_NEUMANN:
            #p[f] = p[mesh.owner_cells[f]]
    return p