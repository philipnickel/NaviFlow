import numpy as np
from numba import njit, prange, jit

@njit(parallel=False, cache=True)
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



@njit(parallel=False, cache=True)
def compute_l2_norm(vec, indices=None):
    """
    Compute the L2 norm of a vector, optionally over a subset of indices.
    Parameters
    ----------
    vec : ndarray
        Input vector.
    indices : ndarray or None
        Indices to include in the norm. If None, use all elements.
    Returns
    -------
    float
        L2 norm over the selected elements.
    """
    if indices is not None:
        total = 0.0
        for i in range(indices.shape[0]):
            total += vec[indices[i]] * vec[indices[i]]
        return np.sqrt(total)
    else:
        total = 0.0
        for i in range(vec.shape[0]):
            total += vec[i] * vec[i]
        return np.sqrt(total)

@njit(parallel=False, cache=True)
def compute_residual(data, indices, indptr, x, b, max_residual=None, norm_indices=None):
    """
    Compute residual field and relative L2 norm: r = b - A @ x.
    If max_residual is provided, compute relative to that instead of ||b||.
    Optionally, compute the norm over a subset of indices.
    Parameters
    ----------
    data, indices, indptr : CSR matrix format (A)
    x : ndarray, solution vector
    b : ndarray, right-hand side vector
    max_residual : float, optional
        Maximum residual to use for relative calculation. If None, uses ||b||
    norm_indices : ndarray or None
        Indices to include in the norm. If None, use all elements.
    Returns
    -------
    rel_res : float
        Relative L2 norm of the residual ||r|| / max_residual if provided, else ||r|| / ||b||
    r : ndarray
        Residual vector: r = b - A @ x
    """
    n = b.shape[0]
    res_field = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        Ax_i = 0.0
        for j in range(indptr[i], indptr[i+1]):
            Ax_i += data[j] * x[indices[j]]
        r_i = b[i] - Ax_i
        res_field[i] = r_i

    L2_res = compute_l2_norm(res_field, norm_indices)
    
    return L2_res, res_field

@njit(parallel=True, cache=True)
def interpolate_to_face(mesh, quantity):
    """
    Optimized interpolation to faces with better memory access patterns.
    Handles both scalar and vector quantities efficiently.
    """
    n_faces = mesh.face_areas.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    
    if quantity.ndim == 1:
        # Scalar field
        interpolated_quantity = np.zeros((n_faces, 1), dtype=np.float64)
        
        # Process internal faces
        for i in prange(n_internal):
            f = mesh.internal_faces[i]
            P = mesh.owner_cells[f]
            N = mesh.neighbor_cells[f]
            gf = mesh.face_interp_factors[f]
            interpolated_quantity[f, 0] = gf * quantity[N] + (1.0 - gf) * quantity[P]

        # Process boundary faces
        for i in prange(n_boundary):
            f = mesh.boundary_faces[i]
            P = mesh.owner_cells[f]
            interpolated_quantity[f, 0] = quantity[P]
            
    else:
        # Vector field - optimized for common 2D case
        n_components = quantity.shape[1]
        interpolated_quantity = np.zeros((n_faces, n_components), dtype=np.float64)
        
        if n_components == 2:
            # Optimized 2D vector case with manual unrolling
            for i in prange(n_internal):
                f = mesh.internal_faces[i]
                P = mesh.owner_cells[f]
                N = mesh.neighbor_cells[f]
                gf = mesh.face_interp_factors[f]
                
                interpolated_quantity[f, 0] = gf * quantity[N, 0] + (1.0 - gf) * quantity[P, 0]
                interpolated_quantity[f, 1] = gf * quantity[N, 1] + (1.0 - gf) * quantity[P, 1]

            for i in prange(n_boundary):
                f = mesh.boundary_faces[i]
                P = mesh.owner_cells[f]
                interpolated_quantity[f, 0] = quantity[P, 0]
                interpolated_quantity[f, 1] = quantity[P, 1]
        else:
            # General vector case
            for i in prange(n_internal):
                f = mesh.internal_faces[i]
                P = mesh.owner_cells[f]
                N = mesh.neighbor_cells[f]
                gf = mesh.face_interp_factors[f]
                
                for c in range(n_components):
                    interpolated_quantity[f, c] = gf * quantity[N, c] + (1.0 - gf) * quantity[P, c]

            for i in prange(n_boundary):
                f = mesh.boundary_faces[i]
                P = mesh.owner_cells[f]
                for c in range(n_components):
                    interpolated_quantity[f, c] = quantity[P, c]

    return interpolated_quantity


@njit(parallel=False, cache=True)
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
BC_OBSTACLE = 4  # Same as wall for our purposes

def set_pressure_boundaries(mesh, p): 
    n_boundary = mesh.boundary_faces.shape[0]
    for i in prange(n_boundary):
        f = mesh.boundary_faces[i]
        bc_type = mesh.boundary_types[f,0]
        P = mesh.owner_cells[f]
        
        if bc_type == BC_OUTLET:
            # Fixed pressure at outlet
            p[f] = mesh.boundary_values[f]
        elif bc_type == BC_WALL or bc_type == BC_OBSTACLE:
            # Zero normal gradient at wall: extrapolate from cell center
            # p_b = p_P + (∇p)_P · d_Cb = p_P since (∇p)_P · n = 0
            p[f] = p[P]
        elif bc_type == BC_INLET:
            # Zero normal gradient at inlet
            p[f] = p[P]
        elif bc_type == BC_NEUMANN:
            # General zero gradient condition
            p[f] = p[P]
    return p

def get_unique_cells_from_faces(mesh, face_indices):
    """
    Given a set of face indices, return the unique cell indices (owners and neighbors) associated with those faces.
    Parameters
    ----------
    mesh : Mesh object
    face_indices : ndarray
        Indices of faces (e.g., internal_faces or boundary_faces)
    Returns
    -------
    unique_cells : ndarray
        Unique cell indices (sorted)
    """
    owners = mesh.owner_cells[face_indices]
    return np.unique(owners)