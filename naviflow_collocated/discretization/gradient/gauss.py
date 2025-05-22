import numpy as np
from numba import njit

@njit
def compute_cell_gradients(mesh, phi, boundary_component_idx=0):
    """
    Green–Gauss linear gradient reconstruction with constant-field safeguard.

    Parameters
    ----------
    phi : ndarray of shape (n_cells,)
        Scalar field at cell centers.

    mesh : MeshData2D
        Structured mesh object with geometry, topology, BCs, etc.

    boundary_component_idx : int
        Index into mesh.boundary_values[f, :] for the scalar field (e.g. 0 for u, 1 for v, 2 for p)

    Returns
    -------
    grad : ndarray of shape (n_cells, 2)
        Gradient of phi at cell centers.
    """

    n_cells = mesh.cell_centers.shape[0]

    # Early exit for constant fields (including all-zero)
    """
    phi0 = phi[0]
    is_constant = True
    for i in range(1, n_cells):
        if abs(phi[i] - phi0) > 1e-12:
            is_constant = False
            break
    if is_constant:
        return np.zeros((n_cells, 2), dtype=np.float64)
    """

    grad = np.zeros((n_cells, 2), dtype=np.float64)

    # === Interior face contributions ===
    for f in mesh.internal_faces:
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        g_f = mesh.face_interp_factors[f]
        phi_f = g_f * phi[N] + (1.0 - g_f) * phi[P]

        Sf = mesh.vector_S_f[f]

        grad[P, 0] += phi_f * Sf[0]
        grad[P, 1] += phi_f * Sf[1]

        grad[N, 0] -= phi_f * Sf[0]
        grad[N, 1] -= phi_f * Sf[1]

    # === Boundary face contributions ===
    for f in mesh.boundary_faces:
        P = mesh.owner_cells[f]

        # Use correct BC component: 0 = u, 1 = v, 2 = p
        phi_b = mesh.boundary_values[f, boundary_component_idx]
        Sf = mesh.vector_S_f[f]

        grad[P, 0] += phi_b * Sf[0]
        grad[P, 1] += phi_b * Sf[1]

    # === Normalize by cell volume ===
    for c in range(n_cells):
        vol = mesh.cell_volumes[c]
        grad[c, 0] /= vol
        grad[c, 1] /= vol

    return grad
