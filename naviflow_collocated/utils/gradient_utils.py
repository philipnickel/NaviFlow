import numpy as np
from numba import njit, prange

_SMALL = 1.0e-12


@njit
def interpolate_scalar_to_face(phi_cell: np.ndarray, mesh, f_idx: int) -> float:
    """
    Interpolate a scalar from cell centers to a specific face.

    Parameters:
    - phi_cell: Scalar field at cell centers.
    - mesh: MeshData2D structure.
    - f_idx: Face index.

    Returns:
    - Interpolated scalar value at the face.
    """
    C = mesh.owner_cells[f_idx]
    F = mesh.neighbor_cells[f_idx]

    if F >= 0:  # Internal face
        fx = mesh.face_interp_factors[f_idx]
        return fx * phi_cell[F] + (1.0 - fx) * phi_cell[C]
    else:  # Boundary face
        # For boundary faces, use cell value (zero gradient assumption for general interpolation)
        # Specific gradient calculations might handle this differently based on BCs.
        return phi_cell[C]


@njit
def calculate_cell_scalar_gradient_green_gauss(
    phi_cell: np.ndarray, mesh, cell_idx: int
) -> np.ndarray:
    """
    Calculate the gradient of a scalar field at a specific cell center using Green-Gauss theorem.

    grad(phi)_cell = (1/Vol_cell) * sum_over_faces(phi_f * S_f)

    Parameters:
    - phi_cell: Scalar field at cell centers.
    - mesh: MeshData2D structure.
    - cell_idx: Index of the cell for which to calculate the gradient.

    Returns:
    - grad_phi: Gradient vector [d(phi)/dx, d(phi)/dy] at the cell center.
    """
    grad_phi_cell = np.zeros(2, dtype=phi_cell.dtype)
    cell_vol = mesh.cell_volumes[cell_idx]

    # Loop over all faces of the current cell
    # mesh.cell_faces[cell_idx] gives indices of faces connected to cell_idx
    # mesh.cell_face_signs[cell_idx] gives the sign (+1 if S_f points out, -1 if S_f points in)
    # However, the standard Green-Gauss formulation sums phi_f * S_f where S_f is the outward normal area vector.
    # The sign in the original code was to adjust for S_f always pointing C->F.
    # Let's use the mesh.face_normals (which are S_f pointing C->F or outward at boundary)
    # and determine if the cell is owner or neighbor.

    for i in range(
        mesh.cell_faces.shape[1]
    ):  # Iterate through local face indices for the cell
        f_idx = mesh.cell_faces[cell_idx, i]
        if f_idx < 0:  # No more faces for this cell
            break

        # Interpolate scalar to the face center
        phi_f = interpolate_scalar_to_face(phi_cell, mesh, f_idx)

        # Get face normal area vector S_f = [S_fx, S_fy]
        # mesh.face_normals[f_idx] points from Owner to Neighbor for internal faces,
        # and outwards for boundary faces (where Owner is the only adjacent cell).
        S_f_vector = mesh.face_normals[f_idx]

        # Determine the sign for the sum: (phi_f * S_f_vector)
        # If current cell_idx is the owner of f_idx, S_f_vector is outward pointing.
        # If current cell_idx is the neighbor of f_idx, S_f_vector is inward pointing, so use -S_f_vector.
        sign = 1.0
        if mesh.owner_cells[f_idx] != cell_idx:
            # This means cell_idx is the neighbor for face f_idx
            # So, the normal S_f (defined C->F) is pointing INTO cell_idx
            sign = -1.0

        # For boundary faces, owner_cells[f_idx] is cell_idx, neighbor_cells[f_idx] is < 0.
        # S_f_vector is already outward pointing for cell_idx.

        grad_phi_cell[0] += sign * phi_f * S_f_vector[0]
        grad_phi_cell[1] += sign * phi_f * S_f_vector[1]

    if cell_vol > _SMALL:
        grad_phi_cell /= cell_vol

    return grad_phi_cell


@njit(parallel=True)
def compute_all_cell_gradients(phi_cell: np.ndarray, mesh) -> np.ndarray:
    """
    Computes the gradient of a scalar field at all cell centers using Green-Gauss.

    Parameters:
    - phi_cell: Scalar field defined at cell centers [n_cells].
    - mesh: MeshData2D structure.

    Returns:
    - grad_phi_all_cells: Gradient field [n_cells, 2].
    """
    n_cells = len(mesh.cell_volumes)
    grad_phi_all_cells = np.zeros((n_cells, 2), dtype=phi_cell.dtype)

    for i in prange(n_cells):
        grad_phi_all_cells[i, :] = calculate_cell_scalar_gradient_green_gauss(
            phi_cell, mesh, i
        )

    return grad_phi_all_cells
