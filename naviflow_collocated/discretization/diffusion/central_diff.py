import numpy as np
from numba import njit


@njit
def compute_central_diffusion_face(
    i_face,
    cell_centers,
    face_centers,
    face_normals,
    face_areas,
    owner_cells,
    neighbor_cells,
    diffusion_coeffs,
    phi,
    gradients,
):
    i_owner = owner_cells[i_face]
    i_neigh = neighbor_cells[i_face]

    n = face_normals[i_face]
    area = face_areas[i_face]
    k = 0.5 * (diffusion_coeffs[i_owner] + diffusion_coeffs[i_neigh])

    # Gradient at face (simple average)
    grad_phi_f = 0.5 * (gradients[i_owner] + gradients[i_neigh])

    # Use dot product directly
    flux = -k * area * np.dot(grad_phi_f, n)
    return flux
