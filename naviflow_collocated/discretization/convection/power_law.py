import numpy as np
from numba import njit


@njit
def power_law_weight(Pe):
    Pe = max(min(Pe, 1e3), -1e3)
    return max(0.0, (1.0 - 0.1 * abs(Pe)) ** 5)


@njit
def compute_convection_face(
    i_face,
    phi,
    density,
    velocity,
    face_areas,
    face_normals,
    owner_cells,
    neighbor_cells,
):
    i_owner = owner_cells[i_face]
    i_neigh = neighbor_cells[i_face]

    flux_vector = velocity[i_face] * face_areas[i_face]  # Vector flux
    flux = np.dot(flux_vector, face_normals[i_face])  # Scalar flux

    phi_upwind = phi[i_owner] if flux >= 0 else phi[i_neigh]
    phi_downwind = phi[i_neigh] if flux >= 0 else phi[i_owner]

    Pe = flux / (1e-20 + abs(phi_downwind - phi_upwind))  # Avoid division by zero
    w = power_law_weight(Pe)

    interpolated_phi = phi_upwind + w * (phi_downwind - phi_upwind)
    return flux * interpolated_phi
