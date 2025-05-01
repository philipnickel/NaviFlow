from numba import njit
import numpy as np


@njit
def compute_cell_gradients(
    cell_centers,
    face_centers,
    face_areas,
    face_normals,
    owner_cells,
    neighbor_cells,
    phi,
):
    n_cells = cell_centers.shape[0]
    n_faces = face_centers.shape[0]
    gradients = np.zeros((n_cells, 2))  # assuming 2D

    for f in range(n_faces):
        C = owner_cells[f]
        F = neighbor_cells[f]
        n = face_normals[f]
        area = face_areas[f]
        Sf = n * area

        if F != -1:
            dphi = phi[F] - phi[C]
        else:
            dphi = 0.0

        gradients[C, 0] += dphi * Sf[0]
        gradients[C, 1] += dphi * Sf[1]

        if F != -1:
            gradients[F, 0] -= dphi * Sf[0]
            gradients[F, 1] -= dphi * Sf[1]

    return gradients
