from numba import njit
import numpy as np


@njit
def compute_divergence_from_face_fluxes(mesh, face_fluxes):
    """
    Compute cell-centered divergence from face-normal mass fluxes.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh structure containing cell-face connectivity.
    face_fluxes : ndarray
        Mass flux at each face.

    Returns
    -------
    divergence : ndarray
        Cell-centered divergence of mass fluxes.
    """
    divergence = np.zeros(mesh.cell_volumes.shape[0])

    for f in range(len(face_fluxes)):
        C = mesh.owner_cells[f]
        F = mesh.neighbor_cells[f]

        divergence[C] -= face_fluxes[f] # Måske fortegnsfejl
        if F >= 0:
            divergence[F] += face_fluxes[f]

    return divergence
