# naviflow_collocated/assembly/momentum_eq.py
import numpy as np
from numba import njit, prange
from naviflow_collocated.discretization.diffusion.central_diff import (
    diffusion_face_contrib,
)
from naviflow_collocated.discretization.convection.power_law import (
    convection_face_contrib,
)


@njit(parallel=True)
def assemble_momentum_matrix(
    n_faces,
    face_owner,
    face_neighbor,
    face_areas,
    face_normals,
    cell_centers,
    face_centers,
    gamma,
    rho,
    u,
    v,
    aP,
    aN,
    Su,
):
    for f in prange(n_faces):
        i = face_owner[f]
        j = face_neighbor[f]

        if j == -1:
            continue  # boundary face: skip or handle elsewhere

        # Face vector quantities
        Sf = face_areas[f] * face_normals[f]
        dx = cell_centers[j] - cell_centers[i]
        d_mag = np.dot(dx, dx) ** 0.5

        # Compute diffusion
        d_contrib = diffusion_face_contrib(gamma, d_mag, face_areas[f])
        aP[i] += d_contrib
        aN[f] = -d_contrib

        # Compute convection
        u_face = 0.5 * (u[i] + u[j])
        v_face = 0.5 * (v[i] + v[j])
        phi = rho * (u_face * Sf[0] + v_face * Sf[1])

        f_contrib = convection_face_contrib(phi)
        aP[i] += f_contrib
        Su[i] += phi  # provisional source term
