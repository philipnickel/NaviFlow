import numpy as np
from numba import njit


@njit(inline="always")
def compute_convective_flux_power_law(f, phi, grad_phi, mesh, rho, u):
    """
    Compute face flux for convection using Power Law scheme.

    Returns:
        phi_flux -- convective face flux (rho u ⋅ S) * interpolated φ
    """
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    alpha = mesh.face_interp_factors[f]
    u_f = (1 - alpha) * u[P] + alpha * u[N]
    Sf = mesh.face_normals[f]
    F = rho * np.dot(np.ascontiguousarray(u_f), np.ascontiguousarray(Sf))

    phi_P = phi[P]
    phi_N = phi[N]
    delta = mesh.delta_PN[f] + 1e-14
    Gamma = 1.0
    D_f = (
        Gamma
        * np.dot(np.ascontiguousarray(Sf), np.ascontiguousarray(mesh.unit_dPN[f]))
        / delta
    )
    Pe = F / (D_f + 1e-14)
    Pe_abs = abs(Pe)
    Pe_clip = min(max(Pe_abs, 0.0), 10.0)
    weight = max((1 - 0.1 * Pe_clip) ** 5, 0.0)

    phi_face = phi_P * weight if F >= 0 else phi_N * weight
    return F * phi_face
