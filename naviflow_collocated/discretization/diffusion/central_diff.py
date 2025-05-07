import numpy as np
from numba import njit


@njit(inline="always")
def compute_diffusive_flux(f, u, grad_u, mesh, mu):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]
    mu_f = mu if isinstance(mu, float) else mu[f]

    delta = mesh.delta_PN[f]  # || d⃗  . ê_f ||
    E = mesh.e_f[f]  # vector projection of d⃗  onto S_f

    # Orthogonal contribution: symmetric stencil
    D_f = mu_f * np.linalg.norm(E) ** 2 / (delta + 1e-14)
    a_PN = -D_f
    a_PP = D_f  # anti-symmetry

    # Non-orthogonal correction (explicit)
    grad_u_P = np.ascontiguousarray(grad_u[P])
    T_cont = np.ascontiguousarray(mesh.non_ortho_correction[f])
    b_corr = -mu_f * np.dot(grad_u_P, T_cont)

    return P, N, a_PP, a_PN, b_corr
