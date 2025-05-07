import numpy as np
from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_diffusive_flux,
)
from naviflow_collocated.discretization.convection.upwind import (
    compute_convective_flux_upwind,
)


def assemble_momentum_matrix(u, grad_u, mesh, rho, mu):
    """
    Assemble momentum matrix A and source vector b using:
    - Central differencing with non-orthogonal correction for diffusion
    - Upwind scheme for convection (can be switched later)
    - Practice B BC handling (modifying coefficients)

    Parameters:
    - u: ndarray (n_cells,) — current velocity field
    - grad_u: ndarray (n_cells, 2) — velocity gradient field
    - mesh: MeshData2D
    - rho: float — fluid density
    - mu: float — dynamic viscosity

    Returns:
    - a_P: ndarray (n_cells,) — diagonal coefficients
    - a_N_list: list of (P, N, value) — off-diagonal entries
    - b_P: ndarray (n_cells,) — source terms
    """
    a_P = np.zeros_like(u)
    b_P = np.zeros_like(u)
    a_N_list = []

    # Internal faces: handle convection and diffusion
    for f in mesh.internal_faces:
        # --- Diffusion ---
        P, N, a_PP_diff, a_PN_diff, b_corr_diff = compute_diffusive_flux(
            f, u, grad_u, mesh, mu
        )
        a_P[P] += a_PP_diff
        a_N_list.append((P, N, a_PN_diff))
        b_P[P] += b_corr_diff

        # --- Convection ---
        uf = np.array([1.0, 0.0])  # Replace with interpolated field later
        P_c, N_c, a_PP_conv, a_PN_conv, b_corr_conv = compute_convective_flux_upwind(
            f, u, mesh, uf, rho
        )
        a_P[P_c] += a_PP_conv
        a_N_list.append((P_c, N_c, a_PN_conv))
        b_P[P_c] += b_corr_conv

    for f in mesh.boundary_faces:
        P = mesh.owner_cells[f]
        bc_val = mesh.boundary_values[f, 0]

        E_f = mesh.e_f[f]
        T_f = mesh.non_ortho_correction[f]
        delta = mesh.d_PB[f] + 1e-14  # Avoid divide-by-zero

        # --- Diffusion ---
        D_f = mu * np.dot(E_f, E_f) / delta
        a_P[P] += D_f
        b_P[P] += D_f * bc_val

        grad_u_P = grad_u[P]  # <- FIXED
        b_corr = -mu * np.dot(grad_u_P, T_f)
        b_P[P] += b_corr

        # --- Convection (inflow only) ---
        uf = np.array([1.0, 0.0])
        S_f_vec = mesh.face_normals[f]
        F = rho * np.dot(uf, S_f_vec)

        if F < 0:  # Inflow
            a_P[P] += -F
            b_P[P] += -F * bc_val

    return a_P, a_N_list, b_P
