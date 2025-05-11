import numpy as np
from numba import njit

EPS = 1.0e-14

BC_DIRICHLET    = 1
BC_NEUMANN      = 2
BC_ZEROGRADIENT = 3
# ──────────────────────────────────────────────────────────────────────────────
# Skew‑corrected interpolation (FMIA Eq. 8.89)
# ──────────────────────────────────────────────────────────────────────────────

@njit
def compute_diffusive_skew_correction(f, grad_phi, mesh, mu):
    """
    Return (P, N, b_skew) — an explicit RHS term correcting for skewness.

        b_skew is SUBTRACTED from the RHS of both owner and neighbour
        (split ½ each, like the non‑orthogonal term).
    """
    P   = mesh.owner_cells[f]
    N   = mesh.neighbor_cells[f]
    muF = mu if isinstance(mu, float) else mu[f]

    # existing geometry
    S_f   = mesh.face_normals[f]
    d_vec = mesh.d_PN[f]
    magd  = np.linalg.norm(d_vec) + EPS
    magS  = np.linalg.norm(S_f)   + EPS
    E_mag = magS / magd           # |E_f| = |S_f| / |d|

    # face gradient (use the same skew‑aware interpolation we already coded)
    g_f = mesh.face_interp_factors[f]
    if N >= 0:
        grad_fprime = (1.0 - g_f) * grad_phi[P] + g_f * grad_phi[N]
    else:
        grad_fprime = grad_phi[P]

    d_skew = mesh.skewness_vectors[f]            # x_f – x_f′
    corr = -muF * np.dot(grad_fprime, d_skew) * E_mag          # |S| / |d|
    return P, N, corr
# ──────────────────────────────────────────────────────────────────────────────
# Internal faces
# ──────────────────────────────────────────────────────────────────────────────
@njit
def compute_diffusive_flux_matrix_entry(f, mesh, mu):
    """
    Return P, N, D_f  with   D_f = Γ_f |S·d| / |d|²  (over‑relaxed mode).
    """
    P   = mesh.owner_cells[f]
    N   = mesh.neighbor_cells[f]
    muf = mu if isinstance(mu, float) else mu[f]

    S_f   = mesh.face_normals[f]
    d_vec = mesh.d_PN[f]

    Sf_dot_d = np.dot(S_f, d_vec) + EPS
    Sf_sq    = np.dot(S_f, S_f)   + EPS

    # Over‑relaxed:  D_f = Γ_f |S_f|² / (S_f·d)
    D_f = muf * Sf_sq / Sf_dot_d
    return P, N, D_f


@njit
def compute_diffusive_correction(f, grad_phi, mesh, mu):
    """Explicit non‑orthogonal correction  −Γ_f ∇φ_f · T_f   (over‑relaxed mode)."""
    P   = mesh.owner_cells[f]
    N   = mesh.neighbor_cells[f]
    muF = mu if isinstance(mu, float) else mu[f]

    S_f = mesh.face_normals[f]
    d   = mesh.d_PN[f]

    Sf_dot_d = np.dot(S_f, d) + EPS
    Sf_sq    = np.dot(S_f, S_f) + EPS
    E_factor = Sf_sq / Sf_dot_d          # λ = |S|² / (S·d)
    T_f  = S_f - E_factor * d            # over‑relaxed: T_f ⟂ S_f

    # use skew‑corrected face gradient: interpolate component‑wise then add skew
    g_f = mesh.face_interp_factors[f]
    grad_fprime = (1.0 - g_f) * grad_phi[P] + g_f * grad_phi[N]

    b_corr = -muF * np.dot(grad_fprime, T_f)
    return P, N, b_corr
# ──────────────────────────────────────────────────────────────────────────────
# Boundary faces
# ──────────────────────────────────────────────────────────────────────────────
@njit
def compute_boundary_diffusive_correction(
        f, phi, grad_phi, mesh, mu, bc_type, bc_val):
    """
    Return (P, a_P, b_P)  —  everything is written to the owner cell only.

       a_P : diagonal coefficient to add
       b_P : RHS increment that will be **subtracted** (b[P]-=b_P)

       (orthogonal‑correction mode)
    """
    P   = mesh.owner_cells[f]
    muF = mu if isinstance(mu, float) else mu[f]
    S_f = mesh.face_normals[f]
    S_mag = np.linalg.norm(S_f) + EPS

    # geometric data
    dPB   = mesh.d_PB[f] + EPS            # distance P–boundary
    n_hat = mesh.unit_dPN[f]              # outward (P→face) unit vector

    # Orthogonal‑correction: E_f magnitude equals |S_f|
    T_f  = S_f - S_mag * n_hat

    if bc_type == BC_DIRICHLET:                                 # Eq. 8.63
        a_P = muF * S_mag / dPB                                 # ⟨FluxCb⟩
        grad_f = (bc_val - phi[P]) / dPB * n_hat
        corr   = -muF * np.dot(grad_f, T_f)

        b_P = a_P * bc_val + corr                               # RHS only φ_B
        return P, a_P, b_P

    if bc_type == BC_NEUMANN:                                   # Eq. 8.66
        b_P = muF * bc_val * S_mag
        return P, 0.0, b_P

    # Zero‑gradient (symmetry / outlet)
    if bc_type == BC_ZEROGRADIENT:
        return P, 0.0, 0.0

    raise ValueError("Unknown BC type")
