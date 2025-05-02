import numpy as np
from numba import njit


@njit
def compute_powerlaw_convection_face_coeffs(
    i_face,
    cell_centers,
    face_centers,
    face_normals,
    face_areas,
    owner_cells,
    neighbor_cells,
    rho_f,
    face_velocity,
    Gamma,
):
    """Per-face power-law *row* contributions (unsymmetric)."""
    C = owner_cells[i_face]
    F = neighbor_cells[i_face]
    if F == -1:
        # physical boundary – handled elsewhere
        return 0.0, 0.0, 0.0, 0.0

    # geometry
    d_vec = cell_centers[F] - cell_centers[C]
    d_mag = np.sqrt(np.dot(d_vec, d_vec))
    if d_mag < 1e-20:
        return 0.0, 0.0, 0.0, 0.0

    A_f = face_areas[i_face]
    n_f = face_normals[i_face]
    rho = rho_f

    # ------ NOTE the leading minus sign ------------------------------
    m_f = -rho * np.dot(face_velocity[i_face], n_f) * A_f  # mass flux

    # diffusive conductance for power-law weight
    Gamma_f = 0.5 * (Gamma[C] + Gamma[F])
    D_f = Gamma_f * A_f / d_mag

    # If no mass flux, convection contribution is zero, scheme defaults to central
    if abs(m_f) < 1e-20:
        fP = 1.0  # Default to central differencing (fP=1)
        D_fp = D_f * fP
        # Convection terms are zero
        diag_C = 0.0 + D_fp
        off_C = 0.0 - D_fp
        diag_F = 0.0 + D_fp
        off_F = 0.0 - D_fp
        return diag_C, off_C, diag_F, off_F

    # Peclet number calculation
    if abs(D_f) < 1e-12:
        P = np.inf * np.sign(m_f)
    else:
        P = m_f / D_f

    fP = max(0.0, (1.0 - 0.1 * np.abs(P)) ** 5)  # Patankar (Restored)

    D_fp = D_f * fP  # Effective diffusion (Restored)

    # --- owner row (C) ------------------------------------------------
    diag_C = (m_f if m_f > 0.0 else 0.0) + D_fp
    off_C = (m_f if m_f < 0.0 else 0.0) - D_fp  # minus → coefficient in matrix

    # --- neighbour row (F) -------------------------------------------
    m_F = -m_f  # flux sign seen from F
    diag_F = (m_F if m_F > 0.0 else 0.0) + D_fp
    off_F = (m_F if m_F < 0.0 else 0.0) - D_fp

    return diag_C, off_C, diag_F, off_F
