import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# 1.  Face conductances & skew-flux -----------------------------------------
# ---------------------------------------------------------------------------
@njit
def compute_central_diffusion_face_coeffs(
    i_face: int,
    cell_centers: np.ndarray,  # (n_cells, 3)
    face_centers: np.ndarray,  # (n_faces, 3)
    face_normals: np.ndarray,  # (n_faces, 3)
    face_areas: np.ndarray,  # (n_faces,)
    owner_cells: np.ndarray,  # (n_faces,)  int32
    neighbor_cells: np.ndarray,  # (n_faces,)  int32  (−1 ⇒ boundary)
    diffusion_coeffs: np.ndarray,  # (n_cells,)  Γ
    gradients: np.ndarray,  # (n_cells, 3)  ∇φ  (from previous sweep)
    phi: np.ndarray,  # (n_cells,)   φ    (only for α, but harmless)
):
    """
    Textbook central-difference conductances  a_C , a_F  **independent of φ**,
    plus the skew flux that must be sent to the RHS.

    Returns
    -------
    aC, aF : float64
        Owner and neighbour coefficients (aC>0, aF<0).
    skew_flux : float64
        Non-orthogonal correction flux (to subtract from RHS of owner and add
        to RHS of neighbour, keeping conservation).
    """
    C = owner_cells[i_face]
    F = neighbor_cells[i_face]

    # (1) boundary face → coefficients are handled elsewhere
    if F == -1:
        return 0.0, 0.0, 0.0

    # geometry
    xC = cell_centers[C]
    xF = cell_centers[F]
    xf = face_centers[i_face]

    d_vec = xF - xC
    d_mag = np.sqrt(np.dot(d_vec, d_vec))
    if d_mag < 1.0e-20:  # degenerate (nearly zero) distance
        return 0.0, 0.0, 0.0

    d_hat = d_vec / d_mag
    n = face_normals[i_face]
    A = face_areas[i_face]

    # (2) orthogonal conductance
    proj = np.abs(np.dot(n, d_hat))  # always ≥ 0, keeps matrix SPD
    Gamma_f = 0.5 * (diffusion_coeffs[C] + diffusion_coeffs[F])
    a = Gamma_f * A * proj / d_mag  # linear, φ-independent
    aC = a
    aF = -a  # symmetric pair

    # (3) skew flux  q_skew = –Γ A (∇φ · s)
    alpha = np.dot(xf - xC, d_vec) / (d_mag * d_mag)
    x_ip = xC + alpha * d_vec  # interpolation point on CF line
    s = xf - x_ip  # skew vector
    grad_f = 0.5 * (gradients[C] + gradients[F])
    skew_flux = -Gamma_f * A * np.dot(grad_f, s)

    return aC, aF, skew_flux
