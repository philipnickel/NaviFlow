"""
Central differencing for diffusion terms on FVM discretization.

This module implements central differencing discretization for diffusion terms,
which is second-order accurate.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# 1.  Face conductances & skew-flux -----------------------------------------
# ---------------------------------------------------------------------------
@njit
def compute_central_diffusion_face_coeffs(mesh, f, mu, use_skew_correction=False):
    """
    Compute diffusion coefficients at face using central differencing.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh data structure
    f : int
        Face index
    mu : float or ndarray
        Diffusion coefficient (e.g., viscosity)
    use_skew_correction : bool, optional
        Flag to include non-orthogonal correction

    Returns
    -------
    aC_diff : float
        Diffusion coefficient for owner cell diagonal
    aF_diff : float
        Diffusion coefficient for neighbor cell diagonal
    skew_correction_flux : float
        Skewness correction to add to right-hand side
    """
    # Small value to prevent division by zero
    _SMALL = 1.0e-12

    # Get owner and neighbor cells
    C = mesh.owner_cells[f]
    F = mesh.neighbor_cells[f]

    # If this is a boundary face, return zeros (handled separately)
    if F < 0:
        return 0.0, 0.0, 0.0

    # Get face area and normal
    area = mesh.face_areas[f]
    nx = mesh.face_normals[f, 0]
    ny = mesh.face_normals[f, 1]

    # Normalize normal vector
    nmag = np.sqrt(nx * nx + ny * ny)
    if nmag > _SMALL:
        nx = nx / nmag
        ny = ny / nmag

    # Get cell centers
    xC = mesh.cell_centers[C, 0]
    yC = mesh.cell_centers[C, 1]
    xF = mesh.cell_centers[F, 0]
    yF = mesh.cell_centers[F, 1]

    # Get face center (variables xf, yf not used)
    # xf = mesh.face_centers[f, 0]
    # yf = mesh.face_centers[f, 1]

    # Vector from owner to neighbor
    eCF_x = xF - xC
    eCF_y = yF - yC
    eCF_mag = np.sqrt(eCF_x * eCF_x + eCF_y * eCF_y)

    # Safety check for degenerate cells
    if eCF_mag < _SMALL:
        return 0.0, 0.0, 0.0

    # Dot product of CF vector and normal to get projection
    eCF_dot_n = eCF_x * nx + eCF_y * ny

    # Distance from cell centers projected onto normal direction
    delta_CF = abs(eCF_dot_n)

    # Safety check for too small distance
    delta_CF = max(delta_CF, _SMALL)

    # Get interpolation factor (fraction of distance from C to F)
    fx = mesh.face_interp_factors[f]

    # Get the diffusion coefficient (viscosity)
    # For scalar mu, just use it directly; otherwise interpolate
    if isinstance(mu, float):
        mu_f = mu
    else:
        # Linear interpolation of viscosity to face
        mu_f = fx * mu[F] + (1.0 - fx) * mu[C]

    # Compute diffusion coefficients
    # Primary coefficients (orthogonal contribution)
    diffusion_coeff = mu_f * area / delta_CF
    aC_diff = diffusion_coeff
    aF_diff = diffusion_coeff

    # Non-orthogonal correction (skewed grid correction)
    skew_correction_flux = 0.0
    if use_skew_correction and mesh.non_ortho_correction is not None:
        # Compute the non-orthogonality correction vectors
        kf_x = mesh.non_ortho_correction[f, 0]
        kf_y = mesh.non_ortho_correction[f, 1]

        # Compute gradients at cells
        grad_C_x = 0.0  # Would be computed from gradient calculation
        grad_C_y = 0.0  # Skipping for now
        grad_F_x = 0.0
        grad_F_y = 0.0

        # Interpolate gradient to face
        grad_f_x = fx * grad_F_x + (1.0 - fx) * grad_C_x
        grad_f_y = fx * grad_F_y + (1.0 - fx) * grad_C_y

        # Dot product of non-orthogonal correction and interpolated gradient
        skew_correction_flux = mu_f * (kf_x * grad_f_x + kf_y * grad_f_y)

    return aC_diff, aF_diff, skew_correction_flux
