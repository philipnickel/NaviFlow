"""
Power Law scheme for convection terms in FVM discretization.

This module implements the Power Law scheme for discretizing
convection terms, which is accurate for a wide range of Peclet numbers.
"""

import numpy as np
from numba import njit


@njit
def compute_powerlaw_convection_face_coeffs(mesh, f, face_flux, rho, mu):
    """
    Compute convection coefficients at face using the Power Law scheme.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh data structure
    f : int
        Face index
    face_flux : float
        Mass flux through the face
    rho : float
        Density
    mu : float or ndarray
        Dynamic viscosity (used for Peclet number calculation)

    Returns
    -------
    dC_conv : float
        Diagonal convection coefficient for owner cell
    oC_conv : float
        Off-diagonal convection coefficient for owner cell
    dF_conv : float
        Diagonal convection coefficient for neighbor cell
    oF_conv : float
        Off-diagonal convection coefficient for neighbor cell
    """
    # Small value to prevent division by zero
    _SMALL = 1.0e-12

    # Get owner and neighbor cells
    C = mesh.owner_cells[f]
    F = mesh.neighbor_cells[f]

    # If this is a boundary face, return zeros (handled separately)
    if F < 0:
        return 0.0, 0.0, 0.0, 0.0

    # Get face area and interpolation factor
    area = mesh.face_areas[f]
    fx = mesh.face_interp_factors[f]

    # Calculate Diffusion Conductance (D_f)
    # Get the diffusion coefficient (viscosity) at the face
    if isinstance(mu, float):
        mu_f = mu
    else:
        mu_f = fx * mu[F] + (1.0 - fx) * mu[C]  # Linear interpolation

    # Distance between cell centers (needed for D_f)
    delta_CF = mesh.delta_CF[f]  # Use precomputed, should not be zero

    D_f = mu_f * area / delta_CF  # Diffusion conductance

    # Use provided face_flux (mass flux = rho * u_n * area)
    flux = face_flux

    # Calculate Peclet number (Pe = F / D = face_flux / D_f)
    if abs(D_f) < _SMALL:
        # Avoid division by zero if diffusion is negligible
        # Treat as pure convection (very high Pe)
        Pe = np.inf * np.sign(flux)
    else:
        Pe = flux / D_f

    # Apply Power Law scheme
    # A(|Pe|) = max(0, (1 - 0.1|Pe|)^5)
    A_Pe = max(0.0, (1.0 - 0.1 * abs(Pe)) ** 5)

    # Calculate coefficients based on flow direction
    if flux >= 0:  # Flow from C to F
        dC_conv = flux
        oC_conv = -flux * A_Pe
        dF_conv = 0.0
        oF_conv = flux * (1.0 - A_Pe)
    else:  # Flow from F to C
        dC_conv = 0.0
        oC_conv = -flux * (1.0 - A_Pe)
        dF_conv = -flux
        oF_conv = flux * A_Pe

    return dC_conv, oC_conv, dF_conv, oF_conv
