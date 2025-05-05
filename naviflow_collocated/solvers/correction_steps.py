"""
Functions for the pressure update and velocity correction steps in the SIMPLE algorithm.
"""

import numpy as np

from naviflow_collocated.mesh import MeshData2D as Mesh


def update_pressure(
    p: np.ndarray, p_prime: np.ndarray, under_relax_p: float
) -> np.ndarray:
    """
    Updates the pressure field using the pressure correction and under-relaxation.

    p_new = p_old + alpha_p * p_prime

    Args:
        p: The pressure field from the previous iteration (or initial guess).
        p_prime: The solved pressure correction field.
        under_relax_p: The under-relaxation factor for pressure.

    Returns:
        The updated pressure field.
    """
    p_new = p + under_relax_p * p_prime
    return p_new


def correct_velocity(
    mesh: Mesh,
    u: np.ndarray,
    v: np.ndarray,
    p_prime: np.ndarray,
    Ap_u: np.ndarray,
    Ap_v: np.ndarray,
    grad_p_prime: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrects the velocity field using the pre-calculated pressure correction gradient.

    u_new = u* - (V / Ap_u) * grad(p')_x
    v_new = v* - (V / Ap_v) * grad(p')_y

    where u*, v* are the velocities solved from the momentum equation.

    Args:
        mesh: The mesh object.
        u: The u-velocity field solved from the momentum equation.
        v: The v-velocity field solved from the momentum equation.
        p_prime: The solved pressure correction field.
        Ap_u: Diagonal coefficients from the u-momentum matrix (unrelaxed form).
        Ap_v: Diagonal coefficients from the v-momentum matrix (unrelaxed form).
        grad_p_prime: Pre-calculated gradient of the pressure correction field.

    Returns:
        Tuple containing the corrected u and v velocity fields.
    """
    # Calculate the correction terms (V / Ap)
    # Ensure Ap terms are not zero to avoid division errors
    vol_over_Ap_u = mesh.cell_volumes / np.maximum(Ap_u, 1e-30)
    vol_over_Ap_v = mesh.cell_volumes / np.maximum(Ap_v, 1e-30)

    # Apply the correction
    u_corrected = u - vol_over_Ap_u * grad_p_prime[:, 0]
    v_corrected = v - vol_over_Ap_v * grad_p_prime[:, 1]

    # Note: Boundary conditions for corrected velocity are implicitly handled
    # by the boundary conditions applied during the momentum solve (for u*, v*)
    # and the boundary conditions used for the p' solve (which affect grad_p_prime near boundaries).
    # No explicit BC application needed here generally.

    return u_corrected, v_corrected
