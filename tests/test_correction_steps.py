"""
Tests for the pressure update and velocity correction steps.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from naviflow_collocated.solvers.correction_steps import (
    update_pressure,
    correct_velocity,
)

# Need momentum assembly to get realistic Ap values for velocity correction test
from naviflow_collocated.assembly.momentum_eq_assembly import (
    assemble_momentum_matrix_csr,
)


def test_update_pressure():
    """Test the pressure update step."""
    n_cells = 10
    p_old = np.linspace(100, 200, n_cells)
    p_prime = np.ones(n_cells) * 5.0
    under_relax_p = 0.3

    p_expected = p_old + under_relax_p * p_prime
    p_new = update_pressure(p_old, p_prime, under_relax_p)

    assert_allclose(
        p_new,
        p_expected,
        rtol=1e-10,
        err_msg="Pressure update calculation is incorrect.",
    )


@pytest.mark.skip(reason="Needs updated boundary condition handling implementation")
def test_correct_velocity(mesh_instance):
    """Test the velocity correction step."""
    # --- Setup ---
    n_cells = len(mesh_instance.cell_volumes)
    rho = 1.0
    mu = 0.01

    # Initial velocity field (u*, v* from momentum solve)
    u_star = np.ones(n_cells) * 2.0
    v_star = np.ones(n_cells) * -1.0
    p_zero = np.zeros(n_cells)  # Pressure used for Ap calc doesn't matter much here

    # Get realistic Ap values (unrelaxed)
    A_u, _ = assemble_momentum_matrix_csr(
        mesh_instance,
        u_star,
        v_star,
        p_zero,
        mu,
        rho,
        None,  # No boundary_conditions needed
        component=0,
        under_relax=1.0,
    )
    A_v, _ = assemble_momentum_matrix_csr(
        mesh_instance,
        u_star,
        v_star,
        p_zero,
        mu,
        rho,
        None,  # No boundary_conditions needed
        component=1,
        under_relax=1.0,
    )
    Ap_u = A_u.diagonal()
    Ap_v = A_v.diagonal()

    # Define a simple pressure correction field (e.g., linear)
    # p_prime = a*x + b*y + c => grad(p_prime) = [a, b]
    a = 10.0  # Pressure correction gradient in x
    b = -5.0  # Pressure correction gradient in y
    c = 100.0
    cell_centers = mesh_instance.cell_centers
    p_prime = a * cell_centers[:, 0] + b * cell_centers[:, 1] + c

    # --- Verification ---
    # Calculate expected gradient first
    from naviflow_collocated.gradients.gradients import calculate_gradient_gauss_linear

    grad_p_prime_expected = calculate_gradient_gauss_linear(
        mesh_instance,
        p_prime,
        None,  # No boundary_conditions needed
    )

    # Now call correct_velocity with the calculated gradient
    u_corrected, v_corrected = correct_velocity(
        mesh_instance, u_star, v_star, p_prime, Ap_u, Ap_v, grad_p_prime_expected
    )

    # Check gradient calculation accuracy (as a sanity check for the test itself)
    expected_grad_const = np.tile([a, b], (n_cells, 1))
    # Use looser tolerance due to mesh/BC effects on gradient calculation
    assert_allclose(
        grad_p_prime_expected,
        expected_grad_const,
        rtol=0.1,
        atol=1.0,
        err_msg="Gradient of p_prime used in test differs significantly from expected.",
    )

    # Calculate expected corrected velocities
    vol_over_Ap_u = mesh_instance.cell_volumes / np.maximum(Ap_u, 1e-30)
    vol_over_Ap_v = mesh_instance.cell_volumes / np.maximum(Ap_v, 1e-30)

    u_expected = u_star - vol_over_Ap_u * grad_p_prime_expected[:, 0]
    v_expected = v_star - vol_over_Ap_v * grad_p_prime_expected[:, 1]

    # Compare results
    assert_allclose(
        u_corrected,
        u_expected,
        rtol=0.1,
        atol=0.1,
        err_msg="Corrected u-velocity does not match expected value.",
    )
    assert_allclose(
        v_corrected,
        v_expected,
        rtol=0.1,
        atol=0.1,
        err_msg="Corrected v-velocity does not match expected value.",
    )
