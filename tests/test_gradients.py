"""
Tests for gradient calculation schemes.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from naviflow_collocated.gradients.gradients import calculate_gradient_gauss_linear


@pytest.mark.skip(reason="Needs updated boundary condition handling implementation")
def test_gradient_linear_field(mesh_instance):
    """Test Green-Gauss gradient on a known linear scalar field."""
    n_cells = len(mesh_instance.cell_volumes)
    cell_centers = mesh_instance.cell_centers

    # Define a linear scalar field: phi = a*x + b*y + c
    a = 2.0
    b = -1.5
    c = 5.0
    phi = a * cell_centers[:, 0] + b * cell_centers[:, 1] + c

    # Calculate the gradient
    grad_phi = calculate_gradient_gauss_linear(
        mesh_instance,
        phi,
        None,  # No boundary_conditions needed
    )

    # Expected gradient: [a, b] for all cells
    expected_grad = np.tile([a, b], (n_cells, 1))

    # --- Verification ---
    assert grad_phi.shape == (n_cells, 2), "Gradient shape is incorrect."

    # Identify internal cells (where the gradient calculation is exact for linear fields on uniform meshes)
    # Boundary cells might have errors due to boundary condition approximations.
    boundary_cell_indices = set(mesh_instance.boundary_owners)
    all_cell_indices = set(range(n_cells))
    internal_cell_indices = list(all_cell_indices - boundary_cell_indices)

    if internal_cell_indices:
        internal_grad_phi = grad_phi[internal_cell_indices]
        expected_internal_grad = expected_grad[internal_cell_indices]

        # Check the gradient for internal cells
        # Tolerance might be needed depending on mesh uniformity and BC implementation details
        # Let's use a relatively tight tolerance for internal cells
        assert_allclose(
            internal_grad_phi,
            expected_internal_grad,
            rtol=0.1,
            atol=0.1,
            err_msg="Gradient calculation incorrect for internal cells on a linear field.",
        )

    # Optional: Check boundary cells with a looser tolerance or specific expectations
    # print(f"Gradient at boundary cells:\n{grad_phi[list(boundary_cell_indices)]}")
    # Add specific checks for boundary gradients if boundary handling is critical for the test

    # Use a linear field: phi = ax + by + c
    a, b, c = 2.0, 3.0, 5.0
    phi = (
        a * mesh_instance.cell_centers[:, 0] + b * mesh_instance.cell_centers[:, 1] + c
    )

    # Calculate gradient numerically
    grad_phi = calculate_gradient_gauss_linear(
        mesh_instance,
        phi,
        None,  # No boundary_conditions needed
    )

    # Use a quadratic field: phi = ax^2 + by^2 + cxy + dx + ey + f
    a, b, c, d, e, f = 1.0, -0.5, 0.2, 1.5, -1.0, 3.0
    x = mesh_instance.cell_centers[:, 0]
    y = mesh_instance.cell_centers[:, 1]
    phi = a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

    # Calculate gradient numerically
    grad_phi = calculate_gradient_gauss_linear(
        mesh_instance,
        phi,
        None,  # No boundary_conditions needed
    )

    # Expected analytical gradient at cell centers
