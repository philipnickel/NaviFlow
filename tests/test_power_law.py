"""
Tests for the Power Law convection scheme.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from naviflow_collocated.discretization.convection.power_law import (
    compute_powerlaw_convection_face_coeffs,
)


def test_power_law_zero_velocity(mesh_instance):
    """Test power law scheme with zero face velocity."""
    # Get an internal face to test
    internal_faces = np.where(mesh_instance.neighbor_cells >= 0)[0]
    if len(internal_faces) == 0:
        pytest.skip("No internal faces in mesh")
    face_idx = internal_faces[0]

    # Set zero flux
    face_flux = 0.0
    rho = 1.0
    mu = 0.01  # Need viscosity for Peclet number

    # Call the function
    diag_C, off_C, diag_F, off_F = compute_powerlaw_convection_face_coeffs(
        mesh_instance, face_idx, face_flux, rho, mu
    )

    # With zero flux (Pe=0), convection coefficients should be zero
    # Power Law defaults to Central Differencing effectively, but only convection part
    assert_allclose(diag_C, 0.0, atol=1e-15)
    assert_allclose(off_C, 0.0, atol=1e-15)
    assert_allclose(diag_F, 0.0, atol=1e-15)
    assert_allclose(off_F, 0.0, atol=1e-15)


def test_power_law_flow(mesh_instance):
    """Test power law scheme with non-zero face velocity."""
    # Get an internal face to test
    internal_faces = np.where(mesh_instance.neighbor_cells >= 0)[0]
    if len(internal_faces) == 0:
        pytest.skip("No internal faces in mesh")
    face_idx = internal_faces[0]

    # Set parameters for a simple case
    face_flux = 1.0  # Positive flux (C to F)
    rho = 1.0
    mu = 0.01

    # Calculate expected A(Pe)
    D_f = mu * mesh_instance.face_areas[face_idx] / mesh_instance.delta_CF[face_idx]
    Pe = face_flux / max(D_f, 1e-12)
    A_Pe = max(0.0, (1.0 - 0.1 * abs(Pe)) ** 5)

    # Call the function
    diag_C, off_C, diag_F, off_F = compute_powerlaw_convection_face_coeffs(
        mesh_instance, face_idx, face_flux, rho, mu
    )

    # Check properties for positive flux
    assert_allclose(diag_C, face_flux, rtol=1e-5)
    assert_allclose(off_C, -face_flux * A_Pe, rtol=1e-5)
    assert_allclose(diag_F, 0.0, atol=1e-15)
    assert_allclose(off_F, face_flux * (1.0 - A_Pe), rtol=1e-5)

    # Test with negative flux (F to C)
    face_flux = -1.0
    Pe = face_flux / max(D_f, 1e-12)
    A_Pe = max(0.0, (1.0 - 0.1 * abs(Pe)) ** 5)

    diag_C, off_C, diag_F, off_F = compute_powerlaw_convection_face_coeffs(
        mesh_instance, face_idx, face_flux, rho, mu
    )

    assert_allclose(diag_C, 0.0, atol=1e-15)
    assert_allclose(off_C, -face_flux * (1.0 - A_Pe), rtol=1e-5)
    assert_allclose(diag_F, -face_flux, rtol=1e-5)
    assert_allclose(off_F, face_flux * A_Pe, rtol=1e-5)


def test_power_law_high_peclet(mesh_instance):
    """Test power law scheme with high Peclet number (convection dominated)."""
    # Get an internal face to test
    internal_faces = np.where(mesh_instance.neighbor_cells >= 0)[0]
    if len(internal_faces) == 0:
        pytest.skip("No internal faces in mesh")

    # Try a few internal faces to find one with reasonable geometry
    found_suitable_face = False
    for face_idx in internal_faces[
        : min(len(internal_faces), 10)
    ]:  # Check up to 10 faces
        if mesh_instance.delta_CF[face_idx] > 1e-10:
            found_suitable_face = True
            break

    if not found_suitable_face:
        pytest.skip(
            f"Could not find an internal face with delta_CF > 1e-10 in the first {min(len(internal_faces), 10)} faces."
        )

    # High flux to simulate high Pe
    face_flux = 100.0
    rho = 1.0
    mu = 0.001  # Low viscosity for high Pe

    # Call the function
    diag_C, off_C, diag_F, off_F = compute_powerlaw_convection_face_coeffs(
        mesh_instance, face_idx, face_flux, rho, mu
    )

    # For high Peclet, A(Pe) approaches 0, Power Law approaches Upwind
    D_f = mu * mesh_instance.face_areas[face_idx] / mesh_instance.delta_CF[face_idx]
    Pe = face_flux / max(D_f, 1e-12)
    A_Pe = max(0.0, (1.0 - 0.1 * abs(Pe)) ** 5)

    # ---- DEBUG PRINT ----
    delta_val = mesh_instance.delta_CF[face_idx]
    area_val = mesh_instance.face_areas[face_idx]
    print(
        f"\nDEBUG High Pe Test (face {face_idx}): mu={mu:.4g}, area={area_val:.4g}, delta_CF={delta_val:.4g}, D_f={D_f:.6g}, Pe={Pe:.6g}, A_Pe={A_Pe:.6g}"
    )

    assert A_Pe < 1e-5, f"A(Pe) should be close to zero for high Pe, got {A_Pe}"

    assert_allclose(diag_C, face_flux, rtol=1e-5)
    assert_allclose(off_C, -face_flux * A_Pe, atol=1e-5)
    assert_allclose(diag_F, 0.0, atol=1e-15)
    assert_allclose(off_F, face_flux * (1.0 - A_Pe), rtol=1e-5)


def test_power_law_low_peclet(mesh_instance):
    """Test power law scheme with low Peclet number (diffusion dominated)."""
    # Get an internal face to test
    internal_faces = np.where(mesh_instance.neighbor_cells >= 0)[0]
    if len(internal_faces) == 0:
        pytest.skip("No internal faces in mesh")

    # Try a few internal faces to find one with reasonable geometry
    found_suitable_face = False
    for face_idx in internal_faces[
        : min(len(internal_faces), 10)
    ]:  # Check up to 10 faces
        if mesh_instance.delta_CF[face_idx] > 1e-10:
            found_suitable_face = True
            break

    if not found_suitable_face:
        pytest.skip(
            f"Could not find an internal face with delta_CF > 1e-10 in the first {min(len(internal_faces), 10)} faces."
        )

    # Low flux to simulate low Pe
    face_flux = 0.01
    rho = 1.0
    mu = 1.0  # High viscosity for low Pe

    # Call the function
    diag_C, off_C, diag_F, off_F = compute_powerlaw_convection_face_coeffs(
        mesh_instance, face_idx, face_flux, rho, mu
    )

    # For low Peclet, A(Pe) approaches 1, Power Law approaches Central Diff.
    D_f = mu * mesh_instance.face_areas[face_idx] / mesh_instance.delta_CF[face_idx]
    Pe = face_flux / max(D_f, 1e-12)
    A_Pe = max(0.0, (1.0 - 0.1 * abs(Pe)) ** 5)

    # Relax tolerance slightly as A_Pe might not be exactly 1 for very small Pe
    assert abs(A_Pe - 1.0) < 1e-2, f"A(Pe) should be close to 1 for low Pe, got {A_Pe}"

    assert_allclose(diag_C, face_flux, rtol=1e-5)
    assert_allclose(off_C, -face_flux * A_Pe, rtol=1e-2)
    assert_allclose(diag_F, 0.0, atol=1e-15)
    assert_allclose(off_F, face_flux * (1.0 - A_Pe), atol=1e-2)


def test_power_law_boundary_face(mesh_instance):
    """Test power law scheme for a boundary face."""
    # Get a boundary face to test
    if len(mesh_instance.boundary_faces) == 0:
        pytest.skip("No boundary faces in mesh")
    face_idx = mesh_instance.boundary_faces[0]

    # Set parameters
    face_flux = 0.5
    rho = 1.0
    mu = 0.01

    # Call the function for a boundary face
    diag_C, off_C, diag_F, off_F = compute_powerlaw_convection_face_coeffs(
        mesh_instance, face_idx, face_flux, rho, mu
    )

    # For a boundary face, all coefficients should be zero
    assert_allclose(diag_C, 0.0, atol=1e-15)
    assert_allclose(off_C, 0.0, atol=1e-15)
    assert_allclose(diag_F, 0.0, atol=1e-15)
    assert_allclose(off_F, 0.0, atol=1e-15)
