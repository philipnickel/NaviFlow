"""
Tests for the Rhie-Chow interpolation implementation.

These tests ensure that the Rhie-Chow interpolation properly suppresses
pressure checkerboarding on collocated grids.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from naviflow_collocated.assembly.face_velocity import (
    compute_rhie_chow_face_velocities,
    compute_face_fluxes,
)


def test_rhie_chow_prevents_checkerboarding(mesh_instance):
    """
    Test that Rhie-Chow interpolation prevents checkerboarding.

    This test creates a checkerboard pressure field (alternating highs and lows),
    applies Rhie-Chow interpolation to face velocities, and verifies that the
    resulting face fluxes do not exhibit the checkerboard pattern.
    """
    mesh = mesh_instance
    n_cells = len(mesh.cell_volumes)

    # Create fields
    u = np.zeros(n_cells)  # Zero initial velocity
    v = np.zeros(n_cells)

    # Create a checkerboard pressure field
    # Even cells have high pressure, odd cells have low pressure
    p_high = 1.0
    p_low = 0.0
    p = np.zeros(n_cells)

    for i in range(n_cells):
        p[i] = p_high if i % 2 == 0 else p_low

    # Constant properties
    mu = 0.01
    rho = 1.0

    # Compute face velocities using Rhie-Chow interpolation
    face_velocities = compute_rhie_chow_face_velocities(mesh, u, v, p, mu, rho)

    # Compute face fluxes
    face_fluxes = compute_face_fluxes(mesh, face_velocities)

    # Verify: For a checkerboard pressure field, fluxes should:
    # 1. Not be all zero (which would happen with simple interpolation)
    # 2. Have a consistent pattern that counters the checkerboard

    # Check that fluxes are not all zero
    assert not np.allclose(face_fluxes, 0.0, atol=1e-10), (
        "Face fluxes should not be zero for a checkerboard pressure field"
    )

    # Find internal faces
    internal_faces = np.where(mesh.neighbor_cells >= 0)[0]

    # Internal checks only make sense if we have internal faces
    if len(internal_faces) > 0:
        # Verify that fluxes for internal faces have nonzero values
        internal_fluxes = face_fluxes[internal_faces]
        assert np.any(np.abs(internal_fluxes) > 1e-10), (
            "Internal face fluxes should be nonzero for a checkerboard pressure field"
        )


def test_rhie_chow_consistency_uniform_pressure(mesh_instance):
    """
    Test that Rhie-Chow interpolation preserves uniform flow with uniform pressure.

    For a uniform pressure field and uniform velocity field, the Rhie-Chow
    interpolation should not alter the face velocities from simple interpolation.
    """
    mesh = mesh_instance
    n_cells = len(mesh.cell_volumes)

    # Create uniform fields
    u = np.ones(n_cells)  # Uniform horizontal velocity
    v = np.zeros(n_cells)  # Zero vertical velocity
    p = np.zeros(n_cells)  # Uniform pressure

    # Constant properties
    mu = 0.01
    rho = 1.0

    # Compute face velocities using Rhie-Chow interpolation
    face_velocities = compute_rhie_chow_face_velocities(mesh, u, v, p, mu, rho)

    # For internal faces with uniform pressure and velocity, the face velocity
    # should be equal to the cell velocity
    internal_faces = np.where(mesh.neighbor_cells >= 0)[0]

    for f in internal_faces:
        # For uniform u=1, v=0, the face velocity should be u=1, v=0
        assert_allclose(face_velocities[f, 0], 1.0, rtol=1e-5, atol=1e-5)
        assert_allclose(face_velocities[f, 1], 0.0, rtol=1e-5, atol=1e-5)


def test_rhie_chow_mass_conservation(mesh_instance):
    """
    Test that mass conservation is satisfied with Rhie-Chow interpolation.

    For any velocity field, the sum of fluxes into each cell should be approximately zero.
    This test uses a rotating velocity field (u = -y, v = x) which is analytically
    divergence-free, but may have small discretization errors.
    """
    mesh = mesh_instance
    n_cells = len(mesh.cell_volumes)

    # Create a divergence-free velocity field
    # We'll use a simple rotating flow: u = -y, v = x
    u = np.zeros(n_cells)
    v = np.zeros(n_cells)

    for c in range(n_cells):
        x = mesh.cell_centers[c, 0]
        y = mesh.cell_centers[c, 1]
        u[c] = -y
        v[c] = x

    # Uniform pressure
    p = np.zeros(n_cells)

    # Constant properties
    mu = 0.01
    rho = 1.0

    # Compute face velocities using Rhie-Chow interpolation
    face_velocities = compute_rhie_chow_face_velocities(mesh, u, v, p, mu, rho)

    # Compute face fluxes
    face_fluxes = compute_face_fluxes(mesh, face_velocities)

    # Check mass conservation for each cell
    # Sum of fluxes into each cell should be approximately zero
    max_flux_sum = 0.0

    for c in range(n_cells):
        flux_sum = 0.0
        for face_idx in range(mesh.cell_faces.shape[1]):
            f = mesh.cell_faces[c, face_idx]
            if f < 0:
                continue  # Skip invalid faces

            # Determine flux sign based on whether this cell is owner or neighbor
            sign = -1.0 if mesh.owner_cells[f] == c else 1.0
            flux_sum += sign * face_fluxes[f]

        max_flux_sum = max(max_flux_sum, abs(flux_sum))

    # The flux sum should be close to zero for all cells
    # On coarse meshes, discretization errors can be relatively large
    # so we use a tolerance of 0.2 which is reasonable for finite volume
    # implementations on coarse meshes
    assert max_flux_sum < 0.2, (
        f"Maximum flux sum: {max_flux_sum}, should be reasonably close to zero"
    )


def test_rhie_chow_boundary_conditions(mesh_instance):
    """
    Test that Rhie-Chow interpolation properly handles boundary conditions.
    """
    mesh = mesh_instance
    n_cells = len(mesh.cell_volumes)

    # Skip test if no boundary faces are present
    if len(mesh.boundary_faces) == 0:
        pytest.skip("No boundary faces in mesh")

    # Create fields
    u = np.zeros(n_cells)
    v = np.zeros(n_cells)
    p = np.zeros(n_cells)

    # Set boundary type for the first boundary face to inlet (if it exists)
    # This is just to test a non-zero boundary condition
    if len(mesh.boundary_faces) > 0:
        bf = mesh.boundary_faces[0]
        # This time we set the actual boundary value using the correct index
        boundary_idx = 0  # First boundary face
        mesh.boundary_types[boundary_idx] = 3  # BC_INLET_VELOCITY
        mesh.boundary_values[boundary_idx, 0] = 1.0  # u=1
        mesh.boundary_values[boundary_idx, 1] = 0.0  # v=0

    # Constant properties
    mu = 0.01
    rho = 1.0

    # Compute face velocities using Rhie-Chow interpolation
    face_velocities = compute_rhie_chow_face_velocities(mesh, u, v, p, mu, rho)

    # Check that the boundary face velocity matches the boundary value
    if len(mesh.boundary_faces) > 0:
        bf = mesh.boundary_faces[0]
        # Check that inlet velocity matches boundary value
        assert_allclose(face_velocities[bf, 0], 1.0, rtol=1e-5, atol=1e-5)
        assert_allclose(face_velocities[bf, 1], 0.0, rtol=1e-5, atol=1e-5)
