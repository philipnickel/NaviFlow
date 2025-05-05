"""
Tests for the central differencing diffusion scheme.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_central_diffusion_face_coeffs,
)


def test_central_diff_coefficients(mesh_instance):
    """Test diffusion coefficient calculation for a face."""
    # Get an internal face to test
    internal_faces = np.where(mesh_instance.neighbor_cells >= 0)[0]
    if len(internal_faces) == 0:
        pytest.skip("No internal faces in mesh")
    face_idx = internal_faces[0]

    # Set diffusion coefficient (viscosity)
    mu = 0.01

    # Call the function
    aC_diff, aF_diff, skew_flux = compute_central_diffusion_face_coeffs(
        mesh_instance, face_idx, mu, use_skew_correction=False
    )

    # Check if coefficients are non-zero (unless degenerate geometry)
    assert isinstance(aC_diff, float)
    assert isinstance(aF_diff, float)
    assert isinstance(skew_flux, float)

    # For a standard mesh, diffusion coeffs should be positive
    if mesh_instance.delta_CF[face_idx] > 1e-10:
        assert aC_diff > 0, "Owner diffusion coeff should be positive"
        assert aF_diff > 0, "Neighbor diffusion coeff should be positive"

    # Skew flux should be zero when correction is off
    assert_allclose(skew_flux, 0.0, atol=1e-15)


def test_central_diff_nonorthogonal(mesh_instance):
    """Test non-orthogonal correction calculation."""
    # Get an internal face to test
    internal_faces = np.where(mesh_instance.neighbor_cells >= 0)[0]
    if len(internal_faces) == 0:
        pytest.skip("No internal faces in mesh")
    face_idx = internal_faces[0]

    mu = 0.01

    # Enable skew correction
    aC_diff, aF_diff, skew_flux = compute_central_diffusion_face_coeffs(
        mesh_instance, face_idx, mu, use_skew_correction=True
    )

    # Skew flux might still be zero if the mesh is orthogonal
    # or if gradient is zero, but it should be calculated
    assert isinstance(skew_flux, float)

    # If the mesh has non-orthogonality vectors, skew flux might be non-zero
    if (
        mesh_instance.non_ortho_correction is not None
        and np.linalg.norm(mesh_instance.non_ortho_correction[face_idx]) > 1e-10
    ):
        # Cannot guarantee non-zero without knowing gradients,
        # just check it's a valid float
        pass
    else:
        # If mesh is orthogonal, skew flux must be zero
        assert_allclose(skew_flux, 0.0, atol=1e-15)


def test_central_diff_boundary_face(mesh_instance):
    """Test behavior for a boundary face."""
    # Get a boundary face to test
    if len(mesh_instance.boundary_faces) == 0:
        pytest.skip("No boundary faces in mesh")
    face_idx = mesh_instance.boundary_faces[0]

    mu = 0.01

    # Call the function for a boundary face
    aC_diff, aF_diff, skew_flux = compute_central_diffusion_face_coeffs(
        mesh_instance, face_idx, mu, use_skew_correction=True
    )

    # For a boundary face, all coefficients should be zero
    assert_allclose(aC_diff, 0.0, atol=1e-15)
    assert_allclose(aF_diff, 0.0, atol=1e-15)
    assert_allclose(skew_flux, 0.0, atol=1e-15)


def test_central_diff_conservation(mesh_instance):
    """Test conservation property (related to symmetry)."""
    # Get an internal face to test
    internal_faces = np.where(mesh_instance.neighbor_cells >= 0)[0]
    if len(internal_faces) == 0:
        pytest.skip("No internal faces in mesh")
    face_idx = internal_faces[0]

    mu = 0.01

    # Call the function
    aC_diff, aF_diff, skew_flux = compute_central_diffusion_face_coeffs(
        mesh_instance, face_idx, mu, use_skew_correction=False
    )

    # Check symmetry of primary coefficients (without skew correction)
    # aC should equal aF for central differencing
    assert_allclose(
        aC_diff,
        aF_diff,
        rtol=1e-8,
        err_msg="Diffusion coeffs aC and aF should be equal for CDS",
    )
