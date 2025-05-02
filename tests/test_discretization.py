import numpy as np

from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_central_diffusion_face_coeffs,
)
from naviflow_collocated.discretization.convection.power_law import (
    compute_powerlaw_convection_face_coeffs,
)
from naviflow_collocated.mesh.mesh_data import mesh_to_data
from naviflow_collocated.discretization.gradient.leastSquares import (
    compute_least_squares_gradients,
)


def estimate_mesh_size(md):
    return np.min(np.sqrt(md.cell_volumes))


def manufactured_linear_field(md):
    """
    Manufactured solution: phi(x, y) = x + y
    - Exact gradient = [1.0, 1.0]
    - Diffusive flux = -Gamma * (grad_phi ⋅ n) * area
    """
    phi = md.cell_centers[:, 0] + md.cell_centers[:, 1]
    exact_grad = np.array([1.0, 1.0])
    return phi, exact_grad


def test_gradient_method_accuracy(mesh_instance):
    """
    Tests the least-squares gradient method on a manufactured linear field.
    For a linear scalar field, gradients should be exact up to machine error.
    """
    md = mesh_to_data(mesh_instance)
    phi, exact_grad = manufactured_linear_field(md)

    gradients = compute_least_squares_gradients(
        md.cell_centers, md.owner_cells, md.neighbor_cells, phi
    )
    error_vectors = gradients - exact_grad
    l2_error = np.sqrt(np.sum(error_vectors**2) / gradients.shape[0])
    h = estimate_mesh_size(md)
    tol = 5 * h  # linear field ⇒ zero truncation, only geometric error
    assert l2_error < tol, f"L2 gradient error too high: {l2_error:.3e} (tol={tol:.3e})"


def test_flux_with_computed_gradient(mesh_instance):
    """
    Uses a manufactured linear solution to validate face-based diffusive flux
    computation. Since the exact gradient is constant, numerical and analytical
    fluxes should agree to within geometric precision.
    """
    md = mesh_to_data(mesh_instance)
    phi, _ = manufactured_linear_field(md)
    Gamma = 1.0
    diffusion_coeffs = Gamma * np.ones_like(phi)

    gradients = compute_least_squares_gradients(
        md.cell_centers, md.owner_cells, md.neighbor_cells, phi
    )

    numeric_fluxes = []
    exact_fluxes = []

    for f in range(len(md.face_areas)):
        C = md.owner_cells[f]
        F = md.neighbor_cells[f]
        if F == -1:
            continue  # Skip boundary

        aC, aF, _ = compute_central_diffusion_face_coeffs(
            f,
            md.cell_centers,
            md.face_centers,
            md.face_normals,
            md.face_areas,
            md.owner_cells,
            md.neighbor_cells,
            diffusion_coeffs,
            gradients,
            phi,
        )
        numeric_flux = aC * phi[C] + aF * phi[F]
        numeric_fluxes.append(numeric_flux)

        exact_flux = -Gamma * np.dot([1.0, 1.0], md.face_normals[f]) * md.face_areas[f]
        exact_fluxes.append(exact_flux)

    error = np.linalg.norm(np.array(numeric_fluxes) - np.array(exact_fluxes)) / np.sqrt(
        len(numeric_fluxes)
    )
    h = estimate_mesh_size(md)
    tol = 5 * h
    assert error < tol, (
        f"L2 flux error too high with computed gradients: {error:.3e} (tol={tol:.3e})"
    )


def test_powerlaw_convection_coeffs_manufactured(mesh_instance):
    """
    Validates per-face power-law convection coefficients against a manufactured
    linear scalar field and uniform flow. Ensures stencil coefficients agree with
    expected upwind-diffusion logic.
    """
    md = mesh_to_data(mesh_instance)
    phi = md.cell_centers[:, 0] + md.cell_centers[:, 1]
    Gamma = np.ones_like(phi) * 0.01
    rho_f = 1.0

    n_faces = md.face_areas.shape[0]
    face_velocity = np.tile(np.array([1.0, 1.0]), (n_faces, 1)) / np.sqrt(2)

    for f in range(n_faces):
        C = md.owner_cells[f]
        F = md.neighbor_cells[f]
        if F == -1:
            continue

        diag_C, off_C, diag_F, off_F = compute_powerlaw_convection_face_coeffs(
            f,
            md.cell_centers,
            md.face_centers,
            md.face_normals,
            md.face_areas,
            md.owner_cells,
            md.neighbor_cells,
            rho_f,
            face_velocity,
            Gamma,
        )

        d_vec = md.cell_centers[F] - md.cell_centers[C]
        d_mag = np.linalg.norm(d_vec)
        A = md.face_areas[f]
        n = md.face_normals[f]
        m_f = -rho_f * np.dot(face_velocity[f], n) * A
        D_f = 0.5 * (Gamma[C] + Gamma[F]) * A / d_mag

        if abs(m_f) < 1e-20:
            fP = 1.0
        elif abs(D_f) < 1e-12:
            fP = 0.0
        else:
            P = m_f / D_f
            fP = max(0.0, (1.0 - 0.1 * abs(P)) ** 5)

        D_fp = D_f * fP
        m_F = -m_f

        expected_diag_C = (m_f if m_f > 0.0 else 0.0) + D_fp
        expected_off_C = (m_f if m_f < 0.0 else 0.0) - D_fp
        expected_diag_F = (m_F if m_F > 0.0 else 0.0) + D_fp
        expected_off_F = (m_F if m_F < 0.0 else 0.0) - D_fp

        assert np.isclose(diag_C, expected_diag_C, rtol=1e-12), (
            f"Face {f}: diag_C mismatch"
        )
        assert np.isclose(off_C, expected_off_C, rtol=1e-12), (
            f"Face {f}: off_C mismatch"
        )
        assert np.isclose(diag_F, expected_diag_F, rtol=1e-12), (
            f"Face {f}: diag_F mismatch"
        )
        assert np.isclose(off_F, expected_off_F, rtol=1e-12), (
            f"Face {f}: off_F mismatch"
        )


def test_upwind_convection_positivity(mesh_instance):
    """
    Confirms that all matrix coefficients assembled from upwind convection are bounded:
    - Diagonal entries are positive
    - Off-diagonal entries are non-positive (no overshoots)
    """
    md = mesh_to_data(mesh_instance)
    n_cells = md.cell_centers.shape[0]
    Gamma = np.ones(n_cells) * 1e-6
    rho_f = 1.0

    n_faces = md.face_areas.shape[0]
    face_velocity = np.tile(np.array([1.0, 0.0]), (n_faces, 1))

    A = np.zeros((n_cells, n_cells))

    for f in range(n_faces):
        C = md.owner_cells[f]
        F = md.neighbor_cells[f]
        if F == -1:
            continue

        diag_C, off_C, diag_F, off_F = compute_powerlaw_convection_face_coeffs(
            f,
            md.cell_centers,
            md.face_centers,
            md.face_normals,
            md.face_areas,
            md.owner_cells,
            md.neighbor_cells,
            rho_f,
            face_velocity,
            Gamma,
        )

        A[C, C] += diag_C
        A[C, F] += off_C
        A[F, F] += diag_F
        A[F, C] += off_F

    for i in range(n_cells):
        assert A[i, i] > 0.0, f"Negative or zero diagonal at cell {i}"

    for i in range(n_cells):
        for j in range(n_cells):
            if i == j:
                continue
            assert A[i, j] <= 0.0 or np.isclose(A[i, j], 0.0), (
                f"Positive off-diagonal A[{i}, {j}] = {A[i, j]:.3e}, violates upwind boundedness"
            )
