import numpy as np

from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_central_diffusion_face,
)
from naviflow_collocated.mesh.mesh_data import mesh_to_data
from naviflow_collocated.discretization.gradient.leastSquares import (
    compute_least_squares_gradients,
)


def test_flux_with_analytic_gradient(mesh_instance):
    md = mesh_to_data(mesh_instance)
    Gamma = 1.0
    phi = md.cell_centers[:, 0] + md.cell_centers[:, 1]
    diffusion_coeffs = Gamma * np.ones_like(phi)
    gradients = np.tile(np.array([1.0, 1.0]), (phi.size, 1))

    numeric_fluxes = []
    exact_fluxes = []

    for f in range(len(md.face_areas)):
        if md.neighbor_cells[f] == -1:
            continue

        numeric_flux = compute_central_diffusion_face(
            f,
            md.cell_centers,
            md.face_centers,
            md.face_normals,
            md.face_areas,
            md.owner_cells,
            md.neighbor_cells,
            diffusion_coeffs,
            phi,
            gradients,
        )
        exact_flux = -Gamma * np.dot([1.0, 1.0], md.face_normals[f]) * md.face_areas[f]
        numeric_fluxes.append(numeric_flux)
        exact_fluxes.append(exact_flux)

    error = np.linalg.norm(np.array(numeric_fluxes) - np.array(exact_fluxes)) / np.sqrt(
        len(numeric_fluxes)
    )
    assert error < 1e-2, f"L2 flux error too high with analytic gradients: {error:.3e}"


def test_gradient_method_accuracy(mesh_instance):
    md = mesh_to_data(mesh_instance)
    phi = md.cell_centers[:, 0] + md.cell_centers[:, 1]
    exact_grad = np.array([1.0, 1.0])
    gradients = compute_least_squares_gradients(
        md.cell_centers,
        md.owner_cells,
        md.neighbor_cells,
        phi,
    )
    error_vectors = gradients - exact_grad
    l2_error = np.sqrt(np.sum(error_vectors**2) / gradients.shape[0])
    assert l2_error < 0.1, f"L2 gradient error too high: {l2_error:.3e}"


def test_flux_with_computed_gradient(mesh_instance):
    md = mesh_to_data(mesh_instance)
    phi = md.cell_centers[:, 0] + md.cell_centers[:, 1]
    Gamma = 1.0
    diffusion_coeffs = Gamma * np.ones_like(phi)

    gradients = compute_least_squares_gradients(
        md.cell_centers,
        md.owner_cells,
        md.neighbor_cells,
        phi,
    )

    numeric_fluxes = []
    exact_fluxes = []
    for f in range(len(md.face_areas)):
        if md.neighbor_cells[f] == -1:
            continue

        numeric_flux = compute_central_diffusion_face(
            f,
            md.cell_centers,
            md.face_centers,
            md.face_normals,
            md.face_areas,
            md.owner_cells,
            md.neighbor_cells,
            diffusion_coeffs,
            phi,
            gradients,
        )

        exact_flux = -Gamma * np.dot([1.0, 1.0], md.face_normals[f]) * md.face_areas[f]
        numeric_fluxes.append(numeric_flux)
        exact_fluxes.append(exact_flux)

    error = np.linalg.norm(np.array(numeric_fluxes) - np.array(exact_fluxes)) / np.sqrt(
        len(numeric_fluxes)
    )
    assert error < 0.05, f"L2 flux error too high with computed gradients: {error:.3e}"


def nonlinear_phi(cell_centers):
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    return x**2 + y**2


def exact_grad_nonlinear(cell_centers):
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    return np.stack([2 * x, 2 * y], axis=1)


def test_nonlinear_gradient_method_accuracy(mesh_instance):
    md = mesh_to_data(mesh_instance)
    phi = nonlinear_phi(md.cell_centers)
    gradients_exact = exact_grad_nonlinear(md.cell_centers)

    gradients_num = compute_least_squares_gradients(
        md.cell_centers,
        md.owner_cells,
        md.neighbor_cells,
        phi,
    )

    error_vectors = gradients_num - gradients_exact
    l2_error = np.sqrt(np.sum(error_vectors**2) / gradients_num.shape[0])
    assert l2_error < 0.2, f"Nonlinear gradient L2 error too high: {l2_error:.3e}"


def test_skewness_magnitude(mesh_instance):
    md = mesh_to_data(mesh_instance)
    skewness = []
    for f in range(len(md.face_centers)):
        if md.neighbor_cells[f] == -1:
            continue
        C = md.owner_cells[f]
        F = md.neighbor_cells[f]
        d = md.cell_centers[F] - md.cell_centers[C]
        d_norm = d / np.linalg.norm(d)
        ip = (
            md.cell_centers[C]
            + np.dot(md.face_centers[f] - md.cell_centers[C], d_norm) * d_norm
        )
        s = md.face_centers[f] - ip
        skewness.append(np.linalg.norm(s))
    skewness = np.array(skewness)
    max_skew = np.max(skewness)
    mean_skew = np.mean(skewness)
    assert max_skew < 0.5, f"Max skewness too large: {max_skew:.3e}"
    assert mean_skew < 0.2, f"Mean skewness too large: {mean_skew:.3e}"
