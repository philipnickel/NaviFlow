import numpy as np

from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_central_diffusion_face_coeffs,
)
from naviflow_collocated.mesh.mesh_data import mesh_to_data
from naviflow_collocated.discretization.gradient.leastSquares import (
    compute_least_squares_gradients,
)


def estimate_mesh_size(md):
    return np.min(np.sqrt(md.cell_volumes))


def test_gradient_method_accuracy(mesh_instance):
    md = mesh_to_data(mesh_instance)
    phi = md.cell_centers[:, 0] + md.cell_centers[:, 1]
    exact_grad = np.array([1.0, 1.0])
    gradients = compute_least_squares_gradients(
        md.cell_centers, md.owner_cells, md.neighbor_cells, phi
    )
    error_vectors = gradients - exact_grad
    l2_error = np.sqrt(np.sum(error_vectors**2) / gradients.shape[0])
    h = estimate_mesh_size(md)
    tol = 5 * h
    assert l2_error < tol, f"L2 gradient error too high: {l2_error:.3e} (tol={tol:.3e})"


def test_flux_with_computed_gradient(mesh_instance):
    md = mesh_to_data(mesh_instance)
    phi = md.cell_centers[:, 0] + md.cell_centers[:, 1]
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
            continue

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
            phi,  # ✅ Added the missing argument
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
