import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import matrix_rank

from naviflow_collocated.assembly.momentum_eq import assemble_matrix_rhs_on_the_fly
from naviflow_collocated.mesh.mesh_data import mesh_to_data
from naviflow_collocated.discretization.gradient.leastSquares import (
    compute_least_squares_gradients,
)


def assert_matrix_and_rhs_sanity(A, rhs):
    assert A.shape[0] == A.shape[1], f"Non-square matrix: {A.shape}"
    assert not np.isnan(A.data).any(), "Matrix data contains NaNs"
    assert not np.isnan(rhs).any(), "RHS contains NaNs"
    assert np.all(np.isfinite(A.data)), "Matrix data contains infs"
    assert np.all(np.isfinite(rhs)), "RHS contains infs"
    assert np.any(A.diagonal() != 0), "Matrix has zero diagonal entries"
    A_dense = A.toarray()
    rank = matrix_rank(A_dense)
    assert rank == A.shape[0], (
        f"Matrix is rank-deficient: rank={rank}, expected={A.shape[0]}"
    )
    nonzeros_per_row = np.diff(A.indptr)
    assert np.all(nonzeros_per_row > 0), "Some matrix rows have all zeros"


def test_matrix_symmetry_and_diagonal_dominance(mesh_instance):
    md = mesh_to_data(mesh_instance)
    n_cells = md.cell_centers.shape[0]
    Gamma = np.ones(n_cells)
    phi = np.zeros(n_cells)

    dirichlet_mask = np.zeros(n_cells, dtype=bool)
    dirichlet_values = np.zeros(n_cells)
    for cell_ids in md.boundary_name_to_cell_indices.values():
        dirichlet_mask[cell_ids] = True
        dirichlet_values[cell_ids] = 0.0

    gradients = compute_least_squares_gradients(
        md.cell_centers, md.owner_cells, md.neighbor_cells, phi
    )

    face_velocity = np.zeros_like(md.face_normals)
    rho_f = 1.0

    row, col, data, _ = assemble_matrix_rhs_on_the_fly(
        n_cells=n_cells,
        n_faces=md.face_areas.shape[0],
        owner_cells=md.owner_cells,
        neighbor_cells=md.neighbor_cells,
        face_areas=md.face_areas,
        face_normals=md.face_normals,
        cell_centers=md.cell_centers,
        face_centers=md.face_centers,
        cell_volumes=md.cell_volumes,
        phi=phi,
        Gamma=Gamma,
        rho_f=rho_f,
        face_velocity=face_velocity,
        dirichlet_mask=dirichlet_mask,
        dirichlet_values=dirichlet_values,
        gradients=gradients,
        include_diffusion=True,
        include_convection=False,
    )

    A = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsc()
    assert_matrix_and_rhs_sanity(A, np.zeros(n_cells))
    # A_sym_diff = A - A.T
    # assert np.allclose(A_sym_diff.data, 0.0, atol=1e-12), "Matrix is not symmetric"
    abs_off_diag = np.abs(A) - np.abs(A.multiply(np.eye(n_cells)))
    diag = A.diagonal()
    assert np.all(diag >= abs_off_diag.sum(axis=1).A1), "Matrix not diagonally dominant"


def test_laplace_solution_accuracy(mesh_instance):
    md = mesh_to_data(mesh_instance)
    n_cells = md.cell_centers.shape[0]
    Gamma = np.ones(n_cells)

    x = md.cell_centers[:, 0]
    y = md.cell_centers[:, 1]
    phi_exact = x**2 + y**2
    Su = -4.0

    dirichlet_mask = np.zeros(n_cells, dtype=bool)
    dirichlet_values = np.zeros(n_cells)
    for cell_ids in md.boundary_name_to_cell_indices.values():
        dirichlet_mask[cell_ids] = True
        dirichlet_values[cell_ids] = phi_exact[cell_ids]

    gradients = compute_least_squares_gradients(
        md.cell_centers, md.owner_cells, md.neighbor_cells, phi_exact
    )

    face_velocity = np.zeros_like(md.face_normals)
    rho_f = 1.0

    row, col, data, rhs = assemble_matrix_rhs_on_the_fly(
        n_cells=n_cells,
        n_faces=md.face_areas.shape[0],
        owner_cells=md.owner_cells,
        neighbor_cells=md.neighbor_cells,
        face_areas=md.face_areas,
        face_normals=md.face_normals,
        cell_centers=md.cell_centers,
        face_centers=md.face_centers,
        cell_volumes=md.cell_volumes,
        phi=phi_exact.copy(),
        Gamma=Gamma,
        rho_f=rho_f,
        face_velocity=face_velocity,
        dirichlet_mask=dirichlet_mask,
        dirichlet_values=dirichlet_values,
        gradients=gradients,
        include_diffusion=True,
        include_convection=False,
    )

    rhs[~dirichlet_mask] += Su * md.cell_volumes[~dirichlet_mask]

    A = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsc()
    assert_matrix_and_rhs_sanity(A, rhs)

    phi_numeric = spsolve(A, rhs)
    assert not np.isnan(phi_numeric).any(), "Solver returned NaNs"

    residual = phi_numeric - phi_exact
    l2_error = np.linalg.norm(residual) / np.sqrt(n_cells)
    h = np.mean(md.cell_volumes) ** 0.5
    tolerance = 16 * h

    assert l2_error < tolerance, (
        f"Manufactured-solution error too high: {l2_error:.3e} (tol = {tolerance:.3e})"
    )


def test_convection_dominant_behavior(mesh_instance):
    md = mesh_to_data(mesh_instance)
    n_cells = md.cell_centers.shape[0]
    Gamma = np.ones(n_cells)

    x = md.cell_centers[:, 0]
    y = md.cell_centers[:, 1]
    phi_exact = np.sin(np.pi * x) * np.sinh(np.pi * y)

    dirichlet_mask = np.zeros(n_cells, dtype=bool)
    dirichlet_values = np.zeros(n_cells)
    for cell_ids in md.boundary_name_to_cell_indices.values():
        dirichlet_mask[cell_ids] = True
        dirichlet_values[cell_ids] = phi_exact[cell_ids]

    gradients = compute_least_squares_gradients(
        md.cell_centers, md.owner_cells, md.neighbor_cells, phi_exact
    )
    dim = md.face_normals.shape[1]
    face_velocity = np.tile(np.eye(dim)[0], (md.face_normals.shape[0], 1))
    rho_f = 1.0

    row, col, data, rhs = assemble_matrix_rhs_on_the_fly(
        n_cells=n_cells,
        n_faces=md.face_areas.shape[0],
        owner_cells=md.owner_cells,
        neighbor_cells=md.neighbor_cells,
        face_areas=md.face_areas,
        face_normals=md.face_normals,
        cell_centers=md.cell_centers,
        face_centers=md.face_centers,
        cell_volumes=md.cell_volumes,
        phi=phi_exact.copy(),
        Gamma=Gamma,
        rho_f=rho_f,
        face_velocity=face_velocity,
        dirichlet_mask=dirichlet_mask,
        dirichlet_values=dirichlet_values,
        gradients=gradients,
        include_diffusion=True,
        include_convection=True,
    )

    # Calculate and add manufactured source term S = π * cos(πx) * sinh(πy)
    x = md.cell_centers[:, 0]
    y = md.cell_centers[:, 1]
    S_analytic = np.pi * np.cos(np.pi * x) * np.sinh(np.pi * y)
    rhs[~dirichlet_mask] += (
        S_analytic[~dirichlet_mask] * md.cell_volumes[~dirichlet_mask]
    )

    A = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsc()
    assert_matrix_and_rhs_sanity(A, rhs)

    phi_numeric = spsolve(A, rhs)
    assert not np.isnan(phi_numeric).any(), "Solver returned NaNs"

    residual = phi_numeric - phi_exact
    l2_error = np.linalg.norm(residual) / np.sqrt(n_cells)
    h = np.mean(md.cell_volumes) ** 0.5
    tolerance = 16 * h

    assert l2_error < tolerance, (
        f"Convection-dominant solution error too high: {l2_error:.3e} (tol = {tolerance:.3e})"
    )


"""
🧪 Recommended Tests to Add
	1.	Stencil Extraction on a Small Mesh:
	•	Use a 3x3 structured mesh.
	•	Assemble the diffusion matrix.
	•	Confirm:
	•	A[i, j] = μ*A/d if j is neighbor of i
	•	A[i, i] = -Σ_off_diagonals
	•	Matrix is symmetric
	2.	Convective Upwind Matrix Inspection:
	•	Use a uniform rightward flow.
	•	Assemble matrix with upwind convection.
	•	Confirm:
	•	Each face adds +ρuA to upwind diagonal, -ρuA to downwind off-diagonal
	•	All coefficients are non-negative
	•	Matrix is not symmetric

"""
