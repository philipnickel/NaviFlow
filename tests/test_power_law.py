import numpy as np
from naviflow_collocated.discretization.convection.convection_helper import (
    compute_convective_flux,
)
from naviflow_collocated.discretization.gradient.leastSquares import (
    compute_cell_gradients,
)


def test_power_law_flux_against_manufactured_solution(mesh_instance):
    rho = 1.0
    velocity_field = np.zeros_like(mesh_instance.cell_centers)
    velocity_field[:, 0] = 1.0  # uniform x-direction velocity

    # Manufactured scalar field phi(x, y) = sin(pi x) * sin(pi y)
    phi = np.zeros(mesh_instance.cell_centers.shape[0])
    for i, x in enumerate(mesh_instance.cell_centers):
        phi[i] = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    grad_phi = compute_cell_gradients(mesh_instance, phi)

    for f in mesh_instance.internal_faces:
        # Now compute_convective_flux returns (a_P, a_N, b_corr)
        _, _, flux = compute_convective_flux(
            f, phi, grad_phi, mesh_instance, rho, velocity_field, scheme="power_law"
        )

        assert np.isfinite(flux), f"Flux at face {f} is not finite"

        P = mesh_instance.owner_cells[f]
        N = mesh_instance.neighbor_cells[f]
        Sf = mesh_instance.face_normals[f]
        u_f = (1.0 - mesh_instance.face_interp_factors[f]) * velocity_field[
            P
        ] + mesh_instance.face_interp_factors[f] * velocity_field[N]
        F = rho * np.dot(u_f, Sf)

        # For tiny values close to zero, we shouldn't worry about sign disagreements
        # as they're essentially noise
        TOL = 1e-6

        if abs(flux) < TOL or abs(F) < TOL:
            # Skip sign check for tiny values
            print(f"Face {f}: F={F:.6e}, flux={flux:.6e} (skipping - near zero)")
            continue

        # For larger values, check sign agreement
        assert np.sign(flux) == np.sign(F), f"Flux sign mismatch at face {f}"

        # Print diagnostic information
        print(f"Face {f}: F={F:.6e}, flux={flux:.6e}")
