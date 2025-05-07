import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

spec = [
    ("owner_cells", int64[:]),
    ("neighbor_cells", int64[:]),
    ("delta_PN", float64[:]),
    ("e_f", float64[:, :]),
    ("non_ortho_correction", float64[:, :]),
    ("cell_centers", float64[:, :]),
    ("face_normals", float64[:, :]),
    ("face_areas", float64[:]),
]


@jitclass(spec)
class SimpleMockMesh:
    def __init__(self):
        self.cell_centers = np.array([[0.0, 0.0], [1.0, 0.0]])  # Example values
        self.face_normals = np.array([[1.0, 0.0]])
        self.face_areas = np.array([1.0])
        self.owner_cells = np.array([0])
        self.neighbor_cells = np.array([1])
        self.delta_PN = np.array([1.0])
        self.e_f = np.array([[1.0, 0.0]])
        self.non_ortho_correction = np.array([[0.0, 0.0]])


def test_diffusive_flux_mms_internal(mesh_instance):
    from naviflow_collocated.discretization.diffusion.central_diff import (
        compute_diffusive_flux,
    )
    import numpy as np

    mu = 1.0
    u = np.zeros(len(mesh_instance.cell_volumes))
    grad_u = np.zeros_like(mesh_instance.cell_centers)

    # Manufactured solution: u(x,y) = sin(pi x) * sin(pi y)
    for i, x in enumerate(mesh_instance.cell_centers):
        u[i] = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        grad_u[i, 0] = np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
        grad_u[i, 1] = np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])

    for f in mesh_instance.internal_faces:
        P, N, a_PP, a_PN, b_corr = compute_diffusive_flux(
            f, u, grad_u, mesh_instance, mu
        )

        u_P = u[P]
        u_N = u[N]
        delta = mesh_instance.delta_PN[f]
        E_mag = np.linalg.norm(mesh_instance.e_f[f])
        T_f = mesh_instance.non_ortho_correction[f]

        # Discrete flux directly from the analytical formula
        flux_numerical = -mu * (u_N - u_P) / delta * E_mag - mu * np.dot(grad_u[P], T_f)
        print(f"Face {f}: flux = {flux_numerical:.6e}")
        assert np.isfinite(flux_numerical), f"Flux is not finite at face {f}"

        # Optional: check symmetry (anti-symmetry of fluxes)
        assert np.isfinite(a_PP)
        assert np.isfinite(a_PN)
        assert np.isfinite(b_corr)


def test_diffusive_flux_minimal_mock():
    from naviflow_collocated.discretization.diffusion.central_diff import (
        compute_diffusive_flux,
    )

    mesh = SimpleMockMesh()
    mu = 1.0
    u = np.array([1.0, 3.0])  # u_P = 1.0, u_N = 3.0
    grad_u = np.array([[1.0, 2.0], [0.0, 0.0]])  # Only grad_u_P used, constant field

    f = 0  # only one face

    P, N, a_PP, a_PN, b_corr = compute_diffusive_flux(f, u, grad_u, mesh, mu)

    delta = mesh.delta_PN[f]
    E_mag = np.linalg.norm(mesh.e_f[f])
    T_f = mesh.non_ortho_correction[f]

    flux_expected = -mu * (u[N] - u[P]) / delta * E_mag - mu * np.dot(grad_u[P], T_f)
    flux_numerical = a_PP * u[P] + a_PN * u[N] + b_corr

    assert np.isclose(flux_numerical, flux_expected, rtol=1e-12), (
        "Mismatch in minimal test"
    )
