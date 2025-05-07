import numpy as np
from naviflow_collocated.discretization.convection.upwind import (
    compute_convective_flux_upwind,
)


def test_compute_convective_flux_upwind_mms(mesh_instance):
    rho = 1.0

    # Manufactured solution: u(x,y) = sin(pi x) * sin(pi y)
    # We will use these cell-centered exact values directly in the test comparison.
    u_exact_field = np.zeros(len(mesh_instance.cell_volumes))
    for i, x_cell in enumerate(mesh_instance.cell_centers):
        u_exact_field[i] = np.sin(np.pi * x_cell[0]) * np.sin(np.pi * x_cell[1])

    for f in mesh_instance.internal_faces:
        uf = np.array(
            [1.0, 0.0], dtype=np.float64
        )  # face velocity, e.g., flow in positive x

        # Call the upwind flux computation
        # P, N, a_PP, a_PN, b_corr are coefficients such that flux = a_PP*u_P + a_PN*u_N + b_corr
        P_idx, N_idx, a_PP_num, a_PN_num, b_corr_num = compute_convective_flux_upwind(
            f, u_exact_field, mesh_instance, uf, rho
        )

        u_P_exact = u_exact_field[P_idx]
        u_N_exact = u_exact_field[N_idx]  # N_idx should be valid for internal faces

        # Numerical flux computed using the scheme's coefficients and exact cell values
        flux_numerical = a_PP_num * u_P_exact + a_PN_num * u_N_exact + b_corr_num

        # Expected flux based on pure upwind principle using exact cell values
        # m_dot_f (F) is rho * dot(uf, Sf)
        Sf_x = mesh_instance.face_normals[f, 0]
        Sf_y = mesh_instance.face_normals[f, 1]
        m_dot_f = rho * (uf[0] * Sf_x + uf[1] * Sf_y)

        flux_expected = 0.0
        if m_dot_f >= 0:  # Flow from P to N (or zero flux)
            flux_expected = m_dot_f * u_P_exact
        else:  # Flow from N to P
            flux_expected = m_dot_f * u_N_exact

        assert np.isclose(flux_numerical, flux_expected, rtol=1e-9, atol=1e-12), (
            f"Mismatch at face {f} (P:{P_idx}, N:{N_idx}) with F={m_dot_f:.4g}:\n"
            f"  Numerical flux = {flux_numerical:.6e}\n"
            f"  Expected flux  = {flux_expected:.6e}\n"
            f"  a_PP_num={a_PP_num:.4g}, a_PN_num={a_PN_num:.4g}, b_corr_num={b_corr_num:.4g}\n"
            f"  u_P_exact={u_P_exact:.4g}, u_N_exact={u_N_exact:.4g}"
        )
