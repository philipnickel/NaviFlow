"""
Face velocity interpolation using Rhie-Chow method for collocated grids.

This module implements the Rhie-Chow interpolation to prevent pressure checkerboarding
on collocated grids. The implementation follows Moukalled's approach as described in:
'The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction
with OpenFOAM and Matlab'.
"""

import numpy as np
from numba import njit, prange


from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients

BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 3
BC_NEUMANN = 4

@njit(parallel=False)
def mdot_correction(mesh, rho, Ap_u_c, Ap_v_c, grad_p_prime):
    n_faces = len(mesh.face_areas)
    mdot_prime = np.zeros(n_faces)

    for f in range(n_faces):
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        # Boundary face: skip correction
        if N < 0:
            mdot_prime[f] = 0.0
            continue

        d_CF_vec = mesh.vector_d_CE[f]
        mag_d_CF = np.linalg.norm(d_CF_vec) + 1e-14
        e_PN = d_CF_vec / mag_d_CF
        gf = mesh.face_interp_factors[f]
        S_f = mesh.vector_S_f[f]

        # Interpolated inverse momentum diagonals
        D_P_u = mesh.cell_volumes[P] / (Ap_u_c[P] + 1e-14)
        D_N_u = mesh.cell_volumes[N] / (Ap_u_c[N] + 1e-14)
        D_P_v = mesh.cell_volumes[P] / (Ap_v_c[P] + 1e-14)
        D_N_v = mesh.cell_volumes[N] / (Ap_v_c[N] + 1e-14)
        D_f = 0.5 * (gf * D_N_u + (1.0 - gf) * D_P_u + gf * D_N_v + (1.0 - gf) * D_P_v)

        # Interpolated gradient of pressure correction
        grad_p_prime_f = gf * grad_p_prime[N] + (1.0 - gf) * grad_p_prime[P]
        grad_dot_n = grad_p_prime_f[0] * e_PN[0] + grad_p_prime_f[1] * e_PN[1]

        # Correction flux
        mdot_prime[f] = rho * D_f * grad_dot_n * (S_f[0] * e_PN[0] + S_f[1] * e_PN[1])

    return mdot_prime


@njit(parallel=False)
def mdot_calculation(mesh, rho, u, v, p, grad_p, Ap_u_c, Ap_v_c, alpha_uv, u_old, v_old):
    """
    Compute face mass fluxes using Rhie–Chow interpolation in a compact, accurate form.
    """
    n_faces = len(mesh.face_areas)
    mdot = np.zeros(n_faces)

    for f in range(n_faces):
        P = mesh.owner_cells[f]
        F = mesh.neighbor_cells[f]

        # Boundary faces: set flux = 0
        if F < 0:
            mdot[f] = 0.0
            continue

        # Geometry
        d_CF = mesh.vector_d_CE[f]
        mag_d_CF = np.linalg.norm(d_CF) + 1e-14
        e_PF = d_CF / mag_d_CF  # face normal unit vector
        S_f = mesh.vector_S_f[f]
        g_f = mesh.face_interp_factors[f]

        # Diagonal-based diffusion coefficients
        D_P_u = mesh.cell_volumes[P] / (Ap_u_c[P] + 1e-14)
        D_F_u = mesh.cell_volumes[F] / (Ap_u_c[F] + 1e-14)
        D_P_v = mesh.cell_volumes[P] / (Ap_v_c[P] + 1e-14)
        D_F_v = mesh.cell_volumes[F] / (Ap_v_c[F] + 1e-14)
        D_f_u = g_f * D_F_u + (1.0 - g_f) * D_P_u
        D_f_v = g_f * D_F_v + (1.0 - g_f) * D_P_v
        D_f = 0.5 * (D_f_u + D_f_v)  # isotropic scalar for now

        # Face-normal velocity interpolation
        u_PF = g_f * u[F] + (1.0 - g_f) * u[P]
        v_PF = g_f * v[F] + (1.0 - g_f) * v[P]
        u_n_f = u_PF * e_PF[0] + v_PF * e_PF[1]

        # Pressure gradient discrepancy
        grad_p_f = g_f * grad_p[F] + (1.0 - g_f) * grad_p[P]
        dp_direct = (p[F] - p[P]) / mag_d_CF
        grad_dot_n = grad_p_f[0] * e_PF[0] + grad_p_f[1] * e_PF[1]
        phi = dp_direct - grad_dot_n  # Rhie–Chow correction

        # Apply correction along face normal
        u_n_corr = u_n_f - D_f * phi

        # Optional deferred temporal correction
        u_old_PF = g_f * u_old[F] + (1.0 - g_f) * u_old[P]
        v_old_PF = g_f * v_old[F] + (1.0 - g_f) * v_old[P]
        u_old_n = u_old_PF * e_PF[0] + v_old_PF * e_PF[1]
        u_bar_n = 0.5 * ((u_old[P] * e_PF[0] + v_old[P] * e_PF[1]) +
                         (u_old[F] * e_PF[0] + v_old[F] * e_PF[1]))
        time_corr = (1.0 - alpha_uv) * (u_bar_n - u_old_n)

        # Final corrected face-normal velocity
        u_n_final = u_n_corr + time_corr

        # Mass flux: rho * (u ⋅ S)
        mdot[f] = rho * u_n_final * (S_f[0] * e_PF[0] + S_f[1] * e_PF[1])

    return mdot


"""
@njit(parallel=False)
def mdot_calculation(mesh, rho, u, v, p, grad_p, Ap_u_c, Ap_v_c, alpha_uv, u_old, v_old):
    
    Compute face mass fluxes using Rhie–Chow interpolation to suppress pressure-velocity decoupling.

    Parameters
    ----------
    mesh : MeshData2D
        Structured or unstructured mesh object.
    rho : float
        Fluid density.
    u, v : ndarray
        Cell-centered velocity components.
    p : ndarray
        Cell-centered pressure field.
    grad_p : ndarray
        Cell-centered gradients of pressure.
    Ap_u_c, Ap_v_c : ndarray
        Physical (unrelaxed) diagonals from the momentum equations.
    alpha_uv : float
        Under-relaxation factor for velocity equations.
    u_old, v_old : ndarray
        Previous time step velocity fields.

    Returns
    -------
    mdot : ndarray
        Mass fluxes on all faces.
    
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    mdot = np.zeros(n_internal + n_boundary)

    for i in range(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        d_CF_vec = mesh.vector_d_CE[f]
        mag_d_CF = np.linalg.norm(d_CF_vec) + 1e-14
        e_PN = d_CF_vec / mag_d_CF
        S_f = mesh.vector_S_f[f]
        gf = mesh.face_interp_factors[f]

        # Diagonal inverse interpolation
        D_P_u = mesh.cell_volumes[P] / (Ap_u_c[P] + 1e-14)
        D_N_u = mesh.cell_volumes[N] / (Ap_u_c[N] + 1e-14)
        D_P_v = mesh.cell_volumes[P] / (Ap_v_c[P] + 1e-14)
        D_N_v = mesh.cell_volumes[N] / (Ap_v_c[N] + 1e-14)
        D_f_u = gf * D_N_u + (1.0 - gf) * D_P_u
        D_f_v = gf * D_N_v + (1.0 - gf) * D_P_v

        # Interpolated face velocities (from current solution)
        u_f = gf * u[N] + (1.0 - gf) * u[P]
        v_f = gf * v[N] + (1.0 - gf) * v[P]

        # Pressure gradient interpolation and directional discrepancy
        grad_p_f = gf * grad_p[N] + (1.0 - gf) * grad_p[P]
        dp_direct = (p[N] - p[P]) / mag_d_CF
        grad_dot_n = np.dot(grad_p_f, e_PN)
        phi = dp_direct - grad_dot_n

        # Directional velocity corrections
        u_corr = D_f_u * phi * e_PN[0]
        v_corr = D_f_v * phi * e_PN[1]

        # Optional temporal correction (deferred)
        u_old_f = gf * u_old[N] + (1.0 - gf) * u_old[P]
        v_old_f = gf * v_old[N] + (1.0 - gf) * v_old[P]
        u_old_bar = 0.5 * (u_old[N] + u_old[P])
        v_old_bar = 0.5 * (v_old[N] + v_old[P])
        u_time_corr = (1.0 - alpha_uv) * (u_old_bar - u_old_f)
        v_time_corr = (1.0 - alpha_uv) * (v_old_bar - v_old_f)

        # Final corrected interpolated face velocity
        u_final = u_f - u_corr  + u_time_corr 
        v_final = v_f - v_corr   + v_time_corr 

        # Mass flux
        mdot[f] = rho * (u_final * S_f[0] + v_final * S_f[1])

    # Zero out boundary face fluxes
    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        mdot[f] = 0.0

    return mdot
"""


"""
@njit(parallel=False)
def mdot_calculation(mesh, rho, u, v, p, grad_p, Ap_u_c, Ap_v_c, alpha_uv, u_old, v_old):
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    mdot = np.zeros(n_internal + n_boundary)

    for i in range(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        # Geometry
        d_CF_vec = mesh.vector_d_CE[f]
        mag_d_CF = np.linalg.norm(d_CF_vec) + 1e-14
        e_PN = d_CF_vec / mag_d_CF
        S_f = mesh.vector_S_f[f]
        gf = mesh.face_interp_factors[f]

        # Diagonal inverse
        D_P_u = mesh.cell_volumes[P] / (Ap_u_c[P] + 1e-14)
        D_N_u = mesh.cell_volumes[N] / (Ap_u_c[N] + 1e-14)
        D_P_v = mesh.cell_volumes[P] / (Ap_v_c[P] + 1e-14)
        D_N_v = mesh.cell_volumes[N] / (Ap_v_c[N] + 1e-14)
        D_f_u = gf * D_N_u + (1.0 - gf) * D_P_u
        D_f_v = gf * D_N_v + (1.0 - gf) * D_P_v

        # Face velocity (interpolated)
        u_f = gf * u[N] + (1.0 - gf) * u[P]
        v_f = gf * v[N] + (1.0 - gf) * v[P]
        uv_f = np.array([u_f, v_f])

        # Pressure gradient interpolation at face
        grad_p_f = gf * grad_p[N] + (1.0 - gf) * grad_p[P]
        dp = p[N] - p[P]
        dp_direct = (dp / mag_d_CF) * e_PN
        correction = dp_direct - grad_p_f
        u_corr = D_f_u * correction[0] #* S_f[0]
        v_corr = D_f_v * correction[1] #* S_f[1]

        uv_corr = np.array([u_corr, v_corr])

        # Temporal correction (Patankar-style implicitness)
        u_old_f = gf * u_old[N] + (1.0 - gf) * u_old[P]
        v_old_f = gf * v_old[N] + (1.0 - gf) * v_old[P]
        u_old_bar = 0.5 * (u_old[N] + u_old[P])
        v_old_bar = 0.5 * (v_old[N] + v_old[P])
        u_time_corr = (1.0 - alpha_uv) * (u_old_bar - u_old_f)
        v_time_corr = (1.0 - alpha_uv) * (v_old_bar - v_old_f)

        uv_time_corr = np.array([u_time_corr, v_time_corr])

        # Final Rhie–Chow interpolated velocity
        uv_final = uv_f - uv_corr# + uv_time_corr

        # Mass flux
        mdot[f] = rho * np.dot(uv_final, S_f)

    # Boundary faces: zero flux or handled elsewhere
    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        mdot[n_internal + f] = 0.0

    return mdot
"""

def compute_face_velocities(mesh, u, v):
    n_faces = len(mesh.face_areas)
    face_velocity = np.zeros((n_faces, 2))
    for f in range(n_faces):
        gf = mesh.face_interp_factors[f]
        face_velocity[f, 0] = gf * u[mesh.neighbor_cells[f]] + (1 - gf) * u[mesh.owner_cells[f]]
        face_velocity[f, 1] = gf * v[mesh.neighbor_cells[f]] + (1 - gf) * v[mesh.owner_cells[f]]
    return face_velocity
