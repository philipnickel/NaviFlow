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


# Small value to prevent division by zero
_SMALL = 1.0e-12


@njit(parallel=False)
def mdot_correction(mesh, rho, Ap_u_c, Ap_v_c, grad_p_prime):
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    mdot_prime = np.zeros(n_internal + n_boundary)

    for i in range(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        # Distance vector and unit direction
        d_CF_vec = mesh.vector_d_CE[f]
        mag_d_CF = np.linalg.norm(d_CF_vec) + 1e-14
        e_PN = d_CF_vec / mag_d_CF

        # Interpolated inverse diagonals for velocity corrections
        D_P_u = mesh.cell_volumes[P] / (Ap_u_c[P] + 1e-14)
        D_N_u = mesh.cell_volumes[N] / (Ap_u_c[N] + 1e-14)
        D_P_v = mesh.cell_volumes[P] / (Ap_v_c[P] + 1e-14)
        D_N_v = mesh.cell_volumes[N] / (Ap_v_c[N] + 1e-14)

        gf = mesh.face_interp_factors[f]
        D_f_u = gf * D_N_u + (1.0 - gf) * D_P_u
        D_f_v = gf * D_N_v + (1.0 - gf) * D_P_v

        # Pressure correction gradient approximation
        grad_p_prime_f = gf * grad_p_prime[N] + (1.0 - gf) * grad_p_prime[P]

        # Apply correction
        u_corr = D_f_u * grad_p_prime_f[0]
        v_corr = D_f_v * grad_p_prime_f[1]
        S_f = mesh.vector_S_f[f]
        mdot_prime[f] = rho * (u_corr * S_f[0] + v_corr * S_f[1])

    # Boundary faces: zero correction for now
    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        mdot_prime[n_internal + f] = 0.0

    return mdot_prime


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
        uv_final = uv_f - uv_corr + uv_time_corr

        # Mass flux
        mdot[f] = rho * np.dot(uv_final, S_f)

    # Boundary faces: zero flux or handled elsewhere
    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        mdot[n_internal + f] = 0.0

    return mdot




@njit(parallel=False)
def compute_rhie_chow_face_velocities(mesh, u, v, p, Ap_u_cell, Ap_v_cell, alpha_uv, v_old, u_old, u_prime=None, v_prime=None, p_prime=None
):

    n_faces = len(mesh.face_areas)
    n_cells = len(mesh.cell_volumes)

    grad_p = compute_cell_gradients(mesh, p)

    face_velocity = np.zeros((n_faces, 2))

    d_u_cell = np.zeros(n_cells)
    d_v_cell = np.zeros(n_cells)
    for c in range(n_cells):
        d_u_cell[c] = mesh.cell_volumes[c] / max(Ap_u_cell[c], _SMALL)  
        d_v_cell[c] = mesh.cell_volumes[c] / max(Ap_v_cell[c], _SMALL)  

    for f in prange(n_faces):
        C = mesh.owner_cells[f]
        F = mesh.neighbor_cells[f]

        if F < 0:
            # eq 15.110 Moukalled
            u_star_C = u[C]
            v_star_C = v[C]
            D_Cv = d_v_cell[C]
            D_Cu = d_u_cell[C]
            # Dv is now diagonal matrix
            Dv = np.diag(np.array([D_Cu, D_Cv], dtype=np.float64))
            # grad_p is gradient of pressure
            grad_p_C = grad_p[C]
            # vector from C to boundary face center
            n_vec = mesh.unit_vector_n[f]
            d_Cb = mesh.d_Cb[f]
            d_Cb_vec = d_Cb * n_vec
            # grad_p at boundary face center
            p_b = grad_p[F] + grad_p_C * d_Cb_vec
            if mesh.boundary_types[f, 0] == BC_OUTLET:
                p_b = p[C]
            else: 
                p_b = p[C] + np.dot(grad_p_C, d_Cb_vec)
            grad_p_b = (p_b - p[C]) / max(d_Cb, _SMALL)
            v_star_b = v_star_C - Dv @ (grad_p_b - grad_p_C)
            face_velocity[f, 1] = v_star_b[1]
            face_velocity[f, 0] = v_star_b[0]

        # For internal faces, use Rhie-Chow interpolation
        else:
            fx  = mesh.face_interp_factors[f]

            # Linear interpolation
            u_interp = fx*u[F] + (1.0 - fx)*u[C]
            v_interp = fx*v[F] + (1.0 - fx)*v[C]

            grad_px = fx*grad_p[F,0] + (1.0 - fx)*grad_p[C,0]
            grad_py = fx*grad_p[F,1] + (1.0 - fx)*grad_p[C,1]

            d_u = 1.0 / (fx/Ap_u_cell[F] + (1.0 - fx)/Ap_u_cell[C])   # harmonic
            d_v = 1.0 / (fx/Ap_v_cell[F] + (1.0 - fx)/Ap_v_cell[C])   # harmonic

            Sx, Sy   = mesh.vector_S_f[f]       # already area-weighted normal
            S_mag    = mesh.face_areas[f]
            nx, ny   = Sx/S_mag, Sy/S_mag

            d_CF_vec = mesh.vector_d_CE[f]      # centre-to-centre vector
            delta_CF = np.linalg.norm(d_CF_vec)

            dp_dn    = (p[F] - p[C]) / max(delta_CF, _SMALL)
            gradp_dot_n = grad_px*nx + grad_py*ny
            phi      = (dp_dn - gradp_dot_n)

            u_corr = d_u * phi * nx
            v_corr = d_v * phi * ny

            face_velocity[f,0] = u_interp - u_corr #+ (1.0 - alpha_uv)*(u_old[F] - u_old[C])
            face_velocity[f,1] = v_interp - v_corr #+ (1.0 - alpha_uv)*(v_old[F] - v_old[C]) 

    return face_velocity

def compute_face_velocities(mesh, u, v):
    n_faces = len(mesh.face_areas)
    face_velocity = np.zeros((n_faces, 2))
    for f in range(n_faces):
        gf = mesh.face_interp_factors[f]
        face_velocity[f, 0] = gf * u[mesh.neighbor_cells[f]] + (1 - gf) * u[mesh.owner_cells[f]]
        face_velocity[f, 1] = gf * v[mesh.neighbor_cells[f]] + (1 - gf) * v[mesh.owner_cells[f]]
    return face_velocity


@njit(parallel=False)
def compute_face_fluxes(mesh, face_velocity, rho):
    """
    Compute mass fluxes at faces from face velocities.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh data structure
    face_velocity : ndarray
        Face velocities [n_faces, 2]
    rho : float
        Density

    Returns
    ------
    face_mass_fluxes : ndarray
        Mass fluxes at faces
    """
    n_faces = len(mesh.face_areas)
    face_mass_fluxes = np.zeros(n_faces)

    for f in prange(n_faces):
        # Get face area and normal vector (Sf_x, Sf_y)
        S_f = np.ascontiguousarray(mesh.vector_S_f[f])  # This is already Area * unit_normal

        vol_flux = np.dot(face_velocity[f], S_f)

        # Calculate mass flux
        face_mass_fluxes[f] = rho * vol_flux

    return face_mass_fluxes
