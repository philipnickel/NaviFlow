"""
Face velocity interpolation using Rhie-Chow method for collocated grids.

This module implements the Rhie-Chow interpolation to prevent pressure checkerboarding
on collocated grids. The implementation follows Moukalled's approach as described in:
'The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction
with OpenFOAM and Matlab'.
"""

import numpy as np
from numba import njit, prange
from naviflow_collocated.mesh.mesh_loader import (
    BC_WALL, BC_DIRICHLET, BC_NEUMANN, BC_ZEROGRADIENT, BC_CONVECTIVE, BC_SYMMETRY
)

from naviflow_collocated.utils.gradient_utils import compute_all_cell_gradients

# Boundary condition type codes (must match momentum_eq_assembly.py)
BC_WALL_NO_SLIP = 1
BC_WALL_SLIP = 2
BC_INLET_VELOCITY = 3
BC_OUTLET_PRESSURE = 4
BC_SYMMETRY = 5

# Small value to prevent division by zero
_SMALL = 1.0e-12


@njit(parallel=True)
def compute_rhie_chow_face_velocities(
    mesh, u, v, p, Ap_u_cell, Ap_v_cell, u_prime=None, v_prime=None, p_prime=None
):
    """
    Compute face velocities using Rhie-Chow interpolation.

    This function calculates face-centered velocities using the Rhie-Chow interpolation
    method to prevent pressure checkerboarding.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh data structure
    u, v : ndarray
        Velocity fields at cell centers (u*, v*)
    p : ndarray
        Pressure field at cell centers (p*)
    Ap_u_cell : ndarray
        Diagonal coefficients from U-momentum equation at cell centers
    Ap_v_cell : ndarray
        Diagonal coefficients from V-momentum equation at cell centers
    u_prime, v_prime : ndarray, optional
        Velocity corrections for SIMPLE algorithm
    p_prime : ndarray, optional
        Pressure correction for SIMPLE algorithm

    Returns
    ------
    face_velocity : ndarray
        Face velocities [n_faces, 2]
    """
    n_faces = len(mesh.face_areas)
    n_cells = len(mesh.cell_volumes)

    # Calculate pressure gradients at cell centers using the utility function
    grad_p = compute_all_cell_gradients(p, mesh)

    # Calculate correction pressure gradients if provided
    grad_p_prime = None
    if p_prime is not None:
        grad_p_prime = compute_all_cell_gradients(p_prime, mesh)

    # Initialize face velocities
    face_velocity = np.zeros((n_faces, 2))

    # Precompute Vol/Ap for each cell and each component
    # These are the d_u, d_v terms from Moukalled (e.g., d_u = Vol/a_P_u)
    d_u_cell = np.zeros(n_cells)
    d_v_cell = np.zeros(n_cells)
    for c in range(n_cells):
        d_u_cell[c] = mesh.cell_volumes[c] / max(Ap_u_cell[c], _SMALL)
        d_v_cell[c] = mesh.cell_volumes[c] / max(Ap_v_cell[c], _SMALL)

    # Compute face velocities for each face
    for f in prange(n_faces):
        C = mesh.owner_cells[f]
        F = mesh.neighbor_cells[f]

        # For boundary faces
        if F < 0:
            # Default boundary values (zero gradient)
            face_velocity[f, 0] = u[C]
            face_velocity[f, 1] = v[C]

            # Find the correct boundary value index
            boundary_idx = -1
            for i in range(len(mesh.boundary_faces)):
                if mesh.boundary_faces[i] == f:
                    boundary_idx = i
                    break

            if boundary_idx >= 0:  # Found the boundary index
                bc_type = mesh.boundary_types[f, 0]  # Use velocity BC type (index 0)

                if bc_type == BC_WALL:  # Wall
                    # No-slip wall: zero velocity
                    face_velocity[f, 0] = 0.0
                    face_velocity[f, 1] = 0.0
                elif bc_type == BC_DIRICHLET:  # Dirichlet
                    # Inlet: use boundary values
                    face_velocity[f, 0] = mesh.boundary_values[f, 0]  # Use face index directly
                    face_velocity[f, 1] = mesh.boundary_values[f, 1]
                elif bc_type == BC_NEUMANN:  # Neumann
                    # Outlet: zero gradient
                    face_velocity[f, 0] = u[C]
                    face_velocity[f, 1] = v[C]
                elif bc_type == BC_ZEROGRADIENT:  # ZeroGradient
                    # Zero gradient
                    face_velocity[f, 0] = u[C]
                    face_velocity[f, 1] = v[C]
                elif bc_type == BC_CONVECTIVE:  # Convective
                    # For now, just use zero gradient for convective boundaries
                    face_velocity[f, 0] = u[C]
                    face_velocity[f, 1] = v[C]
                elif bc_type == BC_SYMMETRY:  # Symmetry
                    # Symmetry: zero normal component, tangential copy
                    normal = mesh.face_normals[f]
                    nmag = np.sqrt(normal[0] ** 2 + normal[1] ** 2)
                    if nmag > _SMALL:
                        nx = normal[0] / nmag
                        ny = normal[1] / nmag

                        # Tangential component
                        tx = -ny
                        ty = nx

                        # Dot product to get tangential velocity
                        u_t = u[C] * tx + v[C] * ty

                        # Set face velocity
                        face_velocity[f, 0] = u_t * tx
                        face_velocity[f, 1] = u_t * ty
                else:  # Default to zero gradient for any other type
                    face_velocity[f, 0] = u[C]
                    face_velocity[f, 1] = v[C]

        # For internal faces, use Rhie-Chow interpolation
        else:
            # Linear interpolation factors
            fx = mesh.face_interp_factors[f]

            # Linear interpolation for velocity and pressure gradients
            u_interp = fx * u[F] + (1.0 - fx) * u[C]
            v_interp = fx * v[F] + (1.0 - fx) * v[C]

            grad_p_x_interp = fx * grad_p[F, 0] + (1.0 - fx) * grad_p[C, 0]
            grad_p_y_interp = fx * grad_p[F, 1] + (1.0 - fx) * grad_p[C, 1]

            # Interpolate d_u = Vol/Ap_u and d_v = Vol/Ap_v to the face
            d_u_face = fx * d_u_cell[F] + (1.0 - fx) * d_u_cell[C]
            d_v_face = fx * d_v_cell[F] + (1.0 - fx) * d_v_cell[C]

            # Face normal and area (normal vector components nx, ny)
            normal = mesh.face_normals[
                f
            ]  # This is S_fx, S_fy (Area * normal_unit_vector)
            face_area_mag = mesh.face_areas[f]  # Magnitude of face area vector

            nx_unit, ny_unit = 0.0, 0.0
            if face_area_mag > _SMALL:
                nx_unit = normal[0] / face_area_mag
                ny_unit = normal[1] / face_area_mag

                # Calculate pressure difference term normal to the face (P_F - P_C)/d_CF
                dp_dCF = (p[F] - p[C]) / max(mesh.delta_CF[f], _SMALL)

                # Rhie-Chow correction for u-component of face velocity
                # (Vol/Ap_u)_f * [ ( (P_F-P_C)/d_CF * nx_unit ) - (gradP_x)_f_bar ]
                rhie_chow_corr_x = d_u_face * (dp_dCF * nx_unit - grad_p_x_interp)

                # Rhie-Chow correction for v-component of face velocity
                # (Vol/Ap_v)_f * [ ( (P_F-P_C)/d_CF * ny_unit ) - (gradP_y)_f_bar ]
                rhie_chow_corr_y = d_v_face * (dp_dCF * ny_unit - grad_p_y_interp)

                # Apply Rhie-Chow correction
                face_velocity[f, 0] = u_interp + rhie_chow_corr_x
                face_velocity[f, 1] = v_interp + rhie_chow_corr_y

                # Add correction velocities if provided (for SIMPLE algorithm velocity update)
                # This part is for U = U* + U' where U' depends on p'
                # The Rhie-Chow form for U' is similar to the one for U*
                if (
                    u_prime is not None  # cell centered u'
                    and v_prime is not None  # cell centered v'
                    and grad_p_prime is not None  # cell centered grad_p'
                ):
                    # Linear interpolation for correction velocities u', v'
                    u_prime_interp = fx * u_prime[F] + (1.0 - fx) * u_prime[C]
                    v_prime_interp = fx * v_prime[F] + (1.0 - fx) * v_prime[C]

                    # Interpolated grad_p_prime
                    grad_p_prime_x_interp = (
                        fx * grad_p_prime[F, 0] + (1.0 - fx) * grad_p_prime[C, 0]
                    )
                    grad_p_prime_y_interp = (
                        fx * grad_p_prime[F, 1] + (1.0 - fx) * grad_p_prime[C, 1]
                    )

                    # Pressure correction difference term normal to the face (P'_F - P'_C)/d_CF
                    # p_prime is cell-centered p'
                    dp_prime_dCF = (p_prime[F] - p_prime[C]) / max(
                        mesh.delta_CF[f], _SMALL
                    )

                    # Velocity correction due to p' (Rhie-Chow form for the U' part)
                    # d_u_face is (Vol/Ap_u)_f from the original momentum equation a_P
                    vel_corr_prime_x = d_u_face * (
                        dp_prime_dCF * nx_unit - grad_p_prime_x_interp
                    )
                    vel_corr_prime_y = d_v_face * (
                        dp_prime_dCF * ny_unit - grad_p_prime_y_interp
                    )

                    # Add to the already corrected U* face velocity
                    # U_f_final = (U_f_interp_star + RC_corr_star) + (U_f_interp_prime + RC_corr_prime)
                    # Or, if u_prime, v_prime are d_u * grad_p', then it's simpler.
                    # Moukalled 7.108: U_f = U_f_rc_star + U_f_rc_prime
                    # U_f_rc_prime = U_f_prime_bar - (Vol/Ap)_f_bar * ( (grad p')_f - (grad p')_f_bar )
                    # where U_f_prime_bar is interpolated from cell-centered u_c' = -(Vol/Ap_c) grad p'_c
                    # The current u_prime, v_prime are just cell-centered velocity corrections, not yet the full u_c'
                    # This section is for the velocity UPDATE step using p', so u_prime, v_prime here are the
                    # dAp/Ap_P * (p'_face_east - p'_face_west) type terms.
                    # For now, let's assume u_prime and v_prime are the FINAL cell-centered corrections.
                    # The velocity update step in SIMPLE is:
                    # u_c_new = u_c_star + u_c_correction_term
                    # U_f_new = U_f_star_rc + correction_term_for_Uf_rc
                    # The correction_term_for_Uf_rc is d_f * (p'_N - p'_S) for a u-velocity face.
                    # This part of the code with u_prime, v_prime, p_prime seems more related to
                    # how velocities are *updated* after p' is found, rather than calculating U_f_star.
                    # Let's simplify for now and assume this function is primarily for U*_rc_f.
                    # The velocity update step will be separate.
                    # So, removing the u_prime, v_prime, p_prime handling from *this* function
                    # makes it purely for calculating U*_rc_f.

                    # REVISITING: The parameters u_prime, v_prime, p_prime are optional.
                    # If they are provided, it means we are calculating the FULLY corrected face velocity.
                    # U_total_f = U*_rc_f + U'_rc_f
                    # U*_rc_f part is already calculated above.
                    # U'_rc_f = U'_f_bar - (Vol/Ap)_f_bar * ( (grad p')_f - (grad p')_f_bar )
                    # where U'_f_bar is interpolated from cell-centered U'_c.
                    # And U'_c = - (Vol/Ap)_c * (grad p')_c.
                    # So, if u_prime, v_prime are cell-centered U'_c, then we interpolate them to get U'_f_bar.
                    # And grad_p_prime is cell-centered grad_p'.

                    # Add interpolated cell-centered velocity correction
                    face_velocity[f, 0] += u_prime_interp  # U_f_bar_prime
                    face_velocity[f, 1] += v_prime_interp  # V_f_bar_prime

                    # Add RC correction for the prime field
                    face_velocity[f, 0] += vel_corr_prime_x
                    face_velocity[f, 1] += vel_corr_prime_y

            else:
                # Fallback to simple interpolation if normal is degenerate
                face_velocity[f, 0] = u_interp
                face_velocity[f, 1] = v_interp
    return face_velocity


@njit(parallel=True)
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
        normal_vector = mesh.face_normals[f]  # This is already Area * unit_normal

        # Dot product of velocity and normal vector (U_f . S_f)
        # S_f = [S_fx, S_fy]
        # U_f = [u_f, v_f]
        # U_f . S_f = u_f * S_fx + v_f * S_fy
        vol_flux = (
            face_velocity[f, 0] * normal_vector[0]
            + face_velocity[f, 1] * normal_vector[1]
        )

        # Calculate mass flux
        face_mass_fluxes[f] = rho * vol_flux

    return face_mass_fluxes
