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

from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients

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

    n_faces = len(mesh.face_areas)
    n_cells = len(mesh.cell_volumes)

    grad_p = compute_cell_gradients(mesh, p)

    grad_p_prime = None
    if p_prime is not None:
        grad_p_prime = compute_cell_gradients(mesh, p_prime)

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
            face_velocity[f, 0] = u[C]
            face_velocity[f, 1] = v[C]

            boundary_idx = -1
            for i in range(len(mesh.boundary_faces)):
                if mesh.boundary_faces[i] == f:
                    boundary_idx = i
                    break

            if boundary_idx >= 0:
                bc_type = mesh.boundary_types[f, 0]

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
                    normal = mesh.vector_S_f[f]
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
            normal = mesh.vector_S_f[f]
            face_area_mag = mesh.face_areas[f]  # Magnitude of face area vector

            nx_unit, ny_unit = 0.0, 0.0
            if face_area_mag > _SMALL:
                nx_unit = normal[0] / face_area_mag
                ny_unit = normal[1] / face_area_mag
                delta_CF = np.linalg.norm(mesh.vector_d_CE[f])

                dp_dCF = (p[F] - p[C]) / max(delta_CF, _SMALL)

                rhie_chow_corr_x = d_u_face * (dp_dCF * nx_unit - grad_p_x_interp)

                rhie_chow_corr_y = d_v_face * (dp_dCF * ny_unit - grad_p_y_interp)

                face_velocity[f, 0] = u_interp + rhie_chow_corr_x
                face_velocity[f, 1] = v_interp + rhie_chow_corr_y

      
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
                        delta_CF, _SMALL
                    )

                    # Velocity correction due to p' (Rhie-Chow form for the U' part)
                    # d_u_face is (Vol/Ap_u)_f from the original momentum equation a_P
                    vel_corr_prime_x = d_u_face * (
                        dp_prime_dCF * nx_unit - grad_p_prime_x_interp
                    )
                    vel_corr_prime_y = d_v_face * (
                        dp_prime_dCF * ny_unit - grad_p_prime_y_interp
                    )
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
        S_f = np.ascontiguousarray(mesh.vector_S_f[f])  # This is already Area * unit_normal

        # Dot product of velocity and normal vector (U_f . S_f)
        # S_f = [S_fx, S_fy]
        # U_f = [u_f, v_f]
        # U_f . S_f = u_f * S_fx + v_f * S_fy
        vol_flux = np.dot(face_velocity[f], S_f)

        # Calculate mass flux
        face_mass_fluxes[f] = rho * vol_flux

    return face_mass_fluxes
