"""
Face velocity interpolation using Rhie-Chow method for collocated grids.

This module implements the Rhie-Chow interpolation to prevent pressure checkerboarding
on collocated grids. The implementation follows Moukalled's approach as described in:
'The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction
with OpenFOAM and Matlab'.
"""

import numpy as np
from numba import njit, prange

# Boundary condition type codes (must match momentum_eq_assembly.py)
BC_WALL_NO_SLIP = 1
BC_WALL_SLIP = 2
BC_INLET_VELOCITY = 3
BC_OUTLET_PRESSURE = 4
BC_SYMMETRY = 5

# Small value to prevent division by zero
_SMALL = 1.0e-12


@njit
def interpolate_scalar_to_face(phi, mesh, f):
    """
    Interpolate a scalar from cell centers to a face.

    Parameters
    ----------
    phi : ndarray
        Scalar field at cell centers
    mesh : MeshData2D
        Mesh data structure
    f : int
        Face index

    Returns
    -------
    phi_f : float
        Interpolated scalar value at the face
    """
    C = mesh.owner_cells[f]
    F = mesh.neighbor_cells[f]

    # For internal faces, use linear interpolation
    if F >= 0:
        fx = mesh.face_interp_factors[f]
        return fx * phi[F] + (1.0 - fx) * phi[C]
    else:
        # For boundary faces, return cell value (zero gradient)
        # This is a simplification; in practice, use proper boundary conditions
        return phi[C]


@njit
def calculate_pressure_gradient(p, mesh, cell_idx):
    """
    Calculate pressure gradient at a cell center.

    Parameters
    ----------
    p : ndarray
        Pressure field
    mesh : MeshData2D
        Mesh data
    cell_idx : int
        Cell index

    Returns
    -------
    grad_p : ndarray
        Pressure gradient vector [dp/dx, dp/dy]
    """
    # Initialize gradient vector
    grad_p = np.zeros(2)

    # Get cell volume
    vol = mesh.cell_volumes[cell_idx]

    # Loop over all faces of the cell
    for face_idx in range(mesh.cell_faces.shape[1]):
        f = mesh.cell_faces[cell_idx, face_idx]
        if f < 0:
            continue  # Skip invalid faces

        # Get face area and normal
        area = mesh.face_areas[f]
        normal = mesh.face_normals[f]

        # Determine face pressure
        p_f = interpolate_scalar_to_face(p, mesh, f)

        # Determine sign based on face orientation
        sign = -1.0 if mesh.owner_cells[f] == cell_idx else 1.0

        # Add contribution to gradient
        grad_p[0] += sign * p_f * area * normal[0]
        grad_p[1] += sign * p_f * area * normal[1]

    # Normalize by cell volume
    if vol > _SMALL:
        grad_p /= vol

    return grad_p


@njit(parallel=True)
def compute_rhie_chow_face_velocities(
    mesh, u, v, p, mu, rho, u_prime=None, v_prime=None, p_prime=None
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
        Velocity fields at cell centers
    p : ndarray
        Pressure field at cell centers
    mu : float or ndarray
        Dynamic viscosity
    rho : float
        Density
    u_prime, v_prime : ndarray, optional
        Velocity corrections for SIMPLE algorithm
    p_prime : ndarray, optional
        Pressure correction for SIMPLE algorithm

    Returns
    -------
    face_velocity : ndarray
        Face velocities [n_faces, 2]
    """
    n_faces = len(mesh.face_areas)
    n_cells = len(mesh.cell_volumes)

    # Calculate pressure gradients at cell centers
    grad_p = np.zeros((n_cells, 2))
    for c in prange(n_cells):
        grad_p[c] = calculate_pressure_gradient(p, mesh, c)

    # Calculate correction pressure gradients if provided
    grad_p_prime = None
    if p_prime is not None:
        grad_p_prime = np.zeros((n_cells, 2))
        for c in prange(n_cells):
            grad_p_prime[c] = calculate_pressure_gradient(p_prime, mesh, c)

    # Initialize face velocities
    face_velocity = np.zeros((n_faces, 2))

    # Calculate A_P for each cell (approximate diagonal coefficient)
    # This is a simplification; in a full solver, A_P would come from matrix assembly
    a_P = np.zeros(n_cells)
    for c in range(n_cells):
        # For diffusion contribution
        a_P_diff = 0.0
        for face_idx in range(mesh.cell_faces.shape[1]):
            f = mesh.cell_faces[c, face_idx]
            if f >= 0:
                a_P_diff += mu * mesh.face_areas[f] / max(mesh.delta_CF[f], _SMALL)

        # For convection contribution (simplified)
        a_P_conv = rho * mesh.cell_volumes[c]

        # Combine and ensure non-zero
        a_P[c] = max(a_P_diff + a_P_conv, _SMALL)

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
                bc_type = mesh.boundary_types[boundary_idx]

                if bc_type == BC_WALL_NO_SLIP:
                    # No-slip wall: zero velocity
                    face_velocity[f, 0] = 0.0
                    face_velocity[f, 1] = 0.0
                elif bc_type == BC_INLET_VELOCITY:
                    # Inlet: use boundary values
                    face_velocity[f, 0] = mesh.boundary_values[boundary_idx, 0]
                    face_velocity[f, 1] = mesh.boundary_values[boundary_idx, 1]
                elif bc_type == BC_OUTLET_PRESSURE:
                    # Outlet: zero gradient
                    face_velocity[f, 0] = u[C]
                    face_velocity[f, 1] = v[C]
                elif bc_type == BC_SYMMETRY:
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

        # For internal faces, use Rhie-Chow interpolation
        else:
            # Linear interpolation factors
            fx = mesh.face_interp_factors[f]

            # Linear interpolation for velocity and pressure gradients
            u_interp = fx * u[F] + (1.0 - fx) * u[C]
            v_interp = fx * v[F] + (1.0 - fx) * v[C]

            grad_p_x_interp = fx * grad_p[F, 0] + (1.0 - fx) * grad_p[C, 0]
            grad_p_y_interp = fx * grad_p[F, 1] + (1.0 - fx) * grad_p[C, 1]

            # Interpolated reciprocal of A_P with safety against division by zero
            reciprocal_aP_C = 1.0 / a_P[C]
            reciprocal_aP_F = 1.0 / a_P[F]
            a_P_interp = fx * reciprocal_aP_F + (1.0 - fx) * reciprocal_aP_C

            # Face normal and area
            normal = mesh.face_normals[f]
            nmag = np.sqrt(normal[0] ** 2 + normal[1] ** 2)
            if nmag > _SMALL:
                nx = normal[0] / nmag
                ny = normal[1] / nmag

                # Calculate pressure difference across the face
                # p_f = interpolate_scalar_to_face(p, mesh, f) # Variable not used
                dp_dn = (p[F] - p[C]) / max(mesh.delta_CF[f], _SMALL)

                # Calculate pressure term at face
                rhie_chow_corr_x = a_P_interp * (dp_dn * nx - grad_p_x_interp)
                rhie_chow_corr_y = a_P_interp * (dp_dn * ny - grad_p_y_interp)

                # Apply Rhie-Chow correction
                face_velocity[f, 0] = u_interp + rhie_chow_corr_x
                face_velocity[f, 1] = v_interp + rhie_chow_corr_y

                # Add correction velocities if provided (for SIMPLE algorithm)
                if (
                    u_prime is not None
                    and v_prime is not None
                    and grad_p_prime is not None
                ):
                    # Linear interpolation for correction velocities
                    u_prime_interp = fx * u_prime[F] + (1.0 - fx) * u_prime[C]
                    v_prime_interp = fx * v_prime[F] + (1.0 - fx) * v_prime[C]

                    # Correction pressure gradient
                    grad_p_prime_x_interp = (
                        fx * grad_p_prime[F, 0] + (1.0 - fx) * grad_p_prime[C, 0]
                    )
                    grad_p_prime_y_interp = (
                        fx * grad_p_prime[F, 1] + (1.0 - fx) * grad_p_prime[C, 1]
                    )

                    # Calculate correction pressure difference
                    dp_prime_dn = (p_prime[F] - p_prime[C]) / max(
                        mesh.delta_CF[f], _SMALL
                    )

                    # Apply correction
                    rhie_chow_corr_x = a_P_interp * (
                        dp_prime_dn * nx - grad_p_prime_x_interp
                    )
                    rhie_chow_corr_y = a_P_interp * (
                        dp_prime_dn * ny - grad_p_prime_y_interp
                    )

                    face_velocity[f, 0] += u_prime_interp + rhie_chow_corr_x
                    face_velocity[f, 1] += v_prime_interp + rhie_chow_corr_y
            else:
                # Fallback to simple interpolation if normal is degenerate
                face_velocity[f, 0] = u_interp
                face_velocity[f, 1] = v_interp

    return face_velocity


@njit(parallel=True)
def compute_face_fluxes(mesh, face_velocity):
    """
    Compute mass fluxes at faces from face velocities.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh data structure
    face_velocity : ndarray
        Face velocities [n_faces, 2]

    Returns
    -------
    face_fluxes : ndarray
        Mass fluxes at faces
    """
    n_faces = len(mesh.face_areas)
    face_fluxes = np.zeros(n_faces)

    for f in prange(n_faces):
        # Get face area and normal
        area = mesh.face_areas[f]
        normal = mesh.face_normals[f]

        # Normalize normal vector
        nmag = np.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if nmag > _SMALL:
            nx = normal[0] / nmag
            ny = normal[1] / nmag

            # Dot product of velocity and normal
            u_n = face_velocity[f, 0] * nx + face_velocity[f, 1] * ny

            # Calculate mass flux
            face_fluxes[f] = area * u_n

    return face_fluxes
