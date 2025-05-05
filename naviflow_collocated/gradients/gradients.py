"""
Gradient calculation schemes for cell-centered scalar fields.
"""

import numpy as np
from numba import njit

# from naviflow_collocated.mesh import MeshData2D as Mesh # F401: Unused


@njit
def _calculate_gradient_gauss_linear_internal(mesh, phi):
    """Internal Numba-jitted gradient calculation."""
    n_cells = len(mesh.cell_volumes)
    n_faces = len(mesh.face_areas)
    n_internal_faces = n_faces - len(mesh.boundary_faces)

    # Initialize gradient vector field
    grad_phi = np.zeros((n_cells, 2))

    # --- Loop over INTERNAL faces ---
    # Accumulate sum(phi_f * Sf) for owner and neighbour
    for i_face in range(n_internal_faces):
        owner = mesh.owner_cells[i_face]
        neighbour = mesh.neighbor_cells[i_face]
        Sf = mesh.face_normals[i_face]  # Face normal vector (magnitude = area)
        Cf = mesh.face_centers[i_face]
        C_owner = mesh.cell_centers[owner]
        C_neighbour = mesh.cell_centers[neighbour]

        # Linear interpolation weight
        d_cf = np.linalg.norm(Cf - C_owner) + np.linalg.norm(Cf - C_neighbour)
        if d_cf < 1e-12:  # Avoid division by zero for coincident points
            fx = 0.5
        else:
            fx = np.linalg.norm(Cf - C_neighbour) / d_cf

        # Interpolate phi to face center
        phi_f = fx * phi[owner] + (1 - fx) * phi[neighbour]

        # Accumulate flux contribution to gradient integral
        flux_vector = phi_f * Sf
        grad_phi[owner] += flux_vector
        grad_phi[neighbour] -= flux_vector  # Sf points out of owner, into neighbour

    # --- Loop over BOUNDARY faces ---
    # Apply boundary conditions for phi_f based on boundary types in mesh
    for i_face in range(n_internal_faces, n_faces):
        owner = mesh.owner_cells[i_face]
        Sf = mesh.face_normals[i_face]

        # Get the index relative to the boundary face list
        boundary_face_local_idx = -1
        for idx, bf in enumerate(mesh.boundary_faces):
            if bf == i_face:
                boundary_face_local_idx = idx
                break

        if boundary_face_local_idx >= 0:
            # Get boundary type from mesh data
            bc_type = mesh.boundary_types[boundary_face_local_idx]
            bc_value = mesh.boundary_values[
                boundary_face_local_idx, 0
            ]  # Use first component as value

            # Determine phi_f based on boundary type
            if bc_type in [
                1,
                3,
            ]:  # BC_WALL_NO_SLIP or BC_INLET_VELOCITY (fixedValue type)
                phi_f = bc_value
            else:
                # Default for outlet, symmetry, etc. - zero gradient
                phi_f = phi[owner]

            # Accumulate flux contribution
            flux_vector = phi_f * Sf
            grad_phi[owner] += flux_vector

    # Divide by cell volume to get the average gradient
    # Avoid division by zero for potentially zero-volume cells (though unlikely)
    non_zero_volumes = np.maximum(mesh.cell_volumes, 1e-30)
    grad_phi[:, 0] /= non_zero_volumes
    grad_phi[:, 1] /= non_zero_volumes

    return grad_phi


def calculate_gradient_gauss_linear(mesh, phi, boundary_conditions=None):
    """
    Calculate the gradient of a scalar field using Gauss theorem with linear interpolation.

    Args:
        mesh: MeshData2D object.
        phi: Scalar field at cell centers.
        boundary_conditions: Not used, kept for backward compatibility.

    Returns:
        ndarray: Gradient vector field at cell centers (n_cells, 2).
    """
    # Simplified implementation that doesn't use boundary_conditions parameter
    return _calculate_gradient_gauss_linear_internal(mesh, phi)
