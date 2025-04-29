import numpy as np

def get_rhs(mesh, rho, u_star, v_star, p, d_avg, bc_manager=None):
    """
    Compute RHS (mass imbalance) from INTERNAL faces using Rhie-Chow interpolation.
    Uses geometrically interpolated average d coefficient.
    """

    n_cells = mesh.n_cells
    n_faces = mesh.n_faces
    owners, neighbors = mesh.get_owner_neighbor()
    face_areas = mesh.get_face_areas()
    face_normals = mesh.get_face_normals()

    rhs = np.zeros(n_cells)

    for face_idx in range(n_faces):
        owner = owners[face_idx]
        neighbor = neighbors[face_idx]
        area = face_areas[face_idx]
        normal = face_normals[face_idx]
        
        # Ensure owner is valid - Skip face if no valid owner cell exists
        if owner < 0 or owner >= n_cells:
            continue # Skip faces with no valid owner

        if normal.shape != (2,):
            raise ValueError(f"Invalid face normal shape {normal.shape} at face {face_idx}")

        if neighbor >= 0 and neighbor < n_cells:
            u_face_avg = 0.5 * (u_star[owner] + u_star[neighbor])
            v_face_avg = 0.5 * (v_star[owner] + v_star[neighbor])
            delta_p = p[neighbor] - p[owner]
            
            # Interpolate d_avg to face using geometric factors
            g_owner, g_neighbor = mesh.get_face_interpolation_factors(face_idx)
            d_owner_val = d_avg[owner]
            # Neighbor must be valid here since it's an internal face
            d_neighbor_val = d_avg[neighbor] 
            d_face = g_owner * d_owner_val + g_neighbor * d_neighbor_val

            pressure_term = d_face * delta_p

            rhie_chow_correction = pressure_term * normal
            face_velocity = np.array([u_face_avg, v_face_avg]) - rhie_chow_correction
            mass_flux = rho * np.dot(face_velocity, normal) * area

            rhs[owner] -= mass_flux
            rhs[neighbor] += mass_flux
        else:
            # Boundary face - Contribution is handled externally based on BC type
            pass # No calculation needed in this core RHS function

    return rhs
