import numpy as np
from scipy import sparse

def get_coeff_mat(mesh, rho, d_avg):
    owner_cells, neighbor_cells = mesh.get_owner_neighbor()
    face_areas = mesh.get_face_areas()
    face_normals = mesh.get_face_normals()
    n_cells = mesh.n_cells
    n_faces = len(face_areas)

    rows = []
    cols = []
    data = []

    for face_idx in range(n_faces):
        owner = owner_cells[face_idx]
        neighbor = neighbor_cells[face_idx]
        area = face_areas[face_idx]

        # Skip faces where owner is invalid
        if owner < 0 or owner >= n_cells:
            continue

        # Get interpolation factors
        g_owner, g_neighbor = mesh.get_face_interpolation_factors(face_idx)
        
        # Interpolate d_avg to face (handle boundary case implicitly via factors)
        d_owner_val = d_avg[owner]
        d_neighbor_val = d_avg[neighbor] if (neighbor >= 0 and neighbor < n_cells) else d_owner_val # Use owner if neighbor invalid
        
        d_face = g_owner * d_owner_val + g_neighbor * d_neighbor_val
        face_coeff = rho * d_face * area

        if neighbor >= 0 and neighbor < n_cells:
            # Internal face
            rows += [owner, neighbor, owner, neighbor]
            cols += [neighbor, owner, owner, neighbor]
            data += [-face_coeff, -face_coeff, face_coeff, face_coeff]
        else:
            # Boundary face (only owner exists)
            # Correctly add the face coefficient to the diagonal term
            # to enforce zero Neumann BC implicitly (a_p = sum(a_nb_internal))
            rows.append(owner)
            cols.append(owner)
            data.append(face_coeff)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells)).tocsr()

    # Apply pressure pinning ONLY if requested (and handle RHS pinning elsewhere)
    # REMOVED PRESSURE PINNING LOGIC - Should be handled externally based on BCs
    # if pin_pressure:
    #     if n_cells > 0:
    #         A = A.tolil()
    #         A[0, :] = 0   # Zero out the first row
    #         A[:, 0] = 0   # Zero out the first column (optional but common)
    #         A[0, 0] = 1.0   # Set diagonal to 1
    #         A = A.tocsr()
    #     else:
    #         print("Warning: Cannot pin pressure for empty matrix.")

    return A
