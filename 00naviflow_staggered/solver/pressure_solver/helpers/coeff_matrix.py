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

        if owner < 0 or owner >= n_cells:
            continue  # Skip invalid owner

        # Interpolation factors for d at face
        g_owner, g_neighbor = mesh.get_face_interpolation_factors(face_idx)
        d_owner_val = d_avg[owner]
        d_neighbor_val = d_avg[neighbor] if (0 <= neighbor < n_cells) else d_owner_val
        d_face = g_owner * d_owner_val + g_neighbor * d_neighbor_val

        face_coeff = rho * d_face * area

        if 0 <= neighbor < n_cells:
            # Internal face
            rows += [owner, owner, neighbor, neighbor]
            cols += [owner, neighbor, neighbor, owner]
            data += [face_coeff, -face_coeff, face_coeff, -face_coeff]
        else:
            # Boundary face (Dirichlet BC → Neumann for pressure correction: zero normal gradient)
            #rows.append(owner)
            #cols.append(owner)
            #data.append(face_coeff)
            continue

    # Assemble sparse matrix
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells)).tocsr()

    # Apply pressure pinning (e.g., cell 0)
    A = A.tolil()
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A = A.tocsr()

    return A
