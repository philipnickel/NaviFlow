import numpy as np
from numba import njit, prange
from naviflow_collocated.core.helpers import BC_NEUMANN

EPS = 1e-20

@njit(parallel=True, cache=True)
def compute_cell_gradients(mesh, u, pinned_idx=0, weighted=True, weight_exponent=1.0, use_limiter=True):
    n_cells = mesh.cell_centers.shape[0]
    grad = np.zeros((n_cells, 2), dtype=np.float64)

    # required mesh views
    cell_faces     = mesh.cell_faces
    owner_cells    = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    cc             = mesh.cell_centers
    face_centers   = mesh.face_centers
    boundary_types = mesh.boundary_types

    for c in prange(n_cells):
        if c == pinned_idx:
            grad[c, 0] = grad[c, 1] = 0.0
            continue

        A00 = A01 = A11 = 0.0
        b0  = b1  = 0.0

        u_c  = u[c]
        x_Px = cc[c, 0]
        x_Py = cc[c, 1]

        umin = u_c
        umax = u_c

        # First pass: compute gradient matrix elements using only internal faces
        for f in cell_faces[c]:
            if f < 0:
                break

            P = owner_cells[f]
            N = neighbor_cells[f]

            if N >= 0:  # Only use internal faces
                other = N if c == P else P
                if other == pinned_idx:
                    continue

                # Cache neighbor values
                other_x = cc[other, 0]
                other_y = cc[other, 1] 
                other_u = u[other]
                
                vec0 = other_x - x_Px
                vec1 = other_y - x_Py
                du   = other_u - u_c

                if use_limiter:
                    if other_u < umin:
                        umin = other_u
                    if other_u > umax:
                        umax = other_u

                r2 = vec0 * vec0 + vec1 * vec1
                if r2 < EPS:
                    continue
                    
                # Distance-weighted least squares
                if weighted:
                    if weight_exponent == 1.0:
                        w = 1.0 / r2  # Optimized for common case
                    else:
                        w = 1.0 / (r2 ** weight_exponent)
                else:
                    w = 1.0 / r2

                w_vec0 = w * vec0
                w_vec1 = w * vec1
                
                A00 += w_vec0 * vec0
                A01 += w_vec0 * vec1  
                A11 += w_vec1 * vec1
                b0  += w_vec0 * du
                b1  += w_vec1 * du

        denom = A00 * A11 - A01 * A01
        if abs(denom) > EPS:
            inv_denom = 1.0 / denom
            gx = (A11 * b0 - A01 * b1) * inv_denom
            gy = (A00 * b1 - A01 * b0) * inv_denom

            # Apply Barth–Jespersen limiter only if enabled and only using internal faces
            phi = 1.0
            if use_limiter and (umax > u_c or umin < u_c):  # Skip if no variation
                for f in cell_faces[c]:
                    if f < 0:
                        break

                    P = owner_cells[f]
                    N = neighbor_cells[f]
                    if N >= 0:  # Only use internal faces for limiting
                        other = N if c == P else P
                        if other == pinned_idx:
                            continue

                        dx = cc[other, 0] - x_Px
                        dy = cc[other, 1] - x_Py
                        delta_u = gx * dx + gy * dy

                        if delta_u > EPS:
                            phi = min(phi, (umax - u_c) / delta_u)
                        elif delta_u < -EPS:
                            phi = min(phi, (umin - u_c) / delta_u)

            grad[c, 0] = phi * gx
            grad[c, 1] = phi * gy
        else:
            grad[c, 0] = 0.0
            grad[c, 1] = 0.0

    return grad
