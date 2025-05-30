import numpy as np
from numba import njit, prange

EPS = 1e-20

@njit(parallel=True)
def compute_cell_gradients(mesh, u, pinned_idx=0):
    n_cells = mesh.cell_centers.shape[0]
    grad = np.zeros((n_cells, 2), dtype=np.float64)

    # required mesh views
    cell_faces     = mesh.cell_faces
    owner_cells    = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    cc             = mesh.cell_centers

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

        for f in cell_faces[c]:
            if f < 0:
                break

            P = owner_cells[f]
            N = neighbor_cells[f]

            if N >= 0:
                other = N if c == P else P
                if other == pinned_idx:
                    continue

                vec0 = cc[other, 0] - x_Px
                vec1 = cc[other, 1] - x_Py
                du   = u[other] - u_c

                umin = min(umin, u[other])
                umax = max(umax, u[other])
            else:
                continue  # boundary face

            r2 = vec0 * vec0 + vec1 * vec1
            if r2 < EPS:
                continue
            w = 1.0 / r2

            A00 += w * vec0 * vec0
            A01 += w * vec0 * vec1
            A11 += w * vec1 * vec1
            b0  += w * vec0 * du
            b1  += w * vec1 * du

        denom = A00 * A11 - A01 * A01
        if abs(denom) > EPS:
            gx = (A11 * b0 - A01 * b1) / denom
            gy = (A00 * b1 - A01 * b0) / denom

            # Barth–Jespersen limiter
            phi = 1.0
            for f in cell_faces[c]:
                if f < 0:
                    break

                P = owner_cells[f]
                N = neighbor_cells[f]
                if N >= 0:
                    other = N if c == P else P
                    if other == pinned_idx:
                        continue
                else:
                    continue

                dx = cc[other, 0] - x_Px
                dy = cc[other, 1] - x_Py
                delta_u = gx * dx + gy * dy

                if delta_u > 0:
                    phi = min(phi, (umax - u_c) / (delta_u + EPS))
                elif delta_u < 0:
                    phi = min(phi, (umin - u_c) / (delta_u + EPS))

            grad[c, 0] = phi * gx
            grad[c, 1] = phi * gy
        else:
            grad[c, :] = 0.0

    return grad
