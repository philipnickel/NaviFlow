import numpy as np
from numba import njit

EPS = 1e-14

@njit
def compute_cell_gradients(mesh, u):
    n_cells = mesh.cell_centers.shape[0]
    grad = np.zeros((n_cells, 2), dtype=np.float64)

    # required mesh views
    cell_faces     = mesh.cell_faces
    owner_cells    = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    cc             = mesh.cell_centers

    for c in range(n_cells):
        A00 = A01 = A11 = 0.0
        b0  = b1  = 0.0

        u_c  = u[c]
        x_Px = cc[c, 0]
        x_Py = cc[c, 1]

        for f in cell_faces[c]:
            if f < 0:
                break

            P = owner_cells[f]
            N = neighbor_cells[f]

            # -------- internal faces only --------
            if N >= 0:           # good – we have a neighbour
                other = N if c == P else P
                vec0  = cc[other, 0] - x_Px
                vec1  = cc[other, 1] - x_Py
                du    = u[other] - u_c
            else:                # boundary face – ignore
                continue

            r2 = vec0*vec0 + vec1*vec1
            if r2 < EPS:
                continue
            w = 1.0 / r2

            A00 += w * vec0 * vec0
            A01 += w * vec0 * vec1
            A11 += w * vec1 * vec1
            b0  += w * vec0 * du
            b1  += w * vec1 * du

        # constant field? → zero gradient
        if abs(b0) < EPS and abs(b1) < EPS:
            continue

        # tiny Tikhonov regularisation
        λ = 1e-8
        A00 += λ
        A11 += λ

        denom = A00 * A11 - A01 * A01
        if abs(denom) > EPS:
            grad[c, 0] = (A11 * b0 - A01 * b1) / denom
            grad[c, 1] = (A00 * b1 - A01 * b0) / denom
        else:
            # ill-posed (likely a boundary cell with <2 neighbours)
            grad[c, 0] = grad[c, 1] = 0.0

    return grad
