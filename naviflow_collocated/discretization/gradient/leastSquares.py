import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def compute_cell_gradients(mesh, u):
    """
    Weighted-least-squares ∇u (serial, optimised).

    Assumes mesh.cell_faces uses -1 as end-of-row sentinel.
    """
    n_cells = mesh.cell_centers.shape[0]
    grad = np.empty((n_cells, 2), dtype=np.float64)

    # local views – cheaper than attribute lookups in the loop
    cell_faces     = mesh.cell_faces
    owner_cells    = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    cc             = mesh.cell_centers
    fc             = mesh.face_centers
    b_types        = mesh.boundary_types
    b_vals         = mesh.boundary_values
    u_arr          = u

    for c in range(n_cells):
        A00 = A01 = A11 = 0.0   # A10 == A01
        b0  = b1  = 0.0

        u_c  = u_arr[c]
        x_Px = cc[c, 0]
        x_Py = cc[c, 1]

        # --- iterate faces of cell c --------------------------------------
        for f in cell_faces[c]:
            if f < 0:          # sentinel reached
                break

            P = owner_cells[f]
            N = neighbor_cells[f]

            # -- decide “other” cell/BC once, branchless ----------
            if N >= 0:                           # internal face
                other = N if c == P else P
                vec0  = cc[other, 0] - x_Px
                vec1  = cc[other, 1] - x_Py
                du    = u_arr[other] - u_c
            else:                                # boundary face
                if b_types[f, 0] < 0:
                    continue                     # unused slot
                vec0 = fc[f, 0] - x_Px
                vec1 = fc[f, 1] - x_Py
                du   = b_vals[f, 0] - u_c

            r2 = vec0*vec0 + vec1*vec1 + 1e-14
            w  = 1.0 / r2

            # accumulate symmetric A in 3 scalars
            A00 += w * vec0 * vec0
            A01 += w * vec0 * vec1      # = A10
            A11 += w * vec1 * vec1

            b0  += w * vec0 * du
            b1  += w * vec1 * du

        # --- solve 2×2 ------------------------------------------
        det = A00*A11 - A01*A01       # since A10=A01
        if np.abs(det) > 1e-14:
            inv00 =  A11 / det
            inv01 = -A01 / det
            inv11 =  A00 / det
            grad[c, 0] = inv00*b0 + inv01*b1
            grad[c, 1] = inv01*b0 + inv11*b1
        else:
            grad[c, 0] = grad[c, 1] = 0.0

    return grad
