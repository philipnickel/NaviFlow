import numpy as np

from numba import njit, prange


@njit(parallel=True)
def compute_cell_gradients(mesh, u):
    """
    Compute the cell-centered gradient of a scalar field using weighted least squares,
    consistent with Moukalled et al.

    Parameters:
    - mesh: MeshData2D
    - u: scalar field (cell-centered, shape = (n_cells,))

    Returns:
    - grad_u: cell-centered gradient of u (shape = (n_cells, 2)); suitable for use in deferred correction (vector form)
    """
    n_cells = mesh.cell_centers.shape[0]
    grad_u = np.zeros((n_cells, 2))

    for c in prange(n_cells):
        A = np.zeros((2, 2))
        b = np.zeros(2)

        for j in range(mesh.cell_faces.shape[1]):
            f = mesh.cell_faces[c, j]
            if f < 0:
                continue

            P = mesh.owner_cells[f]
            N = mesh.neighbor_cells[f]

            x_P = mesh.cell_centers[c]

            if N >= 0:
                if c == P:
                    x_other = mesh.cell_centers[N]
                    u_other = u[N]
                elif c == N:
                    x_other = mesh.cell_centers[P]
                    u_other = u[P]
                else:
                    continue
                vec = x_other - x_P
                du = u_other - u[c]
            else:
                if mesh.boundary_types[f, 0] < 0:
                    continue
                x_B = mesh.face_centers[f]
                u_B = mesh.boundary_values[f, 0]
                u_P = u[c]
                vec = x_B - x_P
                du = u_B - u_P

            r2 = vec[0] * vec[0] + vec[1] * vec[1] + 1e-14
            w = 1.0 / r2

            A[0, 0] += w * vec[0] * vec[0]
            A[0, 1] += w * vec[0] * vec[1]
            A[1, 0] += w * vec[1] * vec[0]
            A[1, 1] += w * vec[1] * vec[1]

            b[0] += w * vec[0] * du
            b[1] += w * vec[1] * du

        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if np.abs(det) > 1e-14:
            A_inv = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det
            grad_u[c] = A_inv @ b
        else:
            grad_u[c] = 0.0

    # Returned gradients are accurate vector fields, suitable for use in ∇φ ⋅ (t_f + d_f) deferred correction terms
    return grad_u
