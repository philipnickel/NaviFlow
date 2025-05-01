import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_least_squares_gradients(cell_centers, owner_cells, neighbor_cells, phi):
    n_cells = cell_centers.shape[0]
    gradients = np.zeros((n_cells, 2))
    counts = np.zeros(n_cells, dtype=np.int32)
    max_neighbors = 20
    neighbors = -1 * np.ones((n_cells, max_neighbors), dtype=np.int32)

    # Serial pass to build neighbor list
    for f in range(len(owner_cells)):
        C = owner_cells[f]
        F = neighbor_cells[f]
        if F != -1:
            idx_C = counts[C]
            neighbors[C, idx_C] = F
            counts[C] += 1

            idx_F = counts[F]
            neighbors[F, idx_F] = C
            counts[F] += 1

    # Parallel gradient estimation
    for i in prange(n_cells):
        A = np.zeros((2, 2))
        b = np.zeros(2)

        for k in range(counts[i]):
            j = neighbors[i, k]
            dx = cell_centers[j] - cell_centers[i]
            dphi = phi[j] - phi[i]

            A[0, 0] += dx[0] * dx[0]
            A[0, 1] += dx[0] * dx[1]
            A[1, 0] += dx[1] * dx[0]
            A[1, 1] += dx[1] * dx[1]

            b[0] += dx[0] * dphi
            b[1] += dx[1] * dphi

        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) > 1e-12:
            invA = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det
            gradients[i] = invA @ b

    return gradients
