import numpy as np
from numba import njit

@njit
def assemble_pressure_correction_matrix(mesh, rho):
    """
    Assemble pressure correction matrix for SIMPLE (Poisson-like).
    Assumes zero-gradient BC (do nothing for boundaries).
    Pins one pressure cell to avoid singular system.
    """
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]

    max_entries = 4 * n_internal + 1
    row = np.zeros(max_entries, dtype=np.int32)
    col = np.zeros(max_entries, dtype=np.int32)
    data = np.zeros(max_entries, dtype=np.float64)
    idx = 0

    for i in range(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        E_f = np.linalg.norm(mesh.vector_E_f[f])
        S_f = mesh.vector_S_f[f]
        S_f_mag = np.linalg.norm(S_f) + 1e-14
        d_CF = np.linalg.norm(mesh.vector_d_CE[f]) + 1e-14
        coeff = rho * E_f / d_CF #* S_f_mag

        row[idx] = P; col[idx] = P; data[idx] =  coeff; idx += 1
        row[idx] = P; col[idx] = N; data[idx] = -coeff; idx += 1
        row[idx] = N; col[idx] = N; data[idx] =  coeff; idx += 1
        row[idx] = N; col[idx] = P; data[idx] = -coeff; idx += 1

    # Pin pressure at one cell
    
    pin = 0
    row[idx] = pin; col[idx] = pin; data[idx] = 1.0; idx += 1

    
    bcorr = 0.0
    return row[:idx], col[:idx], data[:idx], bcorr
