import numpy as np
from numba import njit, prange

@njit(parallel=False)
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

    for i in prange(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        E_f = np.linalg.norm(mesh.vector_E_f[f])
        T_f = mesh.vector_T_f[f]
        d_CF = np.linalg.norm(mesh.vector_d_CE[f]) + 1e-14
        coeff = rho * E_f / d_CF 

        row[idx] = P; col[idx] = P; data[idx] =  coeff; idx += 1
        row[idx] = P; col[idx] = N; data[idx] = -coeff; idx += 1
        row[idx] = N; col[idx] = N; data[idx] =  coeff; idx += 1
        row[idx] = N; col[idx] = P; data[idx] = -coeff; idx += 1

    # Pin pressure at one cell
    
    pin = 0
    row[idx] = pin; col[idx] = pin; data[idx] = 1.0; idx += 1

    
    
    return row[:idx], col[:idx], data[:idx]

@njit(parallel=False)
def pressure_correction_loop_term(mesh, rho, grad_p_prime_f):
    """
    Assembles rhs correction term for second pressure solve
    """
    n_cells = mesh.cell_volumes.shape[0]
    correction_term = np.zeros(n_cells, dtype=np.float64)
    n_internal = mesh.internal_faces.shape[0]

    for i in prange(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        T_f = np.ascontiguousarray(mesh.vector_T_f[f])
        coeff = -rho * np.dot(np.ascontiguousarray(grad_p_prime_f[f]), T_f)
        correction_term[P] += coeff
        correction_term[N] -= coeff

    return correction_term