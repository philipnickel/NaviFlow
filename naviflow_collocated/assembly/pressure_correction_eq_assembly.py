import numpy as np
from numba import njit, prange
import numpy as np
from numba import njit


BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 3
BC_NEUMANN = 4


@njit(parallel=False)
def assemble_pressure_correction_matrix(mesh, rho):
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]

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
        d_CF = np.linalg.norm(mesh.vector_d_CE[f]) #+ 1e-14
        coeff = rho * E_f / d_CF

        row[idx] = P; col[idx] = P; data[idx] =  coeff; idx += 1
        row[idx] = P; col[idx] = N; data[idx] = -coeff; idx += 1
        row[idx] = N; col[idx] = N; data[idx] =  coeff; idx += 1
        row[idx] = N; col[idx] = P; data[idx] = -coeff; idx += 1
    """
    
    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        P = mesh.owner_cells[f]
        E_f = np.linalg.norm(mesh.vector_E_f[f])
        d_CB = mesh.d_Cb[f]
        coeff = rho * E_f / d_CB


        row[idx] = P; col[idx] = P; data[idx] += coeff; idx += 1
    """



    return row[:idx], col[:idx], data[:idx]


@njit(parallel=False)
def pressure_correction_loop_term(mesh, rho, grad_p_prime_f):
    """
    Assembles rhs correction term for second pressure solve
    """
    n_cells = mesh.cell_volumes.shape[0]
    correction_term = np.zeros(n_cells, dtype=np.float64)
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]

    for i in prange(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        T_f = np.ascontiguousarray(mesh.vector_T_f[f])
        coeff = rho * np.dot(np.ascontiguousarray(grad_p_prime_f[f]), T_f)
        correction_term[P] += coeff
        correction_term[N] -= coeff
    """

    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        P = mesh.owner_cells[f]
        T_f = np.ascontiguousarray(mesh.vector_T_f[f])
        coeff = rho * np.dot(np.ascontiguousarray(grad_p_prime_f[f]), T_f)
        correction_term[P] += coeff
    """



    return correction_term

