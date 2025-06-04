import numpy as np
from numba import njit, prange

BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 3
BC_NEUMANN = 4

@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy')
def assemble_pressure_correction_matrix(mesh, rho):
    """
    Assemble pressure correction equation matrix.
    Optimized for memory access patterns with pre-fetched static data.
    """
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]

    # Pessimistic non-zero count 
    max_nnz = 4 * n_internal  
    row = np.zeros(max_nnz, dtype=np.int64)
    col = np.zeros(max_nnz, dtype=np.int64)
    data = np.zeros(max_nnz, dtype=np.float64)

    idx = 0

    # ═══ PRE-FETCH STATIC DATA FOR MEMORY OPTIMIZATION ═══
    internal_faces = mesh.internal_faces
    owner_cells = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    vector_E_f = mesh.vector_E_f
    vector_d_CE = mesh.vector_d_CE

    # ––– internal faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_internal):
        f = internal_faces[i]
        P = owner_cells[f]
        N = neighbor_cells[f]

        # Pre-fetch vector components (single memory access each)
        E_f = vector_E_f[f]
        d_CE = vector_d_CE[f]
        E_f_0 = E_f[0]
        E_f_1 = E_f[1]
        d_CE_0 = d_CE[0]
        d_CE_1 = d_CE[1]

        # Manual norm calculations (faster than np.linalg.norm)
        E_mag = np.sqrt(E_f_0 * E_f_0 + E_f_1 * E_f_1)
        d_mag = np.sqrt(d_CE_0 * d_CE_0 + d_CE_1 * d_CE_1)

        # Compute conductance
        D_f = rho * E_mag / d_mag

        # Matrix coefficients
        row[idx] = P; col[idx] = P; data[idx] = D_f; idx += 1
        row[idx] = P; col[idx] = N; data[idx] = -D_f; idx += 1
        row[idx] = N; col[idx] = N; data[idx] = D_f; idx += 1
        row[idx] = N; col[idx] = P; data[idx] = -D_f; idx += 1

    return row[:idx], col[:idx], data[:idx]

@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy')
def pressure_correction_loop_term(mesh, rho, grad_p_prime_face):
    """
    Compute non-orthogonal pressure correction terms.
    Optimized for memory access patterns.
    """
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]

    source_term = np.zeros(n_cells, dtype=np.float64)

    # ═══ PRE-FETCH STATIC DATA ═══
    internal_faces = mesh.internal_faces
    owner_cells = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    vector_T_f = mesh.vector_T_f

    # ––– internal faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_internal):
        f = internal_faces[i]
        P = owner_cells[f]
        N = neighbor_cells[f]

        # Pre-fetch vector and gradient components
        T_f = vector_T_f[f]
        grad_p_f = grad_p_prime_face[f]
        T_f_0 = T_f[0]
        T_f_1 = T_f[1]
        grad_p_f_0 = grad_p_f[0]
        grad_p_f_1 = grad_p_f[1]

        # Manual dot product (optimized)
        non_ortho_flux = rho * (grad_p_f_0 * T_f_0 + grad_p_f_1 * T_f_1)

        # Distribute to cells
        source_term[P] -= non_ortho_flux
        source_term[N] += non_ortho_flux

    return source_term

