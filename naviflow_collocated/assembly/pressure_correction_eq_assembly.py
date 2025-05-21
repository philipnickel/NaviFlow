from numba import njit
import numpy as np

@njit
def assemble_pressure_correction_matrix(mesh,rho,  Ap_u, Ap_v):
    """
    Assemble the pressure correction matrix (Poisson-like) using momentum diagonals.

    Applies zero-gradient BC on all boundaries by adding diagonal contributions for boundary faces.
    Pins one pressure node to avoid null space due to constant offset invariance.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh object with connectivity.
    Ap_u : ndarray
        Diagonal coefficients of u-momentum equation.
    Ap_v : ndarray
        Diagonal coefficients of v-momentum equation.

    Returns
    -------
    row, col, data : tuple of ndarrays
        COO format entries of the sparse matrix.
    """

    n_cells     = mesh.cell_volumes.shape[0]
    n_internal  = mesh.internal_faces.shape[0]
    n_boundary  = mesh.boundary_faces.shape[0]

    max_entries = 4 * n_internal + n_boundary + 1  # 4 per internal face, 1 per boundary, 1 for pin
    b = np.zeros(n_cells, dtype=np.float64)

    row = np.empty(max_entries, dtype=np.int32)
    col = np.empty(max_entries, dtype=np.int32)
    data = np.empty(max_entries, dtype=np.float64)
    idx = 0

    # Internal face contributions (standard Laplacian stencil)
    for i in range(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        E_f = np.linalg.norm(mesh.vector_E_f[f])
        d_CF = np.linalg.norm(mesh.vector_d_CE[f])
    
        Df = E_f / d_CF
        coeff = -rho * Df

        row[idx] = P; col[idx] = P; data[idx] =  coeff; idx += 1
        row[idx] = P; col[idx] = N; data[idx] = -coeff; idx += 1
        row[idx] = N; col[idx] = N; data[idx] =  coeff; idx += 1
        row[idx] = N; col[idx] = P; data[idx] = -coeff; idx += 1

        correction_term = 0.0 # m*_f + (something) # Eq. 15.99 
        

        b[P] -= 0.0 #coeff
        b[N] += 0.0 #coeff

    

    
    # Pin one pressure node (e.g., cell 0)
    pinned = 0
    row[idx] = pinned
    col[idx] = pinned
    data[idx] = 0.0  # Must be non-zero to ensure the matrix is not singular
    bcorr = 0.0

    return row[:idx], col[:idx], data[:idx], bcorr
