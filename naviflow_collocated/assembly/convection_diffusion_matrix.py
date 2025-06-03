import numpy as np
from numba import njit, prange

from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_diffusive_flux_matrix_entry,
    compute_diffusive_correction,
    compute_boundary_diffusive_correction,
)
from naviflow_collocated.discretization.convection.upwind import (
    compute_convective_stencil,
    compute_boundary_convective_flux,
)

BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 3
BC_OBSTACLE = 4


EPS = 1.0e-14


@njit(fastmath=True, parallel=True, cache=True)
def _assemble_internal_faces_parallel(
    mesh,
    mdot,
    grad_phi,
    u_field,
    rho,
    mu,
    component_idx,
    phi,
    scheme,
    limiter,
):
    n_internal = mesh.internal_faces.shape[0]
    n_cells = mesh.cell_volumes.shape[0]
    
    # Pre-allocate arrays with exact sizes
    row = np.zeros(4 * n_internal, dtype=np.int64)
    col = np.zeros(4 * n_internal, dtype=np.int64)
    data = np.zeros(4 * n_internal, dtype=np.float64)
    rhs = np.zeros(n_cells, dtype=np.float64)

    internal_faces = mesh.internal_faces
    owner_cells = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    face_interp_factors = mesh.face_interp_factors
    vector_d_CE = mesh.vector_d_CE
    vector_skewness = mesh.vector_skewness

    for i in prange(n_internal):
        f = internal_faces[i]
        P = owner_cells[f]
        N = neighbor_cells[f]
        
        # Compute fluxes
        convFlux_P_f, convFlux_N_f, convDC = compute_convective_stencil(
            f, mesh, rho, mdot, u_field, grad_phi, component_idx, phi, scheme=scheme, limiter=limiter
        )
        diffFlux_P_f, diffFlux_N_f = compute_diffusive_flux_matrix_entry(f, grad_phi, mesh, mu)
        diffDC = compute_diffusive_correction(f, grad_phi, mesh, mu)
        
        # Combine fluxes
        Flux_P_f = np.float64(convFlux_P_f + diffFlux_P_f)
        Flux_N_f = np.float64(convFlux_N_f + diffFlux_N_f)
        Flux_V_f = np.float64(convDC + diffDC)
        
        # Store matrix entries (element-wise for Numba parallel)
        base = i * 4
        row[base + 0] = P
        row[base + 1] = P
        row[base + 2] = N
        row[base + 3] = N
        col[base + 0] = P
        col[base + 1] = N
        col[base + 2] = N
        col[base + 3] = P
        data[base + 0] = Flux_P_f
        data[base + 1] = Flux_N_f
        data[base + 2] = -Flux_N_f
        data[base + 3] = -Flux_P_f
        
        # Update RHS
        rhs[P] -= Flux_V_f
        rhs[N] += Flux_V_f

    return row, col, data, rhs

@njit(fastmath=True, cache=True)
def _assemble_boundary_faces_parallel(
    mesh,
    mdot,
    grad_phi,
    u_field,
    rho,
    mu,
    component_idx,
    phi,
    pressure_field,
    grad_pressure_field,
):
    n_boundary = mesh.boundary_faces.shape[0]
    n_cells = mesh.cell_volumes.shape[0]
    
    # Pre-allocate arrays with exact sizes
    row = np.zeros(n_boundary, dtype=np.int64)
    col = np.zeros(n_boundary, dtype=np.int64)
    data = np.zeros(n_boundary, dtype=np.float64)
    rhs = np.zeros(n_cells, dtype=np.float64)

    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        bc_type = mesh.boundary_types[f, 0]
        bc_val = mesh.boundary_values[f, component_idx]
        P = mesh.owner_cells[f]
        
        # Compute geometry
        S_b = mesh.vector_S_f[f]
        mag_S_b = (S_b[0]**2 + S_b[1]**2)**0.5
        d_Cb = mesh.d_Cb[f]
        n = S_b / mag_S_b
        vec_Cb = d_Cb * n
        
        # Compute pressure at boundary
        grad_p = grad_pressure_field[P]
        p_b = pressure_field[P] + np.dot(grad_p, vec_Cb)

        # Compute fluxes
        diffFlux_P_b, diffFlux_N_b = compute_boundary_diffusive_correction(
            f, u_field, grad_phi, mesh, mu, p_b, bc_type, bc_val, component_idx
        )
        convFlux_P_b, convFlux_N_b = compute_boundary_convective_flux(
            f, mesh, rho, mdot, u_field, phi, p_b, bc_type, bc_val, component_idx
        )

        # Store matrix entries
        row[i] = P
        col[i] = P
        data[i] = float(diffFlux_P_b) + float(convFlux_P_b)

        # Update RHS
        if bc_type != BC_OUTLET:
            rhs[P] -= (float(diffFlux_N_b) + float(convFlux_N_b))

    return row, col, data, rhs

@njit(fastmath=True, cache=True)
def assemble_diffusion_convection_matrix(
    mesh,
    mdot,
    grad_phi,
    u_field,
    rho,
    mu,
    component_idx,
    phi,
    scheme="Upwind",
    limiter=None,
    pressure_field = None,
    grad_pressure_field = None,
):
    """Assemble sparse matrix and RHS for a collocated FV discretisation.

    The implementation avoids Python‐level dynamic containers, which drastically
    reduces overhead inside Numba-JIT code and is a prerequisite for safe
    parallelisation with ``prange``.  We pessimistically over-allocate the
    *triplet* (COO) arrays and trim the excess at the end – a single pass keeps
    the code compact while still eliminating all ``append/extend`` operations.

    Parameters
    ----------
    mesh : Mesh
        Mesh object with *internal_faces*, *boundary_faces*, *owner_cells*, …
    grad_phi : ndarray
        Cell-centred gradients of the transported scalar.
    u_field : ndarray
        Face-centred velocity field.
    rho, mu : float
        Density and dynamic viscosity (constant).
    component_idx : int
        Index of the scalar component handled by this call.
    phi : ndarray
        Cell values of the transported scalar.
    beta : float
        Blending factor for deferred-correction convection.

    Returns
    -------
    row, col, data : ndarray
        Triplet format describing the sparse coefficient matrix.
    b : ndarray
        RHS vector.
    """

    n_cells     = mesh.cell_volumes.shape[0]
    n_internal  = mesh.internal_faces.shape[0]
    n_boundary  = mesh.boundary_faces.shape[0]

    # Pre-allocate arrays with exact sizes
    max_nnz = 4 * n_internal + n_boundary
    row  = np.zeros(max_nnz, dtype=np.int64)
    col  = np.zeros(max_nnz, dtype=np.int64)
    data = np.zeros(max_nnz, dtype=np.float64)
    b = np.zeros(n_cells, dtype=np.float64)

    # --- Internal faces (parallel) ---
    int_row, int_col, int_data, rhs = _assemble_internal_faces_parallel(
        mesh, mdot, grad_phi, u_field, rho, mu, component_idx, phi, scheme, limiter
    )
    idx = 0
    for i in range(n_internal * 4):
        row[idx] = int_row[i]
        col[idx] = int_col[i]
        data[idx] = int_data[i]
        idx += 1
    for i in range(n_cells):
        b[i] += rhs[i]

    # --- Boundary faces (parallel) ---
    bnd_row, bnd_col, bnd_data, bnd_rhs = _assemble_boundary_faces_parallel(
        mesh, mdot, grad_phi, u_field, rho, mu, component_idx, phi, pressure_field, grad_pressure_field
    )
    for i in range(n_boundary):
        row[idx] = bnd_row[i]
        col[idx] = bnd_col[i]
        data[idx] = bnd_data[i]
        idx += 1
    for i in range(n_cells):
        b[i] += bnd_rhs[i]

    # ––– trim overallocation –––––––––––––––––––––––––––––––––––––––––––––––
    return row[:idx], col[:idx], data[:idx], b
