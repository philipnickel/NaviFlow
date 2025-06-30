import numpy as np
from numba import njit

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

@njit(cache=True, fastmath=True, nogil=True, parallel=False, boundscheck=False, error_model='numpy')
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
    dt=0.0,
    transient=False,
    time_scheme="Euler"
):
    """Assemble sparse matrix and RHS for a collocated FV discretisation.
    
    Optimized for memory access patterns with pre-fetched static data.
    """

    n_cells     = mesh.cell_volumes.shape[0]
    n_internal  = mesh.internal_faces.shape[0]
    n_boundary  = mesh.boundary_faces.shape[0]

    # ––– pessimistic non-zero count ––––––––––––––––––––––––––––––––––––––––
    max_nnz = 8 * n_internal + 3 * n_boundary + n_cells
    row  = np.zeros(max_nnz, dtype=np.int64)
    col  = np.zeros(max_nnz, dtype=np.int64)
    data = np.zeros(max_nnz, dtype=np.float64)

    idx  = 0  # running write position
    b = np.zeros(n_cells, dtype=np.float64)

    # ═══ PRE-FETCH STATIC DATA (HUGE MEMORY ACCESS OPTIMIZATION) ═══
    # Internal face connectivity (static throughout simulation)
    internal_faces = mesh.internal_faces
    owner_cells = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    
    # Boundary face data (static)
    boundary_faces = mesh.boundary_faces
    boundary_types = mesh.boundary_types
    boundary_values = mesh.boundary_values

    # Determine scaling factor for spatial terms based on time scheme
    time_scheme_factor = 1.0
    if transient and time_scheme == "CrankNicolson":
        time_scheme_factor = 0.5

    # ––– internal faces (OPTIMIZED MEMORY ACCESS) –––––––––––––––––––––––––
    for i in range(n_internal):
        f = internal_faces[i]
        P = owner_cells[f]
        N = neighbor_cells[f]

        # —— convection term (upwind) ——
        convFlux_P_f, convFlux_N_f, convDC = compute_convective_stencil(
            f, mesh, rho, mdot, u_field, grad_phi, component_idx, phi, scheme=scheme, limiter=limiter
        )

        # —— orthogonal diffusion ——
        diffFlux_P_f, diffFlux_N_f = compute_diffusive_flux_matrix_entry(f, grad_phi, mesh, mu)
        # —— non-orthogonal correction (explicit) ——
        diffDC = compute_diffusive_correction(f, grad_phi, mesh, mu)

        # —— face fluxes —— Moukalled 15.72 ——
        Flux_P_f =  convFlux_P_f + diffFlux_P_f
        Flux_N_f =  convFlux_N_f + diffFlux_N_f
        Flux_V_f = convDC + diffDC 

        # Matrix assembly (using pre-fetched P, N)
        row[idx] = P; col[idx] = P; data[idx] = Flux_P_f * time_scheme_factor; idx += 1
        row[idx] = P; col[idx] = N; data[idx] = Flux_N_f * time_scheme_factor; idx += 1
        row[idx] = N; col[idx] = N; data[idx] = -Flux_N_f * time_scheme_factor; idx += 1
        row[idx] = N; col[idx] = P; data[idx] = -Flux_P_f * time_scheme_factor; idx += 1

        b[P] -= Flux_V_f * time_scheme_factor
        b[N] += Flux_V_f * time_scheme_factor

    # ––– boundary faces (OPTIMIZED MEMORY ACCESS) –––––––––––––––––––––––––
    for i in range(n_boundary):
        f = boundary_faces[i]
        bc_type = boundary_types[f, 0]
        bc_val = boundary_values[f, component_idx]
        P = owner_cells[f]
        
        # Pre-fetch geometric vectors (static data)
        S_b = np.ascontiguousarray(mesh.vector_S_f[f])
        E_f = np.ascontiguousarray(mesh.vector_E_f[f])
        T_f = np.ascontiguousarray(mesh.vector_T_f[f])
        mag_S_b = np.linalg.norm(S_b)
        mag_E_f = np.linalg.norm(E_f) + EPS
        d_Cb = mesh.d_Cb[f]
        n = S_b / mag_S_b
        vec_Cb = d_Cb * n
        
        # Boundary pressure calculation (using pre-fetched data)
        uv_b = boundary_values[f]
        grad_p = np.ascontiguousarray(grad_pressure_field[P])
        p_b = pressure_field[P] + np.dot(grad_p, vec_Cb)
 
        diffFlux_P_b, diffFlux_N_b = compute_boundary_diffusive_correction(
            f, u_field, grad_phi, mesh, mu,  p_b,  bc_type, bc_val, component_idx
        )

        convFlux_P_b, convFlux_N_b = compute_boundary_convective_flux(
            f, mesh, rho, mdot, u_field, phi, p_b, bc_type, bc_val, component_idx
        )
        
        row[idx] = P; col[idx] = P; data[idx] = (+diffFlux_P_b + convFlux_P_b) * time_scheme_factor; idx += 1
        if bc_type == BC_OUTLET:
            continue

        b[P] -= (diffFlux_N_b + convFlux_N_b) * time_scheme_factor

    if transient and dt > 0:
        # This loop adds the transient term contribution to the diagonal and source term
        for i in range(n_cells):
            transient_term_coeff = rho * mesh.cell_volumes[i] / dt
            row[idx] = i; col[idx] = i; data[idx] = transient_term_coeff; idx += 1
            b[i] += transient_term_coeff * phi[i]

    # ––– trim overallocation –––––––––––––––––––––––––––––––––––––––––––––––
    return row[:idx], col[:idx], data[:idx], b