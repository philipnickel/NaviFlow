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


@njit
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

    # ––– pessimistic non-zero count ––––––––––––––––––––––––––––––––––––––––
    # internal face: 4 (conv) + 4 (diff) ≤ 8
    # boundary face: ≤ 2 (diff + conv)
    max_nnz = 8 * n_internal + 3 * n_boundary
    row  = np.zeros(max_nnz, dtype=np.int64)
    col  = np.zeros(max_nnz, dtype=np.int64)
    data = np.zeros(max_nnz, dtype=np.float64)


   
    idx  = 0  # running write position
    b = np.zeros(n_cells, dtype=np.float64)

    #===============================================
    #========== 1. Face Fluxes ================
    #===============================================

    # ––– internal faces ––––––––––––––––––––––––––––––––––––––––––––––––––––
    for i in range(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

    # —— convection term (upwind) ——
        convFlux_P_f, convFlux_N_f, convDC= compute_convective_stencil(
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

     

        row[idx] = P; col[idx] = P; data[idx] = Flux_P_f; idx += 1
        row[idx] = P; col[idx] = N; data[idx] = Flux_N_f ; idx += 1
        row[idx] = N; col[idx] = N; data[idx] = -Flux_N_f ; idx += 1
        row[idx] = N; col[idx] = P; data[idx] = -Flux_P_f ; idx += 1

        b[P] -= Flux_V_f #- Flux_T_f
        b[N] += Flux_V_f #+ Flux_T_f

    # ––– boundary faces ––––––––––––––––––––––––––––––––––––––––––––––––––––
    for i in range(n_boundary):
        f        = mesh.boundary_faces[i]
        bc_type  = mesh.boundary_types[f, 0]
        #bc_name = ["Wall", "Dirichlet", "Inlet", "Outlet", "Obstacle"][bc_type]
        #print(f"Boundary type for cell with coordinates: {mesh.cell_centers[P]} is: {bc_name} (type {bc_type})")
        bc_val   = mesh.boundary_values[f, component_idx]
        P = mesh.owner_cells[f]
        S_b = np.ascontiguousarray(mesh.vector_S_f[f])
        E_f = np.ascontiguousarray(mesh.vector_E_f[f])
        T_f = np.ascontiguousarray(mesh.vector_T_f[f])
        mag_S_b = np.linalg.norm(S_b)
        mag_E_f = np.linalg.norm(E_f) + EPS
        d_Cb = mesh.d_Cb[f]
        n = S_b / mag_S_b
        vec_Cb = d_Cb * n
        uv_b = mesh.boundary_values[f]
        grad_p = np.ascontiguousarray(grad_pressure_field[P])
        p_b = pressure_field[P] + np.dot(grad_p, vec_Cb)
 
        
        diffFlux_P_b, diffFlux_N_b = compute_boundary_diffusive_correction(
            f, u_field, grad_phi, mesh, mu,  p_b,  bc_type, bc_val, component_idx
        )

        convFlux_P_b, convFlux_N_b = compute_boundary_convective_flux(
            f, mesh, rho, mdot, u_field, phi, p_b, bc_type, bc_val, component_idx
        )
        
        row[idx] = P; col[idx] = P; data[idx] = +diffFlux_P_b + convFlux_P_b ; idx += 1
        if bc_type == BC_OUTLET:
            #b[P] = -p_b * mag_S_b
            continue

        b[P] -= diffFlux_N_b + convFlux_N_b
        
        

    # ––– trim overallocation –––––––––––––––––––––––––––––––––––––––––––––––
    return row[:idx], col[:idx], data[:idx], b