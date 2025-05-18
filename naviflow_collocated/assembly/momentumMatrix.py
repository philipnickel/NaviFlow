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

BC_DIRICHLET    = 1
BC_NEUMANN      = 2
BC_ZEROGRADIENT = 3
BC_CONVECTIVE   = 4

EPS = 1.0e-14


@njit
def assemble_diffusion_convection_matrix(
    mesh,
    grad_phi,
    u_field,
    rho,
    mu,
    component_idx,
    phi,
    scheme="Upwind",
    limiter=None,
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
    max_nnz = 8 * n_internal + 2 * n_boundary

    row  = np.empty(max_nnz, dtype=np.int64)
    col  = np.empty(max_nnz, dtype=np.int64)
    data = np.empty(max_nnz, dtype=np.float64)
    idx  = 0  # running write position

    b = np.zeros(n_cells, dtype=np.float64)

    # ––– internal faces ––––––––––––––––––––––––––––––––––––––––––––––––––––
    for i in range(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        # —— convection term (upwind) ——
        if rho != 0.0:
            a_P, a_N, b_corr = compute_convective_stencil(
                f, mesh, rho, u_field, grad_phi, component_idx, phi, scheme=scheme, limiter=limiter
            )

            row[idx] = P; col[idx] = P; data[idx] =  a_P; idx += 1
            row[idx] = P; col[idx] = N; data[idx] = -a_P; idx += 1
            row[idx] = N; col[idx] = N; data[idx] =  a_N; idx += 1
            row[idx] = N; col[idx] = P; data[idx] = -a_N; idx += 1

            b[P] -= b_corr
            b[N] += b_corr

            # —— orthogonal diffusion ——
        if mu != 0.0:
            _, _, D_f = compute_diffusive_flux_matrix_entry(f, grad_phi, mesh, mu)

            row[idx] = P; col[idx] = P; data[idx] =  D_f; idx += 1
            row[idx] = P; col[idx] = N; data[idx] = -D_f; idx += 1
            row[idx] = N; col[idx] = N; data[idx] =  D_f; idx += 1
            row[idx] = N; col[idx] = P; data[idx] = -D_f; idx += 1

            # —— non-orthogonal correction (explicit) ——
            _, _, bcorr = compute_diffusive_correction(f, grad_phi, mesh, mu)
            b[P] -= bcorr
            b[N] += bcorr

    # ––– boundary faces ––––––––––––––––––––––––––––––––––––––––––––––––––––
    for i in range(n_boundary):
        f        = mesh.boundary_faces[i]
        bc_type  = mesh.boundary_types[f, 0]
        bc_val   = mesh.boundary_values[f, component_idx]
        P = mesh.owner_cells[f]

        if mu != 0.0:
            P, a_P, b_P = compute_boundary_diffusive_correction(
                f, grad_phi, mesh, mu, bc_type, bc_val
            )

            row[idx] = P; col[idx] = P; data[idx] = a_P; idx += 1
            b[P] -= b_P

        # —— boundary convection (e.g. outlet) ——
        if rho != 0.0:
            aP_cnv, b_cnv = compute_boundary_convective_flux(
                f, mesh, rho, u_field, bc_type, bc_val, component_idx
            )

            row[idx] = P; col[idx] = P; data[idx] = aP_cnv; idx += 1

            b[P] -= b_cnv

    # ––– trim overallocation –––––––––––––––––––––––––––––––––––––––––––––––
    return row[:idx], col[:idx], data[:idx], b
