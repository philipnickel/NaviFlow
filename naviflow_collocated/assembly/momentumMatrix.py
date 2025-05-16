import numpy as np
from numba import njit, prange
from naviflow_collocated.discretization.diffusion.central_diff import (
compute_diffusive_flux_matrix_entry,
compute_diffusive_correction,
compute_boundary_diffusive_correction,
)
from naviflow_collocated.discretization.convection.upwind import (
compute_convective_stencil_upwind,
compute_boundary_convective_flux,
)

BC_DIRICHLET    = 1
BC_NEUMANN      = 2
BC_ZEROGRADIENT = 3
BC_CONVECTIVE   = 4

EPS = 1.0e-14

@njit
def assemble_diffusion_convection_matrix(mesh, grad_phi, u_field, 
    rho, mu, component_idx, phi, beta):



    n_cells = mesh.cell_volumes.shape[0]
    row, col, data = [], [], []
    b = np.zeros(n_cells)

    # Note: Full parallelization requires restructuring away from dynamic Python lists.
    # The loops below are marked prange-ready but still use Python lists and thus are not fully parallel.

    n_internal = len(mesh.internal_faces)
    for i in range(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        # ---------------- convection term (unchanged) ----------------------
        if rho != 0.0:
            a_P, a_N, b_corr = compute_convective_stencil_upwind(
                f, mesh, rho, u_field, grad_phi, component_idx, phi, beta)
            row.extend([P, P, N, N])
            col.extend([P, N, N, P])
            data.extend([a_P, -a_P, a_N, -a_N])

            b[P] -= b_corr
            b[N] += b_corr
        
        # -------- orthogonal (over‑relaxed) Laplacian ----------------------
        P, N, D_f = compute_diffusive_flux_matrix_entry(f, grad_phi, mesh, mu)
        """
        # Compute Peclet number
        Peclet_term = 0.1 * np.abs(a_P/D_f) 
        base = np.maximum(0, 1 - Peclet_term)
        result = np.where(np.abs(a_P) > 1e-10, base**5, 0.0)
        result = np.nan_to_num(result, nan=0.0)

        power_law = False
        if power_law: 
            diff = D_f * result
        else:
        """
        diff = D_f

        row.extend([P, P, N, N])
        col.extend([P, N, N, P])
        data.extend([ diff, -diff, diff, -diff ])


        # -------- cross‑diffusion (non‑orth) ------------------------------
        # explicit part (distributed symmetrically between P and N)
        _, _, bcorr = compute_diffusive_correction(f, grad_phi, mesh, mu)
        b[P] -=   bcorr
        b[N] +=   bcorr


    n_boundary = len(mesh.boundary_faces)
    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        bc_type = mesh.boundary_types[f, 0]
        bc_val = mesh.boundary_values[f, component_idx]

        P, a_P, b_P= compute_boundary_diffusive_correction(
        f, grad_phi, mesh, mu, bc_type, bc_val)

        if a_P: row.append(P); col.append(P); data.append(a_P)
        if b_P: b[P] -= b_P

        # boundary convection (e.g. outlet Neumann)
        if rho != 0.0:
            aP_cnv, b_cnv = compute_boundary_convective_flux(
                f, mesh, rho, u_field, bc_type, bc_val, component_idx)

            if aP_cnv != 0.0:
                row.append(P)
                col.append(P)
                data.append(aP_cnv)
            b[P] -= b_cnv

    
    return row, col, data, b