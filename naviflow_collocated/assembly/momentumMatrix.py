import numpy as np
from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_diffusive_flux_matrix_entry,
    compute_diffusive_correction,
    compute_boundary_diffusive_correction,
    compute_diffusive_skew_correction,
)
from naviflow_collocated.discretization.convection.upwind import (
    compute_convective_stencil_upwind,
    compute_boundary_convective_flux,
)

def assemble_diffusion_convection_matrix(mesh, phi, grad_phi,
                                         rho, mu, u_field):
    n_cells = mesh.cell_volumes.shape[0]
    row, col, data = [], [], []
    b = np.zeros(n_cells)

    # ───────────────────────── internal faces ───────────────────────────────
    for f in mesh.internal_faces:
        P, N, D_f = compute_diffusive_flux_matrix_entry(f, mesh, mu)

        row.extend([P, P, N, N])
        col.extend([P, N, N, P])
        data.extend([ D_f, -D_f,  D_f, -D_f ])

        # explicit non‑orthogonal part — split ½ to each cell
        _, _, bcorr = compute_diffusive_correction(f, grad_phi, mesh, mu)
        b[P] -= 0.5 * bcorr
        b[N] -= 0.5 * bcorr

        # --- NEW: skewness correction --------------------------------------------
        _, _, bskew = compute_diffusive_skew_correction(f, grad_phi, mesh, mu)
        b[P] -= 0.5 *  bskew
        b[N] -= 0.5 * bskew

                # convection (if present)
        if rho != 0.0:
            aP, aN, b_conv = compute_convective_stencil_upwind(
                f, phi, grad_phi, mesh, rho, u_field)

            if aP: row.append(P); col.append(P); data.append(aP)
            if aN: row.append(P); col.append(N); data.append(aN)

            b[P] -= b_conv        # upwind flux goes to owner only

    # ───────────────────────── boundary faces ───────────────────────────────
    for f in mesh.boundary_faces:
        bc_type  = mesh.boundary_types[f, 0]
        bc_val   = mesh.boundary_values[f, 0]

        P, a_P, b_P = compute_boundary_diffusive_correction(
            f, phi, grad_phi, mesh, mu, bc_type, bc_val)

        if a_P: row.append(P); col.append(P); data.append(a_P)
        b[P] -= b_P

        # boundary convection (e.g. outlet Neumann)
        if rho != 0.0:
            aP_cnv, _, b_cnv = compute_boundary_convective_flux(
                f, phi, grad_phi, mesh, rho, u_field, bc_type, bc_val)

            if aP_cnv: row.append(P); col.append(P); data.append(aP_cnv)
            b[P] -= b_cnv

    return row, col, data, b
