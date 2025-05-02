import numpy as np
from numba import njit, prange
from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_central_diffusion_face_coeffs,
)
from naviflow_collocated.discretization.convection.power_law import (
    compute_powerlaw_convection_face_coeffs,
)

_SMALL_DD_EPS = 1.0e-12


@njit(parallel=True)
def assemble_matrix_rhs_on_the_fly(
    n_cells,
    n_faces,
    owner_cells,
    neighbor_cells,
    face_areas,
    face_normals,
    cell_centers,
    face_centers,
    cell_volumes,  # <-- volumes
    phi,
    Gamma,
    rho_f,
    face_velocity,
    dirichlet_mask,
    dirichlet_values,
    gradients,
    include_diffusion=True,
    include_convection=True,
):
    MAX_PER_FACE = 8
    row_face = np.full((n_faces, MAX_PER_FACE), -1, np.int32)
    col_face = np.empty_like(row_face)
    dat_face = np.empty((n_faces, MAX_PER_FACE), np.float64)
    n_filled = np.zeros(n_faces, np.int32)

    rhs = np.zeros(n_cells, np.float64)  # shared, atomics below

    # ------------ parallel face loop ---------------------------------
    for f in prange(n_faces):
        C = owner_cells[f]
        F = neighbor_cells[f]
        if F != -1 and dirichlet_mask[C] and dirichlet_mask[F]:
            continue

        # Initialize coefficients to zero
        aC_d, aF_d, q_skew = 0.0, 0.0, 0.0
        dC_conv, oC_conv, dF_conv, oF_conv = 0.0, 0.0, 0.0, 0.0

        # Always compute diffusion coeffs if needed, primarily for q_skew
        if include_diffusion:
            aC_d, aF_d, q_skew = compute_central_diffusion_face_coeffs(
                f,
                cell_centers,
                face_centers,
                face_normals,
                face_areas,
                owner_cells,
                neighbor_cells,
                Gamma,  # Diffusion coefficient field
                gradients,
                phi,
            )

        # Compute convection coeffs if needed
        if include_convection:
            dC_conv, oC_conv, dF_conv, oF_conv = (
                compute_powerlaw_convection_face_coeffs(
                    f,
                    cell_centers,
                    face_centers,
                    face_normals,
                    face_areas,
                    owner_cells,
                    neighbor_cells,
                    rho_f,
                    face_velocity,
                    Gamma,  # Needed for Peclet number
                )
            )

        if F == -1:  # geometric boundary
            continue

        VC = cell_volumes[C]
        VF = cell_volumes[F]

        # Determine final coefficients based on flags
        final_diag_C = dC_conv if include_convection else aC_d
        final_off_C = oC_conv if include_convection else aF_d
        final_diag_F = (
            dF_conv if include_convection else aC_d
        )  # Note: aC_d is correct for F's diag diffusion
        final_off_F = (
            oF_conv if include_convection else aF_d
        )  # Note: aF_d is correct for F's off-diag diffusion

        cnt = 0
        # --- owner row -------------------------------------------------------
        if not dirichlet_mask[C]:
            row_face[f, cnt] = C
            col_face[f, cnt] = C
            dat_face[f, cnt] = final_diag_C / VC
            cnt += 1

            row_face[f, cnt] = C
            col_face[f, cnt] = F
            dat_face[f, cnt] = final_off_C / VC
            cnt += 1

        # --- neighbour row ---------------------------------------------------
        if not dirichlet_mask[F]:
            row_face[f, cnt] = F
            col_face[f, cnt] = F
            dat_face[f, cnt] = final_diag_F / VF  # Uses F's perspective
            cnt += 1

            row_face[f, cnt] = F
            col_face[f, cnt] = C
            dat_face[f, cnt] = final_off_F / VF  # Uses F's perspective
            cnt += 1

        # ---- RHS contributions (using standard +=) ----------------------
        # Skewness correction is always based on central difference calculation if diffusion included
        if include_diffusion:
            if not dirichlet_mask[C] and not dirichlet_mask[F]:
                rhs[C] += -0.5 * q_skew / VC
                rhs[F] += 0.5 * q_skew / VF
            elif dirichlet_mask[F] and not dirichlet_mask[C]:
                rhs[C] += (
                    -q_skew / VC
                )  # Apply full skew flux correction to non-Dirichlet side
            elif dirichlet_mask[C] and not dirichlet_mask[F]:
                rhs[F] += (
                    q_skew / VF
                )  # Apply full skew flux correction to non-Dirichlet side

        # Boundary condition RHS terms related to convection (if applicable)
        if include_convection:
            if dirichlet_mask[F] and not dirichlet_mask[C]:
                rhs[C] += -(oC_conv / VC) * dirichlet_values[F]
            elif dirichlet_mask[C] and not dirichlet_mask[F]:
                rhs[F] += -(oF_conv / VF) * dirichlet_values[C]

        n_filled[f] = cnt

    # ------------ flatten COO lists (serial) --------------------------
    total_face_nz = np.sum(n_filled)
    extra = 2 * np.sum(dirichlet_mask) + np.sum(~dirichlet_mask)
    nnz = total_face_nz + extra

    row = np.empty(nnz, np.int32)
    col = np.empty(nnz, np.int32)
    dat = np.empty(nnz, np.float64)

    p = 0
    for f in range(n_faces):
        m = n_filled[f]
        if m:
            row[p : p + m] = row_face[f, :m]
            col[p : p + m] = col_face[f, :m]
            dat[p : p + m] = dat_face[f, :m]
            p += m

    # -------- Dirichlet rows -----------------------------------------
    for i in range(n_cells):
        if dirichlet_mask[i]:
            row[p] = i
            col[p] = i
            dat[p] = 1.0
            rhs[i] = dirichlet_values[i]
            p += 1

    # -------- tiny ε for strict DD -----------------------------------
    for i in range(n_cells):
        if not dirichlet_mask[i]:
            row[p] = i
            col[p] = i
            dat[p] = _SMALL_DD_EPS / cell_volumes[i]
            p += 1

    return row[:p], col[:p], dat[:p], rhs
