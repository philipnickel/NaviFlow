import numpy as np
from numba import njit
from naviflow_collocated.discretization.convection.upwind import compute_velocity_gradient_least_squares

EPS = 1.0e-14

BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 3
BC_OBSTACLE = 4

# ──────────────────────────────────────────────────────────────────────────────
# Internal faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always", cache=True, fastmath=True)
def compute_diffusive_flux_matrix_entry(f, grad_phi, mesh, mu):
    """
    Over‑relaxed implicit conductance for one internal face.
    Optimized with pre-fetched mesh data and manual norm calculations.
    """
    # Pre-fetch connectivity and mesh data (better cache locality)
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]
    mu_f = mu 

    # Pre-fetch and cache vector components (single memory access per component)
    vector_E_f = mesh.vector_E_f[f]
    vector_d_CE = mesh.vector_d_CE[f]
    E_f_0 = vector_E_f[0]
    E_f_1 = vector_E_f[1]
    d_CE_0 = vector_d_CE[0]
    d_CE_1 = vector_d_CE[1]

    # Manual norm calculations (faster than np.linalg.norm)
    E_mag = np.sqrt(E_f_0 * E_f_0 + E_f_1 * E_f_1)
    d_mag = np.sqrt(d_CE_0 * d_CE_0 + d_CE_1 * d_CE_1)

    # Over‑relaxed orthogonal conductance (Eq 8.58)
    geoDiff = E_mag / d_mag
    Flux_P_f = mu_f * geoDiff
    Flux_N_f = -mu_f * geoDiff

    return Flux_P_f, Flux_N_f


@njit(inline="always", cache=True, fastmath=True)
def compute_diffusive_correction(f, grad_phi, mesh, mu):
    """
    Compute diffusive correction term with optimized memory access patterns.
    """
    # Pre-fetch connectivity
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]
    muF = mu 
    
    # Pre-fetch vector components (single array access each)
    vector_T_f = mesh.vector_T_f[f]
    vector_skewness = mesh.vector_skewness[f]
    T_f_0 = vector_T_f[0]
    T_f_1 = vector_T_f[1]
    d_skew_0 = vector_skewness[0]
    d_skew_1 = vector_skewness[1]

    # Pre-fetch gradient data (better cache locality)
    grad_phi_P = grad_phi[P]
    grad_phi_N = grad_phi[N]
    gradC_0 = grad_phi_P[0]
    gradC_1 = grad_phi_P[1]
    gradN_0 = grad_phi_N[0]
    gradN_1 = grad_phi_N[1]
    
    # Single access to interpolation factor
    g_f = mesh.face_interp_factors[f]
    grad_f_0 = (1.0 - g_f) * gradC_0 + g_f * gradN_0
    grad_f_1 = (1.0 - g_f) * gradC_1 + g_f * gradN_1

    # Skewness correction with pre-fetched components
    skew_dot = grad_f_0 * d_skew_0 + grad_f_1 * d_skew_1
    grad_f_mark_0 = grad_f_0 + skew_dot * d_skew_0
    grad_f_mark_1 = grad_f_1 + skew_dot * d_skew_1

    # Manual dot product (Moukalled 15.72)
    diffDC = -muF * (grad_f_mark_0 * T_f_0 + grad_f_mark_1 * T_f_1)
    return diffDC

# ──────────────────────────────────────────────────────────────────────────────
# Boundary faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always", cache=True, fastmath=True)
def compute_boundary_diffusive_correction(
        f,U, grad_phi, mesh, mu, p_b, bc_type, bc_val, component_idx):
    """
    Return (P, a_P, b_P)  —  everything is written to the owner cell only.

       a_P : diagonal coefficient to add
       b_P : RHS increment that will be **subtracted** (b[P]-=b_P)

    Supports:
    - BC_DIRICHLET
    - BC_NEUMANN
    - BC_ZEROGRADIENT
    """
    P = mesh.owner_cells[f]
    muF = mu 
    diffFlux_P_b = 0.0
    diffFlux_N_b = 0.0

    E_f = np.ascontiguousarray(mesh.vector_E_f[f])
    T_f = np.ascontiguousarray(mesh.vector_T_f[f])
    d_PB = mesh.d_Cb[f]

    
    if bc_type == BC_DIRICHLET:

        E_mag = np.linalg.norm(E_f)
        diffFlux_P_b = muF * E_mag / (d_PB)
        diffFlux_N_b = -diffFlux_P_b * bc_val  # explicit orthogonal part

        # --- explicit non-orthogonal correction (FluxV_b) ---
        grad_P = grad_phi[P]
        d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        grad_P_mark = grad_P + np.dot(grad_P, d_skew)
        fluxVb = -muF * np.dot(grad_P_mark, T_f)
        diffFlux_N_b += fluxVb
    
    elif bc_type == BC_WALL or bc_type == BC_OBSTACLE:
        # --- wall shear stress ---
        P = mesh.owner_cells[f]
        Sf = np.ascontiguousarray(mesh.vector_S_f[f])
        n = Sf / (np.linalg.norm(Sf) + EPS)
        S_mag = np.linalg.norm(Sf)
        d_Cb = np.ascontiguousarray(mesh.d_Cb[f])
        d_Cb_vec = d_Cb * n
        d_orth = np.dot(d_Cb_vec, n)

        # Get boundary values
        U_b = mesh.boundary_values[f]  # Boundary velocity
        U_C = U[P]  # Cell center velocity

        # Calculate wall conductance (Moukalled 15.125)
        frac = (muF * S_mag) / (d_orth + EPS) 

        if component_idx == 0:  # u-momentum
            term = (1 - n[0]**2)  # Tangential component for u
            # Use boundary values for cross-coupling
            cross_term = (U_C[1] - U_b[1]) * n[1] * n[0]  # Cross-coupling with v
            main_term = U_b[0] * (1 - n[0]**2)  # Wall velocity contribution
        elif component_idx == 1:  # v-momentum
            term = (1 - n[1]**2)  # Tangential component for v
            # Use boundary values for cross-coupling
            cross_term = (U_C[0] - U_b[0]) * n[0] * n[1]  # Cross-coupling with u
            main_term = U_b[1] * (1 - n[1]**2)  # Wall velocity contribution

        # Matrix coefficients for wall shear stress
        diffFlux_P_b += frac * term
        diffFlux_N_b += -frac * (main_term + cross_term) #- p_b * Sf[component_idx]

        # --- explicit non-orthogonal correction (FluxV_b) ---
        #grad_P = grad_phi[P]
        #d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        #grad_P_mark = grad_P + np.dot(grad_P, d_skew)
        #fluxVb = -muF * np.dot(grad_P_mark, T_f)
        #diffFlux_N_b += fluxVb

    elif bc_type == BC_INLET:
        E_mag = np.linalg.norm(E_f)
        diffFlux_P_b = muF * E_mag / (d_PB)
        diffFlux_N_b = -diffFlux_P_b * bc_val  # explicit orthogonal part

        # --- explicit non-orthogonal correction (FluxV_b) Moukalled 8.80 ---
        grad_P = grad_phi[P]
        d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        grad_P_mark = grad_P + np.dot(grad_P, d_skew)
        fluxVb = -muF * np.dot(grad_P_mark, T_f)
        diffFlux_N_b += fluxVb 
    elif bc_type == BC_OUTLET:
        grad_v_b = compute_velocity_gradient_least_squares(mesh, U, U, mesh.face_centers[f], U[P], P, f)
        P = mesh.owner_cells[f]
        Sf = np.ascontiguousarray(mesh.vector_S_f[f])
        E_f = np.ascontiguousarray(mesh.vector_E_f[f])
        e = E_f / np.linalg.norm(E_f)
        d_Pb_vec = d_PB * e
        v_b = U[P] + np.dot(grad_v_b, d_Pb_vec)
        E_mag = np.linalg.norm(E_f) 
        diffFlux_P_b = muF * E_mag / (d_PB )
        diffFlux_N_b = -diffFlux_P_b * v_b[component_idx] #v_b[component_idx] # explicit orthogonal part

        # --- explicit non-orthogonal correction (FluxV_b) ---
        grad_P = grad_phi[P]
        d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        grad_P_mark = grad_P + np.dot(grad_P, d_skew)
        fluxVb = -muF * np.dot(grad_P_mark, T_f)
        diffFlux_N_b += fluxVb 
        diffFlux_N_b = 0.0  
        diffFlux_P_b = 0.0
        ## no diffusive flux at outlet ferziger 8.10.2 Outlet - grid assumed to be orthogonal to the boundary (this may not be entirely true)
    

    return diffFlux_P_b, diffFlux_N_b
