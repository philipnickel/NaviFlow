import numpy as np
from numba import njit

EPS = 1.0e-14

BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 4
BC_NEUMANN = 3

# ──────────────────────────────────────────────────────────────────────────────
# Internal faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always")
def compute_diffusive_flux_matrix_entry(f, grad_phi, mesh, mu):
    """
    Over‑relaxed implicit conductance for one internal face.

    Parameters
    ----------
    f : int
        Face index.
    mesh : MeshData2D
        Pre‑computed geometric data (contains vector_E_f, vector_d_CE, etc.).
    mu : float or array_like
        Diffusion coefficient Γ.  If a scalar, assumed uniform; if an array, face‑based.

    Returns
    -------
    P : int
        Owner cell index.
    N : int
        Neighbour cell index (≥0 for internal faces).
    D_f : float
        Positive conductance that multiplies (φ_N − φ_P) in the matrix
        stencil.  
    """

    P   = mesh.owner_cells[f]
    N   = mesh.neighbor_cells[f]
    mu_f = mu 

    E_f  = mesh.vector_E_f[f]          # orthogonal implicit conductance
    d_CE = mesh.vector_d_CE[f]

    E_mag = np.linalg.norm(E_f) 
    d_mag = np.linalg.norm(d_CE) 

    # ---- over‑relaxed orthogonal conductance (Eq 8.58) --------------------
    # |E_f|  = projection length of S_f on d_CE after over‑relaxed scaling
    geoDiff = E_mag / d_mag
    Flux_P_f = mu_f * geoDiff
    Flux_N_f = -mu_f * geoDiff

    return Flux_P_f, Flux_N_f


@njit(inline="always")
def compute_diffusive_correction(f, grad_phi, mesh, mu):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]
    muF = mu 
    T_f = np.ascontiguousarray(mesh.vector_T_f[f])

    # Compute cross-diffusion term
    grad_P = grad_phi[P]
    grad_N = grad_phi[N]
    g_f = mesh.face_interp_factors[f]
    grad_f = (1 - g_f) * grad_P + g_f * grad_N

    d_skew = np.ascontiguousarray(mesh.vector_skewness[f]) 
    grad_f_mark = grad_f + np.dot(grad_f, d_skew) 

    # Moukalled 15.72
    diffDC = -muF * np.dot(grad_f_mark, T_f)
    return diffDC

# ──────────────────────────────────────────────────────────────────────────────
# Boundary faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always")
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
    elif bc_type == BC_NEUMANN:
        E_mag = np.linalg.norm(E_f) + EPS
        diffFlux_N_b = -muF * bc_val * E_mag
    elif bc_type == BC_WALL:
        """
        P = mesh.owner_cells[f]
        Sf = np.ascontiguousarray(mesh.vector_S_f[f])
        n = Sf / np.linalg.norm(Sf)
        d_Cb = np.ascontiguousarray(mesh.d_Cb[f])
        d_Cb_vec = d_Cb * n
        u_b = mesh.boundary_values[f, 0]
        v_b = mesh.boundary_values[f, 1]

        # no slip wall moukalled 15.125
        d_orth = np.dot(d_Cb_vec, n)
        frac = (muF * np.linalg.norm(Sf)) / (d_orth + EPS)

        if component_idx == 0:  # u-momentum
            term = (1 - n[0]**2)
            cross_term = (U[P][1] - v_b) * n[1] * n[0]
            main_term = u_b * term
        elif component_idx == 1:  # v-momentum
            term = (1 - n[1]**2)
            cross_term = (U[P][0] - u_b) * n[0] * n[1]
            main_term = v_b * term

        diffFlux_P_b = frac * term
        diffFlux_N_b = -frac * (main_term - cross_term)
        """
        E_mag = np.linalg.norm(E_f)
        diffFlux_P_b = muF * E_mag / (d_PB)
        diffFlux_N_b = -diffFlux_P_b * bc_val  # explicit orthogonal part

        # --- explicit non-orthogonal correction (FluxV_b) ---
        grad_P = grad_phi[P]
        d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        grad_P_mark = grad_P + np.dot(grad_P, d_skew)
        fluxVb = -muF * np.dot(grad_P_mark, T_f)
        diffFlux_N_b += fluxVb

        P = mesh.owner_cells[f]
        Sf = np.ascontiguousarray(mesh.vector_S_f[f])
        n = Sf / (np.linalg.norm(Sf) + EPS)
        S_mag = np.linalg.norm(Sf)
        d_Cb = np.ascontiguousarray(mesh.d_Cb[f])
        d_Cb_vec = d_Cb * n
        d_orth = np.dot(d_Cb_vec, n)

        # Implicit contribution
        a_wall = muF * S_mag / (d_orth + EPS)

        # Matrix entry
        #diffFlux_P_b = a_wall

        # RHS contribution from current value (deferred correction)
        #u_C = #U[P][component_idx]
        #diffFlux_N_b -= a_wall * bc_val 


    elif bc_type == BC_INLET:
        E_mag = np.linalg.norm(E_f)
        diffFlux_P_b = muF * E_mag / (d_PB)
        diffFlux_N_b = diffFlux_P_b * bc_val  # explicit orthogonal part

        # --- explicit non-orthogonal correction (FluxV_b) Moukalled 8.80 ---
        grad_P = grad_phi[P]
        d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        grad_P_mark = grad_P + np.dot(grad_P, d_skew)
        fluxVb = -muF * np.dot(grad_P_mark, T_f)
        diffFlux_N_b += fluxVb 
    elif bc_type == BC_OUTLET:
        E_mag = np.linalg.norm(E_f) 
        diffFlux_P_b = muF * E_mag / (d_PB )
        diffFlux_N_b = -diffFlux_P_b * bc_val  # explicit orthogonal part

        # --- explicit non-orthogonal correction (FluxV_b) ---
        grad_P = grad_phi[P]
        d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        grad_P_mark = grad_P + np.dot(grad_P, d_skew)
        fluxVb = -muF * np.dot(grad_P_mark, T_f)
        diffFlux_N_b += fluxVb 



    return diffFlux_P_b, diffFlux_N_b
