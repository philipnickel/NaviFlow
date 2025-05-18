import numpy as np
from numba import njit

EPS = 1.0e-14

BC_DIRICHLET    = 1
BC_NEUMANN      = 2
BC_ZEROGRADIENT = 3
BC_CONVECTIVE   = 4
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

    E_mag = np.linalg.norm(E_f) + EPS
    d_mag = np.linalg.norm(d_CE) + EPS

    # ---- over‑relaxed orthogonal conductance (Eq 8.58) --------------------
    # |E_f|  = projection length of S_f on d_CE after over‑relaxed scaling
    D_f = mu_f * E_mag / d_mag

    return P, N, D_f


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
    b_corr = -muF * np.dot(grad_f_mark, T_f)
    return P, N, b_corr

# ──────────────────────────────────────────────────────────────────────────────
# Boundary faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always")
def compute_boundary_diffusive_correction(
        f, grad_phi, mesh, mu, bc_type, bc_val):
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
    a_P = 0.0
    b_P = 0.0

    E_f = np.ascontiguousarray(mesh.vector_E_f[f])
    T_f = np.ascontiguousarray(mesh.vector_T_f[f])
    d_PB = mesh.d_Cb[f]

    
    if bc_type == BC_DIRICHLET:
        E_mag = np.linalg.norm(E_f) + EPS
        a_P = muF * E_mag / (d_PB + EPS)
        b_P = -a_P * bc_val  # explicit orthogonal part

        # --- explicit non-orthogonal correction (FluxV_b) ---
        grad_P = grad_phi[P]
        d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        grad_P_mark = grad_P + np.dot(grad_P, d_skew)
        fluxVb = -muF * np.dot(grad_P_mark, T_f)
        b_P += fluxVb 
   
        
    elif bc_type == BC_NEUMANN:
        E_mag = np.linalg.norm(E_f) + EPS
        b_P = -muF * bc_val * E_mag
       
    
    elif bc_type == BC_ZEROGRADIENT:
        # Zero gradient (Neumann with zero flux): no contribution to matrix or RHS
        a_P = 0.0
        b_P = 0.0
    
    # Default fallback for any other boundary condition
    else:
        a_P = 0.0
        b_P = 0.0

    return P, a_P, b_P
