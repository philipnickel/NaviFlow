import numpy as np
from numba import njit

EPS = 1.0e-14

BC_DIRICHLET    = 1
BC_NEUMANN      = 2
BC_ZEROGRADIENT = 3

# ──────────────────────────────────────────────────────────────────────────────
# Internal faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always")
def compute_diffusive_flux_matrix_entry(f, mesh, mu):
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

    Notes
    -----
    The sign convention is handled by the assembler: it adds +D_f to the
    diagonal of each cell and −D_f to the off‑diagonal, yielding a
    symmetric negative‑definite Laplacian block.
    """
    P   = mesh.owner_cells[f]
    N   = mesh.neighbor_cells[f]
    mu_f = mu 

    E_f  = mesh.vector_E_f[f]          # over‑relaxed projection along d_CE
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
    T_f = mesh.vector_T_f[f]

    # True interpolation factor using intersection point f' (Eq 8.89)
    x_f      = mesh.face_centers[f]
    d_skew   = mesh.vector_skewness[f]            # f – f′
    x_fprime = x_f - d_skew                        # intersection of PN with face
    x_P      = mesh.cell_centers[P]
    x_N      = mesh.cell_centers[N]

    d_PN     = x_N - x_P
    d_PN_mag = np.linalg.norm(d_PN) + EPS
    e_hat    = d_PN / d_PN_mag                     # unit vector along C→N

    # distance from P to intersection point f′ along PN
    delta_Pf = np.dot(x_fprime - x_P, e_hat)
    g_f      = delta_Pf / d_PN_mag                 # true interpolation factor [0,1]

    # Interpolate gradient using corrected weights
    grad_fmark = (1.0 - g_f) * grad_phi[P] + g_f * grad_phi[N]

    # Apply skewness correction (now uses actual face position)
    d_skew = mesh.vector_skewness[f]
    e_hat = mesh.unit_vector_e[f]
    
    # Project gradient onto skewness direction
    scalar = np.dot(d_skew, grad_fmark)
    grad_f = grad_fmark + scalar * e_hat

    # Compute cross-diffusion term
    b_corr = -muF * np.dot(grad_f, T_f)
    return P, N, b_corr

# ──────────────────────────────────────────────────────────────────────────────
# Boundary faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always")
def compute_boundary_diffusive_correction(
        f, phi, grad_phi, mesh, mu, bc_type, bc_val):
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

    E_f = mesh.vector_E_f[f]
    T_f = mesh.vector_T_f[f]
    d_PB = mesh.d_Cb[f]

    if bc_type == BC_DIRICHLET:
        E_mag = np.linalg.norm(E_f) + EPS
        a_P = muF * E_mag / (d_PB + EPS)
        b_P = -a_P * bc_val  # implicit orthogonal part

        # --- explicit non-orthogonal correction (FluxV_b) ---
        grad_P = grad_phi[P]
        fluxVb = -muF * np.dot(grad_P, T_f)
        b_P += fluxVb
    
    elif bc_type == BC_NEUMANN:
        S_mag = np.linalg.norm(mesh.vector_S_f[f]) + EPS
        b_P = muF * bc_val * S_mag
    
    elif bc_type == BC_ZEROGRADIENT:
        # Zero gradient (Neumann with zero flux): no contribution to matrix or RHS
        a_P = 0.0
        b_P = 0.0
    
    # Default fallback for any other boundary condition
    else:
        a_P = 0.0
        b_P = 0.0

    return P, a_P, b_P

