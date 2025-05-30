import numpy as np
from numba import njit
from naviflow_collocated.assembly.rhie_chow import compute_velocity_gradient_least_squares

BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 4
BC_NEUMANN = 3

@njit(inline="always")
def MUSCL(r):
    return max(0.0, min(2.0, 2.0 * r, 0.5 * (1 + r))) if r > 0 else 0.0

@njit(inline="always")
def OSPRE(r):
    return (3 * r * (r + 1)) / (2 * (r * r + r + 1 + 1e-12)) if r > 0 else 0.0

@njit(inline="always")
def H_Cui(r):
    return (3 * (r + abs(r))) / (2 * (r + 2 + 1e-12)) if r > 0 else 0.0

@njit(inline="always")
def compute_convective_stencil(
    f, mesh, rho, mdot, u_field, grad_phi, component_idx,
    phi, scheme="Upwind", limiter=None
):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    g_f = mesh.face_interp_factors[f]
    d_CE = np.ascontiguousarray(mesh.vector_d_CE[f])
    d_skew = np.ascontiguousarray(mesh.vector_skewness[f])

    # Moukalled 15.72 (negative sign for neighbor handled in matrix assembly)
    Flux_P_f = max(mdot[f], 0)
    Flux_N_f = -max(-mdot[f],0) 


    # stuff for TVD and other HO schemes
    phi_P = phi[P]
    phi_N = phi[N]
    F_low = mdot[f] * (phi_P if mdot[f]  >= 0 else phi_N)

    gradC = grad_phi[P]
    gradN = grad_phi[N]
    grad_f = g_f * gradN + (1 - g_f) * gradC
    grad_f_mark = grad_f + np.dot(grad_f, d_skew)
    d_Cf = d_CE * g_f


    if scheme == "TVD":  
        # Compute the limiter
        if limiter is None:
            psi = 1.0 # numba type safeguard
        #phi_W = 2 * phi_N - phi_P
        phi_W = 2 * phi_P - phi_N
        #r = (phi_N - phi_W )/(phi_N - phi_P + 1e-12) 
        r = (phi_N - phi_P )/(phi_P - phi_W + 1e-12) 
        if limiter == "MUSCL":
            psi = MUSCL(r)
        elif limiter == "OSPRE":
            psi = OSPRE(r)
        elif limiter == "H_Cui":
            psi = H_Cui(r)

        # Apply the limiter
        phi_HO = phi_P + psi * np.dot(grad_f_mark , d_Cf)
        F_high = mdot[f] * phi_HO
        convDC = (F_high - F_low)
    elif scheme == "Upwind": 
        convDC = 0.0 
    elif scheme != "Upwind":
        # set coefficients
        if scheme == "Central difference":
            a = 0.0
            b = 1.0
        elif scheme == "SOU":
            a = 2.0
            b = -1.0
        elif scheme == "QUICK":
            a = 0.5
            b = 0.5
        # Compute the high order term
        phi_HO = phi_P +  np.dot(gradC * a + grad_f_mark * b, d_Cf)
        F_high = mdot[f] * phi_HO
        convDC = (F_high - F_low)
    


    return Flux_P_f, Flux_N_f, convDC

@njit(inline="always")
def compute_boundary_convective_flux(f, mesh, rho, mdot, u_field, phi, p_b, bc_type, bc_value, component_idx):
    """
    First-order upwind boundary convection flux for a specific velocity component.
    Skewness correction is ignored at boundaries.
    """
    P = mesh.owner_cells[f]
    Sf = np.ascontiguousarray(mesh.vector_S_f[f])
    E_f = np.ascontiguousarray(mesh.vector_E_f[f])
    d_Cb = np.ascontiguousarray(mesh.d_Cb[f])
    e = E_f / np.linalg.norm(E_f)
    d_Cb_vec = d_Cb * e
    u_boundary = np.ascontiguousarray(mesh.boundary_values[f, :2])
    phi_P = phi[P]


    mdot_boundary = rho * np.dot(u_boundary, np.ascontiguousarray(Sf))
    mdot_boundary = max(0.0, -mdot_boundary)
    flux = mdot[f]
    phi_B = bc_value  # Dirichlet BC at the boundary

    Flux_C_b = max(mdot[f], 0)
    Flux_N_b = -max(-mdot[f],0) # ghost cell 

    if bc_type == BC_DIRICHLET:
        return 0.0, 0.0#Flux_C_b, Flux_N_b *(2*phi_P-bc_value) #Flux_C_b, -Flux_N_b *bc_value #- 2*phi_P) (only used for MMS tests)
    elif bc_type == BC_NEUMANN:
        return 0.0, 0.0
    elif bc_type == BC_INLET:
        return Flux_C_b, Flux_N_b * (2*phi_P-bc_value) #- 2*phi_P)
    elif bc_type == BC_OUTLET:
        grad_v_b = compute_velocity_gradient_least_squares(mesh, u_field, u_field, mesh.face_centers[f], u_field[P], P, f)
        v_b = u_field[P] + np.dot(grad_v_b, d_Cb_vec)
        term1 = mdot[f] * (v_b - u_field[P])
        term2 = mdot[f] * (2*phi_P - v_b[component_idx])
        term3 = Sf * p_b 
        return -mdot[f], term1[component_idx] + term2 - term3[component_idx]
    elif bc_type == BC_WALL:
        return 0.0, 0.0#-Sf * p_b