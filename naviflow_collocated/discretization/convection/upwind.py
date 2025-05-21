import numpy as np
from numba import njit

# Boundary condition identifiers
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_CONVECTIVE = 4
BC_ZEROGRADIENT = 3
BC_WALL = 0

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
    f, mesh, rho,u_field, grad_phi, component_idx,
    phi, scheme="Upwind", limiter="MUSCL"
):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    g_f = mesh.face_interp_factors[f]
    Sf = np.ascontiguousarray(mesh.vector_S_f[f])
    Ef = np.ascontiguousarray(mesh.vector_E_f[f])
    d_CE = np.ascontiguousarray(mesh.vector_d_CE[f])
    d_skew = np.ascontiguousarray(mesh.vector_skewness[f])

    u_face = (1 - g_f) * u_field[P] + g_f * u_field[N]
    mdot = rho * np.dot(u_face, Sf)
    


    aP = -max(0, mdot)
    aN = -max(0, -mdot)
    b_corr = 0.0

    # stuff for TVD and other HO schemes
    phi_P = phi[P]
    phi_N = phi[N]
    F_low = mdot * (phi_P if mdot >= 0 else phi_N)

    gradC = grad_phi[P]
    gradN = grad_phi[N]
    grad_f = g_f * gradN + (1 - g_f) * gradC
    grad_f_mark = grad_f + np.dot(grad_f, d_skew)
    d_Cf = d_CE * g_f


    if scheme == "TVD":  
        # Compute the limiter
        phi_W = 2 * phi_P - phi_N
        r = (phi_N - phi_P) / (phi_P - phi_W + 1e-12)
        if limiter == "MUSCL":
            psi = MUSCL(r)
        elif limiter == "OSPRE":
            psi = OSPRE(r)
        elif limiter == "H_Cui":
            psi = H_Cui(r)

        # Apply the limiter
        phi_HO = phi_P + psi * np.dot(grad_f_mark , d_Cf)
        F_high = mdot * phi_HO
        b_corr = (F_high - F_low)
    elif scheme == "Upwind": 
        b_corr = 0.0 
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
        F_high = mdot * phi_HO
        b_corr = (F_high - F_low)


    return aP, aN, -b_corr

@njit(inline="always")
def compute_boundary_convective_flux(f, mesh, rho, u_field, phi, bc_type, bc_value, component_idx):
    """
    First-order upwind boundary convection flux for a specific velocity component.
    Skewness correction is ignored at boundaries.
    """
    P = mesh.owner_cells[f]
    Sf = np.ascontiguousarray(mesh.vector_S_f[f])
    u_boundary = np.ascontiguousarray(mesh.boundary_values[f, :2])
    u_P= np.ascontiguousarray(u_field[P])
    phi_P = np.ascontiguousarray(phi[P])


    mdot_boundary = rho * np.dot(u_boundary, np.ascontiguousarray(Sf))
    mdot_boundary = -max(0.0, -mdot_boundary)
    mdot_P = rho * np.dot(u_P, np.ascontiguousarray(Sf))
    mdot_P = -max(0.0, -mdot_P)

    if bc_type == BC_DIRICHLET:
        return mdot_boundary, -mdot_boundary * (2*phi_P[component_idx] - bc_value)
        """
        if mdot_boundary > 0:  # outflow (confirmed)
            return 0.0, 0.0#mdot_boundary* bc_value #mdot, -mdot * bc_value   # outflow 
        else: # inflow
            return mdot_P, -mdot_boundary * bc_value #mdot, -mdot * bc_value   # outflow 
        """
    elif bc_type == BC_ZEROGRADIENT:
        return 0.0, 0.0
    elif bc_type == BC_NEUMANN:
        return 0.0, 0.0
    elif bc_type == BC_CONVECTIVE:
        return 0.0, mdot_boundary * bc_value
    else:
        return 0.0, 0.0