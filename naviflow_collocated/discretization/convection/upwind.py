import numpy as np
from numba import njit

# Boundary condition identifiers
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_CONVECTIVE = 4
BC_ZEROGRADIENT = 3
BC_WALL = 0

@njit(inline="always")
def compute_convective_stencil_upwind(
    f, mesh, rho,u_field, grad_phi, component_idx,
    phi, beta=1.0
):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    g_f = mesh.face_interp_factors[f]
    Sf = np.ascontiguousarray(mesh.vector_S_f[f])
    eF = np.ascontiguousarray(mesh.vector_E_f[f])
    d_CE = np.ascontiguousarray(mesh.vector_d_CE[f])

    u_face = (1 - g_f) * u_field[P] + g_f * u_field[N]
    mdot = rho * np.dot(u_face, Sf)


    aP = -max(0, mdot)
    aN = -max(0, -mdot)

    phi_P = phi[P]
    phi_N = phi[N]
    F_low = mdot * (phi_P if mdot >= 0 else phi_N)

    if beta > 0:

        gradC = grad_phi[P]
        gradN = grad_phi[N]
        grad_f = g_f * gradN + (1 - g_f) * gradC
        d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        grad_f_mark = grad_f + np.dot(grad_f, d_skew)
        d_Cf = d_CE * g_f

        # Apply gradient coefficients properly in the calculation of phi_HO
        phi_HO = phi_P +  np.dot(gradC * 0.5 + grad_f_mark * 0.5, d_Cf)
        F_high = mdot * phi_HO
    else:
        F_high = F_low

    b_corr = -beta * (F_high - F_low)

    return aP, aN, b_corr

@njit(inline="always")
def compute_boundary_convective_flux(f, mesh, rho, u_field, bc_type, bc_value, component_idx):
    """
    First-order upwind boundary convection flux for a specific velocity component.
    Skewness correction is ignored at boundaries.
    """
    P = mesh.owner_cells[f]
    Sf = np.ascontiguousarray(mesh.vector_S_f[f])


    mdot = rho * np.dot(np.ascontiguousarray(Sf), u_field[P])

    # Inflow — depends on boundary type
    if bc_type == BC_DIRICHLET:
        if mdot <= 0:
            return  0.0, 0.0
        else:
            return mdot, -mdot * bc_value
    elif bc_type == BC_ZEROGRADIENT:
        return 0.0, 0.0
    elif bc_type == BC_NEUMANN:
        return 0.0, 0.0
    elif bc_type == BC_CONVECTIVE:
        return 0.0, mdot * bc_value
    else:
        return 0.0, 0.0