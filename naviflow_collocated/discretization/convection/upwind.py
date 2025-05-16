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
    f, mesh, rho,mu,  u_field, grad_phi, component_idx,
    phi, beta=1.0, HO_scheme="SOU"
):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    g_f = mesh.face_interp_factors[f]
    Sf = np.ascontiguousarray(mesh.vector_S_f[f])
    eF = np.ascontiguousarray(mesh.vector_E_f[f])
    d_CE = np.ascontiguousarray(mesh.vector_d_CE[f])

    u_face = (1 - g_f) * u_field[P] + g_f * u_field[N]
    mdot = rho * np.dot(u_face, Sf)

    phi_P = phi[P]
    phi_N = phi[N]

    # Define gradient coefficient dictionary based on HO_scheme
    grad_coeffs = {
        "upwind": np.array([1.0, 0.0]),
        "central_difference": np.array([0.5, 0.5]),
        "SOU": np.array([1.5, -0.5]),
        "FROMM": np.array([0.75, 0.25]),
        "QUICK": np.array([0.75, 0.25]),
        "downwind": np.array([0.0, 1.0]),
    }

    coeffs = grad_coeffs.get(HO_scheme.lower(), grad_coeffs["upwind"])

    aP = -max(0, mdot)
    aN = -max(0, -mdot)

    F_low = mdot * (phi_P if mdot >= 0 else phi_N)

    if beta > 0:
        phi_vals = np.array([phi_P, phi_N])
        phi_f = np.dot(coeffs, phi_vals)

        gradC = grad_phi[P]
        gradN = grad_phi[N]
        grad_f = g_f * gradN + (1 - g_f) * gradC
        d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
        grad_f_mark = grad_f + np.dot(grad_f, d_skew)

        # Apply gradient coefficients properly in the calculation of phi_HO
        phi_HO = phi_f + 0.5 * np.dot(gradC * coeffs[0] + grad_f_mark * coeffs[1], d_CE)
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
