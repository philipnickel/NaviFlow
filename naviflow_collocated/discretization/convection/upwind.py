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
    f, mesh, rho, u_field, grad_phi, component_idx,
    phi, beta=1.0, scheme="quick"
):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    g_f = mesh.face_interp_factors[f]
    Sf = np.ascontiguousarray(mesh.vector_S_f[f])
    d_CE = mesh.vector_d_CE[f]

    u_face = (1 - g_f) * u_field[P] + g_f * u_field[N]
    mdot = rho * np.dot(u_face, Sf)

    phi_P = phi[P]
    phi_N = phi[N]
    phi_f_U = phi_P if mdot >= 0 else phi_N

    if scheme == "powerlaw":
        # Compute diffusive conductance D_f
        d_PN = np.linalg.norm(d_CE)
        mu = 1.0  # Assume unit viscosity; can generalize later
        D_f = mu * np.linalg.norm(Sf) / (d_PN + 1e-14)
        Pe_f = mdot / (D_f + 1e-14)
        f_Pe = max(0.0, (1.0 - 0.1 * abs(Pe_f)) ** 5)
        phi_f_PL = phi_P + f_Pe * (phi_N - phi_P)
        F_low = mdot * phi_f_PL
    else:
        # Standard upwind
        F_low = mdot * phi_f_U

    # Deferred correction using QUICK
    phiC = phi[P]
    gradC = grad_phi[P]
    gradN = grad_phi[N]
    gf = mesh.face_interp_factors[f]
    grad_f = gf * gradN + (1 - gf) * gradC
    d_skew = np.ascontiguousarray(mesh.vector_skewness[f])
    grad_f_mark = grad_f + np.dot(grad_f, d_skew)
    phi_HO = phiC + 0.5 * np.dot(gradC + grad_f_mark, d_CE)
    F_high = mdot * phi_HO

    # Matrix coefficients (still from upwind splitting)
    aP = -max(0, mdot)
    aN = -max(0, -mdot)
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
    e_hat = np.ascontiguousarray(mesh.unit_vector_e[f])

    # Interpolate velocity component at the face
    if bc_type == BC_DIRICHLET:
        phi_f = mesh.boundary_values[f, component_idx]
    else:
        phi_f = u_field[P, component_idx]

    mdot = rho * np.dot(Sf, u_field[P])

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
