import numpy as np
from numba import njit

# Boundary condition identifiers
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_CONVECTIVE = 4
BC_ZEROGRADIENT = 3
BC_WALL = 0

@njit(inline="always")
def compute_convective_stencil_upwind(f, phi, grad_phi, mesh, rho, u_field):
    """
    First-order upwind convection scheme with skewness correction for interior faces.
    """
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    Sf = mesh.vector_S_f[f]
    alpha = mesh.face_interp_factors[f]
    u_f = (1 - alpha) * u_field[P] + alpha * u_field[N]  # TODO: Replace with Rhie-Chow if pressure coupling is used

    F = rho * np.dot(u_f, Sf)  # Mass flux through face

    d_f = mesh.vector_skewness[f]

    if F >= 0:
        grad_upwind = grad_phi[P]
        b_corr = F * np.dot(grad_upwind, d_f)
        return F, -F, b_corr  # a_P, a_N
    else:
        grad_upwind = grad_phi[N]
        b_corr = F * np.dot(grad_upwind, d_f)
        return 0.0, F, b_corr



@njit(inline="always")
def compute_boundary_convective_flux(f, phi, grad_phi, mesh, rho, u_field, bc_type, bc_value):
    """
    Compute convection term stencil and source correction for boundary face.
    """
    P = mesh.owner_cells[f]
    Sf = mesh.vector_S_f[f]
    d_f = mesh.vector_skewness[f]
    grad_P = grad_phi[P]

    # Determine face velocity
    if bc_type == BC_DIRICHLET:
        u_f = mesh.boundary_values[f, :2]
    else:
        u_f = u_field[P]

    F = rho * np.dot(u_f, Sf)

    if F >= 0:
        # Outflow: interior field controls face value
        a_P = F
        b_corr = F * np.dot(grad_P, d_f)
    else:
        # Inflow: external BC prescribes face value
        if bc_type == BC_DIRICHLET:
            a_P = 0.0
            b_corr = F * bc_value  # No skewness correction
        elif bc_type == BC_ZEROGRADIENT:
            a_P = F
            b_corr = 0.0
        elif bc_type == BC_NEUMANN:
            a_P = 0.0
            b_corr = 0.0
        elif bc_type == BC_CONVECTIVE:
            a_P = 0.0
            b_corr = F * bc_value
        else:
            a_P = 0.0
            b_corr = 0.0

    return a_P, 0.0, b_corr
