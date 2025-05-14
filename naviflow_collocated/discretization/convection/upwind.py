import numpy as np
from numba import njit

# Boundary condition identifiers
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_CONVECTIVE = 4
BC_ZEROGRADIENT = 3
BC_WALL = 0

@njit(inline="always")
def compute_convective_stencil_upwind(f, mesh, rho, u_field, grad_phi, component_idx):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    g_f = mesh.face_interp_factors[f]
    Sf = np.ascontiguousarray(mesh.vector_S_f[f])
    d_skew = np.ascontiguousarray(mesh.vector_skewness[f])

    # Interpolate velocity vector at face
    u_face = (1 - g_f) * u_field[P] + g_f * u_field[N]
    F = rho * np.dot(u_face, Sf)
    """
    # Interpolate scalar component with skewness correction
    phi_P = u_field[P, bc_component_idx]
    phi_N = u_field[N, bc_component_idx]
    grad_P = grad_phi[P]
    grad_N = grad_phi[N]
    grad_f = (1 - g_f) * grad_P + g_f * grad_N
    phi_fmark = (1 - g_f) * phi_P + g_f * phi_N
    phi_f = phi_fmark + np.dot(grad_f, d_skew)
    """

    if F >= 0:
        return -F, F, 0.0
    else:
        return -F, F, 0.0

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

    F = rho * phi_f * np.dot(Sf, u_field[P])

    if F >= 0:
        return F, 0.0, 0.0
    else:
        # Inflow — depends on boundary type
        if bc_type == BC_DIRICHLET:
            return 0.0, 0.0, -F * bc_value
        elif bc_type == BC_ZEROGRADIENT:
            return F, 0.0, 0.0
        elif bc_type == BC_NEUMANN:
            return 0.0, 0.0, 0.0
        elif bc_type == BC_CONVECTIVE:
            return 0.0, 0.0, F * bc_value
        else:
            return 0.0, 0.0, 0.0
