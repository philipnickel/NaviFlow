import numpy as np
from numba import njit

# Boundary condition identifiers
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_CONVECTIVE = 4
BC_ZEROGRADIENT = 3
BC_WALL = 0

#@njit(inline="always")
def compute_convective_stencil_upwind(f, phi, grad_phi, mesh, rho, u_field):
    """
    Upwind convection scheme with skewness correction for interior faces.
    """
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    Sf = mesh.face_normals[f]
    
    # Interpolate velocity to face
    alpha = mesh.face_interp_factors[f]
    u_f = (1 - alpha) * u_field[P] + alpha * u_field[N]
    
    # Calculate mass flux through face
    F = rho * np.dot(u_f, Sf)

    # Determine upwind direction and values
    if F >= 0:
        # Flow P -> N
        phi_upwind = phi[P]
        grad_upwind = grad_phi[P]
        a_P = F
        a_N = -F
    else:
        # Flow N -> P
        phi_upwind = phi[N]
        grad_upwind = grad_phi[N]
        a_P = 0.0
        a_N = F
    
    # Skewness correction
    d_f = mesh.skewness_vectors[f]
    b_corr = F * np.dot(grad_upwind, d_f)

    return a_P, a_N, b_corr


#@njit(inline="always")
def compute_boundary_convective_flux(f, phi, grad_phi, mesh, rho, u_field, bc_type, bc_value):
    """
    Compute convection stencil at boundary face.

    Parameters:
        f         : face ID (boundary face)
        phi       : scalar field at cells
        grad_phi  : gradient field at cells
        mesh      : MeshData2D object
        rho       : density (float)
        u_field   : velocity field at cells (n_cells x 2)
        bc_type   : velocity BC type (Dirichlet, Neumann, etc.)
        bc_value  : value for Dirichlet BC (float), ignored for outflow

    Returns:
        a_PP      : matrix coefficient for owner
        a_PN      : always 0.0
        b_corr    : source correction term
    """
    P = mesh.owner_cells[f]
    Sf = mesh.face_normals[f]
    
    # Use prescribed velocity if Dirichlet, otherwise use cell velocity
    if bc_type == BC_DIRICHLET:
        # For Dirichlet, use prescribed boundary value for velocity
        u_f = mesh.boundary_values[f, :2]  # Use boundary velocity vector
    else:
        # For other BCs, use cell velocity
        u_f = u_field[P]
    
    F = rho * np.dot(u_f, Sf)
    phi_P = phi[P]
    grad_P = grad_phi[P]
    d_f = mesh.skewness_vectors[f]

    if F >= 0:
        # Flow leaving domain: upwind from P
        a_P = F
        b_corr = F * np.dot(grad_P, d_f)
    else:
        # Inflow: value from boundary
        if bc_type == BC_DIRICHLET:
            # Use prescribed value at boundary
            a_P = 0.0
            b_corr = F * (bc_value + np.dot(grad_P, d_f))
        elif bc_type == BC_ZEROGRADIENT:
            # For zerogradient, use cell value at boundary face
            a_P = F  # Implicit contribution on P
            b_corr = F * np.dot(grad_P, d_f)
        elif bc_type == BC_NEUMANN:
            # Usually only for pressure
            a_P = 0.0
            b_corr = 0.0
        elif bc_type == BC_CONVECTIVE:
            a_P = 0.0
            b_corr = F * bc_value  # Assuming convective extrapolation already applied
        else:
            raise ValueError(f"Unsupported boundary BC type: {bc_type}")

    return a_P, 0.0, b_corr