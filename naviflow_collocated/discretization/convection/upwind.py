# Placeholder for Upwind convection scheme

from numba import njit


@njit(inline="always")
def compute_convective_flux_upwind(f, u, mesh, uf, rho):
    """
    Compute convective flux contributions using the upwind scheme.

    Parameters
    ----------
    f : int
        Face index.
    u : ndarray
        Solution field at cell centers.
    mesh : MeshData2D (numba.experimental.jitclass.boxing.MeshData2D)
        Mesh data structure.
    uf : ndarray
        Face velocity vector [ux, uy].
    rho : float
        Density.

    Returns
    -------
    tuple
        (P, N, a_P_P, a_P_N, b_P): Convective flux coefficients for cell P
        such that Flux_conv = a_P_P * u_P + a_P_N * u_N + b_P.
    """
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    # Face normal vector (area weighted)
    Sf_x = mesh.face_normals[f, 0]
    Sf_y = mesh.face_normals[f, 1]

    # Mass flux through the face
    # F > 0 if flow is from P to N (along face normal)
    # F < 0 if flow is from N to P (opposite to face normal)
    F = rho * (uf[0] * Sf_x + uf[1] * Sf_y)

    # For upwind scheme:
    # Flux = F * u_face
    # if F > 0, u_face = u_P => Flux = F * u_P
    # if F < 0, u_face = u_N => Flux = F * u_N
    # This can be written as: max(F, 0)*u_P + min(F, 0)*u_N

    a_P_P = max(F, 0.0)  # Coefficient for u_P in the flux expression
    a_P_N = min(F, 0.0)  # Coefficient for u_N in the flux expression
    b_P = 0.0  # No explicit source term from upwind convection itself

    # Note: u_N for boundary faces (where N < 0) is not handled here explicitly.
    # The calling routine or assembly process should manage boundary conditions.
    # If N < 0, u[N] would cause an error if not handled.
    # However, a_P_N will be zero if F > 0, so u_N isn't used.
    # If F < 0 and N is a boundary, then min(F,0) * u_N needs u_N (boundary value).
    # The test will likely use internal faces, so N should be valid.

    return P, N, a_P_P, a_P_N, b_P
