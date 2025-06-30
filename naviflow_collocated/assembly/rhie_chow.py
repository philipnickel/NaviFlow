import numpy as np
from numba import njit, prange

# Boundary condition types (must match convection_diffusion_matrix.py)
BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 3
BC_OBSTACLE = 4


@njit(cache=True)
def compute_velocity_gradient_least_squares(mesh, U_star_rc, U_star, x_P, U_star_C, P, f_exclude):
    """
    Compute 2x2 velocity gradient tensor delta_v at cell P using least squares
    over all face neighbors except the excluded face f_exclude.

    Parameters:
    -----------
    mesh : mesh object
    U_star_rc : (n_faces, 2) array of rhie chow interpolated velocities
    U_star : (n_cells, 2) array of cell-centered velocities from momentum solution
    x_P : (2,) array, cell center of P
    U_star_C : (2,) array, velocity at cell P from momentum solution
    P : int, index of cell
    f_exclude : int, face to exclude (boundary face)

    Returns:
    --------
    grad_v : (2, 2) array, delta_v tensor at cell P
    """
    AtA = np.zeros((2, 2), dtype=np.float64)
    Atb_u = np.zeros(2, dtype=np.float64)
    Atb_v = np.zeros(2, dtype=np.float64)

    for k in range(mesh.cell_faces[P].shape[0]):
        ff = mesh.cell_faces[P][k]
        if ff == f_exclude:
            continue

        x_f = mesh.face_centers[ff]
        dx = x_f[0] - x_P[0]
        dy = x_f[1] - x_P[1]
        U_star_f = U_star_rc[ff]
        du = U_star_f[0] - U_star_C[0]
        dv = U_star_f[1] - U_star_C[1]

        AtA[0, 0] += dx * dx
        AtA[0, 1] += dx * dy
        AtA[1, 0] += dy * dx
        AtA[1, 1] += dy * dy

        Atb_u[0] += dx * du
        Atb_u[1] += dy * du
        Atb_v[0] += dx * dv
        Atb_v[1] += dy * dv

    # Solve AtA x = Atb using analytical 2x2 inverse
    det = AtA[0, 0] * AtA[1, 1] - AtA[0, 1] * AtA[1, 0] + 1e-14
    inv_AtA_00 =  AtA[1, 1] / det
    inv_AtA_01 = -AtA[0, 1] / det
    inv_AtA_10 = -AtA[1, 0] / det
    inv_AtA_11 =  AtA[0, 0] / det

    grad_v = np.zeros((2, 2), dtype=np.float64)
    grad_v[0, 0] = inv_AtA_00 * Atb_u[0] + inv_AtA_01 * Atb_u[1]
    grad_v[0, 1] = inv_AtA_10 * Atb_u[0] + inv_AtA_11 * Atb_u[1]
    grad_v[1, 0] = inv_AtA_00 * Atb_v[0] + inv_AtA_01 * Atb_v[1]
    grad_v[1, 1] = inv_AtA_10 * Atb_v[0] + inv_AtA_11 * Atb_v[1]

    return grad_v


@njit(inline="always", cache=True, fastmath=True, nogil=True)
def rhie_chow_velocity_internal_faces(mesh, U_star, U_star_bar, U_old_bar, U_old_faces, grad_p_bar, grad_p, p, alpha_uv, bold_D_bar):
    """
    Compute Rhie-Chow velocity at internal faces.
    Optimized for memory access patterns with pre-fetched static data.
    """
    n_internal = mesh.internal_faces.shape[0] 
    n_total_faces = mesh.vector_S_f.shape[0]
    U_faces = np.zeros((n_total_faces, 2), dtype=np.float64)

    # ═══ PRE-FETCH STATIC DATA FOR MEMORY OPTIMIZATION ═══
    internal_faces = mesh.internal_faces
    owner_cells = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    vector_S_f = mesh.vector_S_f
    face_interp_factors = mesh.face_interp_factors

    # ––– internal faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_internal):
        f = internal_faces[i]
        P = owner_cells[f]
        N = neighbor_cells[f]
        
        # Pre-fetch interpolation factor (single access)
        g = face_interp_factors[f]
        
        # Pre-fetch face area vector components
        S_f = vector_S_f[f]
        S_f_0 = S_f[0]
        S_f_1 = S_f[1]

        # Manual norm calculation
        mag_S_f = np.sqrt(S_f_0 * S_f_0 + S_f_1 * S_f_1)
        n_f_0 = S_f_0 / mag_S_f
        n_f_1 = S_f_1 / mag_S_f

        # Pre-fetch velocity and pressure gradient data (better cache locality)
        U_star_P = U_star[P]
        U_star_N = U_star[N]
        grad_p_P = grad_p[P]
        grad_p_N = grad_p[N]
        bold_D_P = bold_D_bar[f]  # Already at face

        # Velocity interpolation with pre-fetched data
        U_f_0 = (1.0 - g) * U_star_P[0] + g * U_star_N[0]
        U_f_1 = (1.0 - g) * U_star_P[1] + g * U_star_N[1]

        # Pressure gradient interpolation with cached components
        grad_p_f_0 = (1.0 - g) * grad_p_P[0] + g * grad_p_N[0]
        grad_p_f_1 = (1.0 - g) * grad_p_P[1] + g * grad_p_N[1]

        # Face-centered gradient correction (using pre-fetched gradient)
        grad_p_bar_f = grad_p_bar[f]
        grad_p_f_corr_0 = grad_p_bar_f[0] - grad_p_f_0
        grad_p_f_corr_1 = grad_p_bar_f[1] - grad_p_f_1

        # Rhie-Chow correction with manual operations
        correction_0 = bold_D_P[0] * grad_p_f_corr_0
        correction_1 = bold_D_P[1] * grad_p_f_corr_1

        # Final velocity with optimization
        U_faces[f, 0] = U_f_0 - correction_0
        U_faces[f, 1] = U_f_1 - correction_1

    return U_faces


@njit(inline="always", cache=True, fastmath=True, nogil=True)
def rhie_chow_velocity_boundary_faces(mesh, U_faces, U_star, grad_p_bar, grad_p, p, alpha_uv, bold_D_bar, U_old_bar, U_old_faces):
    """
    Apply boundary conditions to Rhie-Chow velocity.
    Optimized memory access patterns with pre-fetched boundary data.
    """
    n_boundary = mesh.boundary_faces.shape[0]

    # ═══ PRE-FETCH STATIC BOUNDARY DATA ═══
    boundary_faces = mesh.boundary_faces
    owner_cells = mesh.owner_cells
    boundary_types = mesh.boundary_types
    boundary_values = mesh.boundary_values
    vector_S_f = mesh.vector_S_f
    d_Cb = mesh.d_Cb

    # ––– boundary faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_boundary):
        f = boundary_faces[i]
        P = owner_cells[f]
        bc_type = boundary_types[f, 0]

        # Pre-fetch face area components (single memory access)
        S_f = vector_S_f[f]
        S_f_0 = S_f[0]
        S_f_1 = S_f[1]

        # Manual norm calculation
        mag_S_f = np.sqrt(S_f_0 * S_f_0 + S_f_1 * S_f_1)
        n_f_0 = S_f_0 / mag_S_f
        n_f_1 = S_f_1 / mag_S_f

        if bc_type == BC_WALL or bc_type == BC_OBSTACLE:
            # Wall: zero velocity
            U_faces[f, 0] = 0.0
            U_faces[f, 1] = 0.0
            
        elif bc_type == BC_DIRICHLET or bc_type == BC_INLET:
            # Fixed velocity boundary
            boundary_vel = boundary_values[f]
            U_faces[f, 0] = boundary_vel[0]
            U_faces[f, 1] = boundary_vel[1]
            
        elif bc_type == BC_OUTLET:
            # Pressure outlet: use interior velocity with Rhie-Chow correction
            U_star_P = U_star[P]
            grad_p_P = grad_p[P]
            bold_D_f = bold_D_bar[f]
            
            # Distance to boundary
            d_Cb_f = d_Cb[f]
            vec_Cb_0 = d_Cb_f * n_f_0
            vec_Cb_1 = d_Cb_f * n_f_1

            # Extrapolated velocity
            U_b_0 = U_star_P[0] + grad_p_P[0] * vec_Cb_0
            U_b_1 = U_star_P[1] + grad_p_P[1] * vec_Cb_1

            # Gradient correction
            grad_p_bar_f = grad_p_bar[f]
            grad_p_f_0 = grad_p_P[0] + grad_p_P[0] * vec_Cb_0
            grad_p_f_1 = grad_p_P[1] + grad_p_P[1] * vec_Cb_1
            
            correction_0 = bold_D_f[0] * (grad_p_bar_f[0] - grad_p_f_0)
            correction_1 = bold_D_f[1] * (grad_p_bar_f[1] - grad_p_f_1)

            U_faces[f, 0] = U_b_0 - correction_0
            U_faces[f, 1] = U_b_1 - correction_1

    return U_faces


@njit(cache=True, fastmath=True, nogil=True)
def mdot_calculation(mesh, rho, U_f, correction=False):
    """
    Calculate mass flux through faces: mdot = rho * U_f · S_f
    Optimized for memory access patterns with pre-fetched static data.
    """
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    n_faces = n_internal + n_boundary
    
    mdot = np.zeros(n_faces, dtype=np.float64)

    # ═══ PRE-FETCH STATIC DATA FOR MEMORY OPTIMIZATION ═══
    internal_faces = mesh.internal_faces
    boundary_faces = mesh.boundary_faces
    vector_S_f = mesh.vector_S_f
    boundary_types = mesh.boundary_types
    
    # ––– internal faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_internal):
        f = internal_faces[i]
        
        # Pre-fetch velocity and area vector components (single memory access each)
        U_f_vec = U_f[f]
        S_f_vec = vector_S_f[f]
        U_f_0 = U_f_vec[0]
        U_f_1 = U_f_vec[1]
        S_f_0 = S_f_vec[0]
        S_f_1 = S_f_vec[1]
        
        # Manual dot product (avoid np.dot allocation)
        mdot[f] = rho * (U_f_0 * S_f_0 + U_f_1 * S_f_1)

    # ––– boundary faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_boundary):
        f = boundary_faces[i]
        bc_type = boundary_types[f, 0]
        
        # Pre-fetch components
        U_f_vec = U_f[f]
        S_f_vec = vector_S_f[f]
        U_f_0 = U_f_vec[0]
        U_f_1 = U_f_vec[1]
        S_f_0 = S_f_vec[0]
        S_f_1 = S_f_vec[1]
        
        # Manual dot product
        mdot[f] = rho * (U_f_0 * S_f_0 + U_f_1 * S_f_1)

    return mdot


@njit(parallel=False, cache=True)
def compute_face_fluxes(mesh, face_velocity, rho):
    """
    Compute mass fluxes at faces from face velocities.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh data structure
    face_velocity : ndarray
        Face velocities [n_faces, 2]
    rho : float
        Density

    Returns
    ------
    face_mass_fluxes : ndarray
        Mass fluxes at faces
    """
    n_faces = len(mesh.face_areas)
    face_mass_fluxes = np.zeros(n_faces)

    for f in prange(n_faces):
        # Get face area and normal vector (Sf_x, Sf_y)
        S_f = np.ascontiguousarray(mesh.vector_S_f[f])  # This is already Area * unit_normal

        vol_flux = np.dot(face_velocity[f], S_f)

        # Calculate mass flux
        face_mass_fluxes[f] = rho * vol_flux

    return face_mass_fluxes

@njit(parallel=False, cache=True)
def compute_face_velocities(mesh, u, v):
    n_faces = len(mesh.face_areas)
    face_velocity = np.zeros((n_faces, 2))
    for f in range(n_faces):
        gf = mesh.face_interp_factors[f]
        face_velocity[f, 0] = gf * u[mesh.neighbor_cells[f]] + (1 - gf) * u[mesh.owner_cells[f]]
        face_velocity[f, 1] = gf * v[mesh.neighbor_cells[f]] + (1 - gf) * v[mesh.owner_cells[f]]
    return face_velocity

@njit(cache=True, fastmath=True, nogil=True)
def rhie_chow_velocity(mesh, U_star, U_star_bar, U_old_bar, U_old_faces, grad_p_bar, grad_p, p, alpha_uv, bold_D_bar):
    """
    Wrapper function that maintains the same interface while using optimized memory access.
    """
    # Compute internal faces with optimized memory access
    U_faces = rhie_chow_velocity_internal_faces(mesh, U_star, U_star_bar, U_old_bar, U_old_faces, grad_p_bar, grad_p, p, alpha_uv, bold_D_bar)
    
    # Apply boundary conditions with optimized memory access
    U_faces = rhie_chow_velocity_boundary_faces(mesh, U_faces, U_star, grad_p_bar, grad_p, p, alpha_uv, bold_D_bar, U_old_bar, U_old_faces)
    
    return U_faces
