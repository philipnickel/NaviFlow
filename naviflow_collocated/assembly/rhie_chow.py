
import numpy as np
from numba import njit, prange


BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 3
BC_NEUMANN = 4


@njit
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


#@njit(parallel=False)
def rhie_chow_velocity(mesh, U_star, U_star_bar, U_old_bar, U_old_rc, grad_p_bar, p, alpha_uv, bold_D_bar):
    """
    Returns rhie chow interpolated velocity vector at faces 
    U_star: velocities from momentum solution at cell centers
    U_star_bar: velocities from momentum solution interpolated to faces
    U_old_bar: cell centered velocities from previous iteration interpolated to faces
    U_old_rc: rhie chow interpolated velocities from previous iteration (fully corrected)
    grad_p_bar: pressure gradient interpolated to faces
    p: pressure field
    alpha_uv: under-relaxation factor for momentum
    bold_D_bar: bold D at faces (diagonals of bold_D matrix in Moukalled et al. 2015)
    """
    n_internal  = mesh.internal_faces.shape[0]
    n_boundary  = mesh.boundary_faces.shape[0]
    n_faces = n_internal + n_boundary
    U_star_rc = np.zeros((n_faces, 2))

    for i in prange(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        gf = mesh.face_interp_factors[f]
        e_CF = np.ascontiguousarray(mesh.unit_vector_e[f])
        d_CF = mesh.vector_d_CE[f]
        d_mag = np.linalg.norm(d_CF) + 1e-14
        delta_p_f = (p[N] - p[P]) / (d_mag + 1e-14)
        delta_p_f_bar = np.ascontiguousarray(grad_p_bar[f])
        rc_correction = (delta_p_f - np.dot(delta_p_f_bar, e_CF)) * e_CF

        # correct velocity vector
        relax_correction = (1.0 - alpha_uv) * (U_old_rc[f] - U_old_bar[f])
        U_star_rc[f] = U_star_bar[f] - bold_D_bar[f] * rc_correction + relax_correction
    
    for i in prange(n_boundary):
        f = mesh.boundary_faces[i]
        P = mesh.owner_cells[f]
        if mesh.boundary_types[f,0] == BC_WALL or mesh.boundary_types[f,0] == BC_INLET:
            U_star_b = mesh.boundary_values[f,:2]
            U_star_rc[f] = U_star_b 
        elif mesh.boundary_types[f, 0] == BC_OUTLET:
            e_b = np.ascontiguousarray(mesh.unit_vector_e[f])
            U_star_C = U_star[P]
            d_Cb = mesh.d_Cb[f]
            S_b = mesh.vector_S_f[f]
            mag_S_b = np.linalg.norm(S_b) + 1e-14
            n = S_b / mag_S_b
            vec_Cb = d_Cb * n
            x_P = mesh.cell_centers[P]
            grad_U_P = np.ascontiguousarray(compute_velocity_gradient_least_squares(mesh, U_star_rc, U_star, x_P, U_star_C, P, f))


            # Remove normal component 
            grad_U_b = grad_U_P - np.outer(np.dot(grad_U_P, e_b), e_b)

            # Extrapolate to boundary face
            #U_star_b = U_star_C + grad_U_b * vec_Cb
            U_star_b = U_star_C + grad_U_b @ vec_Cb

            U_star_rc[f] = U_star_b


    return U_star_rc


@njit(parallel=False)
def mdot_calculation(mesh, rho, U_star_rc):
    """
    Computes mass fluxes at faces from rhie chow interpolated velocities
    """
    n_internal  = mesh.internal_faces.shape[0]
    n_boundary  = mesh.boundary_faces.shape[0]
    n_faces = n_internal + n_boundary
    mdot_faces = np.zeros(n_faces)

    for i in prange(n_internal):
        f = mesh.internal_faces[i]

        mdot_faces[f] = rho * np.dot(np.ascontiguousarray(U_star_rc[f]), np.ascontiguousarray(mesh.vector_S_f[f]))

    for i in prange(n_boundary):
        f = mesh.boundary_faces[i]
        if mesh.boundary_types[f,0] == BC_WALL:
            mdot_faces[f] = 0.0 
        elif mesh.boundary_types[f,0] == BC_INLET or mesh.boundary_types[f,0] == BC_OUTLET:
            mdot_faces[f] = rho * np.dot(np.ascontiguousarray(U_star_rc[f]), np.ascontiguousarray(mesh.vector_S_f[f]))
    """
        
    sum_mdot_in = 0.0
    sum_mdot_out = 0.0
    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        if mesh.boundary_types[f,0] == BC_INLET:
            sum_mdot_in += mdot_faces[f]
        elif mesh.boundary_types[f,0] == BC_OUTLET:
            sum_mdot_out += mdot_faces[f]
    if sum_mdot_out != 0.0:
        scale = sum_mdot_in / sum_mdot_out
    for i in range(n_boundary):
        f = mesh.boundary_faces[i]
        #if mesh.boundary_types[f,0] == BC_OUTLET:
            #mdot_faces[f] = mdot_faces[f] * scale
    """


    return mdot_faces



@njit(parallel=False)
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

@njit(parallel=False)
def compute_face_velocities(mesh, u, v):
    n_faces = len(mesh.face_areas)
    face_velocity = np.zeros((n_faces, 2))
    for f in range(n_faces):
        gf = mesh.face_interp_factors[f]
        face_velocity[f, 0] = gf * u[mesh.neighbor_cells[f]] + (1 - gf) * u[mesh.owner_cells[f]]
        face_velocity[f, 1] = gf * v[mesh.neighbor_cells[f]] + (1 - gf) * v[mesh.owner_cells[f]]
    return face_velocity
