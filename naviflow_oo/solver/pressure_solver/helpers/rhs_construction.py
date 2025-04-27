import numpy as np

def get_rhs(mesh, rho, u_star, v_star):
    """
    Build the right-hand side vector bp for the pressure-correction equation
    using mesh topology.
    
    Parameters:
    -----------
    mesh : Mesh
        The computational mesh (structured or unstructured)
    rho : float
        Fluid density
    u_star, v_star : ndarray
        Intermediate velocity fields defined on faces
        
    Returns:
    --------
    ndarray
        Right-hand side vector for the pressure correction equation
    """
    # Get mesh topology information
    owner_cells, neighbor_cells = mesh.get_owner_neighbor()
    face_areas = mesh.get_face_areas()
    face_normals = mesh.get_face_normals()
    face_centers = mesh.get_face_centers()
    n_cells = mesh.n_cells
    n_faces = mesh.n_faces
    
    # Initialize the right-hand side vector (mass imbalance for each cell)
    bp = np.zeros(n_cells)
    
    # Compute mass fluxes at all faces
    # Assume velocities are already interpolated to faces (or defined on faces)
    face_velocities = np.zeros((n_faces, 3))  # Will need to be computed from u_star, v_star
    
    # This would be replaced with proper velocity interpolation or extraction
    # For now, we'll use a simplified approach
    for face_idx in range(n_faces):
        # For a proper implementation, we would compute the face velocity
        # from u_star and v_star based on face orientation
        # For now, we'll use a placeholder
        face_velocities[face_idx] = np.array([u_star[face_idx], v_star[face_idx], 0.0])
    
    # Compute mass fluxes and accumulate to cells
    for face_idx in range(n_faces):
        owner = owner_cells[face_idx]
        area = face_areas[face_idx]
        normal = face_normals[face_idx]
        
        # Mass flux = rho * (velocity dot normal) * area
        velocity = face_velocities[face_idx]
        mass_flux = rho * np.dot(velocity, normal) * area
        
        # Add contribution to owner cell (outflow is positive, so negate for continuity)
        bp[owner] -= mass_flux
        
        # Add contribution to neighbor cell if internal face
        if neighbor_cells[face_idx] >= 0:
            neighbor = neighbor_cells[face_idx]
            # Inflow to neighbor is negative of outflow from owner
            bp[neighbor] += mass_flux
    
    # Set reference pressure cell RHS to zero
    bp[0] = 0.0
    
    return bp


# Keep the original function for compatibility with existing code
def get_rhs_structured(imax, jmax, dx, dy, rho, u_star, v_star):
    """
    Build the right-hand side vector bp for the pressure-correction equation
    for structured meshes.
    
    Parameters:
    -----------
    imax, jmax : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    rho : float
        Fluid density
    u_star, v_star : ndarray
        Intermediate velocity fields
        
    Returns:
    --------
    ndarray
        Right-hand side vector for the pressure correction equation
    """
    # Initialize right-hand side vector
    bp = np.zeros(imax*jmax)
    
    # Create 2D matrix first - easier to work with
    bp_2d = np.zeros((imax, jmax))
    
    # Compute entire array at once
    bp_2d = rho * (u_star[:-1, :] * dy - u_star[1:, :] * dy + 
                   v_star[:, :-1] * dx - v_star[:, 1:] * dx)
    
    # Flatten to 1D array in correct order
    bp = bp_2d.flatten('F')  # Fortran-style order (column-major)
    
    # Modify for p_prime(0,0) - pressure at first node is fixed
    bp[0] = 0
    
    return bp


# Alternative version from the original file (keep for reference)
def get_rhs2(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    rho: float,
    u_star: np.ndarray,
    v_star: np.ndarray,
) -> np.ndarray:
    """
    Build the right-hand side vector bp for the pressure-correction equation
    such that  A · p' = bp   (matrix built by `get_coeff_mat`).

    Sign convention matches the coefficient matrix *and* the Rhie–Chow
    velocity correction with  u = u* + d_u ∂p'/∂x   (note the **plus** sign).
    """
    # continuity defect on cell faces (vectorised)
    bp_2d = rho * (
        (u_star[1:, :] - u_star[:-1, :]) * dy
        + (v_star[:, 1:] - v_star[:, :-1]) * dx
    )

    bp = bp_2d.flatten("F")        # Fortran order
    bp[0] = 0.0                    # consistency with pinned pressure node
    return bp
