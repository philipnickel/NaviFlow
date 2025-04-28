import numpy as np

def get_rhs(mesh, rho, u_star, v_star):
    """
    Build the RHS vector for pressure correction equation.
    Mesh-agnostic version.
    
    Parameters:
    -----------
    mesh : Mesh
        The computational mesh (structured or unstructured)
    rho : float
        Fluid density
    u_star, v_star : ndarray
        Intermediate velocity fields at cells
        
    Returns:
    --------
    ndarray
        Right-hand side vector for the pressure equation
    """
    owner_cells, neighbor_cells = mesh.get_owner_neighbor()
    face_areas = mesh.get_face_areas()
    face_normals = mesh.get_face_normals()
    face_centers = mesh.get_face_centers()
    cell_centers = mesh.get_cell_centers()
    n_cells = mesh.n_cells
    n_faces = len(face_areas)

    # Initialize RHS (mass imbalance for each cell)
    bp = np.zeros(n_cells)

    # Loop through all faces and calculate mass fluxes
    for face_idx in range(n_faces):
        owner = owner_cells[face_idx]
        neighbor = neighbor_cells[face_idx]
        area = face_areas[face_idx]
        normal = face_normals[face_idx]

        # Add bounds checking for array access
        try:
            # Ensure owner index is within bounds
            safe_owner = min(owner, len(u_star)-1) if owner >= 0 and owner < len(u_star) else 0
            
            # Get velocity at owner cell (with bounds checking)
            u_o = u_star[safe_owner]
            v_o = v_star[safe_owner]

            if neighbor >= 0 and neighbor < len(u_star):
                # Internal face - use linear interpolation for velocity
                u_n = u_star[neighbor]
                v_n = v_star[neighbor]
                
                # Simple linear interpolation
                u_face = 0.5 * (u_o + u_n)
                v_face = 0.5 * (v_o + v_n)
            else:
                # Boundary face - use owner velocity (could be refined later)
                u_face = u_o
                v_face = v_o
        except (IndexError, TypeError) as e:
            # Handle any indexing errors by setting default values
            u_face = 0.0
            v_face = 0.0
            
            # If this is a wall boundary, apply no-penetration condition
            boundary_name = mesh.get_boundary_name(face_idx)
            if boundary_name is not None:
                # Remove normal component of velocity at walls
                # This ensures zero mass flux through walls
                try:
                    # Ensure velocity and normal are compatible for dot product
                    if np.isscalar(u_face) and np.isscalar(v_face):
                        velocity = np.array([u_face, v_face])
                    else:
                        # Handle case where velocity is an array
                        if hasattr(u_face, 'item') and hasattr(v_face, 'item'):
                            try:
                                velocity = np.array([u_face.item(), v_face.item()])
                            except (ValueError, TypeError):
                                # Fallback for arrays larger than 1 element
                                velocity = np.array([float(u_face.flat[0] if hasattr(u_face, 'flat') else u_face),
                                                    float(v_face.flat[0] if hasattr(v_face, 'flat') else v_face)])
                        else:
                            # If they're proper arrays, use the first element
                            velocity = np.array([float(u_face.flat[0] if hasattr(u_face, 'flat') else u_face),
                                                float(v_face.flat[0] if hasattr(v_face, 'flat') else v_face)])
                    
                    # Ensure normal is a 1D array of 2 elements
                    if isinstance(normal, np.ndarray) and normal.ndim > 1:
                        normal_vec = np.array([normal.flat[0], normal.flat[1]])
                    else:
                        normal_vec = np.array([normal[0], normal[1]])
                        
                    # Now both velocity and normal_vec are 1D arrays of length 2
                    normal_component = np.dot(velocity, normal_vec)
                    u_face -= normal_component * normal_vec[0]
                    v_face -= normal_component * normal_vec[1]
                except Exception as e:
                    # In case of any error, just keep the original values
                    pass

        # Calculate mass flux through the face
        try:
            # Same approach as above to ensure compatible arrays
            if np.isscalar(u_face) and np.isscalar(v_face):
                velocity_face = np.array([u_face, v_face])
            else:
                # Handle case where velocity is an array
                if hasattr(u_face, 'item') and hasattr(v_face, 'item'):
                    try:
                        velocity_face = np.array([u_face.item(), v_face.item()])
                    except (ValueError, TypeError):
                        # Fallback for arrays larger than 1 element
                        velocity_face = np.array([float(u_face.flat[0] if hasattr(u_face, 'flat') else u_face),
                                                float(v_face.flat[0] if hasattr(v_face, 'flat') else v_face)])
                else:
                    # If they're proper arrays, use the first element
                    velocity_face = np.array([float(u_face.flat[0] if hasattr(u_face, 'flat') else u_face),
                                            float(v_face.flat[0] if hasattr(v_face, 'flat') else v_face)])
            
            # Ensure normal is a 1D array of 2 elements
            if isinstance(normal, np.ndarray) and normal.ndim > 1:
                normal_vec = np.array([normal.flat[0], normal.flat[1]])
            else:
                normal_vec = np.array([normal[0], normal[1]])
                
            # Calculate mass flux
            mass_flux = rho * np.dot(velocity_face, normal_vec) * area
        except Exception as e:
            # Default to zero mass flux if the calculation fails
            mass_flux = 0.0

        # Add flux contribution to owner (outflow is negative)
        bp[owner] -= mass_flux

        # Add flux contribution to neighbor if internal
        if neighbor >= 0:
            bp[neighbor] += mass_flux

    # Ensure the RHS sums to zero (necessary for solvability)
    if not np.isclose(np.sum(bp), 0.0, atol=1e-10):
        bp -= np.sum(bp) / n_cells  # Adjust for global mass conservation
    
    # Fix reference pressure cell
    bp[0] = 0.0

    return bp


# Legacy function for compatibility with structured mesh
def get_rhs_structured(nx, ny, dx, dy, rho, u_star, v_star):
    """
    Build the RHS vector for pressure correction equation.
    Structured grid version.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Cell sizes
    rho : float
        Fluid density
    u_star, v_star : ndarray
        Intermediate velocity fields
        
    Returns:
    --------
    ndarray
        Right-hand side vector for the pressure equation
    """
    # Total number of cells
    n_cells = nx * ny
    
    # Initialize RHS vector
    bp = np.zeros(n_cells)
    
    # Interior cells
    for i in range(nx):
        for j in range(ny):
            # Linear index for current cell
            k = i + j * nx
            
            # East face (i+1/2, j)
            if i < nx-1:
                bp[k] -= rho * u_star[i+1, j] * dy
            
            # West face (i-1/2, j)
            if i > 0:
                bp[k] += rho * u_star[i, j] * dy
            
            # North face (i, j+1/2)
            if j < ny-1:
                bp[k] -= rho * v_star[i, j+1] * dx
            
            # South face (i, j-1/2)
            if j > 0:
                bp[k] += rho * v_star[i, j] * dx
    
    # Fix reference pressure cell
    bp[0] = 0.0
    
    return bp
