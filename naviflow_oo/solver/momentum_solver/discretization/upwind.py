import numpy as np

class UpwindDiscretization:
    """
    First-order Upwind discretization for momentum equations.
    """

    def calculate_u_coefficients(self, mesh, fluid, u, v, p):
        return self._calculate_coefficients(mesh, fluid, u, v, p, component='u')

    def calculate_v_coefficients(self, mesh, fluid, u, v, p):
        return self._calculate_coefficients(mesh, fluid, u, v, p, component='v')

    def _calculate_coefficients(self, mesh, fluid, u, v, p, component='u'):
        rho = fluid.get_density()
        mu = fluid.get_viscosity()

        owners, neighbors = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_normals = mesh.get_face_normals()
        face_centers = mesh.get_face_centers()
        cell_centers = mesh.get_cell_centers()
        n_faces = mesh.n_faces
        n_cells = mesh.n_cells

        # Ensure inputs are flattened
        u_flat = u.flatten() if u.ndim > 1 else u.copy()
        v_flat = v.flatten() if v.ndim > 1 else v.copy()
        p_flat = p.flatten() if p.ndim > 1 else p.copy()
        
        if u_flat.size != n_cells or v_flat.size != n_cells or p_flat.size != n_cells:
             raise ValueError(f"Input field sizes mismatch n_cells ({n_cells}): u={u_flat.size}, v={v_flat.size}, p={p_flat.size}")

        a_p = np.zeros(n_cells)
        source = np.zeros(n_cells)
        a_nb = {'face': np.zeros(n_faces)} # Neighbor coeffs (used by some solvers)

        # --- Calculate base coefficients for all faces ---
        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]

            # Skip faces whose owner is outside the primary cell range (e.g., ghost cells if any)
            if owner < 0 or owner >= n_cells: 
                continue

            normal = face_normals[face_idx]
            area = face_areas[face_idx]

            # Calculate distance for diffusion term
            d = 1e-6  # minimal distance
            if neighbor != -1 and neighbor < n_cells:
                # Internal face distance
                d = np.linalg.norm(cell_centers[neighbor] - cell_centers[owner])
            # else: # Boundary face distance (handled below if needed, often implicitly in BC)
                # d = np.linalg.norm(face_centers[face_idx] - cell_centers[owner]) # Example if needed
            d = max(d, 1e-6) # Ensure positive distance

            # --- Convection Term (F) ---
            # Interpolate velocity to face center (simple average for now)
            # Note: This differs from staggered where face velocity is directly available
            if neighbor != -1 and neighbor < n_cells:
                 u_face = 0.5 * (u_flat[owner] + u_flat[neighbor])
                 v_face = 0.5 * (v_flat[owner] + v_flat[neighbor])
            else: # Boundary face - Use owner cell velocity? Needs careful consideration.
                 # For Upwind, maybe use owner cell value is okay? Or depends on BC.
                 u_face = u_flat[owner] 
                 v_face = v_flat[owner]

            vel_dot_n = u_face * normal[0] + v_face * normal[1]
            F = rho * vel_dot_n * area # Mass flux through face

            # --- Diffusion Term (D) ---
            D = mu * area / d # Diffusion conductance

            # --- Assemble Coefficients (based on Upwind logic) ---
            # Contribution TO the owner equation FROM this face
            
            # Upwind logic: coefficient depends on flow direction (F)
            # a_P coefficient increases due to flow *out* of the cell (+ max(F, 0))
            # a_N coefficient involves flow *into* the cell from neighbor (+ max(-F, 0))
            
            if neighbor != -1 and neighbor < n_cells: 
                 # Internal Face
                 # Add convection contribution to a_p based on OUTFLOW from owner
                 a_p[owner] += max(F, 0) 
                 # Add diffusion contribution to a_p
                 a_p[owner] += D
                 
                 # Add convection contribution to a_p of NEIGHBOR based on OUTFLOW from neighbor
                 a_p[neighbor] += max(-F, 0) 
                 # Add diffusion contribution to a_p of NEIGHBOR
                 a_p[neighbor] += D
                 
                 # Set neighbor coefficient a_nb (contribution to owner eqn from neighbor value)
                 # This should be D + max(-F, 0)
                 a_nb_val = D + max(-F, 0)
                 a_nb['face'][face_idx] = a_nb_val # Store face-based neighbor coeff

                 # Pressure Source Term Contribution (for owner cell)
                 p_face = 0.5 * (p_flat[owner] + p_flat[neighbor]) # Simple average pressure at face
                 pressure_force_component = p_face * (normal[0] if component == 'u' else normal[1]) * area
                 source[owner] -= pressure_force_component
                 
                 # Add pressure contribution for neighbor cell (equal and opposite)
                 # This is implicitly handled by doing it for all faces, summing at nodes.
                 # source[neighbor] += pressure_force_component # No, this is double counting.

            else: 
                 # Boundary Face - Treat implicitly for now, modify below
                 # For a simple wall (zero flux, zero velocity gradient approx?)
                 # Or fixed value BC?
                 # Add diffusion to a_p for owner
                 a_p[owner] += D
                 # Add convection based on outflow (if any assumed velocity like u_face=u_owner)
                 a_p[owner] += max(F, 0) # Needs review based on BC type

                 # Pressure Source Term (using owner pressure at boundary face)
                 p_face = p_flat[owner] # Extrapolate pressure to boundary face
                 pressure_force_component = p_face * (normal[0] if component == 'u' else normal[1]) * area
                 source[owner] -= pressure_force_component

        # --- REMOVED HARDCODED BC SECTION --- #
        # Boundary conditions are now handled in the AMGSolver after relaxation.

        # # --- HARDCODED BOUNDARY CONDITIONS for Cavity Flow --- 
        # # This section overrides/modifies the coefficients calculated above for boundary cells

        return {'a_p': a_p, 'a_nb': a_nb, 'source': source}