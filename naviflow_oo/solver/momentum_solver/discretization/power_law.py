"""
Power-law discretization scheme for momentum equations.
"""

import numpy as np
from ....constructor.boundary_conditions import BoundaryConditionManager, BoundaryType

class PowerLawDiscretization:
    """
    Power-law discretization scheme for convection-diffusion terms.
    """
    
    def __init__(self):
        """
        Initialize power law discretization.
        """
        pass
    
    @staticmethod
    def power_law_function(F, D):
        """
        The Power-Law Scheme function A(|P|) where P is the cell Peclet number |F/D|.
        
        Parameters:
        -----------
        F : float or ndarray
            Convection term
        D : float or ndarray
            Diffusion coefficient
            
        Returns:
        --------
        float or ndarray
            A(|P|) value according to power-law scheme
        """
        # Avoid division by zero and potential overflow
        with np.errstate(divide='ignore', invalid='ignore'):
            peclet_term = 0.1 * np.abs(F / D)
            # Ensure the base of the power is not negative to avoid large negative numbers leading to overflow
            base = np.maximum(0.0, 1.0 - peclet_term)
            result = np.where(np.abs(D) > 1e-10, base**5, 0.0)
            # Handle potential NaNs resulting from 0/0 or inf/inf in peclet_term calculation
            result = np.nan_to_num(result, nan=0.0) # Replace NaN with 0
        return result

    def calculate_u_coefficients(self, mesh, fluid, u, v, p, bc_manager=None):
        """
        Calculate coefficients for the u-momentum equation using a mesh-agnostic, face-based method.
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh (structured or unstructured)
        fluid : FluidProperties
            Fluid properties
        u, v : ndarray
            Current velocity fields
        p : ndarray
            Current pressure field
        bc_manager : BoundaryConditionManager, optional
            Boundary conditions
        
        Returns
        -------
        dict
            Dictionary containing 'a_p', 'a_nb', and 'source'
        """
        rho = fluid.get_density()
        mu = fluid.get_viscosity()

        owners, neighbors = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_normals = mesh.get_face_normals()
        face_centers = mesh.get_face_centers()
        cell_centers = mesh.get_cell_centers()
        n_faces = mesh.n_faces
        n_cells = mesh.n_cells
        
        # Get grid shape for safe access if structured
        grid_shape = None
        is_structured = False
        if hasattr(mesh, 'get_dimensions'):
            grid_shape = mesh.get_dimensions() # Returns (nx_cells, ny_cells)
            is_structured = True
        
        # Flatten input fields if they are 2D (structured)
        if is_structured and u.ndim == 2 and u.shape == grid_shape:
            u_flat = u.flatten()
        else:
            u_flat = u # Assume 1D or already flat
        if is_structured and v.ndim == 2 and v.shape == grid_shape:
            v_flat = v.flatten()
        else:
            v_flat = v
        if is_structured and p.ndim == 2 and p.shape == grid_shape:
            p_flat = p.flatten()
        else:
            p_flat = p
        
        # Ensure n_cells doesn't exceed array sizes
        u_size = u_flat.size if hasattr(u_flat, 'size') else 0
        v_size = v_flat.size if hasattr(v_flat, 'size') else 0
        p_size = p_flat.size if hasattr(p_flat, 'size') else 0
        actual_n_cells = min(n_cells, u_size, v_size, p_size) if all([u_size, v_size, p_size]) else n_cells

        # Initialize
        a_p = np.zeros(n_cells)
        source = np.zeros(n_cells)
        a_nb = {'face': np.zeros(n_faces)}

        # Compute convection and diffusion fluxes across faces
        # print(f"\n--- Discretization Debug --- (First 5 faces)") # DEBUG
        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]
            
            # Skip if owner index out of bounds
            if owner >= n_cells:
                 # if face_idx < 5: print(f"Face {face_idx}: Skipping, owner {owner} >= n_cells {n_cells}") # DEBUG
                 continue
                 
            # if face_idx < 5: print(f"\nFace {face_idx}: Owner={owner}, Neighbor={neighbor}") # DEBUG
            # if face_idx < 5: print(f"  Area={area:.4f}, Normal={normal}") # DEBUG

            # Compute relative position vector
            if neighbor != -1 and neighbor < n_cells:
                d_owner = face_centers[face_idx] - cell_centers[owner]
                d_neighbor = cell_centers[neighbor] - face_centers[face_idx]
                d_total = np.linalg.norm(cell_centers[neighbor] - cell_centers[owner])
            else:
                # Boundary face
                d_owner = face_centers[face_idx] - cell_centers[owner]
                d_total = np.linalg.norm(d_owner) # Note: Moukalled uses distance normal to face here?
            # if face_idx < 5: print(f"  d_total={d_total:.4f}") # DEBUG

            # Compute face velocity with direct access
            if neighbor != -1 and neighbor < n_cells:
                # Internal face - simple average of owner and neighbor
                u_own = u_flat[owner]
                u_nei = u_flat[neighbor]
                u_face = 0.5 * (u_own + u_nei)
                    
                v_own = v_flat[owner]
                v_nei = v_flat[neighbor]
                v_face = 0.5 * (v_own + v_nei)
            else:
                # Boundary face - use owner value 
                u_face = u_flat[owner]
                v_face = v_flat[owner]
            # if face_idx < 5: print(f"  u_face={u_face:.4f}, v_face={v_face:.4f}") # DEBUG
            
            # --- Re-assign normal and area just before use to fix NameError --- 
            normal = face_normals[face_idx]
            area = face_areas[face_idx]
            # -------------------------------------------------------------
            
            # Extract normal components (assuming normal is ndarray or None)
            if isinstance(normal, np.ndarray) and normal.size >= 2:
                nx_comp, ny_comp = normal[0], normal[1]
            else:
                nx_comp, ny_comp = 0.0, 0.0
            
            # Calculate dot product manually (assuming scalar u_face/v_face)
            vel_dot_normal = u_face * nx_comp + v_face * ny_comp
                
            # Convective flux (ρ * (U · n) * A)
            F_conv = rho * vel_dot_normal * area

            # Diffusive flux (μ * A / d_total)
            D_diff = mu * area / d_total if d_total > 1e-12 else 0.0
            # if face_idx < 5: print(f"  F_conv={F_conv:.4e}, D_diff={D_diff:.4e}") # DEBUG

            # Power law correction
            if abs(D_diff) < 1e-20:
                 correction = 0.0 # Avoid division by zero in A_Pe if D_diff is effectively zero
                 # if face_idx < 5: print(f"  Peclet -> inf, correction = 0.0") # DEBUG
            else:
                A_Pe = np.abs(F_conv / (D_diff))
                base = np.maximum(0.0, 1.0 - 0.1 * A_Pe)
                correction = base**5
                correction = np.nan_to_num(correction, nan=0.0) # Handle potential NaNs
                # if face_idx < 5: print(f"  Peclet={A_Pe:.4e}, correction={correction:.4e}") # DEBUG
            
            diffusion_term = D_diff * correction

            # Coefficient for this face
            face_coeff = diffusion_term + max(-F_conv, 0)
            # if face_idx < 5: print(f"  diffusion_term={diffusion_term:.4e}, face_coeff={face_coeff:.4e}") # DEBUG

            # Update matrix terms
            if neighbor != -1 and neighbor < n_cells:
                a_p[owner] += face_coeff
                a_p[neighbor] += face_coeff
                a_nb['face'][face_idx] = face_coeff
            else:
                # Boundary face
                boundary_name = mesh.get_boundary_name(face_idx)
                
                if boundary_name is not None and bc_manager is not None:
                    bc_values = bc_manager.get_condition(boundary_name, bc_type='velocity')
                    
                    if bc_values is not None:
                        # Dirichlet boundary condition (e.g., inlet velocity, moving wall)
                        u_wall = bc_values.get('u', 0.0)
                        
                        # Add to diagonal coefficient
                        a_p[owner] += face_coeff
                        
                        # Modify source: move missing neighbor contribution
                        source[owner] += face_coeff * u_wall
                    else:
                        # Wall (zero velocity)
                        a_p[owner] += face_coeff
                        # No source contribution needed for zero velocity
                else:
                    # No boundary name or no BC manager - assume wall (zero velocity)
                    a_p[owner] += face_coeff
            
            # Source term: pressure gradient term
            if neighbor != -1 and neighbor < n_cells:
                p_own = p_flat[owner]
                p_nei = p_flat[neighbor]
                dp = p_nei - p_own
            else:
                # Boundary face
                p_own = p_flat[owner]
                dp = -p_own 
            
            # --- Re-assign normal and area just before use to fix NameError --- 
            normal = face_normals[face_idx]
            area = face_areas[face_idx]
            # -------------------------------------------------------------
            
            # Add pressure contribution to source term (- integral P*nx dA)
            p_face = 0.5 * (p_own + p_nei) if neighbor != -1 else p_own # Simple average for face pressure
            pressure_force_term = p_face * nx_comp * area # Use nx_comp from dot product section
            source[owner] -= pressure_force_term # Pressure force term

            # Add pressure contribution to source term (- integral P*ny dA)
            p_face = 0.5 * (p_own + p_nei) if neighbor != -1 else p_own # Simple average
            pressure_force_term = p_face * ny_comp * area # Use ny_comp from dot product section
            source[owner] -= pressure_force_term # Pressure force term

        # print(f"--- End Discretization Debug --- \n") # DEBUG
        return {
            'a_p': a_p,
            'a_nb': a_nb,
            'source': source
        }

    def calculate_v_coefficients(self, mesh, fluid, u, v, p, bc_manager=None):
        """
        Calculate coefficients for the v-momentum equation using a mesh-agnostic, face-based method.
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh (structured or unstructured)
        fluid : FluidProperties
            Fluid properties
        u, v : ndarray
            Current velocity fields
        p : ndarray
            Current pressure field
        bc_manager : BoundaryConditionManager, optional
            Boundary conditions
        
        Returns
        -------
        dict
            Dictionary containing 'a_p', 'a_nb', and 'source'
        """
        rho = fluid.get_density()
        mu = fluid.get_viscosity()

        owners, neighbors = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_normals = mesh.get_face_normals()
        face_centers = mesh.get_face_centers()
        cell_centers = mesh.get_cell_centers()
        n_faces = mesh.n_faces
        n_cells = mesh.n_cells
        
        # Get grid shape for safe access if structured
        grid_shape = None
        is_structured = False
        if hasattr(mesh, 'get_dimensions'):
            grid_shape = mesh.get_dimensions() # Returns (nx_cells, ny_cells)
            is_structured = True
        
        # Flatten input fields if they are 2D (structured)
        if is_structured and u.ndim == 2 and u.shape == grid_shape:
            u_flat = u.flatten()
        else:
            u_flat = u # Assume 1D or already flat
        if is_structured and v.ndim == 2 and v.shape == grid_shape:
            v_flat = v.flatten()
        else:
            v_flat = v
        if is_structured and p.ndim == 2 and p.shape == grid_shape:
            p_flat = p.flatten()
        else:
            p_flat = p
        
        # Ensure n_cells doesn't exceed array sizes
        u_size = u_flat.size if hasattr(u_flat, 'size') else 0
        v_size = v_flat.size if hasattr(v_flat, 'size') else 0
        p_size = p_flat.size if hasattr(p_flat, 'size') else 0
        actual_n_cells = min(n_cells, u_size, v_size, p_size) if all([u_size, v_size, p_size]) else n_cells

        # Initialize
        a_p = np.zeros(n_cells)
        source = np.zeros(n_cells)
        a_nb = {'face': np.zeros(n_faces)}

        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]
            normal = face_normals[face_idx]
            area = face_areas[face_idx]
            
            # Skip if owner index out of bounds
            if owner >= n_cells:
                continue

            if neighbor != -1 and neighbor < n_cells:
                d_owner = face_centers[face_idx] - cell_centers[owner]
                d_neighbor = cell_centers[neighbor] - face_centers[face_idx]
                d_total = np.linalg.norm(cell_centers[neighbor] - cell_centers[owner])
            else:
                d_owner = face_centers[face_idx] - cell_centers[owner]
                d_total = np.linalg.norm(d_owner)

            # Compute face velocity with direct access
            if neighbor != -1 and neighbor < n_cells:
                # Internal face - simple average of owner and neighbor
                u_own = u_flat[owner]
                u_nei = u_flat[neighbor]
                u_face = 0.5 * (u_own + u_nei)
                    
                v_own = v_flat[owner]
                v_nei = v_flat[neighbor]
                v_face = 0.5 * (v_own + v_nei)
            else:
                # Boundary face - use owner value
                u_face = u_flat[owner]
                v_face = v_flat[owner]
            
            # --- Re-assign normal and area just before use to fix NameError --- 
            normal = face_normals[face_idx]
            area = face_areas[face_idx]
            # -------------------------------------------------------------
            
            # Extract normal components (assuming normal is ndarray or None)
            if isinstance(normal, np.ndarray) and normal.size >= 2:
                nx_comp, ny_comp = normal[0], normal[1]
            else:
                nx_comp, ny_comp = 0.0, 0.0
                
            # Calculate dot product manually (assuming scalar u_face/v_face)
            vel_dot_normal = u_face * nx_comp + v_face * ny_comp
                
            # Convective flux
            F_conv = rho * vel_dot_normal * area

            # Diffusive flux
            D_diff = mu * area / d_total if d_total > 1e-12 else 0.0

            # Power law correction
            A_Pe = np.abs(F_conv / (D_diff + 1e-20))
            correction = (1 - 0.1 * A_Pe)**5
            correction = np.maximum(0.0, correction)
            
            diffusion_term = D_diff * correction

            # Coefficient for this face
            face_coeff = diffusion_term + max(-F_conv, 0)

            # Update matrix terms
            if neighbor != -1 and neighbor < n_cells:
                a_p[owner] += face_coeff
                a_p[neighbor] += face_coeff
                a_nb['face'][face_idx] = face_coeff
            else:
                # Boundary face
                boundary_name = mesh.get_boundary_name(face_idx)
                
                if boundary_name is not None and bc_manager is not None:
                    bc_values = bc_manager.get_condition(boundary_name, bc_type='velocity')
                    
                    if bc_values is not None:
                        # Dirichlet boundary condition (e.g., inlet velocity, moving wall)
                        v_wall = bc_values.get('v', 0.0)
                        
                        # Add to diagonal coefficient
                        a_p[owner] += face_coeff
                        
                        # Modify source: move missing neighbor contribution
                        source[owner] += face_coeff * v_wall
                    else:
                        # Wall (zero velocity)
                        a_p[owner] += face_coeff
                        # No source contribution needed for zero velocity
                else:
                    # No boundary name or no BC manager - assume wall (zero velocity)
                    a_p[owner] += face_coeff

            # Pressure gradient source
            if neighbor != -1 and neighbor < n_cells:
                p_own = p_flat[owner]
                p_nei = p_flat[neighbor]
                dp = p_nei - p_own
            else:
                # Boundary face
                p_own = p_flat[owner]
                dp = -p_own # Mimic original
            
            # --- Re-assign normal and area just before use to fix NameError --- 
            normal = face_normals[face_idx]
            area = face_areas[face_idx]
            # -------------------------------------------------------------
            
            # Add pressure contribution to source term (- integral P*nx dA)
            p_face = 0.5 * (p_own + p_nei) if neighbor != -1 else p_own # Simple average for face pressure
            pressure_force_term = p_face * nx_comp * area # Use nx_comp from dot product section
            source[owner] -= pressure_force_term # Pressure force term

            # Add pressure contribution to source term (- integral P*ny dA)
            p_face = 0.5 * (p_own + p_nei) if neighbor != -1 else p_own # Simple average
            pressure_force_term = p_face * ny_comp * area # Use ny_comp from dot product section
            source[owner] -= pressure_force_term # Pressure force term

        return {
            'a_p': a_p,
            'a_nb': a_nb,
            'source': source
        }
