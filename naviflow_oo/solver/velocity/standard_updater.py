import numpy as np
from ..base_velocity_updater import VelocityUpdater

class StandardVelocityUpdater(VelocityUpdater):
    """
    Standard velocity updater implementing the SIMPLE algorithm's velocity correction step.
    Works with any mesh type by using a mesh-agnostic approach.
    """
    
    def __init__(self, relaxation_factor=0.7):
        """
        Initialize the standard velocity updater.
        
        Parameters
        ----------
        relaxation_factor : float, optional
            Relaxation factor for velocity correction
        """
        super().__init__()
        self.relaxation_factor = relaxation_factor
        self.mesh = None
        self.fluid = None
        
    def setup(self, mesh, fluid_properties):
        """
        Set up the velocity updater with mesh and fluid properties.
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh
        fluid_properties : FluidProperties
            The fluid properties
        """
        self.mesh = mesh
        self.fluid = fluid_properties
    
    def correct_velocity(self, u_star, v_star, p_prime, d_u, d_v):
        """
        Correct the predicted velocities using the pressure correction.
        
        Parameters
        ----------
        u_star, v_star : ndarray
            Predicted velocity fields (1D arrays)
        p_prime : ndarray
            Pressure correction field (1D array)
        d_u, d_v : ndarray
            Diagonal coefficients from momentum matrix (1D arrays)
            
        Returns
        -------
        tuple
            (u_new, v_new) corrected velocity fields as 1D arrays
        """
        # Ensure input arrays are 1D
        u_star = u_star.flatten() if u_star.ndim > 1 else u_star
        v_star = v_star.flatten() if v_star.ndim > 1 else v_star
        p_prime = p_prime.flatten() if p_prime.ndim > 1 else p_prime
        d_u = d_u.flatten() if d_u.ndim > 1 else d_u
        d_v = d_v.flatten() if d_v.ndim > 1 else d_v
        
        # Get cell centers and face areas
        cell_centers = self.mesh.cell_centers
        n_cells = len(cell_centers)
        
        # Initialize corrected velocities
        u_new = np.copy(u_star)
        v_new = np.copy(v_star)
        
        # Calculate velocity corrections from pressure gradients
        # For each internal cell
        for i in range(n_cells):
            # Skip boundary cells where velocities are fixed
            if self.mesh.is_boundary_cell(i):
                continue
                
            # Get cell neighbors and face areas for calculating gradients
            neighbors = self.mesh.get_cell_neighbors(i)
            faces = self.mesh.get_cell_faces(i)
            
            # Initialize pressure gradients
            dp_dx = 0.0
            dp_dy = 0.0
            
            # Calculate pressure gradients using face-based approach
            for j, neighbor_idx in enumerate(neighbors):
                if neighbor_idx < 0:  # Skip if no neighbor (boundary)
                    continue
                    
                # Get face area vector
                face_idx = faces[j]
                face_area = self.mesh.face_areas[face_idx]
                face_normal = self.mesh.face_normals[face_idx]
                
                # Calculate pressure difference across face
                dp = p_prime[neighbor_idx] - p_prime[i]
                
                # Add contribution to pressure gradients
                dp_dx += dp * face_normal[0] * face_area
                dp_dy += dp * face_normal[1] * face_area
                
            # Get cell volume for normalization
            cell_volume = self.mesh.cell_volumes[i]
            
            # Normalize gradients
            if cell_volume > 0:
                dp_dx /= cell_volume
                dp_dy /= cell_volume
            
            # Apply velocity corrections
            if d_u[i] > 0:
                u_correction = -dp_dx / d_u[i]
                u_new[i] = u_star[i] + self.relaxation_factor * u_correction
                
            if d_v[i] > 0:
                v_correction = -dp_dy / d_v[i]
                v_new[i] = v_star[i] + self.relaxation_factor * v_correction
        
        return u_new, v_new
    
    def calculate_mass_flux(self, u, v, p=None, d_u=None, d_v=None):
        """
        Calculate the mass flux through cell faces.
        Optionally applies Rhie-Chow correction if pressure and diagonal coefficients provided.
        
        Parameters
        ----------
        u, v : ndarray
            Velocity fields as 1D arrays
        p : ndarray, optional
            Pressure field as a 1D array (for Rhie-Chow correction)
        d_u, d_v : ndarray, optional
            Diagonal coefficients from momentum matrix (for Rhie-Chow correction)
            
        Returns
        -------
        ndarray
            Mass flux through each face
        """
        # Ensure input arrays are 1D
        u = u.flatten() if u.ndim > 1 else u
        v = v.flatten() if v.ndim > 1 else v
        
        # Get mesh data
        n_faces = self.mesh.n_faces
        face_areas = self.mesh.face_areas
        face_normals = self.mesh.face_normals
        owner_cells = self.mesh.face_owner_cells
        neighbor_cells = self.mesh.face_neighbor_cells
        
        # Initialize mass flux array
        mass_flux = np.zeros(n_faces)
        
        # Apply Rhie-Chow correction if pressure and diagonal coefficients are provided
        use_rhie_chow = (p is not None and d_u is not None and d_v is not None)
        
        if use_rhie_chow:
            # Ensure pressure and diagonal arrays are 1D
            p = p.flatten() if p.ndim > 1 else p
            d_u = d_u.flatten() if d_u.ndim > 1 else d_u
            d_v = d_v.flatten() if d_v.ndim > 1 else d_v
        
        # Calculate mass flux for each face
        for face_idx in range(n_faces):
            owner_idx = owner_cells[face_idx]
            neighbor_idx = neighbor_cells[face_idx]
            
            # Skip boundary faces (handled separately)
            if neighbor_idx < 0:
                # Boundary face handling
                # For boundary faces, use the face-normal velocity directly
                u_face = u[owner_idx]
                v_face = v[owner_idx]
            else:
                # Internal face - interpolate velocity from cell centers
                # Use linear interpolation 
                u_face = 0.5 * (u[owner_idx] + u[neighbor_idx])
                v_face = 0.5 * (v[owner_idx] + v[neighbor_idx])
                
                # Apply Rhie-Chow correction to avoid checkerboard oscillations 
                if use_rhie_chow:
                    # Calculate pressure gradient at face
                    dp_dx = 0.5 * (p[neighbor_idx] - p[owner_idx]) * face_normals[face_idx][0]
                    dp_dy = 0.5 * (p[neighbor_idx] - p[owner_idx]) * face_normals[face_idx][1]
                    
                    # Get average inverse diagonal coefficients
                    inv_d_u_avg = 0.5 * (1.0/max(d_u[owner_idx], 1e-12) + 1.0/max(d_u[neighbor_idx], 1e-12))
                    inv_d_v_avg = 0.5 * (1.0/max(d_v[owner_idx], 1e-12) + 1.0/max(d_v[neighbor_idx], 1e-12))
                    
                    # Apply Rhie-Chow correction
                    u_face -= dp_dx * inv_d_u_avg
                    v_face -= dp_dy * inv_d_v_avg
            
            # Calculate dot product of velocity and face normal to get mass flux
            velocity_dot_normal = u_face * face_normals[face_idx][0] + v_face * face_normals[face_idx][1]
            
            # Multiply by density and face area
            mass_flux[face_idx] = self.fluid.density * velocity_dot_normal * face_areas[face_idx]
        
        return mass_flux 