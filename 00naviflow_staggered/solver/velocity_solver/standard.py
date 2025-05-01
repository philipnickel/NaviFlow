"""
Standard velocity updater implementation.
"""

import numpy as np
from ..velocity_solver.base_velocity_solver import VelocityUpdater
from ...constructor.boundary_conditions import BoundaryConditionManager

class StandardVelocityUpdater(VelocityUpdater):
    """
    Implementation of a standard velocity updater using face-based pressure correction.
    Handles 1D arrays consistently.
    """
    
    def __init__(self):
        """
        Initialize the StandardVelocityUpdater.
        """
        super().__init__()
        
    def update_velocity(self, mesh, u_star, v_star, p_prime, d_u, d_v, boundary_conditions=None):
        """
        Update velocity fields based on pressure correction using face pressure differences.
        Consistent with standard SIMPLE formulation.
        u = u* - d_u * grad(p')_x * Vp  (approximated via face summation)
        v = v* - d_v * grad(p')_y * Vp  (approximated via face summation)
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh
        u_star, v_star : ndarray
            Intermediate velocity fields (1D arrays of size n_cells)
        p_prime : ndarray
            Pressure correction field (1D array of size n_cells)
        d_u, d_v : ndarray
            Momentum diagonal coefficients (1D arrays of size n_cells)
            Assumed definition: d_u = Vp / aP_u_unrelaxed.
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions manager
        
        Returns
        -------
        u_new, v_new : ndarray
            Updated velocity fields (returned as 1D arrays)
        """
        # Ensure all inputs are 1D
        u_star_flat = u_star.flatten() if u_star.ndim > 1 else u_star
        v_star_flat = v_star.flatten() if v_star.ndim > 1 else v_star
        p_prime_flat = p_prime.flatten() if p_prime.ndim > 1 else p_prime
        d_u_flat = d_u.flatten() if d_u.ndim > 1 else d_u
        d_v_flat = d_v.flatten() if d_v.ndim > 1 else d_v
        
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        owners, neighbors = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_normals = mesh.get_face_normals()
        cell_volumes = mesh.get_cell_volumes() # Needed if d_u/d_v don't include Vp
        
        # Initialize new velocity fields as copies of the intermediate fields
        u_new = u_star_flat.copy()
        v_new = v_star_flat.copy()
        
        # --- Velocity Update using Face Pressure Summation --- 
        # grad(p')_x * Vp is approximated by Sum_faces( p'_face * normal_x * Area_face )
        
        # Initialize gradient * volume terms
        grad_p_prime_x_vol = np.zeros(n_cells)
        grad_p_prime_y_vol = np.zeros(n_cells)
        
        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]
            area = face_areas[face_idx]
            normal = face_normals[face_idx]
            
            # Ensure owner is valid
            if owner < 0 or owner >= n_cells:
                continue
                
            # Interpolate p' to face center
            if neighbor >= 0 and neighbor < n_cells:
                # Internal face: Linear interpolation
                # dist_own = np.linalg.norm(mesh.get_face_centers()[face_idx] - mesh.get_cell_centers()[owner])
                # dist_nei = np.linalg.norm(mesh.get_cell_centers()[neighbor] - mesh.get_face_centers()[face_idx])
                # weight_own = dist_nei / (dist_own + dist_nei)
                # weight_nei = dist_own / (dist_own + dist_nei)
                # p_prime_face = weight_own * p_prime_flat[owner] + weight_nei * p_prime_flat[neighbor]
                # --- Use simple average for now --- 
                p_prime_face = 0.5 * (p_prime_flat[owner] + p_prime_flat[neighbor])
            else:
                # Boundary face: Assume zero Neumann gradient (dp'/dn = 0) -> p'_face = p'_owner
                p_prime_face = p_prime_flat[owner]
            
            # Accumulate pressure force term (p' * normal * Area) for the owner cell
            pressure_force_x_face = p_prime_face * normal[0] * area
            pressure_force_y_face = p_prime_face * normal[1] * area
            
            grad_p_prime_x_vol[owner] += pressure_force_x_face
            grad_p_prime_y_vol[owner] += pressure_force_y_face
            
            # If internal face, also add contribution to neighbor (force is opposite)
            if neighbor >= 0 and neighbor < n_cells:
                grad_p_prime_x_vol[neighbor] -= pressure_force_x_face
                grad_p_prime_y_vol[neighbor] -= pressure_force_y_face

        # Perform the velocity update: u = u* - d_u * (Sum P'_f Nx Af) = u* - d_u * (gradP'_x * Vol)
        # Ensure d_u/d_v are defined as Vp/aP before using this directly.
        # If d_u/d_v are 1/aP, need to multiply by Vp here.
        # Let's assume d_u = Vp/aP based on AMGSolver implementation.
        u_new -= d_u_flat * grad_p_prime_x_vol
        v_new -= d_v_flat * grad_p_prime_y_vol
            
        # Apply boundary conditions if provided (using the passed manager)
        if boundary_conditions is not None:
             # Apply boundary conditions directly to the corrected 1D fields
             # Ensure the BC application method exists in the manager
             if hasattr(boundary_conditions, 'apply_velocity_boundary_conditions_to_flat'):
                 u_new, v_new = boundary_conditions.apply_velocity_boundary_conditions_to_flat(u_new, v_new, mesh)
             else:
                 print("Warning: BoundaryConditionManager does not have 'apply_velocity_boundary_conditions_to_flat' method.")
                 # Potentially fall back to structured assumption if needed, but ideally should be unified.
                
        return u_new, v_new
