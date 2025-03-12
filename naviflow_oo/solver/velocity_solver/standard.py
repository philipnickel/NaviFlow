import numpy as np
from ..velocity_solver.base_velocity_solver import VelocityUpdater
from ...constructor.boundary_conditions import BoundaryConditionManager

class StandardVelocityUpdater(VelocityUpdater):
    """
    Standard implementation of velocity updater.
    Uses pressure correction to update velocities.
    """
    
    def update_velocity(self, mesh, u_star, v_star, p_prime, d_u, d_v, boundary_conditions):
        """
        Update velocities based on pressure correction.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        u_star, v_star : ndarray
            Intermediate velocity fields
        p_prime : ndarray
            Pressure correction field
        d_u, d_v : ndarray
            Momentum equation coefficients
        boundary_conditions : dict or BoundaryConditionManager
            Boundary conditions
            
        Returns:
        --------
        u, v : ndarray
            Updated velocity fields
        """
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        
        # For compatibility with existing code
        imax, jmax = nx, ny
        
        # Initialize arrays with u_star and v_star values
        u = u_star.copy()
        v = v_star.copy()
        
        # Vectorized u velocity update for interior nodes
        i_range = np.arange(1, imax)
        j_range = np.arange(1, jmax-1)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        u[i_grid, j_grid] = u_star[i_grid, j_grid] + d_u[i_grid, j_grid] * (p_prime[i_grid-1, j_grid] - p_prime[i_grid, j_grid])
        
        # Vectorized v velocity update for interior nodes
        i_range = np.arange(1, imax-1)
        j_range = np.arange(1, jmax)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        v[i_grid, j_grid] = v_star[i_grid, j_grid] + d_v[i_grid, j_grid] * (p_prime[i_grid, j_grid-1] - p_prime[i_grid, j_grid])
        
        # Apply boundary conditions
        if isinstance(boundary_conditions, BoundaryConditionManager):
            bc_manager = boundary_conditions
        else:
            # Create a temporary boundary condition manager
            bc_manager = BoundaryConditionManager()
            for boundary, conditions in boundary_conditions.items():
                for field_type, values in conditions.items():
                    bc_manager.set_condition(boundary, field_type, values)
        
        # Apply velocity boundary conditions
        u, v = bc_manager.apply_velocity_boundary_conditions(u, v, imax, jmax)
        
        return u, v

 