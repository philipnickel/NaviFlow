import numpy as np
from ..velocity_solver.base_velocity_solver import VelocityUpdater
from ...constructor.boundary_conditions import BoundaryConditionManager

class StandardVelocityUpdater(VelocityUpdater):
    """
    Standard implementation of velocity updater for collocated grids.
    Uses pressure correction to update velocities.
    """
    
    def update_velocity(self, mesh, u_star, v_star, p_prime, d_u, d_v, boundary_conditions):
        """
        Update velocities based on pressure correction (collocated layout).
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh
        u_star, v_star : ndarray
            Intermediate velocity fields (nx, ny)
        p_prime : ndarray
            Pressure correction field (nx, ny)
        d_u, d_v : ndarray
            Momentum equation coefficients (nx, ny)
        boundary_conditions : dict or BoundaryConditionManager
            Boundary conditions
            
        Returns
        -------
        u, v : ndarray
            Updated velocity fields
        """
        # Get cell sizes
        dx, dy = mesh.get_cell_sizes()
        
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        
        # Initialize arrays with u_star and v_star values
        u = u_star.copy()
        v = v_star.copy()
        
        # Calculate pressure gradients using central differences
        dpdx = np.zeros_like(p_prime)
        dpdy = np.zeros_like(p_prime)

        # Internal points: central differences
        dpdx[1:-1, :] = (p_prime[2:, :] - p_prime[:-2, :]) / (2 * dx)
        dpdy[:, 1:-1] = (p_prime[:, 2:] - p_prime[:, :-2]) / (2 * dy)
        
        # Boundary points: one-sided differences
        # Left boundary
        dpdx[0, :] = (p_prime[1, :] - p_prime[0, :]) / dx
        # Right boundary
        dpdx[-1, :] = (p_prime[-1, :] - p_prime[-2, :]) / dx
        # Bottom boundary
        dpdy[:, 0] = (p_prime[:, 1] - p_prime[:, 0]) / dy
        # Top boundary
        dpdy[:, -1] = (p_prime[:, -1] - p_prime[:, -2]) / dy

        # Update velocities based on pressure gradients
        u -= d_u * dpdx
        v -= d_v * dpdy
        
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
        u, v = bc_manager.apply_velocity_boundary_conditions(u, v, nx, ny)
        
        return u, v
