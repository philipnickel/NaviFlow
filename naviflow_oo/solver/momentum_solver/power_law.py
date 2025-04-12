"""
Standard implementation of momentum solver.
"""

import numpy as np
from .base_momentum_solver import MomentumSolver
from ...constructor.boundary_conditions import BoundaryConditionManager

class PowerLawMomentumSolver(MomentumSolver):
    """
    Standard implementation of momentum equations solver.
    Uses power-law scheme for convection-diffusion terms.
    """
    
    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        """
        Solve the u-momentum equation.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        u, v : ndarray
            Current velocity fields
        p : ndarray
            Current pressure field
        relaxation_factor : float, optional
            Relaxation factor for the momentum equation
        boundary_conditions : dict or BoundaryConditionManager, optional
            Boundary conditions
            
        Returns:
        --------
        u_star, d_u : ndarray
            Intermediate velocity field and momentum equation coefficient
        """
        # Get mesh and fluid properties
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho = fluid.get_density()
        mu = fluid.get_viscosity()
        
        # For compatibility with existing code
        imax, jmax = nx, ny
        alpha = relaxation_factor
        
        # Initialize arrays
        u_star = np.zeros((imax+1, jmax))
        d_u = np.zeros((imax+1, jmax))
        
        De = mu * dy / dx   # convective coefficients
        Dw = mu * dy / dx
        Dn = mu * dx / dy
        Ds = mu * dx / dy
        
        # Define the power-law function A (vectorized version)
        def A(F, D):
            return np.maximum(0, (1 - 0.1 * np.abs(F/D))**5)
        
        # Interior points - vectorized computation
        i_range = np.arange(1, imax)
        j_range = np.arange(1, jmax-1)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        # Calculate flow terms
        Fe = 0.5 * rho * dy * (u[i_grid+1, j_grid] + u[i_grid, j_grid])
        Fw = 0.5 * rho * dy * (u[i_grid-1, j_grid] + u[i_grid, j_grid])
        Fn = 0.5 * rho * dx * (v[i_grid, j_grid+1] + v[i_grid-1, j_grid+1])
        Fs = 0.5 * rho * dx * (v[i_grid, j_grid] + v[i_grid-1, j_grid])
        
        # Calculate coefficients
        aE = De * A(Fe, De) + np.maximum(-Fe, 0)
        aW = Dw * A(Fw, Dw) + np.maximum(Fw, 0)
        aN = Dn * A(Fn, Dn) + np.maximum(-Fn, 0)
        aS = Ds * A(Fs, Ds) + np.maximum(Fs, 0)
        aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
        
        pressure_term = (p[i_grid-1, j_grid] - p[i_grid, j_grid]) * dy
        
        # Calculate u_star and d_u
        u_star[i_grid, j_grid] = alpha/aP * ((aE*u[i_grid+1, j_grid] + 
                                            aW*u[i_grid-1, j_grid] + 
                                            aN*u[i_grid, j_grid+1] + 
                                            aS*u[i_grid, j_grid-1]) + 
                                            pressure_term) + (1-alpha)*u[i_grid, j_grid]
        
        d_u[i_grid, j_grid] = alpha * dy / aP
        
        # Bottom boundary (j=0) - can also be vectorized
        j = 0
        i_bottom = np.arange(1, imax)
        Fe_bottom = 0.5 * rho * dy * (u[i_bottom+1, j] + u[i_bottom, j])
        Fw_bottom = 0.5 * rho * dy * (u[i_bottom-1, j] + u[i_bottom, j])
        Fn_bottom = 0.5 * rho * dx * (v[i_bottom, j+1] + v[i_bottom-1, j+1])
        Fs_bottom = 0
        
        aE_bottom = De * A(Fe_bottom, De) + np.maximum(-Fe_bottom, 0)
        aW_bottom = Dw * A(Fw_bottom, Dw) + np.maximum(Fw_bottom, 0)
        aN_bottom = Dn * A(Fn_bottom, Dn) + np.maximum(-Fn_bottom, 0)
        aS_bottom = 0
        aP_bottom = aE_bottom + aW_bottom + aN_bottom + aS_bottom + (Fe_bottom-Fw_bottom) + (Fn_bottom-Fs_bottom)
        d_u[i_bottom, j] = alpha * dy / aP_bottom
        
        # Top boundary (j=jmax-1) - vectorized
        j = jmax-1
        i_top = np.arange(1, imax)
        Fe_top = 0.5 * rho * dy * (u[i_top+1, j] + u[i_top, j])
        Fw_top = 0.5 * rho * dy * (u[i_top-1, j] + u[i_top, j])
        Fn_top = 0
        Fs_top = 0.5 * rho * dx * (v[i_top, j] + v[i_top-1, j])
        
        aE_top = De * A(Fe_top, De) + np.maximum(-Fe_top, 0)
        aW_top = Dw * A(Fw_top, Dw) + np.maximum(Fw_top, 0)
        aN_top = 0
        aS_top = Ds * A(Fs_top, Ds) + np.maximum(Fs_top, 0)
        aP_top = aE_top + aW_top + aN_top + aS_top + (Fe_top-Fw_top) + (Fn_top-Fs_top)
        d_u[i_top, j] = alpha * dy / aP_top
        
        # Apply boundary conditions
        if boundary_conditions:
            if isinstance(boundary_conditions, BoundaryConditionManager):
                bc_manager = boundary_conditions
            else:
                # Create a temporary boundary condition manager
                bc_manager = BoundaryConditionManager()
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)
            
            # Apply velocity boundary conditions
            u_star, _ = bc_manager.apply_velocity_boundary_conditions(u_star, v.copy(), imax, jmax)
        else:
            # Default: initialize all boundaries to zero (wall condition)
            u_star[0, :] = 0.0                      # left wall
            u_star[imax, :] = 0.0                   # right wall
            u_star[:, 0] = 0.0                      # bottom wall
            u_star[:, jmax-1] = 0.0                 # top wall
        
        return u_star, d_u
    
    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        """
        Solve the v-momentum equation.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        u, v : ndarray
            Current velocity fields
        p : ndarray
            Current pressure field
        relaxation_factor : float, optional
            Relaxation factor for the momentum equation
        boundary_conditions : dict or BoundaryConditionManager, optional
            Boundary conditions
            
        Returns:
        --------
        v_star, d_v : ndarray
            Intermediate velocity field and momentum equation coefficient
        """
        # Get mesh and fluid properties
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho = fluid.get_density()
        mu = fluid.get_viscosity()
        
        # For compatibility with existing code
        imax, jmax = nx, ny
        alpha = relaxation_factor
        
        v_star = np.zeros((imax, jmax+1))
        d_v = np.zeros((imax, jmax+1))
        
        De = mu * dy / dx   # convective coefficients
        Dw = mu * dy / dx
        Dn = mu * dx / dy
        Ds = mu * dx / dy
        
        # Define the power-law function A (vectorized version)
        def A(F, D):
            return np.maximum(0, (1 - 0.1 * np.abs(F/D))**5)
        
        # Interior points - vectorized computation
        i_range = np.arange(1, imax-1)
        j_range = np.arange(1, jmax)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        # Calculate flow terms
        Fe = 0.5 * rho * dy * (u[i_grid+1, j_grid] + u[i_grid+1, j_grid-1])
        Fw = 0.5 * rho * dy * (u[i_grid, j_grid] + u[i_grid, j_grid-1])
        Fn = 0.5 * rho * dx * (v[i_grid, j_grid] + v[i_grid, j_grid+1])
        Fs = 0.5 * rho * dx * (v[i_grid, j_grid-1] + v[i_grid, j_grid])
        
        # Calculate coefficients
        aE = De * A(Fe, De) + np.maximum(-Fe, 0)
        aW = Dw * A(Fw, Dw) + np.maximum(Fw, 0)
        aN = Dn * A(Fn, Dn) + np.maximum(-Fn, 0)
        aS = Ds * A(Fs, Ds) + np.maximum(Fs, 0)
        aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
        
        pressure_term = (p[i_grid, j_grid-1] - p[i_grid, j_grid]) * dx
        
        # Calculate v_star and d_v
        v_star[i_grid, j_grid] = alpha/aP * ((aE*v[i_grid+1, j_grid] + 
                                            aW*v[i_grid-1, j_grid] + 
                                            aN*v[i_grid, j_grid+1] + 
                                            aS*v[i_grid, j_grid-1]) + 
                                            pressure_term) + (1-alpha)*v[i_grid, j_grid]
        
        d_v[i_grid, j_grid] = alpha * dx / aP
        
        # Left boundary (i=0) - vectorized
        i = 0
        j_left = np.arange(1, jmax)
        Fe_left = 0.5 * rho * dy * (u[i+1, j_left] + u[i+1, j_left-1])
        Fw_left = 0
        Fn_left = 0.5 * rho * dx * (v[i, j_left+1] + v[i, j_left])
        Fs_left = 0.5 * rho * dx * (v[i, j_left-1] + v[i, j_left])
        
        aE_left = De * A(Fe_left, De) + np.maximum(-Fe_left, 0)
        aW_left = 0
        aN_left = Dn * A(Fn_left, Dn) + np.maximum(-Fn_left, 0)
        aS_left = Ds * A(Fs_left, Ds) + np.maximum(Fs_left, 0)
        aP_left = aE_left + aW_left + aN_left + aS_left + (Fe_left-Fw_left) + (Fn_left-Fs_left)
        d_v[i, j_left] = alpha * dx / aP_left
        
        # Right boundary (i=imax-1) - vectorized
        i = imax-1
        j_right = np.arange(1, jmax)
        Fe_right = 0
        Fw_right = 0.5 * rho * dy * (u[i, j_right] + u[i, j_right-1])
        Fn_right = 0.5 * rho * dx * (v[i, j_right+1] + v[i, j_right])
        Fs_right = 0.5 * rho * dx * (v[i, j_right-1] + v[i, j_right])
        
        aE_right = 0
        aW_right = Dw * A(Fw_right, Dw) + np.maximum(Fw_right, 0)
        aN_right = Dn * A(Fn_right, Dn) + np.maximum(-Fn_right, 0)
        aS_right = Ds * A(Fs_right, Ds) + np.maximum(Fs_right, 0)
        aP_right = aE_right + aW_right + aN_right + aS_right + (Fe_right-Fw_right) + (Fn_right-Fs_right)
        d_v[i, j_right] = alpha * dx / aP_right
        
        # Apply boundary conditions
        if boundary_conditions:
            if isinstance(boundary_conditions, BoundaryConditionManager):
                bc_manager = boundary_conditions
            else:
                # Create a temporary boundary condition manager
                bc_manager = BoundaryConditionManager()
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)
            
            # Apply velocity boundary conditions
            _, v_star = bc_manager.apply_velocity_boundary_conditions(u.copy(), v_star, imax, jmax)
     
        return v_star, d_v 