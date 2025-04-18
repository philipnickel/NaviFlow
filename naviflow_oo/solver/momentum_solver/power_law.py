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
    
    def __init__(self):
        """
        Initialize the momentum solver.
        """
        # Initialize coefficient matrices for u momentum equation
        self.u_a_e = None
        self.u_a_w = None
        self.u_a_n = None
        self.u_a_s = None
        self.u_a_p = None
        self.u_source = None
        
        # Initialize coefficient matrices for v momentum equation
        self.v_a_e = None
        self.v_a_w = None
        self.v_a_n = None
        self.v_a_s = None
        self.v_a_p = None
        self.v_source = None
    
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
        
        # Initialize coefficient arrays for storage
        self.u_a_e = np.zeros((imax+1, jmax))
        self.u_a_w = np.zeros((imax+1, jmax))
        self.u_a_n = np.zeros((imax+1, jmax))
        self.u_a_s = np.zeros((imax+1, jmax))
        self.u_a_p = np.zeros((imax+1, jmax))
        self.u_source = np.zeros((imax+1, jmax))
        
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
        
        # Store coefficients for residual calculation
        self.u_a_e[i_grid, j_grid] = aE
        self.u_a_w[i_grid, j_grid] = aW
        self.u_a_n[i_grid, j_grid] = aN
        self.u_a_s[i_grid, j_grid] = aS
        self.u_a_p[i_grid, j_grid] = aP
        self.u_source[i_grid, j_grid] = pressure_term
        
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
        
        # Store coefficients for bottom boundary
        self.u_a_e[i_bottom, j] = aE_bottom
        self.u_a_w[i_bottom, j] = aW_bottom
        self.u_a_n[i_bottom, j] = aN_bottom
        self.u_a_s[i_bottom, j] = aS_bottom
        self.u_a_p[i_bottom, j] = aP_bottom
        self.u_source[i_bottom, j] = (p[i_bottom-1, j] - p[i_bottom, j]) * dy
        
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
        
        # Store coefficients for top boundary
        self.u_a_e[i_top, j] = aE_top
        self.u_a_w[i_top, j] = aW_top
        self.u_a_n[i_top, j] = aN_top
        self.u_a_s[i_top, j] = aS_top
        self.u_a_p[i_top, j] = aP_top
        self.u_source[i_top, j] = (p[i_top-1, j] - p[i_top, j]) * dy
        
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
            """
            u_star[0, :] = 0.0                      # left wall
            u_star[imax, :] = 0.0                   # right wall
            u_star[:, 0] = 0.0                      # bottom wall
            u_star[:, jmax-1] = 0.0                 # top wall
            """
        
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
        
        # Initialize coefficient arrays for storage
        self.v_a_e = np.zeros((imax, jmax+1))
        self.v_a_w = np.zeros((imax, jmax+1))
        self.v_a_n = np.zeros((imax, jmax+1))
        self.v_a_s = np.zeros((imax, jmax+1))
        self.v_a_p = np.zeros((imax, jmax+1))
        self.v_source = np.zeros((imax, jmax+1))
        
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
        
        # Store coefficients for residual calculation
        self.v_a_e[i_grid, j_grid] = aE
        self.v_a_w[i_grid, j_grid] = aW
        self.v_a_n[i_grid, j_grid] = aN
        self.v_a_s[i_grid, j_grid] = aS
        self.v_a_p[i_grid, j_grid] = aP
        self.v_source[i_grid, j_grid] = pressure_term
        
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
        
        # Store coefficients for left boundary
        self.v_a_e[i, j_left] = aE_left
        self.v_a_w[i, j_left] = aW_left
        self.v_a_n[i, j_left] = aN_left
        self.v_a_s[i, j_left] = aS_left
        self.v_a_p[i, j_left] = aP_left
        self.v_source[i, j_left] = (p[i, j_left-1] - p[i, j_left]) * dx
        
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
        
        # Store coefficients for right boundary
        self.v_a_e[i, j_right] = aE_right
        self.v_a_w[i, j_right] = aW_right
        self.v_a_n[i, j_right] = aN_right
        self.v_a_s[i, j_right] = aS_right
        self.v_a_p[i, j_right] = aP_right
        self.v_source[i, j_right] = (p[i, j_right-1] - p[i, j_right]) * dx
        
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