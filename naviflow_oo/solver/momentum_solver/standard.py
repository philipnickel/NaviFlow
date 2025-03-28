"""
Standard implementation of momentum solver.
"""

import numpy as np
from ..momentum_solver.base_momentum_solver import MomentumSolver
from ...constructor.boundary_conditions import BoundaryConditionManager
from .discretization.convection_schemes import PowerLawDiscretization

class StandardMomentumSolver(MomentumSolver):
    """
    Standard implementation of momentum equations solver.
    Uses a specified discretization scheme for convection-diffusion terms,
    defaulting to the Power Law scheme if none is provided.
    """
    
    def __init__(self, discretization_scheme=None):
        """
        Initialize the standard momentum solver with a discretization scheme.
        
        Parameters:
        -----------
        discretization_scheme : ConvectionDiscretization, optional
            Discretization scheme to use for convection-diffusion terms.
            If None, PowerLawDiscretization will be used.
        """
        super().__init__()
        # Use PowerLawDiscretization as default if none specified
        self.discretization_scheme = discretization_scheme if discretization_scheme else PowerLawDiscretization()
    
    def get_solver_info(self):
        """
        Get information about the solver.
        
        Returns:
        --------
        dict
            Dictionary containing solver information.
        """
        return {
            "solver_type": "Standard Momentum Solver",
            "discretization_scheme": self.discretization_scheme.get_name()
        }
    
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
        
        De = mu * dy / dx   # diffusion coefficients
        Dw = mu * dy / dx
        Dn = mu * dx / dy
        Ds = mu * dx / dy
        
        # Interior points - vectorized computation
        i_range = np.arange(1, imax)
        j_range = np.arange(1, jmax-1)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        # Calculate flow terms
        Fe = 0.5 * rho * dy * (u[i_grid+1, j_grid] + u[i_grid, j_grid])
        Fw = 0.5 * rho * dy * (u[i_grid-1, j_grid] + u[i_grid, j_grid])
        Fn = 0.5 * rho * dx * (v[i_grid, j_grid+1] + v[i_grid-1, j_grid+1])
        Fs = 0.5 * rho * dx * (v[i_grid, j_grid] + v[i_grid-1, j_grid])
        
        # Calculate Peclet numbers
        Pe_e = Fe / De
        Pe_w = Fw / Dw
        Pe_n = Fn / Dn
        Pe_s = Fs / Ds
        
        # Calculate coefficients using the discretization scheme
        aE = self.discretization_scheme.calculate_flux_coefficients(Fe, De, Pe_e)
        aW = self.discretization_scheme.calculate_flux_coefficients(Fw, Dw, Pe_w)
        aN = self.discretization_scheme.calculate_flux_coefficients(Fn, Dn, Pe_n)
        aS = self.discretization_scheme.calculate_flux_coefficients(Fs, Ds, Pe_s)
        
        # Complete the coefficients
        aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
        
        pressure_term = (p[i_grid-1, j_grid] - p[i_grid, j_grid]) * dy
        
        # Calculate u_star and d_u
        u_star[i_grid, j_grid] = alpha/aP * ((aE*u[i_grid+1, j_grid] + 
                                            aW*u[i_grid-1, j_grid] + 
                                            aN*u[i_grid, j_grid+1] + 
                                            aS*u[i_grid, j_grid-1]) + 
                                            pressure_term) + (1-alpha)*u[i_grid, j_grid]
        
        d_u[i_grid, j_grid] = alpha * dy / aP
        
        # Bottom boundary (j=0) - vectorized
        j = 0
        i_bottom = np.arange(1, imax)
        Fe_bottom = 0.5 * rho * dy * (u[i_bottom+1, j] + u[i_bottom, j])
        Fw_bottom = 0.5 * rho * dy * (u[i_bottom-1, j] + u[i_bottom, j])
        Fn_bottom = 0.5 * rho * dx * (v[i_bottom, j+1] + v[i_bottom-1, j+1])
        Fs_bottom = 0
        
        # Calculate Peclet numbers for bottom boundary
        Pe_e_bottom = Fe_bottom / De
        Pe_w_bottom = Fw_bottom / Dw
        Pe_n_bottom = Fn_bottom / Dn
        
        # Calculate coefficients for bottom boundary
        aE_bottom = self.discretization_scheme.calculate_flux_coefficients(Fe_bottom, De, Pe_e_bottom)
        aW_bottom = self.discretization_scheme.calculate_flux_coefficients(Fw_bottom, Dw, Pe_w_bottom)
        aN_bottom = self.discretization_scheme.calculate_flux_coefficients(Fn_bottom, Dn, Pe_n_bottom)
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
        
        # Calculate Peclet numbers for top boundary
        Pe_e_top = Fe_top / De
        Pe_w_top = Fw_top / Dw
        Pe_s_top = Fs_top / Ds
        
        # Calculate coefficients for top boundary
        aE_top = self.discretization_scheme.calculate_flux_coefficients(Fe_top, De, Pe_e_top)
        aW_top = self.discretization_scheme.calculate_flux_coefficients(Fw_top, Dw, Pe_w_top)
        aN_top = 0
        aS_top = self.discretization_scheme.calculate_flux_coefficients(Fs_top, Ds, Pe_s_top)
        
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
        
        De = mu * dy / dx   # diffusion coefficients
        Dw = mu * dy / dx
        Dn = mu * dx / dy
        Ds = mu * dx / dy
        
        # Interior points - vectorized computation
        i_range = np.arange(1, imax-1)
        j_range = np.arange(1, jmax)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        # Calculate flow terms
        Fe = 0.5 * rho * dy * (u[i_grid+1, j_grid] + u[i_grid+1, j_grid-1])
        Fw = 0.5 * rho * dy * (u[i_grid, j_grid] + u[i_grid, j_grid-1])
        Fn = 0.5 * rho * dx * (v[i_grid, j_grid] + v[i_grid, j_grid+1])
        Fs = 0.5 * rho * dx * (v[i_grid, j_grid-1] + v[i_grid, j_grid])
        
        # Calculate Peclet numbers
        Pe_e = Fe / De
        Pe_w = Fw / Dw
        Pe_n = Fn / Dn
        Pe_s = Fs / Ds
        
        # Calculate coefficients using the discretization scheme
        aE = self.discretization_scheme.calculate_flux_coefficients(Fe, De, Pe_e)
        aW = self.discretization_scheme.calculate_flux_coefficients(Fw, Dw, Pe_w)
        aN = self.discretization_scheme.calculate_flux_coefficients(Fn, Dn, Pe_n)
        aS = self.discretization_scheme.calculate_flux_coefficients(Fs, Ds, Pe_s)
        
        # Complete the coefficients
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
        
        # Calculate Peclet numbers for left boundary
        Pe_e_left = Fe_left / De
        Pe_n_left = Fn_left / Dn
        Pe_s_left = Fs_left / Ds
        
        # Calculate coefficients for left boundary
        aE_left = self.discretization_scheme.calculate_flux_coefficients(Fe_left, De, Pe_e_left)
        aW_left = 0
        aN_left = self.discretization_scheme.calculate_flux_coefficients(Fn_left, Dn, Pe_n_left)
        aS_left = self.discretization_scheme.calculate_flux_coefficients(Fs_left, Ds, Pe_s_left)
        
        aP_left = aE_left + aW_left + aN_left + aS_left + (Fe_left-Fw_left) + (Fn_left-Fs_left)
        d_v[i, j_left] = alpha * dx / aP_left
        
        # Right boundary (i=imax-1) - vectorized
        i = imax-1
        j_right = np.arange(1, jmax)
        Fe_right = 0
        Fw_right = 0.5 * rho * dy * (u[i, j_right] + u[i, j_right-1])
        Fn_right = 0.5 * rho * dx * (v[i, j_right+1] + v[i, j_right])
        Fs_right = 0.5 * rho * dx * (v[i, j_right-1] + v[i, j_right])
        
        # Calculate Peclet numbers for right boundary
        Pe_w_right = Fw_right / Dw
        Pe_n_right = Fn_right / Dn
        Pe_s_right = Fs_right / Ds
        
        # Calculate coefficients for right boundary
        aE_right = 0
        aW_right = self.discretization_scheme.calculate_flux_coefficients(Fw_right, Dw, Pe_w_right)
        aN_right = self.discretization_scheme.calculate_flux_coefficients(Fn_right, Dn, Pe_n_right)
        aS_right = self.discretization_scheme.calculate_flux_coefficients(Fs_right, Ds, Pe_s_right)
        
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