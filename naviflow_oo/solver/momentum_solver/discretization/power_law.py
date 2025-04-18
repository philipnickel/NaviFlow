"""
Power-law discretization scheme for momentum equations.
"""

import numpy as np

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
        return np.maximum(0, (1 - 0.1 * np.abs(F/D))**5)
    
    def calculate_u_coefficients(self, mesh, fluid, u, v, p):
        """
        Calculate coefficients for the u-momentum equation using power-law scheme.
        
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
            
        Returns:
        --------
        dict
            Dictionary containing coefficient arrays and source terms
        """
        # Get mesh and fluid properties
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho = fluid.get_density()
        mu = fluid.get_viscosity()
        
        # Initialize coefficient arrays
        a_e = np.zeros((nx+1, ny))
        a_w = np.zeros((nx+1, ny))
        a_n = np.zeros((nx+1, ny))
        a_s = np.zeros((nx+1, ny))
        a_p = np.zeros((nx+1, ny))
        source = np.zeros((nx+1, ny))
        
        # Diffusion coefficients
        De = mu * dy / dx
        Dw = mu * dy / dx
        Dn = mu * dx / dy
        Ds = mu * dx / dy
        
        # Interior points - vectorized computation
        i_range = np.arange(1, nx)
        j_range = np.arange(1, ny-1)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        # Calculate flow terms
        Fe = 0.5 * rho * dy * (u[i_grid+1, j_grid] + u[i_grid, j_grid])
        Fw = 0.5 * rho * dy * (u[i_grid-1, j_grid] + u[i_grid, j_grid])
        Fn = 0.5 * rho * dx * (v[i_grid, j_grid+1] + v[i_grid-1, j_grid+1])
        Fs = 0.5 * rho * dx * (v[i_grid, j_grid] + v[i_grid-1, j_grid])
        
        # Calculate coefficients using power-law scheme
        a_e[i_grid, j_grid] = De * self.power_law_function(Fe, De) + np.maximum(-Fe, 0)
        a_w[i_grid, j_grid] = Dw * self.power_law_function(Fw, Dw) + np.maximum(Fw, 0)
        a_n[i_grid, j_grid] = Dn * self.power_law_function(Fn, Dn) + np.maximum(-Fn, 0)
        a_s[i_grid, j_grid] = Ds * self.power_law_function(Fs, Ds) + np.maximum(Fs, 0)
        a_p[i_grid, j_grid] = a_e[i_grid, j_grid] + a_w[i_grid, j_grid] + a_n[i_grid, j_grid] + a_s[i_grid, j_grid] + (Fe-Fw) + (Fn-Fs)
        source[i_grid, j_grid] = (p[i_grid-1, j_grid] - p[i_grid, j_grid]) * dy
        
        # Bottom boundary (j=0)
        j = 0
        i_bottom = np.arange(1, nx)
        Fe_bottom = 0.5 * rho * dy * (u[i_bottom+1, j] + u[i_bottom, j])
        Fw_bottom = 0.5 * rho * dy * (u[i_bottom-1, j] + u[i_bottom, j])
        Fn_bottom = 0.5 * rho * dx * (v[i_bottom, j+1] + v[i_bottom-1, j+1])
        Fs_bottom = 0
        
        a_e[i_bottom, j] = De * self.power_law_function(Fe_bottom, De) + np.maximum(-Fe_bottom, 0)
        a_w[i_bottom, j] = Dw * self.power_law_function(Fw_bottom, Dw) + np.maximum(Fw_bottom, 0)
        a_n[i_bottom, j] = Dn * self.power_law_function(Fn_bottom, Dn) + np.maximum(-Fn_bottom, 0)
        a_s[i_bottom, j] = 0
        a_p[i_bottom, j] = a_e[i_bottom, j] + a_w[i_bottom, j] + a_n[i_bottom, j] + a_s[i_bottom, j] + (Fe_bottom-Fw_bottom) + (Fn_bottom-Fs_bottom)
        source[i_bottom, j] = (p[i_bottom-1, j] - p[i_bottom, j]) * dy
        
        # Top boundary (j=ny-1)
        j = ny-1
        i_top = np.arange(1, nx)
        Fe_top = 0.5 * rho * dy * (u[i_top+1, j] + u[i_top, j])
        Fw_top = 0.5 * rho * dy * (u[i_top-1, j] + u[i_top, j])
        Fn_top = 0
        Fs_top = 0.5 * rho * dx * (v[i_top, j] + v[i_top-1, j])
        
        a_e[i_top, j] = De * self.power_law_function(Fe_top, De) + np.maximum(-Fe_top, 0)
        a_w[i_top, j] = Dw * self.power_law_function(Fw_top, Dw) + np.maximum(Fw_top, 0)
        a_n[i_top, j] = 0
        a_s[i_top, j] = Ds * self.power_law_function(Fs_top, Ds) + np.maximum(Fs_top, 0)
        a_p[i_top, j] = a_e[i_top, j] + a_w[i_top, j] + a_n[i_top, j] + a_s[i_top, j] + (Fe_top-Fw_top) + (Fn_top-Fs_top)
        source[i_top, j] = (p[i_top-1, j] - p[i_top, j]) * dy
        
        # Return coefficients in a dictionary
        return {
            'a_e': a_e,
            'a_w': a_w,
            'a_n': a_n,
            'a_s': a_s,
            'a_p': a_p,
            'source': source
        }
    
    def calculate_v_coefficients(self, mesh, fluid, u, v, p):
        """
        Calculate coefficients for the v-momentum equation using power-law scheme.
        
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
            
        Returns:
        --------
        dict
            Dictionary containing coefficient arrays and source terms
        """
        # Get mesh and fluid properties
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho = fluid.get_density()
        mu = fluid.get_viscosity()
        
        # Initialize coefficient arrays
        a_e = np.zeros((nx, ny+1))
        a_w = np.zeros((nx, ny+1))
        a_n = np.zeros((nx, ny+1))
        a_s = np.zeros((nx, ny+1))
        a_p = np.zeros((nx, ny+1))
        source = np.zeros((nx, ny+1))
        
        # Diffusion coefficients
        De = mu * dy / dx
        Dw = mu * dy / dx
        Dn = mu * dx / dy
        Ds = mu * dx / dy
        
        # Interior points - vectorized computation
        i_range = np.arange(1, nx-1)
        j_range = np.arange(1, ny)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        # Calculate flow terms
        Fe = 0.5 * rho * dy * (u[i_grid+1, j_grid] + u[i_grid+1, j_grid-1])
        Fw = 0.5 * rho * dy * (u[i_grid, j_grid] + u[i_grid, j_grid-1])
        Fn = 0.5 * rho * dx * (v[i_grid, j_grid] + v[i_grid, j_grid+1])
        Fs = 0.5 * rho * dx * (v[i_grid, j_grid-1] + v[i_grid, j_grid])
        
        # Calculate coefficients using power-law scheme
        a_e[i_grid, j_grid] = De * self.power_law_function(Fe, De) + np.maximum(-Fe, 0)
        a_w[i_grid, j_grid] = Dw * self.power_law_function(Fw, Dw) + np.maximum(Fw, 0)
        a_n[i_grid, j_grid] = Dn * self.power_law_function(Fn, Dn) + np.maximum(-Fn, 0)
        a_s[i_grid, j_grid] = Ds * self.power_law_function(Fs, Ds) + np.maximum(Fs, 0)
        a_p[i_grid, j_grid] = a_e[i_grid, j_grid] + a_w[i_grid, j_grid] + a_n[i_grid, j_grid] + a_s[i_grid, j_grid] + (Fe-Fw) + (Fn-Fs)
        source[i_grid, j_grid] = (p[i_grid, j_grid-1] - p[i_grid, j_grid]) * dx
        
        # Left boundary (i=0)
        i = 0
        j_left = np.arange(1, ny)
        Fe_left = 0.5 * rho * dy * (u[i+1, j_left] + u[i+1, j_left-1])
        Fw_left = 0
        Fn_left = 0.5 * rho * dx * (v[i, j_left+1] + v[i, j_left])
        Fs_left = 0.5 * rho * dx * (v[i, j_left-1] + v[i, j_left])
        
        a_e[i, j_left] = De * self.power_law_function(Fe_left, De) + np.maximum(-Fe_left, 0)
        a_w[i, j_left] = 0
        a_n[i, j_left] = Dn * self.power_law_function(Fn_left, Dn) + np.maximum(-Fn_left, 0)
        a_s[i, j_left] = Ds * self.power_law_function(Fs_left, Ds) + np.maximum(Fs_left, 0)
        a_p[i, j_left] = a_e[i, j_left] + a_w[i, j_left] + a_n[i, j_left] + a_s[i, j_left] + (Fe_left-Fw_left) + (Fn_left-Fs_left)
        source[i, j_left] = (p[i, j_left-1] - p[i, j_left]) * dx
        
        # Right boundary (i=nx-1)
        i = nx-1
        j_right = np.arange(1, ny)
        Fe_right = 0
        Fw_right = 0.5 * rho * dy * (u[i, j_right] + u[i, j_right-1])
        Fn_right = 0.5 * rho * dx * (v[i, j_right+1] + v[i, j_right])
        Fs_right = 0.5 * rho * dx * (v[i, j_right-1] + v[i, j_right])
        
        a_e[i, j_right] = 0
        a_w[i, j_right] = Dw * self.power_law_function(Fw_right, Dw) + np.maximum(Fw_right, 0)
        a_n[i, j_right] = Dn * self.power_law_function(Fn_right, Dn) + np.maximum(-Fn_right, 0)
        a_s[i, j_right] = Ds * self.power_law_function(Fs_right, Ds) + np.maximum(Fs_right, 0)
        a_p[i, j_right] = a_e[i, j_right] + a_w[i, j_right] + a_n[i, j_right] + a_s[i, j_right] + (Fe_right-Fw_right) + (Fn_right-Fs_right)
        source[i, j_right] = (p[i, j_right-1] - p[i, j_right]) * dx
        
        # Return coefficients in a dictionary
        return {
            'a_e': a_e,
            'a_w': a_w,
            'a_n': a_n,
            'a_s': a_s,
            'a_p': a_p,
            'source': source
        } 