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
        Calculate coefficients for the u-momentum equation using power-law scheme.
        Implements Practice B for boundary treatment.
        
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
        bc_manager : BoundaryConditionManager, optional
            Boundary condition manager
            
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
        
        # Diffusion coefficients - constant for all cells
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
        
        # Base a_p calculation (before boundary modifications)
        a_p[i_grid, j_grid] = a_e[i_grid, j_grid] + a_w[i_grid, j_grid] + a_n[i_grid, j_grid] + a_s[i_grid, j_grid] + (Fe-Fw) + (Fn-Fs)
        
        # Pressure gradient source term (before boundary modifications)
        source[i_grid, j_grid] = (p[i_grid-1, j_grid] - p[i_grid, j_grid]) * dy
        
        # Bottom boundary (j=0)
        j = 0
        i_bottom = np.arange(1, nx)
        Fe_bottom = 0.5 * rho * dy * (u[i_bottom+1, j] + u[i_bottom, j])
        Fw_bottom = 0.5 * rho * dy * (u[i_bottom-1, j] + u[i_bottom, j])
        Fn_bottom = 0.5 * rho * dx * (v[i_bottom, j+1] + v[i_bottom-1, j+1])
        Fs_bottom = 0  # No flow through the bottom
        
        a_e[i_bottom, j] = De * self.power_law_function(Fe_bottom, De) + np.maximum(-Fe_bottom, 0)
        a_w[i_bottom, j] = Dw * self.power_law_function(Fw_bottom, Dw) + np.maximum(Fw_bottom, 0)
        a_n[i_bottom, j] = Dn * self.power_law_function(Fn_bottom, Dn) + np.maximum(-Fn_bottom, 0)
        a_s[i_bottom, j] = 0  # No south neighbor at bottom boundary
        a_p[i_bottom, j] = a_e[i_bottom, j] + a_w[i_bottom, j] + a_n[i_bottom, j] + (Fe_bottom-Fw_bottom) + Fn_bottom
        source[i_bottom, j] = (p[i_bottom-1, j] - p[i_bottom, j]) * dy
        
        # Top boundary (j=ny-1)
        j = ny-1
        i_top = np.arange(1, nx)
        Fe_top = 0.5 * rho * dy * (u[i_top+1, j] + u[i_top, j])
        Fw_top = 0.5 * rho * dy * (u[i_top-1, j] + u[i_top, j])
        Fn_top = 0  # No flow through the top
        Fs_top = 0.5 * rho * dx * (v[i_top, j] + v[i_top-1, j])
        
        a_e[i_top, j] = De * self.power_law_function(Fe_top, De) + np.maximum(-Fe_top, 0)
        a_w[i_top, j] = Dw * self.power_law_function(Fw_top, Dw) + np.maximum(Fw_top, 0)
        a_n[i_top, j] = 0  # No north neighbor at top boundary
        a_s[i_top, j] = Ds * self.power_law_function(Fs_top, Ds) + np.maximum(Fs_top, 0)
        a_p[i_top, j] = a_e[i_top, j] + a_w[i_top, j] + a_s[i_top, j] + (Fe_top-Fw_top) - Fs_top
        source[i_top, j] = (p[i_top-1, j] - p[i_top, j]) * dy
        
        # Handle left and right boundaries for all u nodes - PRACTICE B IMPLEMENTATION
        # Only do this if we have a boundary condition manager
        if bc_manager is not None:
            # Cells adjacent to LEFT boundary - modify for i=1 points across all j
            left_bc = bc_manager.get_condition('left')
            if left_bc:
                i_adjacent = 1  # First interior cell
                for j in range(ny):
                    # For u nodes, we already calculated a_w above
                    # Practice B: Move boundary contribution to source term
                    # and zero out a_w for the adjacent cell
                    u_boundary = u[0, j]  # Boundary value
                    # Add boundary contribution to source term
                    source[i_adjacent, j] += a_w[i_adjacent, j] * u_boundary
                    # Set a_w to zero to disconnect this cell from boundary in matrix
                    a_w[i_adjacent, j] = 0.0
                    # Note: a_p was already calculated with the original a_w, no need to adjust

            # Cells adjacent to RIGHT boundary - modify for i=nx-1 points across all j
            right_bc = bc_manager.get_condition('right')
            if right_bc:
                i_adjacent = nx - 1 # Last interior cell (u nodes go up to nx)
                for j in range(ny):
                    # For u nodes, we already calculated a_e above
                    # Practice B: Move boundary contribution to source term
                    # and zero out a_e for the adjacent cell
                    u_boundary = u[nx, j] # Boundary value (u is nx+1 wide)
                    # Add boundary contribution to source term
                    source[i_adjacent, j] += a_e[i_adjacent, j] * u_boundary
                    # Set a_e to zero to disconnect
                    a_e[i_adjacent, j] = 0.0
                    # Note: a_p retains original contribution

            # Handle cells adjacent to BOTTOM boundary
            bottom_bc = bc_manager.get_condition('bottom')
            if bottom_bc:
                j_adjacent = 1  # First interior row
                for i in range(1, nx):  # Skip corners which are handled by left/right
                    # u nodes also live on the bottom boundary, need special treatment
                    u_boundary = u[i, 0]  # Bottom boundary value
                    # Add boundary contribution to source term for cells just above
                    source[i, j_adjacent] += a_s[i, j_adjacent] * u_boundary
                    # Disconnect from boundary in matrix
                    a_s[i, j_adjacent] = 0.0
                    # a_p remains with original a_s contribution
            
            # Handle cells adjacent to TOP boundary
            top_bc = bc_manager.get_condition('top')
            if top_bc:
                j_adjacent = ny-2  # One row below top
                for i in range(1, nx):  # Skip corners
                    # For j=ny-1, u nodes are on the boundary
                    u_boundary = u[i, ny-1]  # Top boundary value
                    # Add boundary contribution to source term for cells just below
                    source[i, j_adjacent] += a_n[i, j_adjacent] * u_boundary
                    # Disconnect from boundary in matrix
                    a_n[i, j_adjacent] = 0.0
                    # a_p remains as is
        
        # Return coefficients in a dictionary
        return {
            'a_e': a_e,
            'a_w': a_w,
            'a_n': a_n,
            'a_s': a_s,
            'a_p': a_p,
            'source': source
        }
    
    def calculate_v_coefficients(self, mesh, fluid, u, v, p, bc_manager=None):
        """
        Calculate coefficients for the v-momentum equation using power-law scheme.
        Implements Practice B for boundary treatment.
        
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
        bc_manager : BoundaryConditionManager, optional
            Boundary condition manager
            
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
        Fw_left = 0  # No flow through left wall
        Fn_left = 0.5 * rho * dx * (v[i, j_left+1] + v[i, j_left])
        Fs_left = 0.5 * rho * dx * (v[i, j_left-1] + v[i, j_left])
        
        a_e[i, j_left] = De * self.power_law_function(Fe_left, De) + np.maximum(-Fe_left, 0)
        a_w[i, j_left] = 0  # No west neighbor
        a_n[i, j_left] = Dn * self.power_law_function(Fn_left, Dn) + np.maximum(-Fn_left, 0)
        a_s[i, j_left] = Ds * self.power_law_function(Fs_left, Ds) + np.maximum(Fs_left, 0)
        a_p[i, j_left] = a_e[i, j_left] + a_n[i, j_left] + a_s[i, j_left] + Fe_left + (Fn_left-Fs_left)
        source[i, j_left] = (p[i, j_left-1] - p[i, j_left]) * dx
        
        # Right boundary (i=nx-1)
        i = nx-1
        j_right = np.arange(1, ny)
        Fe_right = 0  # No flow through right wall
        Fw_right = 0.5 * rho * dy * (u[i, j_right] + u[i, j_right-1])
        Fn_right = 0.5 * rho * dx * (v[i, j_right+1] + v[i, j_right])
        Fs_right = 0.5 * rho * dx * (v[i, j_right-1] + v[i, j_right])
        
        a_e[i, j_right] = 0  # No east neighbor
        a_w[i, j_right] = Dw * self.power_law_function(Fw_right, Dw) + np.maximum(Fw_right, 0)
        a_n[i, j_right] = Dn * self.power_law_function(Fn_right, Dn) + np.maximum(-Fn_right, 0)
        a_s[i, j_right] = Ds * self.power_law_function(Fs_right, Ds) + np.maximum(Fs_right, 0)
        a_p[i, j_right] = a_w[i, j_right] + a_n[i, j_right] + a_s[i, j_right] - Fw_right + (Fn_right-Fs_right)
        source[i, j_right] = (p[i, j_right-1] - p[i, j_right]) * dx
        
        # PRACTICE B IMPLEMENTATION for v-momentum
        if bc_manager is not None:
            # Cells adjacent to BOTTOM boundary - modify for j=1 across all i
            bottom_bc = bc_manager.get_condition('bottom')
            if bottom_bc:
                j_adjacent = 1  # First interior row for v velocity
                for i in range(nx):
                    # For v nodes adjacent to bottom boundary
                    v_boundary = v[i, 0]  # Bottom boundary value
                    # Add boundary contribution to source term
                    source[i, j_adjacent] += a_s[i, j_adjacent] * v_boundary
                    # Set a_s to zero to disconnect
                    a_s[i, j_adjacent] = 0.0
                    # Note: a_p retains original a_s contribution
            
            # Cells adjacent to TOP boundary - modify for j=ny-1 across all i
            top_bc = bc_manager.get_condition('top')
            if top_bc:
                j_adjacent = ny-1  # Last interior row for v velocity
                for i in range(nx):
                    # For v nodes adjacent to top boundary
                    v_boundary = v[i, ny]  # Top boundary value
                    # Add boundary contribution to source term
                    source[i, j_adjacent] += a_n[i, j_adjacent] * v_boundary
                    # Set a_n to zero to disconnect
                    a_n[i, j_adjacent] = 0.0
                    # a_p keeps original contribution
            
            # Handle cells adjacent to LEFT boundary
            left_bc = bc_manager.get_condition('left')
            if left_bc:
                i_adjacent = 1  # First interior column
                for j in range(1, ny):  # Skip corners handled by bottom/top
                    # For v nodes, boundary is at i=0
                    v_boundary = v[0, j]  # Left boundary value
                    # Add boundary contribution to source term
                    source[i_adjacent, j] += a_w[i_adjacent, j] * v_boundary
                    # Disconnect from boundary
                    a_w[i_adjacent, j] = 0.0
                    # a_p unchanged
            
            # Handle cells adjacent to RIGHT boundary
            right_bc = bc_manager.get_condition('right')
            if right_bc:
                i_adjacent = nx-2  # One column before the rightmost
                for j in range(1, ny):  # Skip corners
                    # For v at i=nx-1, we're on the boundary
                    v_boundary = v[nx-1, j]  # Right boundary value
                    # Add boundary contribution to source
                    source[i_adjacent, j] += a_e[i_adjacent, j] * v_boundary
                    # Disconnect from boundary
                    a_e[i_adjacent, j] = 0.0
                    # a_p unchanged
        
        # Return coefficients in a dictionary
        return {
            'a_e': a_e,
            'a_w': a_w,
            'a_n': a_n,
            'a_s': a_s,
            'a_p': a_p,
            'source': source
        } 