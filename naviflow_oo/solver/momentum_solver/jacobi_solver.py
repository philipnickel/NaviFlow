"""
Jacobi-method momentum solver that can use different discretization schemes.
"""

import numpy as np
from .base_momentum_solver import MomentumSolver
from .discretization import power_law
from ...constructor.boundary_conditions import BoundaryConditionManager

class JacobiMomentumSolver(MomentumSolver):
    """
    Momentum solver that uses Jacobi iterations to solve the momentum equations.
    Can use different discretization schemes.
    """
    
    def __init__(self, discretization_scheme='power_law', n_jacobi_sweeps=1):
        """
        Initialize the Jacobi momentum solver.
        
        Parameters:
        -----------
        discretization_scheme : str
            The discretization scheme to use ('power_law' by default)
        n_jacobi_sweeps : int
            Number of Jacobi iterations to perform
        """
        super().__init__()
        self.n_jacobi_sweeps = n_jacobi_sweeps
        
        # Set discretization scheme
        if discretization_scheme == 'power_law':
            self.discretization = power_law.PowerLawDiscretization()
        else:
            raise ValueError(f"Unsupported discretization scheme: {discretization_scheme}")
        
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
        Solve the u-momentum equation using Jacobi iterations.
        
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
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        
        # Initialize arrays
        u_star = np.zeros((imax+1, jmax))
        d_u = np.zeros((imax+1, jmax))
        
        # Calculate coefficients using the specified discretization scheme
        coeffs = self.discretization.calculate_u_coefficients(mesh, fluid, u, v, p)
        
        # Store coefficients for residual calculation
        self.u_a_e = coeffs['a_e']
        self.u_a_w = coeffs['a_w']
        self.u_a_n = coeffs['a_n']
        self.u_a_s = coeffs['a_s']
        self.u_a_p = coeffs['a_p']
        self.u_source = coeffs['source']
        
        # Solve using Jacobi iterations
        u_star_unrelaxed = u.copy()
        
        # Interior points
        i_range = np.arange(1, imax)
        j_range = np.arange(1, jmax-1)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')

        for _ in range(self.n_jacobi_sweeps):
            u_old = u_star_unrelaxed.copy()
            u_star_unrelaxed[i_grid, j_grid] = (
                (self.u_a_e[i_grid, j_grid] * u_old[i_grid + 1, j_grid] +
                self.u_a_w[i_grid, j_grid] * u_old[i_grid - 1, j_grid] +
                self.u_a_n[i_grid, j_grid] * u_old[i_grid, j_grid + 1] +
                self.u_a_s[i_grid, j_grid] * u_old[i_grid, j_grid - 1] + 
                self.u_source[i_grid, j_grid]) / self.u_a_p[i_grid, j_grid]
            )
        
        # Apply relaxation
        u_star[i_grid, j_grid] = alpha * u_star_unrelaxed[i_grid, j_grid] + (1-alpha)*u[i_grid, j_grid]
        d_u[i_grid, j_grid] = alpha * mesh.get_cell_sizes()[1] / self.u_a_p[i_grid, j_grid]
        
        # Bottom boundary (j=0)
        j = 0
        i_bottom = np.arange(1, imax)
        d_u[i_bottom, j] = alpha * mesh.get_cell_sizes()[1] / self.u_a_p[i_bottom, j]
        
        # Top boundary (j=jmax-1)
        j = jmax-1
        i_top = np.arange(1, imax)
        d_u[i_top, j] = alpha * mesh.get_cell_sizes()[1] / self.u_a_p[i_top, j]
        
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
        
        return u_star, d_u

    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        """
        Solve the v-momentum equation using Jacobi iterations.
        
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
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        
        # Initialize arrays
        v_star = np.zeros((imax, jmax+1))
        d_v = np.zeros((imax, jmax+1))
        
        # Calculate coefficients using the specified discretization scheme
        coeffs = self.discretization.calculate_v_coefficients(mesh, fluid, u, v, p)
        
        # Store coefficients for residual calculation
        self.v_a_e = coeffs['a_e']
        self.v_a_w = coeffs['a_w']
        self.v_a_n = coeffs['a_n']
        self.v_a_s = coeffs['a_s']
        self.v_a_p = coeffs['a_p']
        self.v_source = coeffs['source']
        
        # Solve using Jacobi iterations
        v_star_unrelaxed = v.copy()
        
        # Interior points
        i_range = np.arange(1, imax-1)
        j_range = np.arange(1, jmax)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')


        for _ in range(self.n_jacobi_sweeps):
            v_old = v_star_unrelaxed.copy()
            v_star_unrelaxed[i_grid, j_grid] = (
                (self.v_a_e[i_grid, j_grid] * v_old[i_grid + 1, j_grid] +
                self.v_a_w[i_grid, j_grid] * v_old[i_grid - 1, j_grid] +
                self.v_a_n[i_grid, j_grid] * v_old[i_grid, j_grid + 1] +
                self.v_a_s[i_grid, j_grid] * v_old[i_grid, j_grid - 1] + 
                self.v_source[i_grid, j_grid]) / self.v_a_p[i_grid, j_grid]
            )
        
        # Apply relaxation
        v_star[i_grid, j_grid] = alpha * v_star_unrelaxed[i_grid, j_grid] + (1-alpha)*v[i_grid, j_grid]
        d_v[i_grid, j_grid] = alpha * mesh.get_cell_sizes()[0] / self.v_a_p[i_grid, j_grid]
        
        # Left boundary (i=0)
        i = 0
        j_left = np.arange(1, jmax)
        d_v[i, j_left] = alpha * mesh.get_cell_sizes()[0] / self.v_a_p[i, j_left]
        
        # Right boundary (i=imax-1)
        i = imax-1
        j_right = np.arange(1, jmax)
        d_v[i, j_right] = alpha * mesh.get_cell_sizes()[0] / self.v_a_p[i, j_right]
        
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