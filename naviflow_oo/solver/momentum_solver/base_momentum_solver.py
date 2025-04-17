"""
Abstract base class for momentum solvers.
"""

from abc import ABC, abstractmethod
import numpy as np

class MomentumSolver(ABC):
    """
    Base class for momentum solvers.
    """
    def __init__(self):
        """
        Initialize the momentum solver.
        """
        # Storage for coefficients
        self.u_coeffs = None
        self.v_coeffs = None
        self.u_source = None
        self.v_source = None
        # These will hold the coefficient matrices for a_e, a_w, a_n, a_s, a_p
        self.u_a_e = None
        self.u_a_w = None
        self.u_a_n = None
        self.u_a_s = None
        self.u_a_p = None
        self.v_a_e = None
        self.v_a_w = None
        self.v_a_n = None
        self.v_a_s = None
        self.v_a_p = None
        pass
    
    def calculate_coefficients(self, mesh, fluid, u, v, p, boundary_conditions=None, relaxation_factor=None):
        """
        Calculate coefficients for the momentum equations without solving.
        
        This method sets up the coefficient matrices needed for both
        the residual calculation and the momentum equation solves.
        This matches the Rust implementation's approach.
        
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
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions manager
        relaxation_factor : float, optional
            Relaxation factor for the momentum equation
        """
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        # Get fluid properties
        rho = fluid.get_density()
        mu = fluid.get_viscosity()
        
        # Initialize coefficient arrays if needed
        if self.u_a_e is None or self.u_a_e.shape != (nx, ny):
            self.u_a_e = np.zeros((nx, ny))
            self.u_a_w = np.zeros((nx, ny))
            self.u_a_n = np.zeros((nx, ny))
            self.u_a_s = np.zeros((nx, ny))
            self.u_a_p = np.zeros((nx, ny))
            self.u_source = np.zeros((nx, ny))
            
            self.v_a_e = np.zeros((nx, ny))
            self.v_a_w = np.zeros((nx, ny))
            self.v_a_n = np.zeros((nx, ny))
            self.v_a_s = np.zeros((nx, ny))
            self.v_a_p = np.zeros((nx, ny))
            self.v_source = np.zeros((nx, ny))
        
        # Calculate coefficients for each interior cell
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # U-momentum coefficients - using simple central differencing
                self.u_a_e[i,j] = mu * dy / dx
                self.u_a_w[i,j] = mu * dy / dx
                self.u_a_n[i,j] = mu * dx / dy
                self.u_a_s[i,j] = mu * dx / dy
                self.u_a_p[i,j] = self.u_a_e[i,j] + self.u_a_w[i,j] + self.u_a_n[i,j] + self.u_a_s[i,j]
                self.u_source[i,j] = (p[i,j] - p[i+1,j]) * dy
                
                # V-momentum coefficients
                self.v_a_e[i,j] = mu * dy / dx
                self.v_a_w[i,j] = mu * dy / dx
                self.v_a_n[i,j] = mu * dx / dy
                self.v_a_s[i,j] = mu * dx / dy
                self.v_a_p[i,j] = self.v_a_e[i,j] + self.v_a_w[i,j] + self.v_a_n[i,j] + self.v_a_s[i,j]
                self.v_source[i,j] = (p[i,j] - p[i,j+1]) * dx
    
    @abstractmethod
    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7):
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
            
        Returns:
        --------
        u_star, d_u : ndarray
            Intermediate velocity field and momentum equation coefficient
        """
        pass
    
    @abstractmethod
    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7):
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
            
        Returns:
        --------
        v_star, d_v : ndarray
            Intermediate velocity field and momentum equation coefficient
        """
        pass
        
    def calculate_u_residual(self, mesh, fluid, u, v, p, boundary_conditions=None):
        """
        Calculate residual for the u-momentum equation.
        
        This method calculates the L2 norm of the u-momentum equation residuals
        across all interior cells, matching the Rust implementation approach.
        
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
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions manager
            
        Returns:
        --------
        float
            The residual for the u-momentum equation
        """
        # Make sure coefficients are calculated
        if self.u_a_e is None:
            self.calculate_coefficients(mesh, fluid, u, v, p, boundary_conditions)
        
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        
        # Initialize sum of squared residuals
        res_sum = 0.0
        
        # Calculate residuals for each interior cell using the stored coefficients
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Calculate the residual: r = a_e*u_e + a_w*u_w + a_n*u_n + a_s*u_s + a_p*u_p - source_x
                # This matches Rust implementation in residuals.rs
                residual = (self.u_a_e[i,j] * u[i+1, j] + 
                           self.u_a_w[i,j] * u[i-1, j] + 
                           self.u_a_n[i,j] * u[i, j+1] + 
                           self.u_a_s[i,j] * u[i, j-1] + 
                           self.u_a_p[i,j] * u[i, j] - 
                           self.u_source[i,j])
                
                # Add squared residual to sum
                res_sum += residual**2
                
        # Return L2 norm of residuals
        return np.sqrt(res_sum)
    
    def calculate_v_residual(self, mesh, fluid, u, v, p, boundary_conditions=None):
        """
        Calculate residual for the v-momentum equation.
        
        This method calculates the L2 norm of the v-momentum equation residuals
        across all interior cells, matching the Rust implementation approach.
        
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
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions manager
            
        Returns:
        --------
        float
            The residual for the v-momentum equation
        """
        # Make sure coefficients are calculated
        if self.v_a_e is None:
            self.calculate_coefficients(mesh, fluid, u, v, p, boundary_conditions)
        
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        
        # Initialize sum of squared residuals
        res_sum = 0.0
        
        # Calculate residuals for each interior cell using the stored coefficients
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Calculate the residual: r = a_e*v_e + a_w*v_w + a_n*v_n + a_s*v_s + a_p*v_p - source_y
                # This matches Rust implementation in residuals.rs
                residual = (self.v_a_e[i,j] * v[i+1, j] + 
                           self.v_a_w[i,j] * v[i-1, j] + 
                           self.v_a_n[i,j] * v[i, j+1] + 
                           self.v_a_s[i,j] * v[i, j-1] + 
                           self.v_a_p[i,j] * v[i, j] - 
                           self.v_source[i,j])
                
                # Add squared residual to sum
                res_sum += residual**2
                
        # Return L2 norm of residuals
        return np.sqrt(res_sum) 