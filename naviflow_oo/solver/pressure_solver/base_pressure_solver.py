from abc import ABC, abstractmethod
import numpy as np

class PressureSolver(ABC):
    """
    Base class for pressure solvers.
    """
    def __init__(self, tolerance=1e-6, max_iterations=1000):
        """
        Initialize the pressure solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance (for iterative solvers)
        max_iterations : int, optional
            Maximum number of iterations (for iterative solvers)
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        # Initialize attributes needed for boundary conditions
        self.bc_manager = None
        self.p = None
        # Coefficient matrices
        self.a_e = None
        self.a_w = None
        self.a_n = None
        self.a_s = None
        self.a_p = None
        self.mass_imbalance = None
    
    def setup_coefficients(self, mesh, u, v, d_u, d_v):
        """
        Set up coefficient matrices for the pressure correction equation.
        
        This method calculates the coefficients for the pressure correction equation
        which are needed for both residual calculation and solving.
        This follows the Rust implementation approach.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        u, v : ndarray
            Velocity fields
        d_u, d_v : ndarray
            Momentum equation coefficients
        """
        # Save mesh
        self.mesh = mesh
        
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        # Initialize coefficient arrays if needed
        if self.a_e is None or self.a_e.shape != (nx, ny):
            self.a_e = np.zeros((nx, ny))
            self.a_w = np.zeros((nx, ny))
            self.a_n = np.zeros((nx, ny))
            self.a_s = np.zeros((nx, ny))
            self.a_p = np.zeros((nx, ny))
            self.mass_imbalance = np.zeros((nx, ny))
        
        # Calculate coefficients for interior cells
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Simple coefficients for pressure correction equation
                self.a_e[i,j] = dy * dy / dx  # d_u[i+1,j] * rho * dy
                self.a_w[i,j] = dy * dy / dx  # d_u[i,j] * rho * dy
                self.a_n[i,j] = dx * dx / dy  # d_v[i,j+1] * rho * dx
                self.a_s[i,j] = dx * dx / dy  # d_v[i,j] * rho * dx
                
                # Central coefficient is negative sum of neighbors
                self.a_p[i,j] = -(self.a_e[i,j] + self.a_w[i,j] + self.a_n[i,j] + self.a_s[i,j])
                
                # Calculate mass imbalance (RHS of pressure correction equation)
                u_e = u[i+1, j]      # East face u-velocity
                u_w = u[i, j]         # West face u-velocity
                v_n = v[i, j+1]      # North face v-velocity 
                v_s = v[i, j]         # South face v-velocity
                
                self.mass_imbalance[i,j] = -((u_e - u_w) * dy + (v_n - v_s) * dx)

    @abstractmethod
    def solve(self, mesh, rhs, d_u, d_v, p_star, alpha):
        """
        Solve the pressure correction equation.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        rhs : ndarray
            Right-hand side of the equation
        d_u, d_v : ndarray
            Momentum equation coefficients
        p_star : ndarray
            Current pressure field
        alpha : float
            Relaxation factor
            
        Returns:
        --------
        p, p_prime : ndarray
            Updated pressure field and pressure correction
        """
        pass 
        
    def _enforce_pressure_boundary_conditions(self,mesh, p_field, nx=None, ny=None, bc_type='zero_gradient', boundaries=None):
        """
        Apply appropriate pressure boundary conditions based on boundary types.
        
        This method:
        1. Extracts the boundary types from the boundary condition manager
        2. Applies appropriate pressure boundary conditions for each boundary
        3. Sets a reference pressure point to prevent a floating pressure field
        
        By default, it applies Neumann (zero gradient) conditions for all boundaries,
        which is appropriate for most incompressible flow problems.
        
        Parameters:
        -----------
        p_field : ndarray
            Pressure or pressure correction field to apply boundary conditions to
        nx, ny : int, optional
            Mesh dimensions (if not provided, will get from self.mesh)
        bc_type : str, optional
            Type of boundary condition to apply, options are:
            - 'zero_gradient' (default): Neumann condition where ∂p/∂n = 0
            - 'zero_pressure': Dirichlet condition where p = 0
        boundaries : list or str, optional
            When bc_type='zero_pressure', specifies which boundaries to apply zero pressure.
            Can include 'west', 'east', 'south', 'north', or 'all'.
            If None, applies to all boundaries.
        
        Returns:
        --------
        ndarray
            Modified pressure field with boundary conditions applied
        
        Note for derived classes:
        This method should be called after any pressure field updates in the solver
        algorithm implementation to ensure proper pressure boundary treatment.
        Typical places to call this method include:
        - After updating pressure with pressure corrections
        - After solving intermediate pressure fields
        - Before using pressure fields to calculate velocity fields
        """
        nx, ny = mesh.get_dimensions()
        
        boundary_types = {}
        if self.bc_manager:
            boundary_types = self.bc_manager.get_boundary_types()
        
        if bc_type == 'zero_gradient':
            # Apply zero gradient (Neumann) conditions for all boundaries
            # Left boundary (i=0)
            p_field[0, :] = p_field[1, :]
            # Right boundary (i=nx-1)
            p_field[nx-1, :] = p_field[nx-2, :]
            # Bottom boundary (j=0)
            p_field[:, 0] = p_field[:, 1]
            # Top boundary (j=ny-1)
            p_field[:, ny-1] = p_field[:, ny-2]
        
        elif bc_type == 'zero_pressure':
            # Default to all boundaries if not specified
            if boundaries is None:
                boundaries = ['west', 'east', 'south', 'north']
            elif boundaries == 'all':
                boundaries = ['west', 'east', 'south', 'north']
            
            # Apply zero pressure conditions to specified boundaries
            # For other boundaries, apply zero gradient conditions
            
            # Left (west) boundary (i=0)
            if 'west' in boundaries:
                p_field[0, :] = 0.0
            else:
                p_field[0, :] = p_field[1, :]
                
            # Right (east) boundary (i=nx-1)
            if 'east' in boundaries:
                p_field[nx-1, :] = 0.0
            else:
                p_field[nx-1, :] = p_field[nx-2, :]
                
            # Bottom (south) boundary (j=0)
            if 'south' in boundaries:
                p_field[:, 0] = 0.0
            else:
                p_field[:, 0] = p_field[:, 1]
                
            # Top (north) boundary (j=ny-1)
            if 'north' in boundaries:
                p_field[:, ny-1] = 0.0
            else:
                p_field[:, ny-1] = p_field[:, ny-2]
        
        # Return the modified pressure field
        return p_field

    def calculate_residual(self, mesh, u, v, p, p_prime):
        """
        Calculate the residual for the pressure correction equation.
        
        This method calculates the L2 norm of the residuals for the pressure 
        correction equation across all interior cells, matching the Rust implementation.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        u, v : ndarray
            Velocity fields 
        p : ndarray
            Pressure field
        p_prime : ndarray
            Pressure correction field
            
        Returns:
        --------
        float
            The L2 norm of the pressure equation residuals
        """
        # Make sure coefficients are set up
        if self.a_e is None:
            # If coefficients aren't set up yet, we need d_u and d_v
            # Since we don't have them, we'll create dummy ones - this
            # should only happen the first time
            d_u = np.ones_like(u)
            d_v = np.ones_like(v)
            self.setup_coefficients(mesh, u, v, d_u, d_v)
        
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        
        # Initialize sum of squared residuals
        res_sum = 0.0
        
        # For interior cells, compute the residual using pre-calculated coefficients
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Calculate the residual: r = A*p' - b
                # r = a_e*p'_e + a_w*p'_w + a_n*p'_n + a_s*p'_s + a_p*p'_p - mdot
                residual = (self.a_e[i,j] * p_prime[i+1, j] + 
                           self.a_w[i,j] * p_prime[i-1, j] + 
                           self.a_n[i,j] * p_prime[i, j+1] + 
                           self.a_s[i,j] * p_prime[i, j-1] + 
                           self.a_p[i,j] * p_prime[i, j] - 
                           self.mass_imbalance[i,j])
                
                # Add squared residual to sum
                res_sum += residual**2
                
        # Return L2 norm of residuals
        return np.sqrt(res_sum) 