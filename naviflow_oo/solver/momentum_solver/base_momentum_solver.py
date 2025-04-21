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
        # Unrelaxed coefficients for true algebraic residuals
        self.u_a_p_unrelaxed = None
        self.u_source_unrelaxed = None
        self.v_a_p_unrelaxed = None
        self.v_source_unrelaxed = None
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
        
        # --- Get UNRELAXED coefficients from the discretization scheme --- 
        if not hasattr(self, 'discretization') or self.discretization is None:
            raise NotImplementedError("MomentumSolver requires a discretization scheme to be set.")
            
        u_coeffs_unrelaxed = self.discretization.calculate_u_coefficients(mesh, fluid, u, v, p)
        v_coeffs_unrelaxed = self.discretization.calculate_v_coefficients(mesh, fluid, u, v, p)
        
        # Extract unrelaxed coefficients
        temp_u_a_e = u_coeffs_unrelaxed['a_e']
        temp_u_a_w = u_coeffs_unrelaxed['a_w']
        temp_u_a_n = u_coeffs_unrelaxed['a_n']
        temp_u_a_s = u_coeffs_unrelaxed['a_s']
        temp_u_a_p_unrelaxed = u_coeffs_unrelaxed['a_p']
        temp_u_source_unrelaxed = u_coeffs_unrelaxed['source']
        
        temp_v_a_e = v_coeffs_unrelaxed['a_e']
        temp_v_a_w = v_coeffs_unrelaxed['a_w']
        temp_v_a_n = v_coeffs_unrelaxed['a_n']
        temp_v_a_s = v_coeffs_unrelaxed['a_s']
        temp_v_a_p_unrelaxed = v_coeffs_unrelaxed['a_p']
        temp_v_source_unrelaxed = v_coeffs_unrelaxed['source']
        
        # Store unrelaxed coeffs first (always needed for Rust-like residual)
        self.u_a_e = temp_u_a_e
        self.u_a_w = temp_u_a_w
        self.u_a_n = temp_u_a_n
        self.u_a_s = temp_u_a_s
        self.u_a_p_unrelaxed = temp_u_a_p_unrelaxed
        self.u_source_unrelaxed = temp_u_source_unrelaxed
        
        self.v_a_e = temp_v_a_e
        self.v_a_w = temp_v_a_w
        self.v_a_n = temp_v_a_n
        self.v_a_s = temp_v_a_s
        self.v_a_p_unrelaxed = temp_v_a_p_unrelaxed
        self.v_source_unrelaxed = temp_v_source_unrelaxed
        
        # --- Apply Relaxation to store relaxed versions (potentially used by solver) --- 
        alpha_u = relaxation_factor
        
        if alpha_u is not None and alpha_u < 1.0:
            # Modify a_p and source term for U-momentum
            safe_alpha_u = max(alpha_u, 1e-12) 
            self.u_a_p_relaxed = self.u_a_p_unrelaxed / safe_alpha_u
            self.u_source_relaxed = self.u_source_unrelaxed + (1.0 - alpha_u) * self.u_a_p_relaxed * u[:,:] # Use current u as u_old
            
            # Modify a_p and source term for V-momentum
            self.v_a_p_relaxed = self.v_a_p_unrelaxed / safe_alpha_u
            self.v_source_relaxed = self.v_source_unrelaxed + (1.0 - alpha_u) * self.v_a_p_relaxed * v[:,:] # Use current v as v_old

            # Store relaxed versions. If solver needs unrelaxed, it uses self.u_a_p_unrelaxed etc.
            self.u_a_p = self.u_a_p_relaxed
            self.u_source = self.u_source_relaxed
            self.v_a_p = self.v_a_p_relaxed
            self.v_source = self.v_source_relaxed

        else:
            # No relaxation: store unrelaxed coefficients also in the primary attributes
            self.u_a_p = self.u_a_p_unrelaxed
            self.u_source = self.u_source_unrelaxed
            self.v_a_p = self.v_a_p_unrelaxed
            self.v_source = self.v_source_unrelaxed
            # Ensure relaxed attributes exist even if None (or set to unrelaxed?)
            self.u_a_p_relaxed = None 
            self.u_source_relaxed = None
            self.v_a_p_relaxed = None
            self.v_source_relaxed = None
        
        # --- Boundary conditions --- 
        # Note: Boundary condition modification should ideally happen *after* 
        #       relaxation is applied to the source term, potentially modifying 
        #       both relaxed and unrelaxed coefficients/sources depending on BC type.
        #       This base implementation doesn't modify coefficients for BCs yet.

    def calculate_u_algebraic_residual(self, u):
        """
        Compute true algebraic residual ||b - A*u||_2 / ||b||_2 for the x-momentum equation.
        This is a normalized (relative) residual that provides better convergence information.
        
        Parameters:
        -----------
        u : ndarray
            Current x-velocity field
        
        Returns:
        --------
        float
            Normalized L2 norm of the algebraic residual
        """
        if self.u_a_p_unrelaxed is None or self.u_source_unrelaxed is None:
            raise ValueError("Unrelaxed coefficients have not been calculated. Call calculate_coefficients first.")
        
        # Get dimensions from coefficient matrices
        nx, ny = self.u_a_p_unrelaxed.shape[1], self.u_a_p_unrelaxed.shape[0]
        
        # Initialize arrays for residual and RHS
        residual_vector = []
        b_vector = []
        
        # Use unrelaxed coefficients for true algebraic residual
        a_e = self.u_a_e
        a_w = self.u_a_w
        a_n = self.u_a_n
        a_s = self.u_a_s
        a_p = self.u_a_p_unrelaxed
        source = self.u_source_unrelaxed
        
        # Only calculate for interior points to avoid boundary issues
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Ensure we're not accessing out of bounds
                if i+1 >= u.shape[0] or j+1 >= u.shape[1] or i-1 < 0 or j-1 < 0:
                    continue
                    
                # Calculate A*u
                Au = (a_p[i,j] * u[i,j] -
                      a_e[i,j] * u[i+1,j] -
                      a_w[i,j] * u[i-1,j] -
                      a_n[i,j] * u[i,j+1] -
                      a_s[i,j] * u[i,j-1])
                
                # Calculate residual: b - A*u
                residual = float(source[i,j] - Au)
                residual_vector.append(residual)
                b_vector.append(float(source[i,j]))
        
        # Calculate normalized residual
        r_norm = np.linalg.norm(residual_vector, 2)
        b_norm = np.linalg.norm(b_vector, 2)
        res_norm = r_norm / b_norm if b_norm > 0 else r_norm
        
        return res_norm
    
    def calculate_v_algebraic_residual(self, v):
        """
        Compute true algebraic residual ||b - A*v||_2 / ||b||_2 for the y-momentum equation.
        This is a normalized (relative) residual that provides better convergence information.
        
        Parameters:
        -----------
        v : ndarray
            Current y-velocity field
        
        Returns:
        --------
        float
            Normalized L2 norm of the algebraic residual
        """
        if self.v_a_p_unrelaxed is None or self.v_source_unrelaxed is None:
            raise ValueError("Unrelaxed coefficients have not been calculated. Call calculate_coefficients first.")
        
        # Get dimensions from coefficient matrices
        nx, ny = self.v_a_p_unrelaxed.shape[1], self.v_a_p_unrelaxed.shape[0]
        
        # Initialize arrays for residual and RHS
        residual_vector = []
        b_vector = []
        
        # Use unrelaxed coefficients for true algebraic residual
        a_e = self.v_a_e
        a_w = self.v_a_w
        a_n = self.v_a_n
        a_s = self.v_a_s
        a_p = self.v_a_p_unrelaxed
        source = self.v_source_unrelaxed
        
        # Only calculate for interior points to avoid boundary issues
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Ensure we're not accessing out of bounds
                if i+1 >= v.shape[0] or j+1 >= v.shape[1] or i-1 < 0 or j-1 < 0:
                    continue
                    
                # Calculate A*v
                Av = (a_p[i,j] * v[i,j] -
                      a_e[i,j] * v[i+1,j] -
                      a_w[i,j] * v[i-1,j] -
                      a_n[i,j] * v[i,j+1] -
                      a_s[i,j] * v[i,j-1])
                
                # Calculate residual: b - A*v
                residual = float(source[i,j] - Av)
                residual_vector.append(residual)
                b_vector.append(float(source[i,j]))
        
        # Calculate normalized residual
        r_norm = np.linalg.norm(residual_vector, 2)
        b_norm = np.linalg.norm(b_vector, 2)
        res_norm = r_norm / b_norm if b_norm > 0 else r_norm
        
        return res_norm
    
    @abstractmethod
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
        u_star : ndarray
            Intermediate velocity field
        d_u : ndarray
            Momentum equation coefficient
        u_residual : float
            Algebraic residual for the u-momentum equation
        """
        pass
    
    @abstractmethod
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
        v_star : ndarray
            Intermediate velocity field
        d_v : ndarray
            Momentum equation coefficient
        v_residual : float
            Algebraic residual for the v-momentum equation
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