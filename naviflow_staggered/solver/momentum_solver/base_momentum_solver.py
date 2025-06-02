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
        