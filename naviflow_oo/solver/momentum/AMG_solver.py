import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import pyamg
from ..base_momentum_solver import MomentumSolver

class AMGMomentumSolver(MomentumSolver):
    """
    Algebraic Multigrid solver for momentum equations.
    """
    
    def __init__(self, discretization_scheme, tolerance=1e-10, max_iterations=1000):
        """
        Initialize the AMG momentum solver with a specified discretization scheme.
        
        Parameters
        ----------
        discretization_scheme : MomentumDiscretization
            The discretization scheme to use for constructing the coefficient matrix
        tolerance : float, optional
            Convergence tolerance for the multigrid solver
        max_iterations : int, optional
            Maximum number of iterations for the solver
        """
        super().__init__()
        self.discretization_scheme = discretization_scheme
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.residual_history = []
        self.u = None
        self.v = None
        self.p = None
        self._matrix_u = None
        self._matrix_v = None
        self._rhs_u = None
        self._rhs_v = None
        self._d_u = None
        self._d_v = None
    
    def setup(self, mesh, fluid_properties, boundary_conditions=None):
        """
        Set up the solver with mesh, fluid properties, and boundary conditions.
        
        Parameters
        ----------
        mesh : Mesh
            Computational mesh
        fluid_properties : FluidProperties
            Fluid properties 
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions
        """
        self.mesh = mesh
        self.fluid = fluid_properties
        self.bc_manager = boundary_conditions
        
        # Initialize velocity fields if not already done
        n_cells = mesh.n_cells
        if self.u is None:
            self.u = np.zeros(n_cells)
        if self.v is None:
            self.v = np.zeros(n_cells)
        if self.p is None:
            self.p = np.zeros(n_cells)
    
    def solve_momentum(self, pressure=None, relaxation_factor=1.0):
        """
        Solve the momentum equations using the AMG solver.
        
        Parameters
        ----------
        pressure : ndarray, optional
            Pressure field to use (if None, uses the internal pressure field)
        relaxation_factor : float, optional
            Relaxation factor for the momentum solution
            
        Returns
        -------
        dict
            Dictionary containing the velocity fields and solver information
        """
        # Use internal pressure if none provided
        if pressure is None:
            pressure = self.p
        
        # Ensure all fields are 1D arrays
        pressure_flat = pressure.flatten() if pressure.ndim > 1 else pressure
        u_flat = self.u.flatten() if self.u.ndim > 1 else self.u
        v_flat = self.v.flatten() if self.v.ndim > 1 else self.v
        
        # Apply boundary conditions
        self._apply_boundary_conditions(u_flat, v_flat)
        
        # Construct discretization matrices
        discretization_info = self.discretization_scheme.discretize(
            self.mesh, 
            self.fluid, 
            u_flat, 
            v_flat, 
            pressure_flat, 
            self.bc_manager
        )
        
        # Extract matrices, RHS, and diagonal coefficients
        self._matrix_u = discretization_info['matrix_u']
        self._matrix_v = discretization_info['matrix_v']
        self._rhs_u = discretization_info['rhs_u']
        self._rhs_v = discretization_info['rhs_v']
        self._d_u = discretization_info['d_u']
        self._d_v = discretization_info['d_v']
        
        # Create AMG solvers for velocity components
        ml_u = pyamg.smoothed_aggregation_solver(self._matrix_u, max_coarse=10)
        ml_v = pyamg.smoothed_aggregation_solver(self._matrix_v, max_coarse=10)
        
        # Solve for u and v velocity components
        u_new, u_info = ml_u.solve(
            self._rhs_u, 
            x0=u_flat, 
            tol=self.tolerance, 
            maxiter=self.max_iterations, 
            return_info=True
        )
        
        v_new, v_info = ml_v.solve(
            self._rhs_v, 
            x0=v_flat, 
            tol=self.tolerance, 
            maxiter=self.max_iterations, 
            return_info=True
        )
        
        # Apply under-relaxation
        if relaxation_factor < 1.0:
            u_new = relaxation_factor * u_new + (1.0 - relaxation_factor) * u_flat
            v_new = relaxation_factor * v_new + (1.0 - relaxation_factor) * v_flat
        
        # Update internal velocity fields
        self.u = u_new
        self.v = v_new
        
        # Create residual information for reporting
        residual_u = self._rhs_u - self._matrix_u @ u_new
        residual_v = self._rhs_v - self._matrix_v @ v_new
        
        residual_info = {
            'u': {
                'abs_norm': np.linalg.norm(residual_u),
                'rel_norm': np.linalg.norm(residual_u) / max(np.linalg.norm(self._rhs_u), 1e-12),
                'iterations': u_info['iterations'],
                'field': residual_u
            },
            'v': {
                'abs_norm': np.linalg.norm(residual_v),
                'rel_norm': np.linalg.norm(residual_v) / max(np.linalg.norm(self._rhs_v), 1e-12),
                'iterations': v_info['iterations'],
                'field': residual_v
            }
        }
        
        # Add to residual history for convergence tracking
        self.residual_history.append({
            'u_res': residual_info['u']['rel_norm'],
            'v_res': residual_info['v']['rel_norm']
        })
        
        return {
            'u': u_new,
            'v': v_new,
            'd_u': self._d_u,
            'd_v': self._d_v,
            'residual_info': residual_info
        }
    
    def get_diagonal_coefficients(self):
        """
        Get the diagonal coefficients for the momentum equations.
        
        Returns
        -------
        tuple
            (d_u, d_v) diagonal coefficients as 1D arrays
        """
        if self._d_u is None or self._d_v is None:
            # We need to run a discretization first to get coefficients
            self.solve_momentum(relaxation_factor=0.0)  # No update of velocity fields
            
        return self._d_u, self._d_v
    
    def _apply_boundary_conditions(self, u, v):
        """
        Apply boundary conditions to the velocity fields.
        
        Parameters
        ----------
        u, v : ndarray
            Velocity fields as 1D arrays
        """
        if self.bc_manager is None:
            return
            
        # Apply Dirichlet boundary conditions for each boundary
        for boundary_name in self.bc_manager.get_boundary_names():
            boundary_type = self.bc_manager.get_boundary_type(boundary_name)
            
            if boundary_type == 'velocity':
                bc_values = self.bc_manager.get_condition(boundary_name, 'velocity')
                if bc_values is not None:
                    bc_u = bc_values.get('u')
                    bc_v = bc_values.get('v')
                    
                    # Get cells adjacent to this boundary
                    boundary_cells = self.mesh.get_boundary_cells(boundary_name)
                    
                    # Apply the boundary values
                    if bc_u is not None and len(boundary_cells) > 0:
                        u[boundary_cells] = bc_u
                    if bc_v is not None and len(boundary_cells) > 0:
                        v[boundary_cells] = bc_v 