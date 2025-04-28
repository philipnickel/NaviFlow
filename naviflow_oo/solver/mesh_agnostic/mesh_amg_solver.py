"""
Mesh-agnostic Algebraic Multigrid (AMG) momentum solver.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pyamg  # PyAMG library for AMG solvers

from ..momentum_solver.base_momentum_solver import MomentumSolver
from .power_law import MeshAgnosticPowerLaw

class MeshAgnosticAMGSolver(MomentumSolver):
    """
    Mesh-agnostic momentum solver using Algebraic Multigrid (AMG).
    Works with any mesh topology, not just structured grids.
    """

    def __init__(self, discretization_scheme='power_law', tolerance=1e-8, max_iterations=100):
        """
        Initialize the mesh-agnostic AMG momentum solver.

        Parameters:
        -----------
        discretization_scheme : str, optional
            The discretization scheme to use (default: 'power_law')
        tolerance : float, optional
            Convergence tolerance for the AMG solver (default: 1e-8)
        max_iterations : int, optional
            Maximum number of iterations for the AMG solver (default: 100)
        """
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # For now, only power_law is implemented in mesh-agnostic form
        if discretization_scheme == 'power_law':
            self.discretization_scheme = MeshAgnosticPowerLaw()
        else:
            raise ValueError(f"Unsupported mesh-agnostic discretization scheme: {discretization_scheme}. "
                           "Available options: 'power_law'")

        # Store coefficients and matrices
        self.matrix = None
        self.rhs = None
        self.coefficients = None

    def _build_sparse_matrix(self, mesh, coefficients, relaxation_factor=1.0, component='u'):
        """
        Build a sparse matrix from the coefficients for the momentum equation.
        
        Parameters:
        -----------
        mesh : Mesh
            The computational mesh (structured or unstructured)
        coefficients : dict
            Coefficients dictionary containing a_nb, neighbor_indices, a_p, and source
        relaxation_factor : float, optional
            Relaxation factor for under-relaxation
        component : str, optional
            Component ('u' or 'v') for which to build the matrix
            
        Returns:
        --------
        matrix : scipy.sparse.csr_matrix
            Sparse matrix for the linear system
        rhs : ndarray
            Right-hand side vector
        """
        # Extract coefficients
        a_nb = coefficients['a_nb']
        neighbor_indices = coefficients['neighbor_indices']
        a_p = coefficients['a_p']
        
        # Get the correct source term based on component
        if component == 'u':
            source = coefficients.get('source_x', coefficients.get('source', np.zeros(mesh.n_cells)))
        else:  # 'v'
            source = coefficients.get('source_y', coefficients.get('source', np.zeros(mesh.n_cells)))
        
        # Apply relaxation to diagonal (a_p)
        a_p_relaxed = a_p / relaxation_factor
        
        # Number of cells
        n_cells = mesh.n_cells
        
        # Initialize arrays for COO matrix format
        rows = []
        cols = []
        data = []
        
        # Build sparse matrix
        for i in range(n_cells):
            # Diagonal entry
            rows.append(i)
            cols.append(i)
            data.append(a_p_relaxed[i])
            
            # Off-diagonal entries
            for j, nb_idx in enumerate(neighbor_indices[i]):
                if j < len(a_nb[i]):  # Check if we have a coefficient for this neighbor
                    rows.append(i)
                    cols.append(nb_idx)
                    # Off-diagonal coefficients are negative
                    data.append(-a_nb[i][j])
        
        # Create sparse matrix
        matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        matrix = matrix.tocsr()
        
        # Return matrix and RHS
        return matrix, source

    def solve_momentum(self, mesh, fluid, velocity_field, pressure_field, 
                       component='u', relaxation_factor=0.7, boundary_conditions=None):
        """
        Solve the momentum equation for a specific component using AMG.
        
        Parameters:
        -----------
        mesh : Mesh
            The computational mesh (structured or unstructured)
        fluid : FluidProperties
            Fluid properties
        velocity_field : VectorField
            Current velocity field
        pressure_field : ScalarField
            Current pressure field
        component : str
            Component to solve ('u' or 'v')
        relaxation_factor : float, optional
            Relaxation factor for under-relaxation
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions
            
        Returns:
        --------
        velocity_new : ndarray
            Updated velocity field for the specified component
        d_coeff : ndarray
            Momentum equation coefficient (for use in pressure equation)
        residual_info : dict
            Dictionary with residual information
        """
        # Get the appropriate velocity component
        if component == 'u':
            u_cells = velocity_field.get_u_at_cells()
            initial_guess = u_cells.copy()
        elif component == 'v':
            v_cells = velocity_field.get_v_at_cells()
            initial_guess = v_cells.copy()
        else:
            raise ValueError(f"Unsupported component: {component}. Must be 'u' or 'v'.")
        
        # Calculate coefficients using the discretization scheme
        coefficients = self.discretization_scheme.calculate_momentum_coefficients(
            mesh, fluid, velocity_field, pressure_field, boundary_conditions
        )
        self.coefficients = coefficients
        
        # Build sparse matrix
        matrix, rhs = self._build_sparse_matrix(mesh, coefficients, relaxation_factor, component)
        self.matrix = matrix
        self.rhs = rhs
        
        # Create the AMG solver hierarchy
        ml = pyamg.smoothed_aggregation_solver(matrix)
        
        # Solve the linear system
        solution = ml.solve(rhs, x0=initial_guess, tol=self.tolerance, maxiter=self.max_iterations)
        
        # Calculate residual
        r_unrelaxed = rhs - matrix.dot(solution)
        r_norm = np.linalg.norm(r_unrelaxed)
        res_field = r_unrelaxed.reshape(-1)
        
        # Calculate relative norm
        rhs_norm = np.linalg.norm(rhs)
        rel_norm = r_norm / rhs_norm if rhs_norm > 0 else r_norm
        
        # Prepare residual info
        residual_info = {
            'rel_norm': rel_norm,
            'field': res_field
        }
        
        # Calculate d coefficient for use in pressure equation
        # This is 1/a_p for the specified component
        safe_ap = np.where(np.abs(coefficients['a_p']) > 1e-12, 
                           coefficients['a_p'], 1e-12)
        d_coeff = 1.0 / safe_ap
        
        return solution, d_coeff, residual_info
    
    def solve_u_momentum(self, mesh, fluid, velocity_field, pressure_field, relaxation_factor=0.7, 
                         boundary_conditions=None, return_dict=True):
        """
        Solve the u-momentum equation using the mesh-agnostic AMG solver.
        This is a wrapper around solve_momentum for compatibility.
        
        Parameters:
        -----------
        mesh : Mesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        velocity_field : VectorField
            Current velocity field object
        pressure_field : ScalarField
            Current pressure field object
        relaxation_factor : float, optional
            Relaxation factor for under-relaxation
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions
        return_dict : bool, optional
            If True, returns residual information in dictionary format
            
        Returns:
        --------
        u_star : ndarray
            Intermediate u-velocity field
        d_u : ndarray
            Momentum equation coefficient
        residual_info : dict
            Dictionary with residual information
        """
        # Solve for u component
        u_star, d_u, residual_info = self.solve_momentum(
            mesh, fluid, velocity_field, pressure_field,
            component='u', relaxation_factor=relaxation_factor,
            boundary_conditions=boundary_conditions
        )
        
        return u_star, d_u, residual_info
    
    def solve_v_momentum(self, mesh, fluid, velocity_field, pressure_field, relaxation_factor=0.7, 
                         boundary_conditions=None, return_dict=True):
        """
        Solve the v-momentum equation using the mesh-agnostic AMG solver.
        This is a wrapper around solve_momentum for compatibility.
        
        Parameters:
        -----------
        mesh : Mesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        velocity_field : VectorField
            Current velocity field object
        pressure_field : ScalarField
            Current pressure field object
        relaxation_factor : float, optional
            Relaxation factor for under-relaxation
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions
        return_dict : bool, optional
            If True, returns residual information in dictionary format
            
        Returns:
        --------
        v_star : ndarray
            Intermediate v-velocity field
        d_v : ndarray
            Momentum equation coefficient
        residual_info : dict
            Dictionary with residual information
        """
        # Solve for v component
        v_star, d_v, residual_info = self.solve_momentum(
            mesh, fluid, velocity_field, pressure_field,
            component='v', relaxation_factor=relaxation_factor,
            boundary_conditions=boundary_conditions
        )
        
        return v_star, d_v, residual_info 