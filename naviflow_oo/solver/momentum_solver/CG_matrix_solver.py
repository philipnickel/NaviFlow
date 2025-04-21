"""
Jacobi-method momentum solver that can use different discretization schemes.
"""

import numpy as np
from scipy import sparse
from .base_momentum_solver import MomentumSolver
from .discretization import power_law
from ...constructor.boundary_conditions import BoundaryConditionManager
from scipy.sparse.linalg import cg
import pyamg

class CGMatrixMomentumSolver(MomentumSolver):
    """
    Momentum solver that uses Preconditioned Conjugate Gradient (PCG) 
    with AMG preconditioning to solve the momentum equations.
    Can use different discretization schemes.
    """

    def __init__(self, discretization_scheme='power_law', tolerance=1e-6, max_iterations=1000):
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        if discretization_scheme == 'power_law':
            self.discretization = power_law.PowerLawDiscretization()
        else:
            raise ValueError(f"Unsupported discretization scheme: {discretization_scheme}")

        self.u_a_e = None
        self.u_a_w = None
        self.u_a_n = None
        self.u_a_s = None
        self.u_a_p = None
        self.u_source = None
        self.u_a_p_unrelaxed = None
        self.u_source_unrelaxed = None
        self.u_matrix = None
        self.u_rhs = None

        self.v_a_e = None
        self.v_a_w = None
        self.v_a_n = None
        self.v_a_s = None
        self.v_a_p = None
        self.v_source = None
        self.v_a_p_unrelaxed = None
        self.v_source_unrelaxed = None
        self.v_matrix = None
        self.v_rhs = None

    def _build_sparse_matrix(self, a_e, a_w, a_n, a_s, a_p, source, nx, ny, is_u=True):
        """
        Build a sparse matrix from the coefficients using vectorized operations.

        Parameters:
        -----------
        a_e, a_w, a_n, a_s, a_p : ndarray
            Coefficient arrays
        source : ndarray
            Source term
        nx, ny : int
            Mesh dimensions
        is_u : bool
            Flag to indicate if this is for u (True) or v (False) momentum

        Returns:
        --------
        tuple
            (sparse_matrix, rhs_vector, flattened_indices_map)
        """
        if is_u:
            rows, cols = nx + 1, ny
        else:
            rows, cols = nx, ny + 1

        n_cells = rows * cols
        idx_map = np.arange(n_cells).reshape(rows, cols)

        data = []
        row_indices = []
        col_indices = []

        # 1. Diagonal entries (A_p)
        flat_idx = idx_map.flatten()
        data.extend(a_p.flatten())
        row_indices.extend(flat_idx)
        col_indices.extend(flat_idx)

        # Create meshgrid for indices
        i_grid, j_grid = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

        # 2. East neighbors (-A_e)
        # Ensure a_e has the correct shape corresponding to the grid
        mask_e = i_grid < rows - 1
        valid_i_e, valid_j_e = i_grid[mask_e], j_grid[mask_e]
        if valid_i_e.size > 0: # Check if there are any east neighbors
            row_idx_e = idx_map[valid_i_e, valid_j_e]
            col_idx_e = idx_map[valid_i_e + 1, valid_j_e]
            # Ensure indices match a_e's shape/domain
            data_e = -a_e[valid_i_e, valid_j_e]
            data.extend(data_e)
            row_indices.extend(row_idx_e)
            col_indices.extend(col_idx_e)

        # 3. West neighbors (-A_w)
        mask_w = i_grid > 0
        valid_i_w, valid_j_w = i_grid[mask_w], j_grid[mask_w]
        if valid_i_w.size > 0: # Check if there are any west neighbors
            row_idx_w = idx_map[valid_i_w, valid_j_w]
            col_idx_w = idx_map[valid_i_w - 1, valid_j_w]
            # Ensure indices match a_w's shape/domain
            data_w = -a_w[valid_i_w, valid_j_w]
            data.extend(data_w)
            row_indices.extend(row_idx_w)
            col_indices.extend(col_idx_w)

        # 4. North neighbors (-A_n)
        mask_n = j_grid < cols - 1
        valid_i_n, valid_j_n = i_grid[mask_n], j_grid[mask_n]
        if valid_i_n.size > 0: # Check if there are any north neighbors
            row_idx_n = idx_map[valid_i_n, valid_j_n]
            col_idx_n = idx_map[valid_i_n, valid_j_n + 1]
            # Ensure indices match a_n's shape/domain
            data_n = -a_n[valid_i_n, valid_j_n]
            data.extend(data_n)
            row_indices.extend(row_idx_n)
            col_indices.extend(col_idx_n)

        # 5. South neighbors (-A_s)
        mask_s = j_grid > 0
        valid_i_s, valid_j_s = i_grid[mask_s], j_grid[mask_s]
        if valid_i_s.size > 0: # Check if there are any south neighbors
            row_idx_s = idx_map[valid_i_s, valid_j_s]
            col_idx_s = idx_map[valid_i_s, valid_j_s - 1]
            # Ensure indices match a_s's shape/domain
            data_s = -a_s[valid_i_s, valid_j_s]
            data.extend(data_s)
            row_indices.extend(row_idx_s)
            col_indices.extend(col_idx_s)

        # 6. RHS vector
        rhs = source.flatten()

        # 7. Create sparse matrix
        # Using COO format first for construction
        matrix_coo = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))
        # Convert to CSR for efficient arithmetic operations later
        matrix_csr = matrix_coo.tocsr()
        
        # Sum duplicates in case any arise (e.g. if an index is targeted by multiple coefficient additions)
        matrix_csr.sum_duplicates() 

        return matrix_csr, rhs, idx_map

    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        u_star = np.zeros((imax+1, jmax))
        d_u = np.zeros((imax+1, jmax))

        # Apply BCs before coefficient generation
        if boundary_conditions:
            if isinstance(boundary_conditions, BoundaryConditionManager):
                bc_manager = boundary_conditions
            else:
                bc_manager = BoundaryConditionManager()
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)
            u, v = bc_manager.apply_velocity_boundary_conditions(u.copy(), v.copy(), imax, jmax)

        self.calculate_coefficients(mesh, fluid, u, v, p, boundary_conditions, relaxation_factor)
        
        # Build sparse matrix for u-momentum
        self.u_matrix, self.u_rhs, idx_map = self._build_sparse_matrix(
            self.u_a_e, self.u_a_w, self.u_a_n, self.u_a_s, self.u_a_p, 
            self.u_source, nx, ny, is_u=True
        )

        i_range = np.arange(1, imax)
        j_range = np.arange(1, jmax-1)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        # Flatten u for matrix operations
        u_flat_initial_guess = u.flatten() # Use current u as initial guess

        # Create AMG preconditioner
        ml = pyamg.smoothed_aggregation_solver(self.u_matrix)
        M = ml.aspreconditioner(cycle='V')

        # Solve using CG with AMG preconditioner
        u_flat, info = cg(self.u_matrix, self.u_rhs, 
                          x0=u_flat_initial_guess, 
                          atol=self.tolerance, 
                          maxiter=self.max_iterations, 
                          M=M)
        
        if info > 0:
            print(f"Warning: U-momentum CG solver did not converge within {self.max_iterations} iterations. Info code: {info}")
        elif info < 0:
            print(f"Error: U-momentum CG solver encountered illegal input or breakdown. Info code: {info}")
        
        # Reshape back to 2D
        u_star_unrelaxed = u_flat.reshape((imax+1, jmax))

        # Calculate residual BEFORE relaxation: r = b - Ax
        Ax_unrelaxed = self.u_matrix @ u_flat # Use the solved u_flat
        r = self.u_rhs - Ax_unrelaxed
        r_norm = np.linalg.norm(r) # Default L2 norm
        b_norm = np.linalg.norm(self.u_rhs)
        u_residual = r_norm / b_norm if b_norm > 1e-12 else r_norm # Avoid division by zero
        
        # Apply relaxation to the internal cells
        # Note: Using u (previous iteration velocity) for the (1-alpha) part
        u_star[i_grid, j_grid] = alpha * u_star_unrelaxed[i_grid, j_grid] + (1-alpha)*u[i_grid, j_grid]
        
        # Keep boundary values from unrelaxed solution (or apply BCs later)
        # This depends on how BCs interact with relaxation
        # For now, copy boundary values directly from the unrelaxed solution
        # This assumes BCs are fixed and shouldn't be relaxed towards the previous iteration.
        u_star[0, :] = u_star_unrelaxed[0, :]
        u_star[imax, :] = u_star_unrelaxed[imax, :]
        u_star[:, 0] = u_star_unrelaxed[:, 0]
        u_star[:, jmax-1] = u_star_unrelaxed[:, jmax-1] # Adjusted index for jmax

        d_u[i_grid, j_grid] = alpha * mesh.get_cell_sizes()[1] / self.u_a_p[i_grid, j_grid]

        j = 0
        i_bottom = np.arange(1, imax)
        d_u[i_bottom, j] = alpha * mesh.get_cell_sizes()[1] / self.u_a_p[i_bottom, j]

        j = jmax-1
        i_top = np.arange(1, imax)
        d_u[i_top, j] = alpha * mesh.get_cell_sizes()[1] / self.u_a_p[i_top, j]

        if boundary_conditions:
            # Re-apply BCs after relaxation and potential modification
            u_star, _ = bc_manager.apply_velocity_boundary_conditions(u_star, v.copy(), imax, jmax)

        # Residual calculation moved before relaxation
        # Return the calculated residual
        return u_star, d_u, u_residual

    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        v_star = np.zeros((imax, jmax+1))
        d_v = np.zeros((imax, jmax+1))

        if boundary_conditions:
            if isinstance(boundary_conditions, BoundaryConditionManager):
                bc_manager = boundary_conditions
            else:
                bc_manager = BoundaryConditionManager()
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)
            u, v = bc_manager.apply_velocity_boundary_conditions(u.copy(), v.copy(), imax, jmax)

        if self.v_a_p is None:
            self.calculate_coefficients(mesh, fluid, u, v, p, boundary_conditions, relaxation_factor)
        
        # Build sparse matrix for v-momentum
        self.v_matrix, self.v_rhs, idx_map = self._build_sparse_matrix(
            self.v_a_e, self.v_a_w, self.v_a_n, self.v_a_s, self.v_a_p, 
            self.v_source, nx, ny, is_u=False
        )

        i_range = np.arange(1, imax-1)
        j_range = np.arange(1, jmax)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
        
        # Flatten v for matrix operations
        v_flat_initial_guess = v.flatten() # Use current v as initial guess

        # Create AMG preconditioner
        ml = pyamg.smoothed_aggregation_solver(self.v_matrix)
        M = ml.aspreconditioner(cycle='V')

        # Solve using CG with AMG preconditioner
        v_flat, info = cg(self.v_matrix, self.v_rhs, 
                          x0=v_flat_initial_guess, 
                          atol=self.tolerance, 
                          maxiter=self.max_iterations, 
                          M=M)
        
        if info > 0:
            print(f"Warning: V-momentum CG solver did not converge within {self.max_iterations} iterations. Info code: {info}")
        elif info < 0:
            print(f"Error: V-momentum CG solver encountered illegal input or breakdown. Info code: {info}")

        # Reshape back to 2D
        v_star_unrelaxed = v_flat.reshape((imax, jmax+1))

        # Calculate residual BEFORE relaxation: r = b - Ax
        Ax_unrelaxed = self.v_matrix @ v_flat # Use the solved v_flat
        r = self.v_rhs - Ax_unrelaxed
        r_norm = np.linalg.norm(r) # Default L2 norm
        b_norm = np.linalg.norm(self.v_rhs)
        v_residual = r_norm / b_norm if b_norm > 1e-12 else r_norm # Avoid division by zero

        # Apply relaxation to the internal cells
        # Note: Using v (previous iteration velocity) for the (1-alpha) part
        v_star[i_grid, j_grid] = alpha * v_star_unrelaxed[i_grid, j_grid] + (1-alpha)*v[i_grid, j_grid]

        # Keep boundary values from unrelaxed solution (or apply BCs later)
        # Similar logic as in u-momentum
        v_star[0, :] = v_star_unrelaxed[0, :]
        v_star[imax-1, :] = v_star_unrelaxed[imax-1, :] # Adjusted index for imax
        v_star[:, 0] = v_star_unrelaxed[:, 0]
        v_star[:, jmax] = v_star_unrelaxed[:, jmax]

        d_v[i_grid, j_grid] = alpha * mesh.get_cell_sizes()[0] / self.v_a_p[i_grid, j_grid]

        i = 0
        j_left = np.arange(1, jmax)
        d_v[i, j_left] = alpha * mesh.get_cell_sizes()[0] / self.v_a_p[i, j_left]

        i = imax-1
        j_right = np.arange(1, jmax)
        d_v[i, j_right] = alpha * mesh.get_cell_sizes()[0] / self.v_a_p[i, j_right]

        if boundary_conditions:
            # Re-apply BCs after relaxation and potential modification
            _, v_star = bc_manager.apply_velocity_boundary_conditions(u.copy(), v_star, imax, jmax)

        # Residual calculation moved before relaxation
        # Return the calculated residual
        return v_star, d_v, v_residual