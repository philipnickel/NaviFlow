"""
Jacobi-method momentum solver that can use different discretization schemes.
"""

import numpy as np
from scipy import sparse
from .base_momentum_solver import MomentumSolver
from .discretization import power_law
from ...constructor.boundary_conditions import BoundaryConditionManager, BoundaryLocation, BoundaryType

class JacobiMatrixMomentumSolver(MomentumSolver):
    """
    Momentum solver that uses Jacobi iterations to solve the momentum equations.
    Can use different discretization schemes. Correctly incorporates BCs into the matrix.
    """

    def __init__(self, discretization_scheme='power_law', n_jacobi_sweeps=1):
        super().__init__()
        self.n_jacobi_sweeps = n_jacobi_sweeps

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

    def _apply_bc_to_matrix_system(self, matrix, rhs, idx_map, field, nx, ny, bc_manager, is_u=True):
        """
        Modifies the sparse matrix and RHS vector to enforce Dirichlet BCs.

        Parameters:
        -----------
        matrix : sparse.csr_matrix
            The sparse matrix A
        rhs : ndarray
            The right-hand side vector b
        idx_map : ndarray
            Map from (i, j) grid index to flattened 1D index
        field : ndarray
            The field (u or v) *with boundary values already set*
        nx, ny : int
            Mesh dimensions
        bc_manager : BoundaryConditionManager
            Boundary condition manager instance
        is_u : bool
            True if modifying for u-momentum, False for v-momentum

        Returns:
        --------
        tuple
            (modified_matrix, modified_rhs)
        """
        rows, cols = matrix.shape[0], matrix.shape[1]
        modified_matrix = matrix.copy().tolil() # LIL is efficient for row modifications
        modified_rhs = rhs.copy()

        grid_rows, grid_cols = idx_map.shape

        # Flatten the field to easily get BC values using flat indices
        field_flat = field.flatten()

        boundary_nodes = []
        boundary_values = []

        # Identify boundary nodes and their corresponding values
        # --- U-Momentum ---
        if is_u:
            # Left boundary (i=0)
            if bc_manager.get_condition('left'): # Check if any BC is set for the left boundary
                j_indices = np.arange(grid_cols) # All j values at i=0
                flat_indices = idx_map[0, j_indices]
                boundary_nodes.extend(flat_indices)
                boundary_values.extend(field_flat[flat_indices]) # Use pre-set values

            # Right boundary (i=nx)
            if bc_manager.get_condition('right'): # Check if any BC is set for the right boundary
                 # u is defined up to i=nx
                 j_indices = np.arange(grid_cols) # All j values at i=nx
                 flat_indices = idx_map[nx, j_indices]
                 boundary_nodes.extend(flat_indices)
                 boundary_values.extend(field_flat[flat_indices])

            # Bottom boundary (j=0)
            if bc_manager.get_condition('bottom'):
                 # u is defined from i=1 to nx-1 on the bottom face, but the matrix includes all i
                 # Typically, u at corners (0,0), (nx,0) is handled by left/right BCs.
                 # Interior nodes on bottom edge: i=1 to nx-1
                 i_indices = np.arange(1, grid_rows -1) # Exclude corners if handled by vertical BCs
                 flat_indices = idx_map[i_indices, 0]
                 boundary_nodes.extend(flat_indices)
                 boundary_values.extend(field_flat[flat_indices])

            # Top boundary (j=ny-1)
            if bc_manager.get_condition('top'):
                 # Interior nodes on top edge: i=1 to nx-1
                 i_indices = np.arange(1, grid_rows -1) # Exclude corners if handled by vertical BCs
                 flat_indices = idx_map[i_indices, grid_cols - 1] # j = ny-1
                 boundary_nodes.extend(flat_indices)
                 boundary_values.extend(field_flat[flat_indices])

        # --- V-Momentum ---
        else: # is_v
             # Left boundary (i=0)
             if bc_manager.get_condition('left'):
                 # v is defined from j=1 to ny-1 on the left face
                 # Interior nodes on left edge: j=1 to ny-1
                 j_indices = np.arange(1, grid_cols - 1) # Exclude corners
                 flat_indices = idx_map[0, j_indices]
                 boundary_nodes.extend(flat_indices)
                 boundary_values.extend(field_flat[flat_indices])

             # Right boundary (i=nx-1)
             if bc_manager.get_condition('right'):
                 # Interior nodes on right edge: j=1 to ny-1
                 j_indices = np.arange(1, grid_cols - 1) # Exclude corners
                 flat_indices = idx_map[grid_rows - 1, j_indices] # i = nx-1
                 boundary_nodes.extend(flat_indices)
                 boundary_values.extend(field_flat[flat_indices])

             # Bottom boundary (j=0)
             if bc_manager.get_condition('bottom'):
                 # All i values at j=0
                 i_indices = np.arange(grid_rows)
                 flat_indices = idx_map[i_indices, 0]
                 boundary_nodes.extend(flat_indices)
                 boundary_values.extend(field_flat[flat_indices])

             # Top boundary (j=ny)
             if bc_manager.get_condition('top'):
                 # v is defined up to j=ny
                 i_indices = np.arange(grid_rows)
                 flat_indices = idx_map[i_indices, ny]
                 boundary_nodes.extend(flat_indices)
                 boundary_values.extend(field_flat[flat_indices])

        # Remove duplicates if corners are added by multiple boundaries
        unique_boundary_nodes = list(set(boundary_nodes))
        unique_boundary_values_map = dict(zip(boundary_nodes, boundary_values)) # Use a map for lookup

        # Modify matrix and RHS for unique boundary nodes
        for node_idx in unique_boundary_nodes:
            # Zero out the row corresponding to the boundary node
            modified_matrix.rows[node_idx] = []
            modified_matrix.data[node_idx] = []
            # Set the diagonal element to 1
            modified_matrix[node_idx, node_idx] = 1.0
            # Set the RHS value to the known boundary value
            modified_rhs[node_idx] = unique_boundary_values_map[node_idx]

        return modified_matrix.tocsr(), modified_rhs # Convert back to CSR

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
            # Check if a_e has expected dimensions before indexing
            if a_e.shape[0] > np.max(valid_i_e) and a_e.shape[1] > np.max(valid_j_e):
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
            if a_w.shape[0] > np.max(valid_i_w) and a_w.shape[1] > np.max(valid_j_w):
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
            if a_n.shape[0] > np.max(valid_i_n) and a_n.shape[1] > np.max(valid_j_n):
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
            if a_s.shape[0] > np.max(valid_i_s) and a_s.shape[1] > np.max(valid_j_s):
                 data_s = -a_s[valid_i_s, valid_j_s]
                 data.extend(data_s)
                 row_indices.extend(row_idx_s)
                 col_indices.extend(col_idx_s)

        # 6. RHS vector
        rhs = source.flatten()

        # 7. Create sparse matrix
        matrix_coo = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))
        matrix_csr = matrix_coo.tocsr()
        matrix_csr.sum_duplicates()

        return matrix_csr, rhs, idx_map

    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        # Use current u as initial guess for u_star
        u_star = u.copy()
        d_u = np.zeros((imax+1, jmax))

        # Ensure we have a BC manager instance
        if isinstance(boundary_conditions, BoundaryConditionManager):
            bc_manager = boundary_conditions
        else:
            # Create one if needed (e.g., from a dict)
            bc_manager = BoundaryConditionManager()
            if boundary_conditions:
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)

        # Apply BCs to u and v *before* coefficient calculation
        u_bc, v_bc = bc_manager.apply_velocity_boundary_conditions(u.copy(), v.copy(), imax, jmax)

        # Calculate coefficients using velocities with BCs applied
        coeffs = self.discretization.calculate_u_coefficients(mesh, fluid, u_bc, v_bc, p)
        u_a_e = coeffs['a_e']
        u_a_w = coeffs['a_w']
        u_a_n = coeffs['a_n']
        u_a_s = coeffs['a_s']
        u_a_p_unrelaxed = coeffs['a_p']
        u_source_unrelaxed = coeffs['source']

        # Apply under-relaxation to coefficients
        self.u_a_p = u_a_p_unrelaxed / alpha
        u_source = u_source_unrelaxed + (1 - alpha) * u_a_p_unrelaxed / alpha * u_bc # Use u_bc here

        # Build the initial sparse matrix and RHS (without BC modification yet)
        matrix_no_bc, rhs_no_bc, idx_map = self._build_sparse_matrix(
            u_a_e, u_a_w, u_a_n, u_a_s, self.u_a_p, # Use relaxed a_p
            u_source, # Use relaxed source
            nx, ny, is_u=True
        )

        # Modify the matrix and RHS to enforce boundary conditions
        self.u_matrix, self.u_rhs = self._apply_bc_to_matrix_system(
            matrix_no_bc, rhs_no_bc, idx_map, u_bc, nx, ny, bc_manager, is_u=True
        )

        # --- Jacobi Iteration ---
        u_flat = u_star.flatten() # Use current u (with BCs) as initial guess
        diag = self.u_matrix.diagonal()
        diag_inv = np.zeros_like(diag)
        nonzero_mask = np.abs(diag) > 1e-12 # Avoid division by zero
        diag_inv[nonzero_mask] = 1.0 / diag[nonzero_mask]
        diag_inv_mat = sparse.diags(diag_inv)
        off_diag = self.u_matrix - sparse.diags(diag)

        for _ in range(self.n_jacobi_sweeps):
            u_old_flat = u_flat.copy()
            # Jacobi step: u_new = D^-1 * (b - (A - D) * u_old)
            u_flat = diag_inv_mat @ (self.u_rhs - off_diag @ u_old_flat)

        # Reshape result back to 2D
        u_star = u_flat.reshape((imax+1, jmax))
        
        # Calculate d_u (using the relaxed a_p before BC modification)
        # Need to handle boundaries where a_p might be zero or modified
        d_u.fill(np.nan) # Initialize with NaN
        valid_ap_mask = np.abs(self.u_a_p) > 1e-12
        dy = mesh.get_cell_sizes()[1]
        d_u[valid_ap_mask] = dy / self.u_a_p[valid_ap_mask] # Note: alpha is implicitly included in self.u_a_p now

        # Calculate residual: r = b - Ax (using the final u_star)
        u_star_flat_final = u_star.flatten()
        Ax = self.u_matrix @ u_star_flat_final
        r = self.u_rhs - Ax # Residual vector (1D)
        
        # Calculate normalized L2 norm of the residual
        # Exclude boundary nodes from residual calculation for a better measure of interior convergence
        interior_mask_flat = np.ones_like(self.u_rhs, dtype=bool)
        # Get unique boundary node indices from the modification step
        # Need to pass idx_map and bc_manager to get boundary nodes again, or store them
        # For simplicity, let's recalculate them here (can be optimized)
        _, _, boundary_node_indices = self._get_boundary_nodes(idx_map, nx, ny, bc_manager, is_u=True)
        interior_mask_flat[boundary_node_indices] = False

        r_interior = r[interior_mask_flat]
        b_interior = self.u_rhs[interior_mask_flat]

        r_norm = np.linalg.norm(r_interior)
        b_norm = np.linalg.norm(b_interior)
        u_residual_norm = r_norm / b_norm if b_norm > 1e-12 else r_norm

        # Reshape residual field to 2D (still contains boundary residuals)
        u_residual_field = r.reshape((imax+1, jmax))

        return u_star, d_u, u_residual_norm, u_residual_field

    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        # Use current v as initial guess for v_star
        v_star = v.copy()
        d_v = np.zeros((imax, jmax+1))

        # Ensure we have a BC manager instance
        if isinstance(boundary_conditions, BoundaryConditionManager):
            bc_manager = boundary_conditions
        else:
            # Create one if needed (e.g., from a dict)
            bc_manager = BoundaryConditionManager()
            if boundary_conditions:
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)

        # Apply BCs to u and v *before* coefficient calculation
        u_bc, v_bc = bc_manager.apply_velocity_boundary_conditions(u.copy(), v.copy(), imax, jmax)

        # Calculate coefficients using velocities with BCs applied
        coeffs = self.discretization.calculate_v_coefficients(mesh, fluid, u_bc, v_bc, p)
        v_a_e = coeffs['a_e']
        v_a_w = coeffs['a_w']
        v_a_n = coeffs['a_n']
        v_a_s = coeffs['a_s']
        v_a_p_unrelaxed = coeffs['a_p']
        v_source_unrelaxed = coeffs['source']

        # Apply under-relaxation to coefficients
        self.v_a_p = v_a_p_unrelaxed / alpha
        v_source = v_source_unrelaxed + (1 - alpha) * v_a_p_unrelaxed / alpha * v_bc # Use v_bc here

        # Build the initial sparse matrix and RHS (without BC modification yet)
        matrix_no_bc, rhs_no_bc, idx_map = self._build_sparse_matrix(
            v_a_e, v_a_w, v_a_n, v_a_s, self.v_a_p, # Use relaxed a_p
            v_source, # Use relaxed source
            nx, ny, is_u=False
        )

        # Modify the matrix and RHS to enforce boundary conditions
        self.v_matrix, self.v_rhs = self._apply_bc_to_matrix_system(
            matrix_no_bc, rhs_no_bc, idx_map, v_bc, nx, ny, bc_manager, is_u=False
        )

        # --- Jacobi Iteration ---
        v_flat = v_star.flatten() # Use current v (with BCs) as initial guess
        diag = self.v_matrix.diagonal()
        diag_inv = np.zeros_like(diag)
        nonzero_mask = np.abs(diag) > 1e-12 # Avoid division by zero
        diag_inv[nonzero_mask] = 1.0 / diag[nonzero_mask]
        diag_inv_mat = sparse.diags(diag_inv)
        off_diag = self.v_matrix - sparse.diags(diag)

        for _ in range(self.n_jacobi_sweeps):
            v_old_flat = v_flat.copy()
            # Jacobi step: v_new = D^-1 * (b - (A - D) * v_old)
            v_flat = diag_inv_mat @ (self.v_rhs - off_diag @ v_old_flat)

        # Reshape result back to 2D
        v_star = v_flat.reshape((imax, jmax+1))

        # Calculate d_v (using the relaxed a_p before BC modification)
        d_v.fill(np.nan) # Initialize with NaN
        valid_ap_mask = np.abs(self.v_a_p) > 1e-12
        dx = mesh.get_cell_sizes()[0]
        d_v[valid_ap_mask] = dx / self.v_a_p[valid_ap_mask] # Note: alpha is implicitly included in self.v_a_p now

        # Calculate residual: r = b - Ax (using the final v_star)
        v_star_flat_final = v_star.flatten()
        Ax = self.v_matrix @ v_star_flat_final
        r = self.v_rhs - Ax # Residual vector (1D)

        # Calculate normalized L2 norm of the residual (excluding boundaries)
        interior_mask_flat = np.ones_like(self.v_rhs, dtype=bool)
        # Get unique boundary node indices
        _, _, boundary_node_indices = self._get_boundary_nodes(idx_map, nx, ny, bc_manager, is_u=False)
        interior_mask_flat[boundary_node_indices] = False
        
        r_interior = r[interior_mask_flat]
        b_interior = self.v_rhs[interior_mask_flat]

        r_norm = np.linalg.norm(r_interior)
        b_norm = np.linalg.norm(b_interior)
        v_residual_norm = r_norm / b_norm if b_norm > 1e-12 else r_norm

        # Reshape residual field to 2D (still contains boundary residuals)
        v_residual_field = r.reshape((imax, jmax+1))

        return v_star, d_v, v_residual_norm, v_residual_field

    def _get_boundary_nodes(self, idx_map, nx, ny, bc_manager, is_u=True):
        """ Helper to get boundary node indices without modifying matrix """
        boundary_nodes = []
        grid_rows, grid_cols = idx_map.shape

        if is_u:
            # Left (i=0)
            if bc_manager.get_condition('left'):
                j_indices = np.arange(grid_cols)
                boundary_nodes.extend(idx_map[0, j_indices])
            # Right (i=nx)
            if bc_manager.get_condition('right'):
                j_indices = np.arange(grid_cols)
                boundary_nodes.extend(idx_map[nx, j_indices])
            # Bottom (j=0)
            if bc_manager.get_condition('bottom'):
                i_indices = np.arange(1, grid_rows - 1) # Interior i
                boundary_nodes.extend(idx_map[i_indices, 0])
            # Top (j=ny-1)
            if bc_manager.get_condition('top'):
                i_indices = np.arange(1, grid_rows - 1) # Interior i
                boundary_nodes.extend(idx_map[i_indices, grid_cols - 1])
        else: # is_v
            # Left (i=0)
            if bc_manager.get_condition('left'):
                j_indices = np.arange(1, grid_cols - 1) # Interior j
                boundary_nodes.extend(idx_map[0, j_indices])
            # Right (i=nx-1)
            if bc_manager.get_condition('right'):
                j_indices = np.arange(1, grid_cols - 1) # Interior j
                boundary_nodes.extend(idx_map[grid_rows - 1, j_indices])
            # Bottom (j=0)
            if bc_manager.get_condition('bottom'):
                i_indices = np.arange(grid_rows)
                boundary_nodes.extend(idx_map[i_indices, 0])
            # Top (j=ny)
            if bc_manager.get_condition('top'):
                i_indices = np.arange(grid_rows)
                boundary_nodes.extend(idx_map[i_indices, ny])

        unique_nodes = list(set(boundary_nodes))
        # This helper doesn't need values or modify matrix, just returns indices
        return None, None, unique_nodes