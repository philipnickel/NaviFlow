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
    Can use different discretization schemes. Uses Practice B to incorporate BCs.
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
        # Practice B is implemented within these coefficient functions
        coeffs = self.discretization.calculate_u_coefficients(mesh, fluid, u_bc, v_bc, p, bc_manager)
        u_a_e = coeffs['a_e']
        u_a_w = coeffs['a_w']
        u_a_n = coeffs['a_n']
        u_a_s = coeffs['a_s']
        u_a_p_unrelaxed = coeffs['a_p']
        u_source_unrelaxed = coeffs['source']

        # Apply under-relaxation to coefficients
        self.u_a_p = u_a_p_unrelaxed / alpha
        u_source = u_source_unrelaxed + (1 - alpha) * u_a_p_unrelaxed / alpha * u_bc # Use u_bc here

        # Build the matrix system directly (Practice B handles BCs in coefficients)
        self.u_matrix, self.u_rhs, idx_map = self._build_sparse_matrix(
            u_a_e, u_a_w, u_a_n, u_a_s, self.u_a_p, # Use relaxed a_p
            u_source, # Use relaxed source
            nx, ny, is_u=True
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
  
        
        # Calculate d_u (using the relaxed a_p)
        # Need to handle boundaries where a_p might be zero or modified
        d_u.fill(np.nan) # Initialize with NaN
        valid_ap_mask = np.abs(self.u_a_p) > 1e-12
        dy = mesh.get_cell_sizes()[1]
        d_u[valid_ap_mask] = dy / self.u_a_p[valid_ap_mask] # Note: alpha is implicitly included in self.u_a_p now

        # Calculate residual: r = b - Ax (using the final u_star)
        u_star_flat_final = u_star.flatten()
        Ax = self.u_matrix @ u_star_flat_final
        r = self.u_rhs - Ax # Residual vector (1D)
        
        # --- Calculate normalized L2 norm explicitly excluding boundaries --- 
        # 1. Reshape residual and RHS to 2D fields
        u_residual_field_full = r.reshape((imax+1, jmax))
        rhs_field_full = self.u_rhs.reshape((imax+1, jmax))

        # 2. Create copies and zero out boundary values
        u_residual_field_masked = u_residual_field_full.copy()
        rhs_field_masked = rhs_field_full.copy()

        u_residual_field_masked[0, :] = 0.0  # Left boundary
        u_residual_field_masked[nx, :] = 0.0  # Right boundary
        u_residual_field_masked[:, 0] = 0.0  # Bottom boundary
        u_residual_field_masked[:, ny-1] = 0.0 # Top boundary

        rhs_field_masked[0, :] = 0.0
        rhs_field_masked[nx, :] = 0.0
        rhs_field_masked[:, 0] = 0.0
        rhs_field_masked[:, ny-1] = 0.0
        
        # 3. Calculate L2 norm of the masked fields
        r_norm = np.linalg.norm(u_residual_field_masked)
        b_norm = np.linalg.norm(rhs_field_masked)

        # 4. Calculate normalized residual norm
        u_residual_norm = r_norm / (b_norm + 1e-15) if (b_norm + 1e-15) > 0 else r_norm
        # --- End of explicit boundary exclusion norm calculation ---

        # Reshape the original full residual field for returning
        u_residual_field = u_residual_field_full 

        # Zero out residuals at boundary and adjacent nodes in the RETURNED field
        u_residual_field[0, :] = 0.0
        u_residual_field[1, :] = 0.0
        if nx > 1:
            u_residual_field[nx-1, :] = 0.0 # Adjacent to right
        u_residual_field[nx, :] = 0.0  # Right boundary
        # Note: Keep residuals at top/bottom as they showed no bleed

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
        # Practice B is implemented within these coefficient functions
        coeffs = self.discretization.calculate_v_coefficients(mesh, fluid, u_bc, v_bc, p, bc_manager)
        v_a_e = coeffs['a_e']
        v_a_w = coeffs['a_w']
        v_a_n = coeffs['a_n']
        v_a_s = coeffs['a_s']
        v_a_p_unrelaxed = coeffs['a_p']
        v_source_unrelaxed = coeffs['source']

        # Apply under-relaxation to coefficients
        self.v_a_p = v_a_p_unrelaxed / alpha
        v_source = v_source_unrelaxed + (1 - alpha) * v_a_p_unrelaxed / alpha * v_bc # Use v_bc here

        # Build the matrix system directly (Practice B handles BCs in coefficients)
        self.v_matrix, self.v_rhs, idx_map = self._build_sparse_matrix(
            v_a_e, v_a_w, v_a_n, v_a_s, self.v_a_p, # Use relaxed a_p
            v_source, # Use relaxed source
            nx, ny, is_u=False
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

        # Calculate d_v (using the relaxed a_p)
        d_v.fill(np.nan) # Initialize with NaN
        valid_ap_mask = np.abs(self.v_a_p) > 1e-12
        dx = mesh.get_cell_sizes()[0]
        d_v[valid_ap_mask] = dx / self.v_a_p[valid_ap_mask] # Note: alpha is implicitly included in self.v_a_p now

        # Calculate residual: r = b - Ax (using the final v_star)
        v_star_flat_final = v_star.flatten()
        Ax = self.v_matrix @ v_star_flat_final
        r = self.v_rhs - Ax # Residual vector (1D)

        # --- Calculate normalized L2 norm explicitly excluding boundaries ---
        # 1. Reshape residual and RHS to 2D fields
        v_residual_field_full = r.reshape((imax, jmax+1))
        rhs_field_full = self.v_rhs.reshape((imax, jmax+1))

        # 2. Create copies and zero out boundary values
        v_residual_field_masked = v_residual_field_full.copy()
        rhs_field_masked = rhs_field_full.copy()

        v_residual_field_masked[0, :] = 0.0   # Left boundary
        v_residual_field_masked[nx-1, :] = 0.0 # Right boundary
        v_residual_field_masked[:, 0] = 0.0   # Bottom boundary
        v_residual_field_masked[:, ny] = 0.0   # Top boundary

        rhs_field_masked[0, :] = 0.0
        rhs_field_masked[nx-1, :] = 0.0
        rhs_field_masked[:, 0] = 0.0
        rhs_field_masked[:, ny] = 0.0

        # 3. Calculate L2 norm of the masked fields
        r_norm = np.linalg.norm(v_residual_field_masked)
        b_norm = np.linalg.norm(rhs_field_masked)

        # 4. Calculate normalized residual norm
        v_residual_norm = r_norm / (b_norm + 1e-15) if (b_norm + 1e-15) > 0 else r_norm
        # --- End of explicit boundary exclusion norm calculation ---

        # Reshape the original full residual field for returning
        v_residual_field = v_residual_field_full

        # Zero out residuals at boundary and adjacent nodes in the RETURNED field
        v_residual_field[:, 0] = 0.0
        v_residual_field[:, 1] = 0.0
        if ny > 1:
            v_residual_field[:, ny-1] = 0.0 # Adjacent to top
        v_residual_field[:, ny] = 0.0  # Top boundary
        # Note: Keep residuals at left/right as they showed no bleed

        return v_star, d_v, v_residual_norm, v_residual_field