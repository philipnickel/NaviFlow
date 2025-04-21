"""
Algebraic Multigrid (AMG) momentum solver.
"""

import numpy as np
from scipy import sparse
from .base_momentum_solver import MomentumSolver
from .discretization import power_law
from ...constructor.boundary_conditions import BoundaryConditionManager
import pyamg # PyAMG library for AMG solvers

class AMGMomentumSolver(MomentumSolver):
    """
    Momentum solver that uses Algebraic Multigrid (AMG) to solve the momentum equations.
    Uses Practice B to incorporate BCs.
    """

    def __init__(self, discretization_scheme='power_law', tolerance=1e-8, max_iterations=100):
        """
        Initialize the AMG momentum solver.

        Parameters:
        -----------
        discretization_scheme : str, optional
            The discretization scheme to use (default: 'power_law').
        tolerance : float, optional
            Convergence tolerance for the AMG solver (default: 1e-8).
        max_iterations : int, optional
            Maximum number of iterations for the AMG solver (default: 100).
        """
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        if discretization_scheme == 'power_law':
            self.discretization = power_law.PowerLawDiscretization()
        else:
            raise ValueError(f"Unsupported discretization scheme: {discretization_scheme}")

        # Store coefficients and matrices similar to JacobiMatrixSolver
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
        (Copied from JacobiMatrixMomentumSolver - should be identical)
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
        mask_e = i_grid < rows - 1
        valid_i_e, valid_j_e = i_grid[mask_e], j_grid[mask_e]
        if valid_i_e.size > 0:
            row_idx_e = idx_map[valid_i_e, valid_j_e]
            col_idx_e = idx_map[valid_i_e + 1, valid_j_e]
            if a_e.shape[0] > np.max(valid_i_e) and a_e.shape[1] > np.max(valid_j_e):
                data_e = -a_e[valid_i_e, valid_j_e]
                data.extend(data_e)
                row_indices.extend(row_idx_e)
                col_indices.extend(col_idx_e)

        # 3. West neighbors (-A_w)
        mask_w = i_grid > 0
        valid_i_w, valid_j_w = i_grid[mask_w], j_grid[mask_w]
        if valid_i_w.size > 0:
            row_idx_w = idx_map[valid_i_w, valid_j_w]
            col_idx_w = idx_map[valid_i_w - 1, valid_j_w]
            if a_w.shape[0] > np.max(valid_i_w) and a_w.shape[1] > np.max(valid_j_w):
                data_w = -a_w[valid_i_w, valid_j_w]
                data.extend(data_w)
                row_indices.extend(row_idx_w)
                col_indices.extend(col_idx_w)

        # 4. North neighbors (-A_n)
        mask_n = j_grid < cols - 1
        valid_i_n, valid_j_n = i_grid[mask_n], j_grid[mask_n]
        if valid_i_n.size > 0:
            row_idx_n = idx_map[valid_i_n, valid_j_n]
            col_idx_n = idx_map[valid_i_n, valid_j_n + 1]
            if a_n.shape[0] > np.max(valid_i_n) and a_n.shape[1] > np.max(valid_j_n):
                data_n = -a_n[valid_i_n, valid_j_n]
                data.extend(data_n)
                row_indices.extend(row_idx_n)
                col_indices.extend(col_idx_n)

        # 5. South neighbors (-A_s)
        mask_s = j_grid > 0
        valid_i_s, valid_j_s = i_grid[mask_s], j_grid[mask_s]
        if valid_i_s.size > 0:
            row_idx_s = idx_map[valid_i_s, valid_j_s]
            col_idx_s = idx_map[valid_i_s, valid_j_s - 1]
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
        matrix_csr.sum_duplicates() # Important for CSR format

        return matrix_csr, rhs, idx_map

    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        # Use current u as initial guess
        u_initial_guess = u.copy()
        d_u = np.zeros((imax+1, jmax))

        # Ensure we have a BC manager instance
        if isinstance(boundary_conditions, BoundaryConditionManager):
            bc_manager = boundary_conditions
        else:
            bc_manager = BoundaryConditionManager()
            if boundary_conditions:
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)

        # Apply BCs to u and v *before* coefficient calculation
        u_bc, v_bc = bc_manager.apply_velocity_boundary_conditions(u.copy(), v.copy(), imax, jmax)

        # Calculate coefficients using velocities with BCs applied
        coeffs = self.discretization.calculate_u_coefficients(mesh, fluid, u_bc, v_bc, p, bc_manager)
        u_a_e = coeffs['a_e']
        u_a_w = coeffs['a_w']
        u_a_n = coeffs['a_n']
        u_a_s = coeffs['a_s']
        u_a_p_unrelaxed = coeffs['a_p']
        u_source_unrelaxed = coeffs['source']

        # Apply under-relaxation to coefficients (ORIGINAL METHOD)
        self.u_a_p = u_a_p_unrelaxed / alpha
        # Need to handle potential division by zero if alpha is 0 or a_p_unrelaxed is 0
        safe_ap_unrelaxed = np.where(np.abs(u_a_p_unrelaxed) > 1e-12, u_a_p_unrelaxed, 1e-12)
        self.u_a_p = safe_ap_unrelaxed / alpha 
        u_source = u_source_unrelaxed + (1 - alpha) * self.u_a_p * u_bc

        # Build the sparse matrix system using RELAXED coefficients
        self.u_matrix, self.u_rhs, idx_map = self._build_sparse_matrix(
            u_a_e, u_a_w, u_a_n, u_a_s, self.u_a_p, # Use RELAXED a_p
            u_source, # Use RELAXED source
            nx, ny, is_u=True
        )

        # --- AMG Solver ---
        # Create the AMG solver hierarchy
        ml = pyamg.smoothed_aggregation_solver(self.u_matrix)

        # Solve the RELAXED system Ax = b, using initial guess
        u_flat = ml.solve(self.u_rhs, x0=u_initial_guess.flatten(), tol=self.tolerance, maxiter=self.max_iterations)

        # Reshape result back to 2D
        u_star = u_flat.reshape((imax+1, jmax))

        # Apply boundary conditions explicitly AFTER the solve to ensure they are met
        # (AMG might not perfectly preserve them, especially if BCs were implicitly handled)
        u_star, _ = bc_manager.apply_velocity_boundary_conditions(u_star, v.copy(), imax, jmax)
        
        # Calculate d_u (using the RELAXED a_p, as this is consistent with the solved system)
        d_u.fill(np.nan)
        valid_ap_mask = np.abs(self.u_a_p) > 1e-12
        dy = mesh.get_cell_sizes()[1]
        d_u[valid_ap_mask] = dy / self.u_a_p[valid_ap_mask]

        # Calculate residual: r = b - Ax (using the RELAXED system and the final u_star)
        # Re-flatten u_star after potentially applying BCs
        Ax = self.u_matrix @ u_star.flatten() 
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
        u_residual_field_masked[:, ny-1] = 0.0 # Top boundary (u lives up to j=ny-1)
        
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

        # Reshape the original full residual field for returning (will be zeroed later)
        u_residual_field = u_residual_field_full 

        # Zero out residuals at boundary and adjacent nodes in the RETURNED field
        u_residual_field[0, :] = 0.0
        u_residual_field[1, :] = 0.0
        if nx > 1:
            u_residual_field[nx-1, :] = 0.0
        u_residual_field[nx, :] = 0.0

        return u_star, d_u, u_residual_norm, u_residual_field

    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        # Use current v as initial guess
        v_initial_guess = v.copy()
        d_v = np.zeros((imax, jmax+1))

        # Ensure we have a BC manager instance
        if isinstance(boundary_conditions, BoundaryConditionManager):
            bc_manager = boundary_conditions
        else:
            bc_manager = BoundaryConditionManager()
            if boundary_conditions:
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)

        # Apply BCs to u and v *before* coefficient calculation
        u_bc, v_bc = bc_manager.apply_velocity_boundary_conditions(u.copy(), v.copy(), imax, jmax)

        # Calculate coefficients using velocities with BCs applied
        coeffs = self.discretization.calculate_v_coefficients(mesh, fluid, u_bc, v_bc, p, bc_manager)
        v_a_e = coeffs['a_e']
        v_a_w = coeffs['a_w']
        v_a_n = coeffs['a_n']
        v_a_s = coeffs['a_s']
        v_a_p_unrelaxed = coeffs['a_p']
        v_source_unrelaxed = coeffs['source']

        # Apply under-relaxation to coefficients (ORIGINAL METHOD)
        safe_ap_unrelaxed_v = np.where(np.abs(v_a_p_unrelaxed) > 1e-12, v_a_p_unrelaxed, 1e-12)
        self.v_a_p = safe_ap_unrelaxed_v / alpha
        v_source = v_source_unrelaxed + (1 - alpha) * self.v_a_p * v_bc

        # Build the sparse matrix system using RELAXED coefficients
        self.v_matrix, self.v_rhs, idx_map = self._build_sparse_matrix(
            v_a_e, v_a_w, v_a_n, v_a_s, self.v_a_p, # Use RELAXED a_p
            v_source, # Use RELAXED source
            nx, ny, is_u=False
        )

        # --- AMG Solver ---
        # Create the AMG solver hierarchy
        ml = pyamg.smoothed_aggregation_solver(self.v_matrix)

        # Solve the RELAXED system Ax = b, using initial guess
        v_flat = ml.solve(self.v_rhs, x0=v_initial_guess.flatten(), tol=self.tolerance, maxiter=self.max_iterations)

        # Reshape result back to 2D
        v_star = v_flat.reshape((imax, jmax+1))

        # Apply boundary conditions explicitly AFTER the solve
        _, v_star = bc_manager.apply_velocity_boundary_conditions(u.copy(), v_star, imax, jmax)

        # Calculate d_v (using the RELAXED a_p)
        d_v.fill(np.nan)
        valid_ap_mask = np.abs(self.v_a_p) > 1e-12
        dx = mesh.get_cell_sizes()[0]
        d_v[valid_ap_mask] = dx / self.v_a_p[valid_ap_mask]

        # Calculate residual: r = b - Ax (using the RELAXED system and final v_star)
        Ax = self.v_matrix @ v_star.flatten()
        r = self.v_rhs - Ax # Residual vector (1D)

        # --- Calculate normalized L2 norm explicitly excluding boundaries --- 
        # 1. Reshape residual and RHS to 2D fields
        v_residual_field_full = r.reshape((imax, jmax+1))
        rhs_field_full = self.v_rhs.reshape((imax, jmax+1))

        # 2. Create copies and zero out boundary values
        v_residual_field_masked = v_residual_field_full.copy()
        rhs_field_masked = rhs_field_full.copy()

        v_residual_field_masked[0, :] = 0.0   # Left boundary
        v_residual_field_masked[nx-1, :] = 0.0 # Right boundary (v lives up to i=nx-1)
        v_residual_field_masked[:, 0] = 0.0   # Bottom boundary
        v_residual_field_masked[:, ny] = 0.0   # Top boundary (v lives up to j=ny)

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

        # Reshape the original full residual field for returning (will be zeroed later)
        v_residual_field = v_residual_field_full

        # Zero out residuals at boundary and adjacent nodes in the RETURNED field
        v_residual_field[:, 0] = 0.0
        v_residual_field[:, 1] = 0.0
        if ny > 1:
            v_residual_field[:, ny-1] = 0.0
        v_residual_field[:, ny] = 0.0

        return v_star, d_v, v_residual_norm, v_residual_field 