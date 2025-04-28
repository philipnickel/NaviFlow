"""
Algebraic Multigrid (AMG) momentum solver.
"""

import numpy as np
from scipy import sparse
from .base_momentum_solver import MomentumSolver
from .discretization import power_law
from .discretization import quick
from .discretization import second_order_upwind
from .discretization import second_order_upwind
from ...constructor.boundary_conditions import BoundaryConditionManager
import pyamg  # PyAMG library for AMG solvers

class AMGMomentumSolver(MomentumSolver):
    """
    Momentum solver that uses Algebraic Multigrid (AMG) to solve the momentum equations.
    Uses Practice B to incorporate BCs.
    Supports power_law, quick, upwind, and second_order_upwind discretization schemes.
    """

    def __init__(self, discretization_scheme='power_law', tolerance=1e-8, max_iterations=100):
        """
        Initialize the AMG momentum solver.

        Parameters:
        -----------
        discretization_scheme : str, optional
            The discretization scheme to use (default: 'power_law').
            Options: 'power_law', 'quick', 'upwind', 'second_order_upwind'
        tolerance : float, optional
            Convergence tolerance for the AMG solver (default: 1e-8).
        max_iterations : int, optional
            Maximum number of iterations for the AMG solver (default: 100).
        """
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        if discretization_scheme == 'power_law':
            self.discretization_scheme = power_law.PowerLawDiscretization()
        elif discretization_scheme == 'quick':
            self.discretization_scheme = quick.QUICKDiscretization()
        elif discretization_scheme == 'upwind':
            self.discretization_scheme = second_order_upwind.UpwindDiscretization()
        elif discretization_scheme == 'second_order_upwind':
            self.discretization_scheme = second_order_upwind.SecondOrderUpwindDiscretization()
        else:
            raise ValueError(f"Unsupported discretization scheme: {discretization_scheme}. "
                           "Available options: 'power_law', 'quick', 'upwind', 'second_order_upwind'")

        # Store coefficients and matrices
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
        Handles standard and higher-order discretization schemes with second-neighbor coefficients.
        
        Parameters:
        -----------
        a_e, a_w, a_n, a_s : ndarray
            Standard coefficients for east, west, north, south neighbors
        a_p : ndarray
            Diagonal coefficients
        source : ndarray
            Source term vector
        nx, ny : int
            Mesh dimensions
        is_u : bool
            True if building matrix for u-momentum, False for v-momentum
            
        Returns:
        --------
        matrix_csr : csr_matrix
            Sparse matrix in CSR format
        rhs : ndarray
            Right-hand side vector
        idx_map : ndarray
            Mapping from grid indices to matrix indices
        """
        # Determine matrix dimensions based on velocity component
        if is_u:
            rows, cols = nx + 1, ny  # u-velocity grid dimensions
        else:
            rows, cols = nx, ny + 1  # v-velocity grid dimensions

        n_cells = rows * cols
        idx_map = np.arange(n_cells).reshape(rows, cols)

        # Initialize storage for matrix elements
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

        # 6. Handle higher-order schemes (QUICK or second-order upwind)
        if hasattr(self.discretization_scheme, '__class__') and \
           self.discretization_scheme.__class__.__name__ in ['QUICKDiscretization', 'SecondOrderUpwindDiscretization']:
            # Get second-neighbor coefficients if they exist
            a_ee = getattr(self, 'u_a_ee' if is_u else 'v_a_ee', np.zeros_like(a_e))
            a_ww = getattr(self, 'u_a_ww' if is_u else 'v_a_ww', np.zeros_like(a_w))
            a_nn = getattr(self, 'u_a_nn' if is_u else 'v_a_nn', np.zeros_like(a_n))
            a_ss = getattr(self, 'u_a_ss' if is_u else 'v_a_ss', np.zeros_like(a_s))

            # East-East neighbors (-A_ee)
            mask_ee = i_grid < rows - 2
            valid_i_ee, valid_j_ee = i_grid[mask_ee], j_grid[mask_ee]
            if valid_i_ee.size > 0:
                row_idx_ee = idx_map[valid_i_ee, valid_j_ee]
                col_idx_ee = idx_map[valid_i_ee + 2, valid_j_ee]
                if a_ee.shape[0] > np.max(valid_i_ee) and a_ee.shape[1] > np.max(valid_j_ee):
                    data_ee = -a_ee[valid_i_ee, valid_j_ee]
                    data.extend(data_ee)
                    row_indices.extend(row_idx_ee)
                    col_indices.extend(col_idx_ee)

            # West-West neighbors (-A_ww)
            mask_ww = i_grid > 1
            valid_i_ww, valid_j_ww = i_grid[mask_ww], j_grid[mask_ww]
            if valid_i_ww.size > 0:
                row_idx_ww = idx_map[valid_i_ww, valid_j_ww]
                col_idx_ww = idx_map[valid_i_ww - 2, valid_j_ww]
                if a_ww.shape[0] > np.max(valid_i_ww) and a_ww.shape[1] > np.max(valid_j_ww):
                    data_ww = -a_ww[valid_i_ww, valid_j_ww]
                    data.extend(data_ww)
                    row_indices.extend(row_idx_ww)
                    col_indices.extend(col_idx_ww)

            # North-North neighbors (-A_nn)
            mask_nn = j_grid < cols - 2
            valid_i_nn, valid_j_nn = i_grid[mask_nn], j_grid[mask_nn]
            if valid_i_nn.size > 0:
                row_idx_nn = idx_map[valid_i_nn, valid_j_nn]
                col_idx_nn = idx_map[valid_i_nn, valid_j_nn + 2]
                if a_nn.shape[0] > np.max(valid_i_nn) and a_nn.shape[1] > np.max(valid_j_nn):
                    data_nn = -a_nn[valid_i_nn, valid_j_nn]
                    data.extend(data_nn)
                    row_indices.extend(row_idx_nn)
                    col_indices.extend(col_idx_nn)

            # South-South neighbors (-A_ss)
            mask_ss = j_grid > 1
            valid_i_ss, valid_j_ss = i_grid[mask_ss], j_grid[mask_ss]
            if valid_i_ss.size > 0:
                row_idx_ss = idx_map[valid_i_ss, valid_j_ss]
                col_idx_ss = idx_map[valid_i_ss, valid_j_ss - 2]
                if a_ss.shape[0] > np.max(valid_i_ss) and a_ss.shape[1] > np.max(valid_j_ss):
                    data_ss = -a_ss[valid_i_ss, valid_j_ss]
                    data.extend(data_ss)
                    row_indices.extend(row_idx_ss)
                    col_indices.extend(col_idx_ss)

        # 7. RHS vector
        rhs = source.flatten()

        # 8. Create sparse matrix
        matrix_coo = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))
        matrix_csr = matrix_coo.tocsr()
        matrix_csr.sum_duplicates()  # Important for CSR format to merge duplicates

        return matrix_csr, rhs, idx_map

    def _build_sparse_matrix_face_based(self, mesh, a_p, a_nb, source, is_u=True):
        """
        Build a mesh-agnostic sparse matrix for momentum equations.
        
        Parameters
        ----------
        mesh : Mesh
            Mesh object with owner/neighbor information.
        a_p : ndarray
            Diagonal coefficients for cells.
        a_nb : dict
            Dictionary with neighbor coefficients: keys 'east', 'west', 'north', 'south'.
        source : ndarray
            Source term.
        is_u : bool
            Whether solving for u (True) or v (False).
        
        Returns
        -------
        A_csr : csr_matrix
            System matrix.
        rhs : ndarray
            Right-hand side vector.
        """
        from scipy.sparse import coo_matrix

        owners, neighbors = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces

        data = []
        row = []
        col = []

        # Diagonal terms
        for cell_idx in range(n_cells):
            data.append(a_p[cell_idx])
            row.append(cell_idx)
            col.append(cell_idx)

        # Off-diagonal neighbor terms
        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]

            if neighbor != -1:
                # Internal face → interaction between owner and neighbor
                # Assume uniform treatment for now
                coeff = a_nb.get('face', np.zeros(n_faces))[face_idx]  # if a_nb['face'] exists
                if coeff == 0:  # fallback
                    coeff = 0.5 * (a_p[owner] + a_p[neighbor])  # crude approximation

                data.append(-coeff)
                row.append(owner)
                col.append(neighbor)

                data.append(-coeff)
                row.append(neighbor)
                col.append(owner)

        # Build sparse matrix
        A = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()

        # Right-hand side
        rhs = source.flatten()

        return A, rhs

    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None, return_dict=True):
        """
        Solve the u-momentum equation using AMG for a collocated grid.
        """
        alpha = relaxation_factor

        if isinstance(boundary_conditions, BoundaryConditionManager):
            bc_manager = boundary_conditions
        else:
            bc_manager = BoundaryConditionManager()
            if boundary_conditions:
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)

        u_initial_guess = u.copy()
        n_cells = mesh.n_cells

        # Pass original u,v to discretization; BCs handled by discretization scheme (Practice B)
        # u_bc, v_bc = bc_manager.apply_velocity_boundary_conditions(...) # Removed

        coeffs = self.discretization_scheme.calculate_u_coefficients(mesh, fluid, u, v, p, bc_manager)
        # Assume coeffs are returned as 1D arrays of size n_cells
        a_p_unrelaxed = coeffs['a_p']
        source_unrelaxed = coeffs['source']
        if a_p_unrelaxed.size != n_cells or source_unrelaxed.size != n_cells:
             raise ValueError(f"Coefficient array sizes ({a_p_unrelaxed.size}, {source_unrelaxed.size}) mismatch n_cells ({n_cells})")

        self.u_a_p_unrelaxed = a_p_unrelaxed 
        self.u_source_unrelaxed = source_unrelaxed

        # --- Prepare full relaxed coefficients (1D arrays) ---
        safe_ap_full = np.where(np.abs(a_p_unrelaxed) > 1e-12, a_p_unrelaxed, 1e-12)
        relaxed_a_p_full = safe_ap_full / alpha
        
        # Calculate relaxed source (all 1D arrays)
        # Use original u field (flattened if needed) for relaxation term
        u_flat = u.flatten() if u.ndim > 1 else u
        if relaxed_a_p_full.shape != u_flat.shape:
             raise ValueError("Unexpected relaxed_a_p_full dimensions")
        if source_unrelaxed.shape != relaxed_a_p_full.shape:
             raise ValueError("Unexpected relaxed_source_full dimensions")
        relaxed_source_full = source_unrelaxed + (1.0 - alpha) * relaxed_a_p_full * u_flat
        
        # --- Build sparse system using FACE-BASED method for ALL cells ---
        A, b = self._build_sparse_matrix_face_based(
            mesh, relaxed_a_p_full, coeffs.get('a_nb', {}), relaxed_source_full, is_u=True
        )
        
        # --- Solve using AMG --- 
        ml = pyamg.smoothed_aggregation_solver(A)
        
        # Ensure initial guess is flat
        u_initial_guess_flat = u_initial_guess.flatten()
        if u_initial_guess_flat.shape != b.shape:
             if u_initial_guess.size == b.size:
                  u_initial_guess_flat = u_initial_guess.reshape(b.shape)
             else:
                  raise ValueError(f"Shape mismatch for solver: initial guess {u_initial_guess_flat.shape} (orig: {u_initial_guess.shape}) vs rhs {b.shape}")

        u_full_flat = ml.solve(
            b, 
            x0=u_initial_guess_flat, 
            tol=self.tolerance, 
            maxiter=self.max_iterations
        )

        # --- Return 1D solution --- 
        # Reshaping is left to the caller
        u_star = u_full_flat 

        # --- d_u calculation (1D) ---
        d_u = np.zeros(n_cells) # Return 1D array
        # Use Vp / aP definition
        cell_volumes = mesh.get_cell_volumes()
        if hasattr(self, 'u_a_p_unrelaxed') and self.u_a_p_unrelaxed is not None:
            temp_ap_unrelaxed = self.u_a_p_unrelaxed # Assumed 1D
            if temp_ap_unrelaxed.size == d_u.size and cell_volumes.size == d_u.size:
                valid_ap_mask = np.abs(temp_ap_unrelaxed) > 1e-12
                # d_u = Vp / aP_u
                d_u[valid_ap_mask] = cell_volumes[valid_ap_mask] / temp_ap_unrelaxed[valid_ap_mask]
            else:
                print(f"Warning: Size mismatch for d_u calculation: a_p {temp_ap_unrelaxed.size}, vol {cell_volumes.size}, d_u {d_u.size}")

        # Residual info
        r = b - A @ u_full_flat
        norm_r = np.linalg.norm(r)
        norm_b = np.linalg.norm(b)
        rel_norm = norm_r / max(norm_b, 1e-10)

        residual_info = {
            'rel_norm': rel_norm,
            'field': r # Return 1D residual field from full system
        }

        return u_star, d_u, residual_info
    
    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None, return_dict=True):
        """
        Solve the v-momentum equation using AMG for a collocated grid.
        """
        alpha = relaxation_factor

        if isinstance(boundary_conditions, BoundaryConditionManager):
            bc_manager = boundary_conditions
        else:
            bc_manager = BoundaryConditionManager()
            if boundary_conditions:
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)

        v_initial_guess = v.copy()
        n_cells = mesh.n_cells

        # Pass original u,v to discretization; BCs handled by discretization scheme (Practice B)
        # u_bc, v_bc = bc_manager.apply_velocity_boundary_conditions(...) # Removed

        coeffs = self.discretization_scheme.calculate_v_coefficients(mesh, fluid, u, v, p, bc_manager)
        # Assume coeffs are returned as 1D arrays of size n_cells
        a_p_unrelaxed = coeffs['a_p']
        source_unrelaxed = coeffs['source']
        if a_p_unrelaxed.size != n_cells or source_unrelaxed.size != n_cells:
             raise ValueError(f"Coefficient array sizes ({a_p_unrelaxed.size}, {source_unrelaxed.size}) mismatch n_cells ({n_cells})")

        self.v_a_p_unrelaxed = a_p_unrelaxed 
        self.v_source_unrelaxed = source_unrelaxed

        # --- Prepare full relaxed coefficients (1D arrays) ---
        safe_ap_full = np.where(np.abs(a_p_unrelaxed) > 1e-12, a_p_unrelaxed, 1e-12)
        relaxed_a_p_full = safe_ap_full / alpha
        
        # Calculate relaxed source (all 1D arrays)
        # Use original v field (flattened if needed) for relaxation term
        v_flat = v.flatten() if v.ndim > 1 else v
        if relaxed_a_p_full.shape != v_flat.shape:
             raise ValueError("Unexpected relaxed_a_p_full dimensions")
        if source_unrelaxed.shape != relaxed_a_p_full.shape:
             raise ValueError("Unexpected relaxed_source_full dimensions")
        relaxed_source_full = source_unrelaxed + (1.0 - alpha) * relaxed_a_p_full * v_flat
        
        # --- Build sparse system using FACE-BASED method for ALL cells ---
        A, b = self._build_sparse_matrix_face_based(
             mesh, relaxed_a_p_full, coeffs.get('a_nb', {}), relaxed_source_full, is_u=False
        )

        # --- Solve using AMG --- 
        ml = pyamg.smoothed_aggregation_solver(A)
        
        # Ensure initial guess is flat
        v_initial_guess_flat = v_initial_guess.flatten()
        if v_initial_guess_flat.shape != b.shape:
             if v_initial_guess.size == b.size:
                  v_initial_guess_flat = v_initial_guess.reshape(b.shape)
             else:
                  raise ValueError(f"Shape mismatch for solver: initial guess {v_initial_guess_flat.shape} (orig: {v_initial_guess.shape}) vs rhs {b.shape}")

        v_full_flat = ml.solve(
             b, 
             x0=v_initial_guess_flat, 
             tol=self.tolerance, 
             maxiter=self.max_iterations
        )

        # --- Return 1D solution --- 
        v_star = v_full_flat

        # --- d_v calculation (1D) ---
        d_v = np.zeros(n_cells) # Return 1D array
        # Use Vp / aP definition
        cell_volumes = mesh.get_cell_volumes()
        if hasattr(self, 'v_a_p_unrelaxed') and self.v_a_p_unrelaxed is not None:
            temp_ap_unrelaxed = self.v_a_p_unrelaxed # Assumed 1D
            if temp_ap_unrelaxed.size == d_v.size and cell_volumes.size == d_v.size:
                 valid_ap_mask = np.abs(temp_ap_unrelaxed) > 1e-12
                 # d_v = Vp / aP_v
                 d_v[valid_ap_mask] = cell_volumes[valid_ap_mask] / temp_ap_unrelaxed[valid_ap_mask]
            else:
                print(f"Warning: Size mismatch for d_v calculation: a_p {temp_ap_unrelaxed.size}, vol {cell_volumes.size}, d_v {d_v.size}")

        r = b - A @ v_full_flat
        norm_r = np.linalg.norm(r)
        norm_b = np.linalg.norm(b)
        rel_norm = norm_r / max(norm_b, 1e-10)

        residual_info = {
            'rel_norm': rel_norm,
            'field': r # Return 1D residual field from full system
        }

        return v_star, d_v, residual_info
