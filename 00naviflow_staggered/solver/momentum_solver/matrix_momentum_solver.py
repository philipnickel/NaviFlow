"""
Matrix-based momentum solver with PETSc-only iterative methods.
"""

import numpy as np
from .base_momentum_solver import MomentumSolver
from .discretization import power_law
from .discretization import quick
from .discretization import upwind
from ...constructor.boundary_conditions import BoundaryConditionManager
from scipy import sparse  # Still needed for CSR matrix construction

# PETSc import
try:
    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc
    HAS_PETSC = True
except ImportError:
    HAS_PETSC = False
    raise ImportError("PETSc is required for this solver. Please install petsc4py.")

class MatrixMomentumSolver(MomentumSolver):
    """
    Momentum solver that uses matrix-based iterative methods from PETSc to solve the momentum equations.
    Supports GMRES, CG, BiCG, and BiCGSTAB (BCGS) solvers with optional preconditioning.
    Uses Practice B to incorporate BCs.
    Supports power_law, quick, upwind, and second_order_upwind discretization schemes.
    """

    def __init__(self, solver_type='gmres', discretization_scheme='power_law', tolerance=1e-8, 
                 max_iterations=100, use_preconditioner=True, print_its=False, restart=30,
                 petsc_pc_type='ilu'):
        """
        Initialize the matrix momentum solver with PETSc.

        Parameters:
        -----------
        solver_type : str, optional
            The PETSc iterative solver to use (default: 'gmres').
            Options: 'gmres', 'cg', 'bicg', 'bcgs', 'preonly', 'gamg', 'mg'
            When 'gamg' is used, an algebraic multigrid approach is employed.
            When 'mg' is used, a geometric multigrid approach is employed with a GMRES Krylov solver.
        discretization_scheme : str, optional
            The discretization scheme to use (default: 'power_law').
            Options: 'power_law', 'quick', 'upwind', 'second_order_upwind'
        tolerance : float, optional
            Convergence tolerance for the solver (default: 1e-8).
        max_iterations : int, optional
            Maximum number of iterations for the solver (default: 100).
        use_preconditioner : bool, optional
            Whether to use preconditioner for the solver (default: True).
        print_its : bool, optional
            Whether to print the number of iterations needed for convergence (default: False).
        restart : int, optional
            Restart parameter for GMRES (default: 30).
        petsc_pc_type : str, optional
            PETSc preconditioner type (default: 'ilu')
            Options: 'none', 'ilu', 'jacobi', 'sor', 'bjacobi', 'mg', 'gamg', 'hypre'
        """
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.use_preconditioner = use_preconditioner
        self.print_its = print_its
        self.restart = restart
        self.petsc_pc_type = petsc_pc_type
        
        if not HAS_PETSC:
            raise ImportError("PETSc is required for this solver. Please install petsc4py.")
        
        # Validate and set solver type
        self.solver_type = solver_type.lower()
        valid_solvers = ['gmres', 'cg', 'bicg', 'bcgs', 'preonly', 'gamg', 'mg']
        if self.solver_type not in valid_solvers:
            raise ValueError(f"Unsupported solver type: {solver_type}. "
                           f"Available options: {', '.join(valid_solvers)}")

        if discretization_scheme == 'power_law':
            self.discretization_scheme = power_law.PowerLawDiscretization()
        elif discretization_scheme == 'quick':
            self.discretization_scheme = quick.QUICKDiscretization()
        elif discretization_scheme == 'upwind':
            self.discretization_scheme = upwind.UpwindDiscretization()
        elif discretization_scheme == 'second_order_upwind':
            self.discretization_scheme = upwind.SecondOrderUpwindDiscretization()
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

    def _calculate_unrelaxed_residual(self, u_star, a_e, a_w, a_n, a_s, a_p_unrelaxed, source_unrelaxed, nx, ny, is_u):
        """Calculates the unrelaxed residual norm and field."""
        if is_u:
            imax, jmax = nx + 1, ny
            shape = (imax, jmax)
        else:
            imax, jmax = nx, ny + 1
            shape = (imax, jmax)

        # Build the unrelaxed system to calculate residual: r_unrelaxed = b_unrelaxed - A_unrelaxed * x
        matrix_unrelaxed, rhs_unrelaxed_flat, _ = self._build_sparse_matrix(
             a_e, a_w, a_n, a_s, a_p_unrelaxed, source_unrelaxed, nx, ny, is_u=is_u
        )
        Ax_unrelaxed = matrix_unrelaxed @ u_star.flatten()
        r_unrelaxed_flat = rhs_unrelaxed_flat - Ax_unrelaxed
        r_unrelaxed_field = r_unrelaxed_flat.reshape(shape)

        # Masking and normalization (using unrelaxed source field as the base)
        # Use slicing to extract interior points ONLY
        if is_u:
            # Extract interior points for u (from 1:nx, 1:ny-1)
            r_unrelaxed_interior = r_unrelaxed_field[1:nx, 1:ny-1]
            b_unrelaxed_interior = source_unrelaxed[1:nx, 1:ny-1]
        else:
            # Extract interior points for v (from 1:nx-1, 1:ny)
            r_unrelaxed_interior = r_unrelaxed_field[1:nx-1, 1:ny]
            b_unrelaxed_interior = source_unrelaxed[1:nx-1, 1:ny]

        # Calculate L2 norm of interior points only
        r_unrelaxed_norm_val = np.linalg.norm(r_unrelaxed_interior)
        b_unrelaxed_norm_val = np.linalg.norm(b_unrelaxed_interior)

        # Calculate normalized residual norm
        residual_norm_unrelaxed = r_unrelaxed_norm_val
        
        # Zero out boundaries in the returned field
        r_unrelaxed_field_final = r_unrelaxed_field.copy()
        if is_u:
            r_unrelaxed_field_final[0, :] = 0.0
            r_unrelaxed_field_final[1, :] = 0.0  # Adjacent
            if nx > 1:
                 r_unrelaxed_field_final[nx-1, :] = 0.0  # Adjacent
            r_unrelaxed_field_final[nx, :] = 0.0
            # Zero top/bottom boundaries as well in the returned field
            r_unrelaxed_field_final[:, 0] = 0.0
            r_unrelaxed_field_final[:, ny-1] = 0.0
        else:  # is_v
            r_unrelaxed_field_final[0, :] = 0.0
            r_unrelaxed_field_final[nx-1, :] = 0.0
            r_unrelaxed_field_final[:, 0] = 0.0
            r_unrelaxed_field_final[:, 1] = 0.0  # Adjacent
            if ny > 1:
                 r_unrelaxed_field_final[:, ny-1] = 0.0  # Adjacent
            r_unrelaxed_field_final[:, ny] = 0.0

        return residual_norm_unrelaxed, r_unrelaxed_field_final

    def _solve_matrix_system(self, matrix, rhs, initial_guess, component_name):
        """
        Solve a linear system using PETSc solvers.
        
        Parameters:
        -----------
        matrix : csr_matrix
            The sparse coefficient matrix
        rhs : ndarray
            The right-hand side vector
        initial_guess : ndarray
            Initial guess for the solution
        component_name : str
            Name of the component being solved ('u' or 'v') for logging
            
        Returns:
        --------
        solution : ndarray
            Solution vector
        iteration_count : int
            Number of iterations performed
        """
        # Flatten initial guess if needed
        if initial_guess.ndim > 1:
            initial_guess = initial_guess.flatten()
            
        # Solve using PETSc
        try:
            return self._solve_with_petsc(matrix, rhs, initial_guess, component_name)
        except Exception as e:
            # If PETSc solver fails, raise the exception
            error_msg = f"PETSc solver failed: {str(e)}"
            if self.print_its:
                print(error_msg)
            raise RuntimeError(error_msg)

    def _solve_with_petsc(self, matrix, rhs, initial_guess, component_name):
        """
        Solve a linear system using PETSc solvers.
        
        Parameters:
        -----------
        matrix : csr_matrix
            The sparse coefficient matrix
        rhs : ndarray
            The right-hand side vector
        initial_guess : ndarray
            Initial guess for the solution
        component_name : str
            Name of the component being solved ('u' or 'v') for logging
            
        Returns:
        --------
        solution : ndarray
            Solution vector
        iteration_count : int
            Number of iterations performed
        """
        # Convert to PETSc objects
        n = matrix.shape[0]
        
        # Create PETSc matrix
        A = PETSc.Mat().createAIJ(size=matrix.shape, 
                                 csr=(matrix.indptr, matrix.indices, matrix.data))
        A.assemblyBegin()
        A.assemblyEnd()
        
        # Create PETSc vectors
        b = PETSc.Vec().createWithArray(rhs)
        x = PETSc.Vec().createWithArray(initial_guess.copy())
        
        # Create and configure the KSP (linear solver)
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        
        # Set solver type based on user configuration
        if self.solver_type == 'gmres':
            ksp.setType('gmres')
            ksp.setGMRESRestart(self.restart)
            
            # Additional GMRES optimizations
            opts = PETSc.Options()
            # Use modified Gram-Schmidt (more stable than default)
            opts.setValue('ksp_gmres_modifiedgramschmidt', True)
            # Use the Krylov approximation only for residual calculations, improves performance
            opts.setValue('ksp_gmres_krylov_monitor', False)
            # Adaptive restart to improve convergence (when restart is large)
            if self.restart > 50:
                opts.setValue('ksp_gmres_classicalgramschmidt', 1)
                opts.setValue('ksp_gmres_restart_adaptive', True)
            ksp.setFromOptions()
        elif self.solver_type == 'cg':
            ksp.setType('cg')
        elif self.solver_type == 'bicg':
            ksp.setType('bicg')
        elif self.solver_type == 'bcgs':  # BiCGSTAB
            ksp.setType('bcgs')
            
            # BiCGSTAB optimizations
            opts = PETSc.Options()
            # Use the Chronopoulos/Gear variant of BiCGSTAB for improved stability
            opts.setValue('ksp_bcgs_variant', 'chronopoulos')
            # Enable flexible BiCGSTAB for better performance with inexact preconditioners
            opts.setValue('ksp_bcgs_flexible', True)
            ksp.setFromOptions()
        elif self.solver_type == 'preonly':  # Use preconditioner directly as solver
            ksp.setType('preonly')
        elif self.solver_type == 'gamg':  # Use GAMG directly as the solver
            ksp.setType('gmres')  # We still need a Krylov solver type
            ksp.setGMRESRestart(self.restart)
            pc = ksp.getPC()
            pc.setType('gamg')
            pc.setGAMGType('agg')
            pc.setGAMGLevels(4)
            # Set options to make GAMG more effective as a solver
            opts = PETSc.Options()
            opts.setValue('pc_gamg_threshold', 0.01)
            opts.setValue('pc_gamg_square_graph', 1)
            opts.setValue('pc_gamg_agg_nsmooths', 1)
            pc.setFromOptions()
        elif self.solver_type == 'mg':  # Use geometric multigrid (MG) directly as the solver
            ksp.setType('gmres')  # We still need a Krylov solver type
            ksp.setGMRESRestart(self.restart)
            pc = ksp.getPC()
            pc.setType('mg')
            
            # Configure MG options
            pc.setMGLevels(4)  # Number of levels in the multigrid hierarchy
            pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)  # Can also be ADDITIVE or FULL
            
            # Configure the smoother for each level
            opts = PETSc.Options()
            # Smoother options for MG
            opts.setValue('mg_levels_ksp_type', 'chebyshev')  # Use Chebyshev as smoother
            opts.setValue('mg_levels_pc_type', 'sor')  # SOR preconditioner for the smoother
            opts.setValue('mg_levels_ksp_max_it', 3)  # Number of smoother iterations
            
            # Coarse grid solver options
            opts.setValue('mg_coarse_ksp_type', 'preonly')  # Direct solve on the coarsest grid
            opts.setValue('mg_coarse_pc_type', 'lu')  # Use LU factorization for the coarse grid solver
            
            pc.setFromOptions()
        else:
            # Default to GMRES if somehow we got here with an invalid type
            ksp.setType('gmres')
            ksp.setGMRESRestart(self.restart)
        
        # Setup convergence criteria
        ksp.setInitialGuessNonzero(True)
        ksp.setTolerances(rtol=0.0, atol=self.tolerance, divtol=1e10, max_it=self.max_iterations)
        
        # Configure preconditioner
        pc = ksp.getPC()
        if self.use_preconditioner:
            # Set the requested preconditioner type
            pc_type = self.petsc_pc_type.lower()
            valid_pc_types = ['none', 'jacobi', 'sor', 'ilu', 'icc', 'bjacobi', 'mg', 'gamg', 'hypre', 'asm']
            
            if pc_type not in valid_pc_types:
                if self.print_its:
                    print(f"Warning: Unsupported preconditioner type '{pc_type}'. Falling back to 'ilu'.")
                pc_type = 'ilu'
                
            pc.setType(pc_type)
            
            # Additional configuration for specific preconditioners
            if pc_type == 'sor':
                pc.setSORType(PETSc.PC.SORType.SYMMETRIC)
                pc.setFactor(1.5)  # Set omega parameter
            elif pc_type == 'ilu':
                pc.setFactorLevels(1)  # ILU(1) has better convergence than ILU(0)
            elif pc_type == 'gamg':
                pc.setGAMGType('agg')
                pc.setGAMGLevels(4)
                
                # Optimize GAMG when used as a preconditioner
                opts = PETSc.Options()
                opts.setValue('pc_gamg_threshold', 0.02)       # Slightly higher for preconditioner
                opts.setValue('pc_gamg_square_graph', 1)
                opts.setValue('pc_gamg_agg_nsmooths', 1)
                opts.setValue('pc_gamg_reuse_interpolation', True)  # Reuse interpolation for efficiency
                opts.setValue('pc_gamg_sym_graph', True)       # Use symmetric graph if possible
                
                # Configure smoother for GAMG
                opts.setValue('mg_levels_ksp_type', 'chebyshev')
                opts.setValue('mg_levels_pc_type', 'sor')
                opts.setValue('mg_levels_ksp_max_it', 2)      # Fewer iterations when used as preconditioner
                
                # Coarse grid solver
                opts.setValue('mg_coarse_ksp_type', 'preonly')
                opts.setValue('mg_coarse_pc_type', 'lu')
                
                pc.setFromOptions()
            elif pc_type == 'mg':
                pc.setMGLevels(4)
                pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
                
                # Configure MG smoother and coarse grid solver
                opts = PETSc.Options()
                opts.setValue('mg_levels_ksp_type', 'chebyshev')
                opts.setValue('mg_levels_pc_type', 'sor')
                opts.setValue('mg_levels_ksp_max_it', 2)
                opts.setValue('mg_coarse_ksp_type', 'preonly')
                opts.setValue('mg_coarse_pc_type', 'lu')
                
                pc.setFromOptions()
            elif pc_type == 'asm':
                # Configure ASM (Additive Schwarz Method) for optimal performance
                
                # Set the number of sub-domains based on problem size
                # For this CFD problem, using small number of blocks works well
                pc.setASMType(PETSc.PC.ASMType.BASIC)  # BASIC is typically faster than RESTRICT
                pc.setASMOverlap(1)  # Minimal overlap for better performance
                
                # Configure the local solvers for each subdomain
                opts = PETSc.Options()
                # Use ILU as the sub-solver for each block - faster than default
                opts.setValue('sub_pc_type', 'ilu')
                opts.setValue('sub_pc_factor_levels', 1)  # ILU(1) for better convergence
                # Direct solve on each subdomain
                opts.setValue('sub_ksp_type', 'preonly')
                
                pc.setFromOptions()
        else:
            pc.setType('none')
        
        # Monitor convergence if requested
        if self.print_its:
            def monitor(ksp, it, rnorm):
                print(f"{component_name.upper()}-momentum PETSc {self.solver_type.upper()} iteration {it}, residual norm = {rnorm}")
                return 0
            
            ksp.setMonitor(monitor)
        
        # Solve the system
        ksp.solve(b, x)
        iteration_count = ksp.getIterationNumber()
        
        # Get the solution back to numpy array
        solution = x.getArray().copy()
        
        # Print convergence information if requested
        if self.print_its:
            reason = ksp.getConvergedReason()
            # Simple mapping of PETSc convergence reasons
            if reason > 0:  # Converged
                reason_string = f"Success ({reason})"
            else:  # Diverged or other issue
                reason_string = f"Failed ({reason})"
                
            if reason < 0:
                print(f"{component_name.upper()}-momentum PETSc {self.solver_type.upper()} failed to converge. Reason: {reason_string}")
            else:
                print(f"{component_name.upper()}-momentum PETSc {self.solver_type.upper()} converged in {iteration_count} iterations. Reason: {reason_string}")
        
        return solution, iteration_count

    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None, return_dict=True):
        """
        Solve the u-momentum equation using the selected iterative solver.
        
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
            Relaxation factor for under-relaxation
        boundary_conditions : dict or BoundaryConditionManager, optional
            Boundary conditions
        return_dict : bool, optional
            If True, returns residual information in a dictionary format (default).
            If False, returns separate residual values (deprecated).
            
        Returns:
        --------
        u_star : ndarray
            Intermediate u-velocity field
        d_u : ndarray
            Momentum equation coefficient
        residual_info : dict
            Dictionary with residual information: 
            - 'rel_norm': l2(r)/max(l2(r))
            - 'field': residual field
        """
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
        coeffs = self.discretization_scheme.calculate_u_coefficients(mesh, fluid, u_bc, v_bc, p, bc_manager)
        u_a_e = coeffs['a_e']
        u_a_w = coeffs['a_w']
        u_a_n = coeffs['a_n']
        u_a_s = coeffs['a_s']
        u_a_p_unrelaxed = coeffs['a_p']
        u_source_unrelaxed = coeffs['source']

        # Apply under-relaxation to coefficients
        safe_ap_unrelaxed = np.where(np.abs(u_a_p_unrelaxed) > 1e-12, u_a_p_unrelaxed, 1e-12)
        self.u_a_p = safe_ap_unrelaxed / alpha 
        u_source = u_source_unrelaxed + (1 - alpha) * self.u_a_p * u_bc

        # Store unrelaxed coefficients for residual calculation later
        self.u_a_e = u_a_e
        self.u_a_w = u_a_w
        self.u_a_n = u_a_n
        self.u_a_s = u_a_s
        self.u_a_p_unrelaxed = u_a_p_unrelaxed
        self.u_source_unrelaxed = u_source_unrelaxed

        # Build the sparse matrix system using RELAXED coefficients
        self.u_matrix, self.u_rhs, idx_map = self._build_sparse_matrix(
            u_a_e, u_a_w, u_a_n, u_a_s, self.u_a_p,
            u_source,
            nx, ny, is_u=True
        )

        # Solve the matrix system using the selected solver
        u_flat, iteration_count = self._solve_matrix_system(
            self.u_matrix, self.u_rhs, u_initial_guess, 'u'
        )

        # Reshape result back to 2D
        u_star = u_flat.reshape((imax+1, jmax))

        # Apply boundary conditions explicitly AFTER the solve
        u_star, _ = bc_manager.apply_velocity_boundary_conditions(u_star, v.copy(), imax, jmax)
        
        # Calculate d_u (using the RELAXED a_p)
        d_u.fill(np.nan)
        valid_ap_mask = np.abs(self.u_a_p) > 1e-12
        dy = mesh.get_cell_sizes()[1]
        d_u[valid_ap_mask] = dy / self.u_a_p[valid_ap_mask]

        # Calculate residual norm and field using unrelaxed system
        _, u_residual_field = self._calculate_unrelaxed_residual(
            u_star, self.u_a_e, self.u_a_w, self.u_a_n, self.u_a_s,
            self.u_a_p_unrelaxed, self.u_source_unrelaxed, nx, ny, is_u=True
        )

        # Calculate L2 norm of the interior residual field
        u_interior_residual = u_residual_field[1:nx, 1:ny-1]
        u_current_l2 = np.linalg.norm(u_interior_residual)
        
        # Keep track of the maximum L2 norm for relative scaling
        if not hasattr(self, 'u_max_l2'):
            self.u_max_l2 = u_current_l2
        else:
            self.u_max_l2 = max(self.u_max_l2, u_current_l2)
        
        # Calculate relative norm
        u_rel_norm = u_current_l2 / np.linalg.norm(u_source_unrelaxed)
        
        # Create the minimal residual information dictionary
        residual_info = {
            'rel_norm': u_rel_norm,  # l2(r)/max(l2(r))
            'field': u_residual_field,  # Absolute residual field
            'iterations': iteration_count  # Add iteration count to the info dictionary
        }

        return u_star, d_u, residual_info

    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None, return_dict=True):
        """
        Solve the v-momentum equation using the selected iterative solver.
        
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
            Relaxation factor for under-relaxation
        boundary_conditions : dict or BoundaryConditionManager, optional
            Boundary conditions
        return_dict : bool, optional
            If True, returns residual information in a dictionary format (default).
            If False, returns separate residual values (deprecated).
            
        Returns:
        --------
        v_star : ndarray
            Intermediate v-velocity field
        d_v : ndarray
            Momentum equation coefficient
        residual_info : dict
            Dictionary with residual information: 
            - 'rel_norm': l2(r)/max(l2(r))
            - 'field': residual field
        """
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
        coeffs = self.discretization_scheme.calculate_v_coefficients(mesh, fluid, u_bc, v_bc, p, bc_manager)
        v_a_e = coeffs['a_e']
        v_a_w = coeffs['a_w']
        v_a_n = coeffs['a_n']
        v_a_s = coeffs['a_s']
        v_a_p_unrelaxed = coeffs['a_p']
        v_source_unrelaxed = coeffs['source']

        # Apply under-relaxation to coefficients
        safe_ap_unrelaxed_v = np.where(np.abs(v_a_p_unrelaxed) > 1e-12, v_a_p_unrelaxed, 1e-12)
        self.v_a_p = safe_ap_unrelaxed_v / alpha
        if v_bc.shape != self.v_a_p.shape:
             v_source = v_source_unrelaxed + (1 - alpha) * self.v_a_p * v_bc
        else:
             v_source = v_source_unrelaxed + (1 - alpha) * self.v_a_p * v_bc

        # Store unrelaxed coefficients for residual calculation later
        self.v_a_e = v_a_e
        self.v_a_w = v_a_w
        self.v_a_n = v_a_n
        self.v_a_s = v_a_s
        self.v_a_p_unrelaxed = v_a_p_unrelaxed
        self.v_source_unrelaxed = v_source_unrelaxed

        # Build the sparse matrix system using RELAXED coefficients
        self.v_matrix, self.v_rhs, idx_map = self._build_sparse_matrix(
            v_a_e, v_a_w, v_a_n, v_a_s, self.v_a_p,
            v_source,
            nx, ny, is_u=False
        )
        
        # Solve the matrix system using the selected solver
        v_flat, iteration_count = self._solve_matrix_system(
            self.v_matrix, self.v_rhs, v_initial_guess, 'v'
        )

        # Reshape result back to 2D
        v_star = v_flat.reshape((imax, jmax+1))

        # Apply boundary conditions explicitly AFTER the solve
        _, v_star = bc_manager.apply_velocity_boundary_conditions(u.copy(), v_star, imax, jmax)

        # Calculate d_v (using the RELAXED a_p)
        d_v.fill(np.nan)
        valid_ap_mask = np.abs(self.v_a_p) > 1e-12
        dx = mesh.get_cell_sizes()[0]
        d_v[valid_ap_mask] = dx / self.v_a_p[valid_ap_mask]

        # Calculate residual norm and field using unrelaxed system
        _, v_residual_field = self._calculate_unrelaxed_residual(
            v_star, self.v_a_e, self.v_a_w, self.v_a_n, self.v_a_s,
            self.v_a_p_unrelaxed, self.v_source_unrelaxed, nx, ny, is_u=False
        )

        # Calculate L2 norm of the interior residual field
        v_interior_residual = v_residual_field[1:nx-1, 1:ny]
        v_current_l2 = np.linalg.norm(v_interior_residual)
        
        # Keep track of the maximum L2 norm for relative scaling
        if not hasattr(self, 'v_max_l2'):
            self.v_max_l2 = v_current_l2
        else:
            self.v_max_l2 = max(self.v_max_l2, v_current_l2)
        
        # Calculate relative norm
        v_rel_norm = v_current_l2 / np.linalg.norm(v_source_unrelaxed)
        
        # Create the minimal residual information dictionary
        residual_info = {
            'rel_norm': v_rel_norm,  # l2(r)/max(l2(r))
            'field': v_residual_field,  # Absolute residual field
            'iterations': iteration_count  # Add iteration count to the info dictionary
        }

        return v_star, d_v, residual_info