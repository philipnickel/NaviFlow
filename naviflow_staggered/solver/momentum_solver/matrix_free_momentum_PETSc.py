import numpy as np
from .base_momentum_solver import MomentumSolver
from .discretization import power_law, quick, second_order_upwind
from ...constructor.boundary_conditions import BoundaryConditionManager

# PETSc import
try:
    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc
    HAS_PETSC = True
except ImportError:
    HAS_PETSC = False
    raise ImportError("PETSc is required for this solver. Please install petsc4py.")

__all__ = ["MatrixFreeMomentumSolver"]

class MatrixFreeMomentumSolverPETSc(MomentumSolver):
    """Matrix-free momentum solver using PETSc's matrix-free capabilities."""
    
    def __init__(
        self,
        discretization_scheme: str = "power_law",
        tolerance: float = 1e-8,
        max_iterations: int = 200,
        solver_type: str = "bcgs",
        use_preconditioner: bool = True,
        petsc_pc_type: str = "asm",
        print_its: bool = False,
        restart: int = 30,
    ) -> None:
        super().__init__()
        if not HAS_PETSC:
            raise ImportError("PETSc is required for this solver. Please install petsc4py.")
            
        self.tol = float(tolerance)
        self.maxiter = int(max_iterations)
        self.restart = int(restart)
        self.use_preconditioner = use_preconditioner
        self.petsc_pc_type = petsc_pc_type
        self.print_its = print_its
        
        # Validate solver type
        solver_type = solver_type.lower()
        valid_solvers = ['gmres', 'bcgs', 'cg', 'bicg', 'lsqr', 'tfqmr', 'cr']
        if solver_type not in valid_solvers:
            raise ValueError(f"Unsupported solver type: {solver_type}. "
                           f"Available options: {', '.join(valid_solvers)}")
        self.solver_type = solver_type

        # Set up discretization scheme
        schemes = {
            "power_law": power_law.PowerLawDiscretization,
            "quick": quick.QUICKDiscretization,
            "second_order_upwind": second_order_upwind.SecondOrderUpwindDiscretization,
        }
        try:
            self.discretization_scheme = schemes[discretization_scheme]()
        except KeyError as exc:
            raise ValueError(
                f"Unsupported discretization scheme: {discretization_scheme}"
            ) from exc

    # ───────────────────── matrix-vector kernels ─────────────────────
    @staticmethod
    def _matvec_u(x, a_e, a_w, a_n, a_s, a_p, nx, ny):
        """Matrix-vector product for u-momentum equation."""
        # Create a new vector for the result
        y = x.duplicate()
        y.set(0.0)
        
        # Get arrays for computation
        x_array = x.getArray().copy()  # Create a copy of the array to avoid lock issues
        x_array = x_array.reshape((nx + 1, ny))
        y_array = np.zeros_like(x_array)
        
        # Apply the matrix operation
        y_array[1:-1, 1:-1] = (
            a_p[1:-1, 1:-1] * x_array[1:-1, 1:-1]
            - a_e[1:-1, 1:-1] * x_array[2:, 1:-1]
            - a_w[1:-1, 1:-1] * x_array[:-2, 1:-1]
            - a_n[1:-1, 1:-1] * x_array[1:-1, 2:]
            - a_s[1:-1, 1:-1] * x_array[1:-1, :-2]
        )
        # Apply boundary conditions
        y_array[[0, -1], :] = x_array[[0, -1], :]
        y_array[:, [0, -1]] = x_array[:, [0, -1]]
        
        # Set the resulting vector
        y.setArray(y_array.ravel())
        return y

    @staticmethod
    def _matvec_v(x, a_e, a_w, a_n, a_s, a_p, nx, ny):
        """Matrix-vector product for v-momentum equation."""
        # Create a new vector for the result
        y = x.duplicate()
        y.set(0.0)
        
        # Get arrays for computation
        x_array = x.getArray().copy()  # Create a copy of the array to avoid lock issues
        x_array = x_array.reshape((nx, ny + 1))
        y_array = np.zeros_like(x_array)
        
        # Apply the matrix operation
        y_array[1:-1, 1:-1] = (
            a_p[1:-1, 1:-1] * x_array[1:-1, 1:-1]
            - a_e[1:-1, 1:-1] * x_array[2:, 1:-1]
            - a_w[1:-1, 1:-1] * x_array[:-2, 1:-1]
            - a_n[1:-1, 1:-1] * x_array[1:-1, 2:]
            - a_s[1:-1, 1:-1] * x_array[1:-1, :-2]
        )
        # Apply boundary conditions
        y_array[[0, -1], :] = x_array[[0, -1], :]
        y_array[:, [0, -1]] = x_array[:, [0, -1]]
        
        # Set the resulting vector
        y.setArray(y_array.ravel())
        return y

    # ──────────────────── PETSc Sparse Approximation ────────────────────
    def _create_sparse_matrix(self, a_e, a_w, a_n, a_s, a_p, nx, ny, is_u):
        """Create a PETSc sparse matrix for the stencil."""
        if is_u:
            nrows = (nx + 1) * ny
            ncols = nrows
            
            # Pre-allocate with estimated non-zeros per row (5-point stencil)
            mat = PETSc.Mat().create()
            mat.setSizes((nrows, ncols))
            mat.setType('aij')  # Sparse AIJ format
            mat.setPreallocationNNZ(5)  # 5-point stencil
            
            # Reshape coefficients
            a_e_flat = a_e.flatten() 
            a_w_flat = a_w.flatten()
            a_n_flat = a_n.flatten()
            a_s_flat = a_s.flatten()
            a_p_flat = a_p.flatten()
            
            # Set values in the matrix
            for j in range(ny):
                for i in range(nx + 1):
                    row = i * ny + j
                    
                    # Skip boundary points - they'll be handled separately
                    if i == 0 or i == nx or j == 0 or j == ny - 1:
                        mat.setValues(row, row, 1.0)
                        continue
                    
                    # Set diagonal
                    mat.setValues(row, row, a_p_flat[row])
                    
                    # East connection
                    if i < nx:
                        east = (i + 1) * ny + j
                        mat.setValues(row, east, -a_e_flat[row])
                    
                    # West connection
                    if i > 0:
                        west = (i - 1) * ny + j
                        mat.setValues(row, west, -a_w_flat[row])
                    
                    # North connection
                    if j < ny - 1:
                        north = i * ny + (j + 1)
                        mat.setValues(row, north, -a_n_flat[row])
                    
                    # South connection
                    if j > 0:
                        south = i * ny + (j - 1)
                        mat.setValues(row, south, -a_s_flat[row])
        else:
            nrows = nx * (ny + 1)
            ncols = nrows
            
            # Pre-allocate with estimated non-zeros per row (5-point stencil)
            mat = PETSc.Mat().create()
            mat.setSizes((nrows, ncols))
            mat.setType('aij')  # Sparse AIJ format
            mat.setPreallocationNNZ(5)  # 5-point stencil
            
            # Reshape coefficients
            a_e_flat = a_e.flatten() 
            a_w_flat = a_w.flatten()
            a_n_flat = a_n.flatten()
            a_s_flat = a_s.flatten()
            a_p_flat = a_p.flatten()
            
            # Set values in the matrix
            for i in range(nx):
                for j in range(ny + 1):
                    row = i * (ny + 1) + j
                    
                    # Skip boundary points
                    if i == 0 or i == nx - 1 or j == 0 or j == ny:
                        mat.setValues(row, row, 1.0)
                        continue
                    
                    # Set diagonal
                    mat.setValues(row, row, a_p_flat[row])
                    
                    # East connection
                    if i < nx - 1:
                        east = (i + 1) * (ny + 1) + j
                        mat.setValues(row, east, -a_e_flat[row])
                    
                    # West connection
                    if i > 0:
                        west = (i - 1) * (ny + 1) + j
                        mat.setValues(row, west, -a_w_flat[row])
                    
                    # North connection
                    if j < ny:
                        north = i * (ny + 1) + (j + 1)
                        mat.setValues(row, north, -a_n_flat[row])
                    
                    # South connection
                    if j > 0:
                        south = i * (ny + 1) + (j - 1)
                        mat.setValues(row, south, -a_s_flat[row])
        
        # Assemble the matrix
        mat.assemblyBegin()
        mat.assemblyEnd()
        
        return mat

    # ──────────────────── PETSc Preconditioner Construction ────────────────────
    def _configure_preconditioner(self, ksp, is_u):
        """Configure the preconditioner for the KSP solver."""
        if not self.use_preconditioner:
            pc = ksp.getPC()
            pc.setType('none')
            return
            
        pc = ksp.getPC()
        pc_type = self.petsc_pc_type.lower()
        
        # With sparse matrices, we can use any preconditioner
        valid_pc_types = ['none', 'jacobi', 'sor', 'ilu', 'icc', 'bjacobi', 'asm', 'gamg', 'hypre']
        
        if pc_type not in valid_pc_types:
            if self.print_its:
                print(f"Warning: Unsupported preconditioner type '{pc_type}'. Falling back to 'ilu'.")
            pc_type = 'ilu'
            
        pc.setType(pc_type)
        
        # Configure specific preconditioners
        if pc_type == 'sor':
            pc.setSORType(PETSc.PC.SORType.SYMMETRIC)
            pc.setFactor(1.5)  # Set omega parameter
        elif pc_type == 'ilu':
            pc.setFactorLevels(1)  # ILU(1) has better convergence than ILU(0)
        elif pc_type == 'asm':
            # Additive Schwarz Method
            pc.setASMType(PETSc.PC.ASMType.BASIC)
            pc.setASMOverlap(1)
            
            # Configure sub-solvers
            opts = PETSc.Options()
            opts.setValue('sub_pc_type', 'ilu')
            opts.setValue('sub_pc_factor_levels', 1)
            opts.setValue('sub_ksp_type', 'preonly')
        elif pc_type == 'gamg':
            # Algebraic Multigrid
            pc.setGAMGType('agg')
            pc.setGAMGLevels(4)
            
            # Configure GAMG options
            opts = PETSc.Options()
            opts.setValue('pc_gamg_threshold', 0.02)
            opts.setValue('pc_gamg_square_graph', 1)
            opts.setValue('pc_gamg_agg_nsmooths', 1)
            
        # Apply options
        pc.setFromOptions()

    # ───────────────────────── PETSc KSP Solver ────────────────────────
    def _solve_with_petsc(self, mat, rhs, initial_guess, nx, ny, is_u, component_name):
        """Solve a linear system using PETSc KSP solvers."""
        # Create PETSc vectors for rhs and solution
        size = mat.getSize()[0]
        b = PETSc.Vec().createWithArray(rhs)
        x = PETSc.Vec().createWithArray(initial_guess.copy())
        
        # Create KSP solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(mat)
        
        # Set solver type
        if self.solver_type == 'gmres':
            ksp.setType('gmres')
            ksp.setGMRESRestart(self.restart)
        elif self.solver_type == 'bcgs':
            ksp.setType('bcgs')
        elif self.solver_type == 'cg':
            ksp.setType('cg')
        elif self.solver_type == 'bicg':
            ksp.setType('bicg')
        elif self.solver_type == 'lsqr':
            ksp.setType('lsqr')
        elif self.solver_type == 'tfqmr':
            ksp.setType('tfqmr')
        elif self.solver_type == 'cr':
            ksp.setType('cr')
        else:
            # Default to BCGS if unsupported
            ksp.setType('bcgs')
            
        # Configure solver
        ksp.setInitialGuessNonzero(True)
        ksp.setTolerances(rtol=0.0, atol=self.tol, divtol=1e10, max_it=self.maxiter)
        
        # Configure preconditioner
        self._configure_preconditioner(ksp, is_u)
        
        # Set up convergence monitoring
        if self.print_its:
            def monitor(ksp, it, rnorm):
                print(f"{component_name.upper()}-momentum PETSc {self.solver_type.upper()} iteration {it}, residual norm = {rnorm}")
                return 0
            ksp.setMonitor(monitor)
            
        # Solve the system
        ksp.solve(b, x)
        iteration_count = ksp.getIterationNumber()
        
        # Get solution array
        solution = x.getArray().copy()
        
        # Print convergence info
        if self.print_its:
            reason = ksp.getConvergedReason()
            if reason > 0:
                reason_string = f"Success ({reason})"
            else:
                reason_string = f"Failed ({reason})"
                
            if reason < 0:
                print(f"{component_name.upper()}-momentum PETSc {self.solver_type.upper()} failed to converge. Reason: {reason_string}")
            else:
                print(f"{component_name.upper()}-momentum PETSc {self.solver_type.upper()} converged in {iteration_count} iterations. Reason: {reason_string}")
                
        return solution, iteration_count

    # ─────────────────── unrelaxed residual helper ──────────────────
    def _calculate_unrelaxed_residual(self, star_field, a_e, a_w, a_n, a_s, 
                                    a_p_un, source_un, nx, ny, is_u):
        """Calculate residual using unrelaxed system."""
        # Create PETSc Vec for the solution
        size = (nx + 1) * ny if is_u else nx * (ny + 1)
        sol_vec = PETSc.Vec().createWithArray(star_field.flatten())
        
        # Create the matrix-free operator
        A = self._create_sparse_matrix(a_e, a_w, a_n, a_s, a_p_un, nx, ny, is_u)
        
        # Apply the matrix operation: r = b - Ax
        Ax = sol_vec.duplicate()
        A.mult(sol_vec, Ax)
        
        # Convert to numpy arrays for residual calculation
        Ax_np = Ax.getArray().copy()
        b_np = source_un.flatten()
        
        # Compute the residual
        residual = b_np - Ax_np
        r_field = residual.reshape(source_un.shape)
        
        # Zero out boundaries
        if is_u:
            r_field[0, :] = 0.0; r_field[1, :] = 0.0
            if nx > 1: r_field[-2, :] = 0.0
            r_field[-1, :] = 0.0; r_field[:, 0] = 0.0; r_field[:, -1] = 0.0
            interior = r_field[1:nx, 1:ny-1]
        else:
            r_field[0, :] = 0.0; r_field[-1, :] = 0.0
            r_field[:, 0] = 0.0; r_field[:, 1] = 0.0
            if ny > 1: r_field[:, -2] = 0.0
            r_field[:, -1] = 0.0
            interior = r_field[1:nx-1, 1:ny]
        
        # Calculate norm and return
        norm = np.linalg.norm(interior)
        return r_field, norm

    # ─────────────────────────── u-momentum ─────────────────────────
    def solve_u_momentum(
        self, mesh, fluid, u, v, p,
        relaxation_factor=0.7, boundary_conditions=None,
        return_dict=True,
    ):
        nx, ny = mesh.get_dimensions()
        α = relaxation_factor

        # BC manager
        bc = (boundary_conditions if isinstance(boundary_conditions, BoundaryConditionManager) 
              else BoundaryConditionManager())
        if boundary_conditions and not isinstance(boundary_conditions, BoundaryConditionManager):
            for b, c in boundary_conditions.items():
                for fld, val in c.items():
                    bc.set_condition(b, fld, val)

        u_bc, v_bc = bc.apply_velocity_boundary_conditions(u.copy(), v.copy(), nx + 1, ny)

        # coefficients
        coeffs = self.discretization_scheme.calculate_u_coefficients(
            mesh, fluid, u_bc, v_bc, p, bc
        )
        a_e, a_w, a_n, a_s = coeffs["a_e"], coeffs["a_w"], coeffs["a_n"], coeffs["a_s"]
        a_p_un, src_un = coeffs["a_p"], coeffs["source"]

        # Apply under-relaxation
        a_p = np.where(np.abs(a_p_un) > 1e-12, a_p_un, 1e-12) / α
        src = src_un + (1 - α) * a_p * u_bc

        # Create matrix-free operator
        mat = self._create_sparse_matrix(a_e, a_w, a_n, a_s, a_p, nx, ny, is_u=True)
        
        # Solve with PETSc
        u_star, iterations = self._solve_with_petsc(
            mat, src.ravel(), u.ravel(), nx, ny, True, 'u'
        )
        
        # Reshape and apply boundary conditions
        u_star = u_star.reshape((nx + 1, ny))
        u_star, _ = bc.apply_velocity_boundary_conditions(u_star, v_bc, nx + 1, ny)

        # Calculate d_u coefficients
        dy = mesh.get_cell_sizes()[1]
        d_u = np.where(np.abs(a_p) > 1e-12, dy / a_p, 0.0)

        # Residual calculation
        r_un, norm_un = self._calculate_unrelaxed_residual(
            u_star, a_e, a_w, a_n, a_s, a_p_un, src_un, nx, ny, True
        )
        rel_norm = norm_un / np.linalg.norm(src_un)

        if not hasattr(self, 'u_max_l2'):
            self.u_max_l2 = norm_un
        else:
            self.u_max_l2 = max(self.u_max_l2, norm_un)

        res_info = {
            "rel_norm": rel_norm,
            "field": r_un,
            "iterations": iterations,
            "solver_type": self.solver_type
        }
            
        return (u_star, d_u, res_info) if return_dict else (u_star, d_u, rel_norm)

    # ─────────────────────────── v-momentum ─────────────────────────
    def solve_v_momentum(
        self, mesh, fluid, u, v, p,
        relaxation_factor=0.7, boundary_conditions=None,
        return_dict=True,
    ):
        nx, ny = mesh.get_dimensions()
        α = relaxation_factor

        # BC manager
        bc = (boundary_conditions if isinstance(boundary_conditions, BoundaryConditionManager) 
              else BoundaryConditionManager())
        if boundary_conditions and not isinstance(boundary_conditions, BoundaryConditionManager):
            for b, c in boundary_conditions.items():
                for fld, val in c.items():
                    bc.set_condition(b, fld, val)

        u_bc, v_bc = bc.apply_velocity_boundary_conditions(u.copy(), v.copy(), nx + 1, ny)

        # coefficients
        coeffs = self.discretization_scheme.calculate_v_coefficients(
            mesh, fluid, u_bc, v_bc, p, bc
        )
        a_e, a_w, a_n, a_s = coeffs["a_e"], coeffs["a_w"], coeffs["a_n"], coeffs["a_s"]
        a_p_un, src_un = coeffs["a_p"], coeffs["source"]

        # Apply under-relaxation
        a_p = np.where(np.abs(a_p_un) > 1e-12, a_p_un, 1e-12) / α
        src = src_un + (1 - α) * a_p * v_bc

        # Create matrix-free operator
        mat = self._create_sparse_matrix(a_e, a_w, a_n, a_s, a_p, nx, ny, is_u=False)
        
        # Solve with PETSc
        v_star, iterations = self._solve_with_petsc(
            mat, src.ravel(), v.ravel(), nx, ny, False, 'v'
        )
        
        # Reshape and apply boundary conditions
        v_star = v_star.reshape((nx, ny + 1))
        _, v_star = bc.apply_velocity_boundary_conditions(u_bc, v_star, nx + 1, ny)

        # Calculate d_v coefficients
        dx = mesh.get_cell_sizes()[0]
        d_v = np.where(np.abs(a_p) > 1e-12, dx / a_p, 0.0)

        # Residual calculation
        r_un, norm_un = self._calculate_unrelaxed_residual(
            v_star, a_e, a_w, a_n, a_s, a_p_un, src_un, nx, ny, False
        )
        rel_norm = norm_un / np.linalg.norm(src_un)

        if not hasattr(self, 'v_max_l2'):
            self.v_max_l2 = norm_un
        else:
            self.v_max_l2 = max(self.v_max_l2, norm_un)

        res_info = {
            "rel_norm": rel_norm,
            "field": r_un,
            "iterations": iterations,
            "solver_type": self.solver_type
        }
            
        return (v_star, d_v, res_info) if return_dict else (v_star, d_v, rel_norm)