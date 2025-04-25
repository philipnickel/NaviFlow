import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, bicgstab, spilu
from .base_momentum_solver import MomentumSolver
from .discretization import power_law, quick, second_order_upwind
from ...constructor.boundary_conditions import BoundaryConditionManager

__all__ = ["MatrixFreeMomentumSolver"]


class MatrixFreeMomentumSolver(MomentumSolver):
    """Matrix-free momentum solver with ILU preconditioning based on sparse stencil approximation."""
    
    def __init__(
        self,
        discretization_scheme: str = "power_law",
        tolerance: float = 1e-8,
        max_iterations: int = 200,
        solver_type: str = "bicgstab",
        ilu_drop_tol: float = 1e-3,
        ilu_fill_factor: int = 15
    ) -> None:
        super().__init__()
        self.tol = float(tolerance)
        self.maxiter = int(max_iterations)
        self.ilu_drop_tol = float(ilu_drop_tol)
        self.ilu_fill_factor = int(ilu_fill_factor)
        
        solver_type = solver_type.lower()
        if solver_type not in {"gmres", "bicgstab"}:
            raise ValueError("solver_type must be 'gmres' or 'bicgstab'")
        self.solver_type = solver_type

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
    def _matvec_u(flat_x, a_e, a_w, a_n, a_s, a_p, nx, ny):
        """Matrix-vector product for u-momentum equation."""
        x = flat_x.reshape((nx + 1, ny))
        y = np.zeros_like(x)
        y[1:-1, 1:-1] = (
            a_p[1:-1, 1:-1] * x[1:-1, 1:-1]
            - a_e[1:-1, 1:-1] * x[2:, 1:-1]
            - a_w[1:-1, 1:-1] * x[:-2, 1:-1]
            - a_n[1:-1, 1:-1] * x[1:-1, 2:]
            - a_s[1:-1, 1:-1] * x[1:-1, :-2]
        )
        y[[0, -1], :] = x[[0, -1], :]
        y[:, [0, -1]] = x[:, [0, -1]]
        return y.ravel()

    @staticmethod
    def _matvec_v(flat_x, a_e, a_w, a_n, a_s, a_p, nx, ny):
        """Matrix-vector product for v-momentum equation."""
        x = flat_x.reshape((nx, ny + 1))
        y = np.zeros_like(x)
        y[1:-1, 1:-1] = (
            a_p[1:-1, 1:-1] * x[1:-1, 1:-1]
            - a_e[1:-1, 1:-1] * x[2:, 1:-1]
            - a_w[1:-1, 1:-1] * x[:-2, 1:-1]
            - a_n[1:-1, 1:-1] * x[1:-1, 2:]
            - a_s[1:-1, 1:-1] * x[1:-1, :-2]
        )
        y[[0, -1], :] = x[[0, -1], :]
        y[:, [0, -1]] = x[:, [0, -1]]
        return y.ravel()

    # ──────────────────── Sparse Matrix Approximation ────────────────────
    def _build_sparse_approx(self, a_e, a_w, a_n, a_s, a_p, nx, ny, is_u):
        """Construct sparse approximation from stencil coefficients with exact dimensions."""
        if is_u:
            size = (nx + 1) * ny
            # Main diagonal (always present)
            diags_data = [a_p.flatten()]
            offsets = [0]
            
            # East connections (-a_e)
            if nx >= 1:
                east_data = -a_e[:-1, :].flatten()
                if east_data.size == size - ny:  # Check correct length
                    diags_data.append(east_data)
                    offsets.append(ny)
            
            # West connections (-a_w)
            if nx >= 1:
                west_data = -a_w[1:, :].flatten()
                if west_data.size == size - ny:  # Check correct length
                    diags_data.append(west_data)
                    offsets.append(-ny)
            
            # North connections (-a_n)
            if ny > 1:
                north_data = -a_n[:, :-1].flatten()
                if north_data.size == size - 1:  # Check correct length
                    diags_data.append(north_data)
                    offsets.append(1)
            
            # South connections (-a_s)
            if ny > 1:
                south_data = -a_s[:, 1:].flatten()
                if south_data.size == size - 1:  # Check correct length
                    diags_data.append(south_data)
                    offsets.append(-1)
        else:
            size = nx * (ny + 1)
            # Main diagonal
            diags_data = [a_p.flatten()]
            offsets = [0]
            
            # East connections (-a_e)
            if nx > 1:
                east_data = -a_e[:-1, :].flatten()
                if east_data.size == size - (ny + 1):
                    diags_data.append(east_data)
                    offsets.append(ny + 1)
            
            # West connections (-a_w)
            if nx > 1:
                west_data = -a_w[1:, :].flatten()
                if west_data.size == size - (ny + 1):
                    diags_data.append(west_data)
                    offsets.append(-(ny + 1))
            
            # North connections (-a_n)
            if ny >= 1:
                north_data = -a_n[:, :-1].flatten()
                if north_data.size == size - 1:
                    diags_data.append(north_data)
                    offsets.append(1)
            
            # South connections (-a_s)
            if ny >= 1:
                south_data = -a_s[:, 1:].flatten()
                if south_data.size == size - 1:
                    diags_data.append(south_data)
                    offsets.append(-1)
        
        # Final validation
        valid_diags = []
        valid_offsets = []
        for i, (diag, offset) in enumerate(zip(diags_data, offsets)):
            expected_length = size - abs(offset)
            if diag.size == expected_length:
                valid_diags.append(diag)
                valid_offsets.append(offset)
            else:
                print(f"Warning: Dropping diagonal {i} with length {diag.size} (expected {expected_length})")
        
        if not valid_diags:
            raise ValueError("No valid diagonals found for sparse matrix construction")
        
        return diags(valid_diags, valid_offsets, shape=(size, size), format='csr')

    # ──────────────────── ILU Preconditioner Construction ────────────────────
    def _create_ilu_preconditioner(self, a_e, a_w, a_n, a_s, a_p, nx, ny, is_u):
        """Create ILU preconditioner from sparse matrix approximation."""
        A_sparse = self._build_sparse_approx(a_e, a_w, a_n, a_s, a_p, nx, ny, is_u)
        ilu = spilu(A_sparse, drop_tol=self.ilu_drop_tol, fill_factor=self.ilu_fill_factor)
        return LinearOperator(A_sparse.shape, ilu.solve)

    # ───────────────────────── Krylov driver ────────────────────────
    def _solve_krylov(self, A: LinearOperator, rhs: np.ndarray, x0: np.ndarray, M: LinearOperator = None):
        """Driver for GMRES/BiCGSTAB solvers with preconditioning."""
        if self.solver_type == "gmres":
            sol, info = gmres(
                A, rhs, M=M, x0=x0, atol=self.tol, restart=60, maxiter=self.maxiter
            )
        else:
            sol, info = bicgstab(
                A, rhs, M=M, x0=x0, atol=self.tol, maxiter=self.maxiter
            )
        if info != 0:
            raise RuntimeError(f"{self.solver_type.upper()} failed (info={info}).")
        return sol

    # ─────────────────── unrelaxed residual helper ──────────────────
    def _calculate_unrelaxed_residual(self, star_field, a_e, a_w, a_n, a_s, 
                                    a_p_un, source_un, nx, ny, is_u):
        """Calculate residual using unrelaxed system (matches AMG exactly)."""
        if is_u:
            Ax_un = self._matvec_u(star_field.ravel(), a_e, a_w, a_n, a_s, a_p_un, nx, ny)
            r = source_un - Ax_un.reshape(source_un.shape)
            r[0, :] = 0.0; r[1, :] = 0.0
            if nx > 1: r[-2, :] = 0.0
            r[-1, :] = 0.0; r[:, 0] = 0.0; r[:, -1] = 0.0
            interior = r[1:nx, 1:ny-1]
        else:
            Ax_un = self._matvec_v(star_field.ravel(), a_e, a_w, a_n, a_s, a_p_un, nx, ny)
            r = source_un - Ax_un.reshape(source_un.shape)
            r[0, :] = 0.0; r[-1, :] = 0.0
            r[:, 0] = 0.0; r[:, 1] = 0.0
            if ny > 1: r[:, -2] = 0.0
            r[:, -1] = 0.0
            interior = r[1:nx-1, 1:ny]
        
        norm = np.linalg.norm(interior)
        return r, norm

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

        # Create ILU preconditioner
        #M = self._create_ilu_preconditioner(a_e, a_w, a_n, a_s, a_p, nx, ny, is_u=True)
        M = None
        # Linear operator for relaxed system
        A = LinearOperator(
            ((nx + 1) * ny, (nx + 1) * ny),
            matvec=lambda x: self._matvec_u(x, a_e, a_w, a_n, a_s, a_p, nx, ny),
            dtype=np.float64,
        )

        # Solve with preconditioning
        u_star = self._solve_krylov(A, src.ravel(), x0=u.ravel(), M=M).reshape((nx + 1, ny))
        u_star, _ = bc.apply_velocity_boundary_conditions(u_star, v_bc, nx + 1, ny)

        # Calculate d_u coefficients
        dy = mesh.get_cell_sizes()[1]
        d_u = np.where(np.abs(a_p) > 1e-12, dy / a_p, 0.0)

        # Residual calculation
        r_un, norm_un = self._calculate_unrelaxed_residual(
            u_star, a_e, a_w, a_n, a_s, a_p_un, src_un, nx, ny, True
        )
        src_un_interior = src_un[1:nx, 1:ny-1]
        rel_norm = norm_un #/ (np.linalg.norm(src_un_interior) + 1e-16)

        if not hasattr(self, 'u_max_l2'):
            self.u_max_l2 = norm_un
        else:
            self.u_max_l2 = max(self.u_max_l2, norm_un)

        res_info = {
            "rel_norm": rel_norm,
            "field": r_un,
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

        # Create ILU preconditioner
        #M = self._create_ilu_preconditioner(a_e, a_w, a_n, a_s, a_p, nx, ny, is_u=False)
        M = None
        # Linear operator for relaxed system
        A = LinearOperator(
            (nx * (ny + 1), nx * (ny + 1)),
            matvec=lambda x: self._matvec_v(x, a_e, a_w, a_n, a_s, a_p, nx, ny),
            dtype=np.float64,
        )

        # Solve with preconditioning
        v_star = self._solve_krylov(A, src.ravel(), x0=v.ravel(), M=M).reshape((nx, ny + 1))
        _, v_star = bc.apply_velocity_boundary_conditions(u_bc, v_star, nx + 1, ny)

        # Calculate d_v coefficients
        dx = mesh.get_cell_sizes()[0]
        d_v = np.where(np.abs(a_p) > 1e-12, dx / a_p, 0.0)

        # Residual calculation
        r_un, norm_un = self._calculate_unrelaxed_residual(
            v_star, a_e, a_w, a_n, a_s, a_p_un, src_un, nx, ny, False
        )
        src_un_interior = src_un[1:nx-1, 1:ny]
        rel_norm = norm_un #/ (np.linalg.norm(src_un_interior) + 1e-16)

        if not hasattr(self, 'v_max_l2'):
            self.v_max_l2 = norm_un
        else:
            self.v_max_l2 = max(self.v_max_l2, norm_un)

        res_info = {
            "rel_norm": rel_norm,
            "field": r_un,
        }
        return (v_star, d_v, res_info) if return_dict else (v_star, d_v, rel_norm)