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
        ilu_fill_factor: int = 15,
        idrs_s: int = 4
    ) -> None:
        super().__init__()
        self.tol = float(tolerance)
        self.maxiter = int(max_iterations)
        self.ilu_drop_tol = float(ilu_drop_tol)
        self.ilu_fill_factor = int(ilu_fill_factor)
        self.idrs_s = int(idrs_s)
        
        solver_type = solver_type.lower()
        if solver_type not in {"gmres", "bicgstab", "idrs"}:
            raise ValueError("solver_type must be 'gmres', 'bicgstab', or 'idrs'")
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

    # ────────────────── IDR(s) solver implementation ──────────────────
    def _idrs(self, A, b, x0=None, tol=1e-5, s=4, maxiter=None, M=None):
        """
        Implementation of the IDR(s) method for solving linear systems.
        Based on the algorithm by Sonneveld and van Gijzen.
        
        Parameters:
        -----------
        A : LinearOperator
            Matrix of the linear system
        b : ndarray
            Right-hand side vector
        x0 : ndarray, optional
            Initial guess
        tol : float, optional
            Convergence tolerance
        s : int, optional
            Dimension of the shadow space
        maxiter : int, optional
            Maximum number of iterations
        M : LinearOperator, optional
            Preconditioner
            
        Returns:
        --------
        x : ndarray
            Solution vector
        info : int
            Convergence info: 0 for success, >0 for non-convergence (iterations)
        residual_history : list
            History of residual norms
        """
        # Initialize parameters
        n = len(b)
        if maxiter is None:
            maxiter = n * 10
            
        # Apply preconditioner if provided
        if M is not None:
            psolve = M.matvec
        else:
            psolve = lambda x: x
            
        matvec = A.matvec
        residual_history = []
        
        # Create random shadow space
        np.random.seed(0)  # For reproducibility
        P = np.random.randn(s, n)
        
        # Check for zero rhs
        bnrm = np.linalg.norm(b)
        if bnrm == 0.0:
            return np.zeros(n, dtype=b.dtype), 0, residual_history
            
        # Initial solution
        if x0 is None:
            x = np.zeros(n, dtype=b.dtype)
            r = b.copy()
        else:
            x = x0.copy()
            r = b - matvec(x)
            
        # Check initial residual
        rnrm = np.linalg.norm(r)
        residual_history.append(rnrm)
            
        # Relative tolerance
        tolb = tol * bnrm
        if rnrm < tolb:
            return x, 0, residual_history
            
        # Initialization for IDR(s)
        angle = 0.7  # Angle for the plane rotation
        G = np.zeros((n, s), dtype=b.dtype)  # G-space vectors
        U = np.zeros((n, s), dtype=b.dtype)  # Preconditioned G-space vectors
        Ms = np.eye(s, dtype=b.dtype)  # Coefficients for orthogonalization
        om = 1.0  # Initial omega
        iter_ = 0
        
        # Main iteration loop, build G-spaces
        while rnrm >= tolb and iter_ < maxiter:
            # New right-hand side for small system
            f = P.dot(r)
            
            # Process each dimension of the shadow space
            for k in range(s):
                # Solve small system and make v orthogonal to P
                c = np.linalg.solve(Ms[k:s, k:s], f[k:s])
                v = r - G[:, k:s].dot(c)
                
                # Preconditioning
                v = psolve(v)
                
                # Compute new basis vector
                U[:, k] = v
                if k > 0:
                    U[:, k] += U[:, k:s].dot(c) * om
                
                # Matrix-vector product
                G[:, k] = matvec(U[:, k])
                
                # Bi-orthogonalize the new basis vectors
                for i in range(k):
                    alpha = P[i, :].dot(G[:, k]) / Ms[i, i]
                    G[:, k] = G[:, k] - alpha * G[:, i]
                    U[:, k] = U[:, k] - alpha * U[:, i]
                    
                # New column of M = P'*G (first k-1 entries are zero)
                for i in range(k, s):
                    Ms[i, k] = P[i, :].dot(G[:, k])
                    
                # Check for breakdown
                if Ms[k, k] == 0.0:
                    return x, -1, residual_history
                    
                # Make r orthogonal to g_i, i = 1..k
                beta = f[k] / Ms[k, k]
                x = x + beta * U[:, k]
                r = r - beta * G[:, k]
                rnrm = np.linalg.norm(r)
                residual_history.append(rnrm)
                
                iter_ += 1
                if rnrm < tolb or iter_ >= maxiter:
                    break
                    
                # New f = P'*r (first k components are zero)
                if k < s - 1:
                    f[k+1:s] = f[k+1:s] - beta * Ms[k+1:s, k]
                    
            # Now we have sufficient vectors in G_j to compute residual in G_j+1
            if rnrm < tolb or iter_ >= maxiter:
                break
                
            # Preconditioning
            v = psolve(r)
            
            # Matrix-vector product
            t = matvec(v)
            
            # Computation of a new omega
            nr = np.linalg.norm(r)
            nt = np.linalg.norm(t)
            ts = t.dot(r)
            rho = abs(ts / (nt * nr))
            om = ts / (nt * nt)
            
            # Adjust omega if the cosine of the angle is too small
            if rho < angle:
                om = om * angle / rho
                
            # New vector in G_j+1
            x = x + om * v
            r = r - om * t
            rnrm = np.linalg.norm(r)
            residual_history.append(rnrm)
            
            iter_ += 1
            
        # Set return info based on convergence
        if rnrm >= tolb:
            info = iter_
        else:
            info = 0
            
        return x, info, residual_history

    # ───────────────────────── Krylov driver ────────────────────────
    def _solve_krylov(self, A: LinearOperator, rhs: np.ndarray, x0: np.ndarray, M: LinearOperator = None):
        """Driver for GMRES/BiCGSTAB/IDR(s) solvers with preconditioning."""
        residual_history = []
        
        def callback(x):
            # Calculate residual
            r = np.linalg.norm(A.matvec(x) - rhs)
            residual_history.append(r)
            
        if self.solver_type == "gmres":
            sol, info = gmres(
                A, rhs, M=M, x0=x0, atol=self.tol, restart=60, maxiter=self.maxiter,
                callback=callback
            )
            return sol, residual_history
        elif self.solver_type == "bicgstab":
            sol, info = bicgstab(
                A, rhs, M=M, x0=x0, atol=self.tol, maxiter=self.maxiter,
                callback=callback
            )
            return sol, residual_history
        elif self.solver_type == "idrs":
            # Use the IDR(s) implementation
            sol, info, residual_history = self._idrs(
                A, rhs, x0=x0, tol=self.tol, s=self.idrs_s, maxiter=self.maxiter, M=M
            )
            
            if info > 0:
                print(f"Warning: IDR(s) did not converge after {info} iterations")
            elif info < 0:
                raise RuntimeError(f"IDR(s) failed (info={info}).")
                
            return sol, residual_history
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

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
        M = self._create_ilu_preconditioner(a_e, a_w, a_n, a_s, a_p, nx, ny, is_u=True)
        
        # Linear operator for relaxed system
        A = LinearOperator(
            ((nx + 1) * ny, (nx + 1) * ny),
            matvec=lambda x: self._matvec_u(x, a_e, a_w, a_n, a_s, a_p, nx, ny),
            dtype=np.float64,
        )

        # Solve with preconditioning
        u_star, residual_history = self._solve_krylov(A, src.ravel(), x0=u.ravel(), M=M)
        u_star = u_star.reshape((nx + 1, ny))
        u_star, _ = bc.apply_velocity_boundary_conditions(u_star, v_bc, nx + 1, ny)

        # Calculate d_u coefficients
        dy = mesh.get_cell_sizes()[1]
        d_u = np.where(np.abs(a_p) > 1e-12, dy / a_p, 0.0)

        # Residual calculation
        r_un, norm_un = self._calculate_unrelaxed_residual(
            u_star, a_e, a_w, a_n, a_s, a_p_un, src_un, nx, ny, True
        )
        rel_norm = norm_un

        if not hasattr(self, 'u_max_l2'):
            self.u_max_l2 = norm_un
        else:
            self.u_max_l2 = max(self.u_max_l2, norm_un)

        res_info = {
            "rel_norm": rel_norm,
            "field": r_un,
            "iterations": len(residual_history),
            "solver_type": self.solver_type
        }
        
        if self.solver_type == "idrs":
            res_info["idrs_s"] = self.idrs_s
            
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
        M = self._create_ilu_preconditioner(a_e, a_w, a_n, a_s, a_p, nx, ny, is_u=False)
        
        # Linear operator for relaxed system
        A = LinearOperator(
            (nx * (ny + 1), nx * (ny + 1)),
            matvec=lambda x: self._matvec_v(x, a_e, a_w, a_n, a_s, a_p, nx, ny),
            dtype=np.float64,
        )

        # Solve with preconditioning
        v_star, residual_history = self._solve_krylov(A, src.ravel(), x0=v.ravel(), M=M)
        v_star = v_star.reshape((nx, ny + 1))
        _, v_star = bc.apply_velocity_boundary_conditions(u_bc, v_star, nx + 1, ny)

        # Calculate d_v coefficients
        dx = mesh.get_cell_sizes()[0]
        d_v = np.where(np.abs(a_p) > 1e-12, dx / a_p, 0.0)

        # Residual calculation
        r_un, norm_un = self._calculate_unrelaxed_residual(
            v_star, a_e, a_w, a_n, a_s, a_p_un, src_un, nx, ny, False
        )
        rel_norm = norm_un

        if not hasattr(self, 'v_max_l2'):
            self.v_max_l2 = norm_un
        else:
            self.v_max_l2 = max(self.v_max_l2, norm_un)

        res_info = {
            "rel_norm": rel_norm,
            "field": r_un,
            "iterations": len(residual_history),
            "solver_type": self.solver_type
        }
        
        if self.solver_type == "idrs":
            res_info["idrs_s"] = self.idrs_s
            
        return (v_star, d_v, res_info) if return_dict else (v_star, d_v, rel_norm)