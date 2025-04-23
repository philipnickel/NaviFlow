"""
Matrix-free conjugate gradient solver for pressure correction equation.
"""

import numpy as np
from scipy.sparse.linalg import cg, bicgstab, LinearOperator
from .base_pressure_solver import PressureSolver
from .helpers.matrix_free import compute_Ap_product
from .helpers.rhs_construction import get_rhs
from .multigrid import MultiGridSolver
from .gauss_seidel import GaussSeidelSolver
class MatrixFreeBiCGSTABSolver(PressureSolver):
    """
    Matrix-free conjugate gradient solver for pressure correction equation.
    
    This solver uses the conjugate gradient method without explicitly forming
    the coefficient matrix, which can be more memory-efficient for large problems.
    """
    
    def __init__(self, tolerance=1e-7, max_iterations=1000, use_preconditioner=False, 
                 preconditioner='jacobi', mg_pre_smoothing=2, mg_post_smoothing=2, 
                 mg_cycles=1, mg_cycle_type='v', mg_cycle_type_buildup='v',
                 mg_max_cycles_buildup=1, mg_coarsest_grid_size=7, 
                 mg_restriction_method='restrict_full_weighting',
                 mg_interpolation_method='interpolate_linear', 
                 smoother_relaxation=0.8, smoother_method_type='red_black'):
        """$
        Initialize the matrix-free conjugate gradient solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the conjugate gradient method
        max_iterations : int, optional
            Maximum number of iterations for the conjugate gradient method
        use_preconditioner : bool, optional
            Whether to use a preconditioner
        preconditioner : str, optional
            Type of preconditioner to use: 'jacobi' or 'multigrid'
        mg_pre_smoothing : int, optional
            Number of pre-smoothing steps in multigrid preconditioner
        mg_post_smoothing : int, optional
            Number of post-smoothing steps in multigrid preconditioner
        mg_cycles : int, optional
            Number of multigrid cycles to apply for each preconditioning step
        mg_cycle_type : str, optional
            Type of multigrid cycle to use: 'v', 'w', or 'f'
        mg_cycle_type_buildup : str, optional
            Type of cycle to use during buildup phase if mg_cycle_type is 'f'
        mg_max_cycles_buildup : int, optional
            Number of cycles to perform at each level during buildup
        mg_coarsest_grid_size : int, optional
            Size of the coarsest grid (must be odd and >= 3)
        mg_restriction_method : str, optional
            Method for restriction: 'restrict_inject' or 'restrict_full_weighting'
        mg_interpolation_method : str, optional
            Method for interpolation: 'interpolate_linear' or 'interpolate_cubic'
        smoother_relaxation : float, optional
            Relaxation parameter for the smoother in multigrid (0 < omega < 2)
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.residual_history = []
        self.use_preconditioner = use_preconditioner
        self.preconditioner = preconditioner
        
        # Multigrid parameters
        self.mg_pre_smoothing = mg_pre_smoothing
        self.mg_post_smoothing = mg_post_smoothing
        self.mg_cycles = mg_cycles
        self.mg_cycle_type = mg_cycle_type
        self.mg_cycle_type_buildup = mg_cycle_type_buildup
        self.mg_max_cycles_buildup = mg_max_cycles_buildup
        self.mg_coarsest_grid_size = mg_coarsest_grid_size
        self.mg_restriction_method = mg_restriction_method
        self.mg_interpolation_method = mg_interpolation_method
        self.smoother_relaxation = smoother_relaxation
        
        # Initialize multigrid preconditioner if needed
        self.mg_precond = None
        if self.use_preconditioner and self.preconditioner == 'multigrid':
            # Create a Jacobi smoother for the multigrid preconditioner
            smoother = GaussSeidelSolver(omega=smoother_relaxation, method_type=smoother_method_type)
            # Initialize the multigrid preconditioner
            self.mg_precond = MultiGridSolver(
                smoother=smoother,
                tolerance=tolerance,
                max_iterations=1,  # Only need 1 iteration as we'll call the cycle methods directly
                pre_smoothing=mg_pre_smoothing,
                post_smoothing=mg_post_smoothing,
                cycle_type=mg_cycle_type,
                cycle_type_buildup=mg_cycle_type_buildup,
                max_cycles_buildup=mg_max_cycles_buildup,
                coarsest_grid_size=mg_coarsest_grid_size,
                restriction_method=mg_restriction_method,
                interpolation_method=mg_interpolation_method
            )
            
            # Store omega from smoother
            self.omega = smoother.omega
    
    def _create_jacobi_preconditioner(self, nx, ny, dx, dy, rho, d_u, d_v):
        """
        Create a Jacobi (diagonal) preconditioner for the pressure Poisson equation.
        
        Returns a LinearOperator that applies the preconditioner.
        """
        # Create diagonal approximation based on the standard 5-point stencil
        # For each interior point, the diagonal entry is approximately:
        # 2/dx^2 + 2/dy^2
        diag = np.ones(nx * ny, dtype=np.float64)  # Initialize with 1s for boundary points
        
        # Create index arrays for interior points
        i_interior = np.repeat(np.arange(1, nx-1), ny-2)
        j_interior = np.tile(np.arange(1, ny-1), nx-2)
        idx_interior = i_interior + j_interior * nx
        
        # Set interior point values vectorized
        diag[idx_interior] = 2.0/(dx*dx) + 2.0/(dy*dy)
        
        # Invert the diagonal for the preconditioner
        diag_inv = 1.0 / diag
        
        # Create preconditioner function
        def apply_preconditioner(vec):
            return diag_inv * vec.astype(np.float64)
        
        return LinearOperator((len(diag), len(diag)), matvec=apply_preconditioner, dtype=np.float64)

    def _create_multigrid_preconditioner(self, mesh, rho, d_u, d_v):
        """
        Create a geometric multigrid preconditioner for the pressure Poisson equation.
        
        Returns a LinearOperator that applies the preconditioner.
        """
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        
        # Create a preconditioner function using multigrid
        def M_x(x):
            # Ensure x is float64 to avoid type issues
            x = x.astype(np.float64)
            
            # Initialize a zero solution
            u = np.zeros_like(x, dtype=np.float64)
            
            # Apply multiple cycles if requested
            for _ in range(self.mg_cycles):
                if self.mg_cycle_type == 'v':
                    u = self.mg_precond._v_cycle(
                        p=u, 
                        rhs=x,  # x is the right-hand side vector b in A⁻¹b
                        mesh=mesh, 
                        rho=rho, 
                        d_u=d_u, 
                        d_v=d_v, 
                        omega=self.omega,
                        pre_smoothing=self.mg_pre_smoothing, 
                        post_smoothing=self.mg_post_smoothing,
                        level=0
                    )
                elif self.mg_cycle_type == 'w':
                    u = self.mg_precond._w_cycle(
                        u=u, 
                        f=x,
                        mesh=mesh, 
                        rho=rho, 
                        d_u=d_u, 
                        d_v=d_v, 
                        omega=self.omega,
                        pre_smoothing=self.mg_pre_smoothing, 
                        post_smoothing=self.mg_post_smoothing,
                        level=0
                    )
                elif self.mg_cycle_type == 'f' or self.mg_cycle_type == 'fmg':
                    u = self.mg_precond._fmg_cycle(
                        rhs_fine=x,
                        mesh_fine=mesh, 
                        d_u_fine=d_u, 
                        d_v_fine=d_v, 
                        nx_finest=mesh.get_dimensions()[0],
                        ny_finest=mesh.get_dimensions()[1],
                        dx_finest=mesh.get_cell_sizes()[0],
                        dy_finest=mesh.get_cell_sizes()[1]
                    )
            
            return u
        
        return LinearOperator((nx * ny, nx * ny), matvec=M_x, dtype=np.float64)
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star, return_dict=True):
        """
        Solve the pressure correction equation using the matrix-free BiCGSTAB method.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        u_star, v_star : ndarray
            Intermediate velocity fields
        d_u, d_v : ndarray
            Momentum equation coefficients
        p_star : ndarray
            Current pressure field
        return_dict : bool, optional
            If True, returns a dictionary with complete residual information (default)
            If False, returns separate residual values (deprecated)
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        residual_info : dict
            Dictionary with residual information: 
            - 'rel_norm': l2(r)/max(l2(r))
            - 'field': residual field
        """

        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho = 1.0  # This should come from fluid properties
        
        # Reset residual history
        self.residual_history = []
        
        # Ensure arrays have consistent float64 data type
        u_star = u_star.astype(np.float64)
        v_star = v_star.astype(np.float64)
        d_u = d_u.astype(np.float64)
        d_v = d_v.astype(np.float64)
        
        # Get right-hand side of pressure correction equation
        rhs = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Initial guess
        x0 = np.zeros_like(rhs, dtype=np.float64)
        
        # Create a callback function to track convergence
        def callback(xk):
            # Compute residual: ||Ax - b||
            r = compute_Ap_product(xk.astype(np.float64), nx, ny, dx, dy, rho, d_u, d_v) - rhs
            res_norm = np.linalg.norm(r)
            self.residual_history.append(res_norm)
            return False  # Continue iteration
        
        # Create a lambda function for the matrix-vector product
        def mv_product(v):
            return compute_Ap_product(v.astype(np.float64), nx, ny, dx, dy, rho, d_u, d_v)
        
        # Create a LinearOperator to represent our matrix operation
        A_op = LinearOperator((len(rhs), len(rhs)), matvec=mv_product, dtype=np.float64)
        
        # Create preconditioner if requested
        M_op = None
        if self.use_preconditioner:
            if self.preconditioner == 'jacobi':
                M_op = self._create_jacobi_preconditioner(nx, ny, dx, dy, rho, d_u, d_v)
            elif self.preconditioner == 'multigrid':
                M_op = self._create_multigrid_preconditioner(mesh, rho, d_u, d_v)
        
        # Use bicgstab to solve system
        p_prime_flat, info = bicgstab(
            A_op, 
            rhs, 
            x0=x0, 
            atol=self.tolerance, 
            maxiter=self.max_iterations,
            callback=callback,
            M=M_op
        )
        
        if info != 0:
            print(f"Warning: BiCGSTAB did not converge, info={info}")
        else:
            cycles_type = self.preconditioner if self.use_preconditioner else "none"
            print(f"BiCGSTAB converged in {len(self.residual_history)} iterations with preconditioner: {cycles_type}")
        
        # Reshape to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        # Calculate residual for tracking
        r = rhs - compute_Ap_product(p_prime_flat, nx, ny, dx, dy, rho, d_u, d_v)
        r_field_full = r.reshape((nx, ny), order='F')  # Reshape residual field
        
        # Calculate L2 norm on interior points only
        r_interior = r_field_full[1:nx-1, 1:ny-1]
        p_current_l2 = np.linalg.norm(r_interior)
        
        # Keep track of the maximum L2 norm for relative scaling
        if not hasattr(self, 'p_max_l2'):
            self.p_max_l2 = p_current_l2
        else:
            self.p_max_l2 = max(self.p_max_l2, p_current_l2)
        
        # Calculate relative norm as l2(r)/max(l2(r))
        p_rel_norm = p_current_l2 / self.p_max_l2 if self.p_max_l2 > 0 else 1.0
        
        # Create the residual information dictionary
        residual_info = {
            'rel_norm': p_rel_norm,  # l2(r)/max(l2(r))
            'field': r_field_full     # Absolute residual field
        }
        
        return p_prime, residual_info
        
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics
        """
        info = {
            'name': 'MatrixFreeCGSolver',
            'iterations': len(self.residual_history),
            'convergence_rate': self._calculate_convergence_rate() if self.residual_history else None
        }
        
        # Add solver-specific information
        info['solver_specific'] = {
            'method': 'bicgstab',
            'use_preconditioner': self.use_preconditioner,
            'preconditioner': self.preconditioner
        }
        
        # Add multigrid-specific info if using multigrid preconditioner
        if self.use_preconditioner and self.preconditioner == 'multigrid':
            info['solver_specific'].update({
                'mg_pre_smoothing': self.mg_pre_smoothing,
                'mg_post_smoothing': self.mg_post_smoothing,
                'mg_cycles': self.mg_cycles,
                'mg_cycle_type': self.mg_cycle_type,
                'mg_coarsest_grid_size': self.mg_coarsest_grid_size,
                'smoother_relaxation': self.smoother_relaxation
            })
        
        return info
    
    def _calculate_convergence_rate(self):
        """Calculate the average convergence rate from residual history."""
        if len(self.residual_history) < 3:
            return None
            
        # Calculate convergence rate as geometric mean of ratios
        ratios = [self.residual_history[i] / self.residual_history[i-1] 
                 for i in range(1, len(self.residual_history))
                 if self.residual_history[i-1] > 0]
        
        if not ratios:
            return None
            
        # Filter out extreme values that might skew the average
        valid_ratios = [r for r in ratios if 0 < r < 1.0]
        
        if not valid_ratios:
            return None
            
        # Calculate the geometric mean
        return np.exp(np.mean(np.log(valid_ratios))) 