"""
Preconditioned conjugate gradient solver with geometric multigrid preconditioner.

This solver uses the conjugate gradient method with a geometric multigrid
method as a preconditioner, which can significantly improve convergence rates.
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.coeff_matrix import get_coeff_mat
from .multigrid import MultiGridSolver

class GeoMultigridPrecondCGSolver(PressureSolver):
    """
    Preconditioned conjugate gradient solver for pressure correction equation.
    
    This solver uses the conjugate gradient method with geometric multigrid as a
    preconditioner. This combines the robustness of CG with the efficiency of 
    multigrid methods, especially for problems with complex boundary conditions
    or highly anisotropic grids.
    """
    
    def __init__(self, tolerance=1e-5, max_iterations=500, mg_pre_smoothing=3, 
                 mg_post_smoothing=2, mg_cycles=1, mg_cycle_type='f', mg_cycle_type_buildup='w',
                 mg_max_cycles_buildup=1, mg_coarsest_grid_size=7, mg_restriction_method='restrict_inject',
                 mg_interpolation_method='interpolate_cubic', smoother=None):
        """
        Initialize the multigrid-preconditioned conjugate gradient solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the conjugate gradient method
        max_iterations : int, optional
            Maximum number of iterations for the conjugate gradient method
        mg_pre_smoothing : int, optional
            Number of pre-smoothing steps in multigrid preconditioner
        mg_post_smoothing : int, optional
            Number of post-smoothing steps in multigrid preconditioner
        mg_cycles : int, optional
            Number of multigrid cycles to apply for each preconditioning step
        mg_cycle_type : str, optional
            Type of multigrid cycle to use: 'v', 'w', or 'f'
        smoother : PressureSolver, optional
            Smoother to use in the multigrid preconditioner
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.residual_history = []
        self.inner_iterations = []
        self.mg_cycles = mg_cycles
        
        # Initialize the multigrid preconditioner
        self.mg_precond = MultiGridSolver(
            tolerance=tolerance*0.1,
            max_iterations=1,  # Only need 1 iteration as we'll call the cycle methods directly
            pre_smoothing=mg_pre_smoothing,
            post_smoothing=mg_post_smoothing,
            smoother=smoother,
            cycle_type=mg_cycle_type,
            cycle_type_buildup=mg_cycle_type_buildup,
            max_cycles_buildup=mg_max_cycles_buildup,
            coarsest_grid_size=mg_coarsest_grid_size,
            restriction_method=mg_restriction_method,
            interpolation_method=mg_interpolation_method
        )
        
        # Get omega from smoother if it exists
        self.omega = getattr(smoother, 'omega', 1.0) if smoother else 1.0
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using the preconditioned conjugate gradient method.
        
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
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho = 1.0  # This should come from fluid properties
        
        # Reset residual history
        self.residual_history = []
        
        # Get right-hand side of pressure correction equation
        b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Initial guess
        x0 = np.zeros_like(b, dtype=np.float64)
        
        # Fix reference pressure at (0,0)
        b[0] = 0.0
        
        # Construct the coefficient matrix explicitly
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Fix reference pressure at (0,0) - use lil_matrix for efficient modification
        A_lil = A.tolil()
        A_lil[0, :] = 0
        A_lil[0, 0] = 1
        A = A_lil.tocsr()
        
        # Create a preconditioner function using multigrid
        def M_x(x):
            # Ensure x is float64 to avoid type issues
            x = x.astype(np.float64)
            
            # Apply multigrid V-cycle directly as a preconditioner
            # This approximates A⁻¹x which is what we need for effective preconditioning
            
            # Initialize a zero solution
            u = np.zeros_like(x, dtype=np.float64)
            
            # Apply multiple cycles if requested
            for _ in range(self.mg_cycles):
                if self.mg_precond.cycle_type == 'v':
                    u = self.mg_precond._v_cycle(
                        p=u, 
                        rhs=x,  # x is the right-hand side vector b in A⁻¹b
                        mesh=mesh, 
                        rho=rho, 
                        d_u=d_u, 
                        d_v=d_v, 
                        omega=self.omega,
                        pre_smoothing=self.mg_precond.pre_smoothing, 
                        post_smoothing=self.mg_precond.post_smoothing,
                        level=0
                    )
                elif self.mg_precond.cycle_type == 'w':
                    u = self.mg_precond._w_cycle(
                        u=u, 
                        f=x,
                        mesh=mesh, 
                        rho=rho, 
                        d_u=d_u, 
                        d_v=d_v, 
                        omega=self.omega,
                        pre_smoothing=self.mg_precond.pre_smoothing, 
                        post_smoothing=self.mg_precond.post_smoothing,
                        level=0
                    )
                elif self.mg_precond.cycle_type == 'fmg':
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
        
        # Create a LinearOperator to represent our preconditioner
        M = LinearOperator((len(b), len(b)), matvec=M_x, dtype=np.float64)
        
        # Create a callback function to track convergence
        def callback(xk):
            # Compute residual: ||Ax - b||
            r = A.dot(xk) - b
            res_norm = np.linalg.norm(r)
            self.residual_history.append(res_norm)
            return False  # Continue iteration
        
        # Use conjugate gradient with preconditioner
        p_prime_flat, info = cg(
            A,
            b,
            x0=x0,
            M=M, 
            atol=self.tolerance,
            maxiter=self.max_iterations,
            callback=callback
        )
        
        self.inner_iterations.append(len(self.residual_history))
        
        if info != 0:
            print(f"Warning: Geo-Multigrid Preconditioned CG did not converge, info={info}")
        else:
            print(f"Geo-Multigrid Preconditioned CG converged in {len(self.residual_history)} iterations")
        
        # Reshape to 2D
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        return p_prime
    
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics
        """
        info = {
            'name': 'GeoMultigridPrecondCGSolver',
            'inner_iterations_history': self.inner_iterations,
            'total_inner_iterations': sum(self.inner_iterations),
            'convergence_rate': self._calculate_convergence_rate() if self.residual_history else None
        }
        
        # Add solver-specific information
        info['solver_specific'] = {
            'method': 'conjugate_gradient',
            'preconditioner': 'geometric_multigrid',
            'pre_smoothing': self.mg_precond.pre_smoothing,
            'post_smoothing': self.mg_precond.post_smoothing,
            'cycle_type': self.mg_precond.cycle_type
        }
        
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