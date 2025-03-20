"""
Matrix-free geometric multigrid solver for pressure correction equation.
"""

import numpy as np
import math
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product
from .helpers.multigrid_helpers import restrict, restrict_coefficient, interpolate, solve_directly
from .jacobi import JacobiSolver


class MultiGridSolver(PressureSolver):
    """
    Matrix-free geometric multigrid solver for pressure correction equation.
    
    This solver uses a geometric multigrid approach with V-cycles or F-cycles to solve
    the pressure correction equation. Grid sizes should be 2^k-1 (e.g., 3, 7, 15, ...)
    to ensure a clean coarsening down to 1x1, but the code can handle slight variations.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=100, 
                 pre_smoothing=2, post_smoothing=2,
                 smoother_iterations=2, smoother_omega=0.8,
                 smoother=None, cycle_type='v'):
        """
        Initialize the multigrid solver.
        
        Parameters:
        -----------
        tolerance : float
            Convergence tolerance for the overall solver
        max_iterations : int
            Maximum number of iterations for the overall solver
        pre_smoothing : int
            Number of pre-smoothing steps (each step applies `smoother_iterations` inside)
        post_smoothing : int
            Number of post-smoothing steps
        smoother_iterations : int
            Number of iterations for the smoother
        smoother_omega : float
            Relaxation factor for the smoother (e.g., 0.8 for Jacobi)
        smoother : PressureSolver or None
            External smoother to use (if None, will use the built-in JacobiSolver)
        cycle_type : str
            Type of multigrid cycle to use ('v' or 'f')
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.pre_smoothing = pre_smoothing
        self.post_smoothing = post_smoothing
        self.smoother_iterations = smoother_iterations
        self.smoother_omega = smoother_omega
        self.residual_history = []
        self.cycle_type = cycle_type.lower()
        
        # Initialize smoother
        self.smoother = smoother if smoother is not None else JacobiSolver(
            tolerance=1e-4,   # not used for fixed iteration mode
            max_iterations=1000, 
            omega=smoother_omega
        )
        
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using the matrix-free multigrid method.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            Must provide get_dimensions() -> (nx, ny) and get_cell_sizes() -> (dx, dy)
        u_star, v_star : ndarray
            Intermediate velocity fields used to construct the RHS
        d_u, d_v : ndarray
            Momentum equation coefficients
        p_star : ndarray
            Current pressure field (not used as an initial guess here, we start from zero)
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field shaped (nx, ny).
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho = 1.0  # Usually from fluid properties or mesh/flow state
        
        # Build RHS for Poisson
        b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Initialize guess for pressure correction
        x = np.zeros_like(b)
        
        # Track residuals
        self.residual_history = []
        
        
        # V-cycle iterations
        for k in range(self.max_iterations):
            x = self._v_cycle(x, b, nx, ny, dx, dy, rho, d_u, d_v)
        
            # Check convergence
            Ax = compute_Ap_product(x, nx, ny, dx, dy, rho, d_u, d_v)
            r = b - Ax
            res_norm = np.linalg.norm(r, 2) / (np.linalg.norm(b, 2))
            self.residual_history.append(res_norm)
            print(f"Residual: {res_norm}")
            if res_norm < self.tolerance:
                print(f"[MultiGridSolver] Converged in {k+1} iterations (V-cycles).")
                break
        
        p_prime = x.reshape((nx, ny), order='F')
        return p_prime

    # --------------------------------------------------------------------
    # V-cycle
    # --------------------------------------------------------------------
    def _v_cycle(self, u, f, nx, ny, dx, dy, rho, d_u, d_v):
        """Perform one V-cycle."""
        # Base case: coarsest grid
        if nx <= 3 and ny <= 3:
            u = self._smooth(u, f, nx, ny, dx, dy, rho, d_u, d_v)
            return u.flatten(order='F')
        
        # 1) Pre-smoothing
        u = self._smooth(u, f, nx, ny, dx, dy, rho, d_u, d_v)
        
        # 2) Compute residual & restrict
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        r_coarse, nx_c, ny_c, dx_c, dy_c = restrict(r, nx, ny, dx, dy)
        
        # 3) Restrict coefficients d_u and d_v
        d_u_coarse = restrict_coefficient(d_u, nx+1, ny, nx_c+1, ny_c)
        d_v_coarse = restrict_coefficient(d_v, nx, ny+1, nx_c, ny_c+1)
        
        # 4) Solve error on coarse grid (recursive V-cycle)
        e_coarse = np.zeros_like(r_coarse)
        e_coarse = self._v_cycle(e_coarse, r_coarse, nx_c, ny_c, dx_c, dy_c, rho, d_u_coarse, d_v_coarse)
        
        # 5) Prolong and update
        e_fine = interpolate(e_coarse, nx_c, ny_c, nx, ny)
       
        u = u + e_fine
        
        # 6) Post-smoothing
        u = self._smooth(u, f, nx, ny, dx, dy, rho, d_u, d_v)
       
        return u

    # --------------------------------------------------------------------
    # Smoothing: wraps around a user-supplied or built-in Jacobi/Gauss-Seidel
    # --------------------------------------------------------------------
    def _smooth(self, u, f, nx, ny, dx, dy, rho, d_u, d_v):
        """Apply the chosen smoother for a set number of iterations."""
        print(f"Smoothing: {nx}x{ny} with {self.smoother_iterations} iterations")
        # pin the pressure at the reference cell
        # Match the interface expected by the smoother
        return self.smoother.solve(
            nx=nx, 
            ny=ny, 
            dx=dx, 
            dy=dy, 
            rho=rho, 
            d_u=d_u, 
            d_v=d_v, 
            p=u, 
            b=f, 
            num_iterations=self.smoother_iterations,
            track_residuals=True
        )

    def get_solver_info(self):
        """Return dictionary with solver performance info."""
        info = {
            'name': 'MultiGridSolver',
            'residual_history': self.residual_history,
            'final_residual': self.residual_history[-1] if self.residual_history else None,
            'iterations': len(self.residual_history),
            'cycle_type': self.cycle_type,
            'pre_smoothing': self.pre_smoothing,
            'post_smoothing': self.post_smoothing,
            'smoother_iterations': self.smoother_iterations,
            'smoother_omega': self.smoother_omega,
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations
        }
        return info