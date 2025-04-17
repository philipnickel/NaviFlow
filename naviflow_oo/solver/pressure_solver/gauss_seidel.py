"""
Matrix-free Gauss-Seidel solver for pressure correction equation.
"""

import numpy as np
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product

class GaussSeidelSolver(PressureSolver):
    """
    Matrix-free Red-Black Gauss-Seidel solver for pressure correction equation.
    
    This solver uses the Red-Black Gauss-Seidel iterative method without explicitly forming
    the coefficient matrix, which is memory-efficient for large problems.
    Red-Black Gauss-Seidel (RBGS) allows for better vectorization while preserving the
    Gauss-Seidel property that updates depend on the latest values of neighbors.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=1000, omega=1.0):
        """
        Initialize the Red-Black Gauss-Seidel solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the method
        max_iterations : int, optional
            Maximum number of iterations
        omega : float, optional
            Relaxation factor for SOR (Successive Over-Relaxation)
            omega=1.0 for standard Gauss-Seidel, 1.0 < omega < 2.0 for SOR
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        self.omega = omega
        self.residual_history = []
        self.inner_iterations_history = []
        self.total_inner_iterations = 0
        self.convergence_rates = []
        
        # Initialize coefficient matrices for pressure correction equation
        self.p_a_e = None
        self.p_a_w = None
        self.p_a_n = None
        self.p_a_s = None
        self.p_a_p = None
        self.p_source = None
    
    def solve(self, mesh=None, u_star=None, v_star=None, d_u=None, d_v=None, p_star=None, 
              p=None, b=None, nx=None, ny=None, dx=None, dy=None, rho=1.0, num_iterations=None, 
              track_residuals=True):
        """
        Solve the pressure correction equation using the Red-Black Gauss-Seidel method.
        
        Parameters:
        -----------
        mesh : StructuredMesh, optional
            The computational mesh
        u_star, v_star : ndarray, optional
            Intermediate velocity fields
        d_u, d_v : ndarray
            Momentum equation coefficients
        p_star : ndarray, optional
            Current pressure field (not used)
        p : ndarray, optional
            Initial pressure field
        b : ndarray, optional
            Right-hand side
        nx, ny : int, optional
            Grid dimensions
        dx, dy : float, optional
            Grid spacing
        rho : float, optional
            Fluid density (default: 1.0)
        num_iterations : int, optional
            Number of iterations to perform
        track_residuals : bool, optional
            Whether to track residuals (default: True)
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        """
        # Get grid dimensions and spacing if mesh is provided
        if mesh is not None:
            nx, ny = mesh.get_dimensions()
            dx, dy = mesh.get_cell_sizes()
            
        # Set number of iterations
        if num_iterations is None:
            num_iterations = self.max_iterations
            
        # Get right-hand side
        if b is None:
            b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
            
        # Initial guess
        if p is None:
            p = np.zeros_like(b)
            
        # Track inner iterations and reset residuals if needed
        inner_iterations = 0
        if track_residuals:
            self.residual_history = []
        
        # Convert to 2D arrays for computation
        if p.ndim == 1:
            p_2d = p.reshape((nx, ny), order='F')
        else:
            p_2d = p.copy()
            
        if b.ndim == 1:
            b_2d = b.reshape((nx, ny), order='F')
        else:
            b_2d = b.copy()
        
        # Pre-compute coefficient arrays
        aE, aW, aN, aS, aP = self._precompute_coefficients(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Store coefficient arrays for residual calculation
        self.p_a_e = aE.copy()
        self.p_a_w = aW.copy()
        self.p_a_n = aN.copy()
        self.p_a_s = aS.copy()
        self.p_a_p = aP.copy()
        
        # Store RHS (mass imbalance) for residual calculation
        self.p_source = b_2d.copy()
        
        # Fix reference pressure at (0,0)
        p_2d[0, 0] = 0.0
        
        # Create red and black masks for vectorized computation
        i_indices, j_indices = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        red_mask = ((i_indices + j_indices) % 2 == 0)
        red_mask[0, 0] = False  # Exclude reference point from updates
        black_mask = ~red_mask
        
        # Main iteration loop
        old_res_norm = float('inf')
        for k in range(num_iterations):
            # Perform one Red-Black Gauss-Seidel iteration
            self._rb_gauss_seidel_step(p_2d, b_2d, aE, aW, aN, aS, aP, red_mask, black_mask)
            
            # Track convergence if needed
            if track_residuals:
                # Calculate residual using Ax - b
                p_flat = p_2d.flatten('F')
                r = b_2d.flatten('F') - compute_Ap_product(p_flat, nx, ny, dx, dy, rho, d_u, d_v)
                res_norm = np.linalg.norm(r)
                self.residual_history.append(res_norm)
                
                # Calculate convergence rate
                if k > 0 and old_res_norm > 0:
                    conv_rate = res_norm / old_res_norm
                    self.convergence_rates.append(conv_rate)
                old_res_norm = res_norm
                
                # Check residual-based convergence
                if res_norm < self.tolerance:
                    print(f"Gauss-Seidel converged in {k+1} iterations, residual: {res_norm:.6e}")
                    break
                #print(f"Gauss-Seidel iteration {k+1}, residual: {res_norm:.6e}")
        
        return p_2d
    
    def _precompute_coefficients(self, nx, ny, dx, dy, rho, d_u, d_v):
        """
        Precompute coefficient arrays for the pressure equation.
        This exactly matches the coefficient calculation in compute_Ap_product.
        
        Returns:
        --------
        tuple
            (aE, aW, aN, aS, inv_aP) coefficient arrays
        """
        # Create coefficient arrays
        aE = np.zeros((nx, ny))
        aW = np.zeros((nx, ny))
        aN = np.zeros((nx, ny))
        aS = np.zeros((nx, ny))
        aP = np.zeros((nx, ny))
        
        # East coefficients (aE) - For interior cells: i < nx-1
        # Using explicit loops to match compute_Ap_product exactly
        # East coefficients (aE) - For interior cells: i < nx-1
        aE[:-1, :] = rho * d_u[1:nx, :] * dy
        
        # West coefficients (aW) - For interior cells: i > 0
        aW[1:, :] = rho * d_u[1:nx, :] * dy
        
        # North coefficients (aN) - For interior cells: j < ny-1
        aN[:, :-1] = rho * d_v[:, 1:ny] * dx
        
        # South coefficients (aS) - For interior cells: j > 0
        aS[:, 1:] = rho * d_v[:, 1:ny] * dx
        # Apply boundary conditions by modifying coefficients
        # West boundary (i=0)
        aP[0, :] += aE[0, :]
        aE[0, :] = 0
        
        # East boundary (i=nx-1)
        aP[nx-1, :] += aW[nx-1, :]
        aW[nx-1, :] = 0
        
        # South boundary (j=0)
        aP[:, 0] += aN[:, 0]
        aN[:, 0] = 0
        
        # North boundary (j=ny-1)
        aP[:, ny-1] += aS[:, ny-1]
        aS[:, ny-1] = 0
        
        # Diagonal term is sum of all coefficients
        aP += aE + aW + aN + aS
        
        # IMPORTANT: No special handling for reference pressure point here
        # Reference point is handled in the main solve loop with p[0, 0] = 0.0
        
        # Avoid division by zero
        aP[aP < 1e-15] = 1.0
        
        return aE, aW, aN, aS, aP
    
    def _rb_gauss_seidel_step(self, p, b, aE, aW, aN, aS, aP, red_mask, black_mask):
        """
        Perform one Red-Black Gauss-Seidel iteration with SOR.
        
        This optimized implementation updates all "red" points first, then all "black" points.
        Red points are those where i+j is even, black points where i+j is odd.
        """
        # Compute inverse of aP once for efficiency
        inv_aP = 1.0 / aP
        
        # Pre-allocate arrays for vectorized operations
        east = np.zeros_like(p)
        west = np.zeros_like(p)
        north = np.zeros_like(p)
        south = np.zeros_like(p)
        
        # Update RED points first
        east[:-1, :] = aE[:-1, :] * p[1:, :]
        west[1:, :] = aW[1:, :] * p[:-1, :]
        north[:, :-1] = aN[:, :-1] * p[:, 1:]
        south[:, 1:] = aS[:, 1:] * p[:, :-1]
        
        # Calculate new values for red points
        p_new = (b + east + west + north + south) * inv_aP
        p[red_mask] = p[red_mask] + self.omega * (p_new[red_mask] - p[red_mask])
        
        # Update BLACK points next
        east[:-1, :] = aE[:-1, :] * p[1:, :]
        west[1:, :] = aW[1:, :] * p[:-1, :]
        north[:, :-1] = aN[:, :-1] * p[:, 1:]
        south[:, 1:] = aS[:, 1:] * p[:, :-1]
        
        # Calculate new values for black points
        p_new = (b + east + west + north + south) * inv_aP
        p[black_mask] = p[black_mask] + self.omega * (p_new[black_mask] - p[black_mask])
        
        # Ensure reference point stays fixed
        p[0, 0] = 0.0
    
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics
        """
        # Calculate convergence rate if available
        if len(self.convergence_rates) > 0:
            # Use the average of the last few iterations for stability
            last_rates = self.convergence_rates[-min(10, len(self.convergence_rates)):]
            avg_rate = sum(last_rates) / len(last_rates)
        else:
            avg_rate = None
            
        return {
            'name': 'GaussSeidelSolver',
            'inner_iterations_history': self.inner_iterations_history,
            'total_inner_iterations': self.total_inner_iterations,
            'convergence_rate': avg_rate,
            'omega': self.omega,
            'method': 'Red-Black'
        } 