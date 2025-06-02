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
    
    def __init__(self, tolerance=1e-6, max_iterations=1000, omega=1.0, method_type='red_black'):
        """
        Initialize the Gauss-Seidel solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for the method
        max_iterations : int, optional
            Maximum number of iterations
        omega : float, optional
            Relaxation factor for SOR (Successive Over-Relaxation)
            omega=1.0 for standard Gauss-Seidel, 1.0 < omega < 2.0 for SOR
        method_type : str, optional
            Type of Gauss-Seidel iteration: 'red_black', 'standard', or 'symmetric'
            (default: 'red_black')
        """
        super().__init__(tolerance=tolerance, max_iterations=max_iterations)
        if method_type not in ['red_black', 'standard', 'symmetric']:
            raise ValueError("method_type must be one of 'red_black', 'standard', or 'symmetric'")
        self.omega = omega
        self.method_type = method_type
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
              track_residuals=True, return_dict=True):
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
        return_dict : bool, optional
            If True, returns a dictionary with complete residual information (default)
            If False, returns only the pressure correction field (deprecated)
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        residual_info : dict
            Dictionary with residual information: 
            - 'rel_norm': l2(r)/max(l2(r))
            - 'field': residual field
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
        final_residual_field = None
        for k in range(num_iterations):
            # Perform one Gauss-Seidel iteration
            if self.method_type == 'red_black':
                self._rb_gauss_seidel_step(p_2d, b_2d, aE, aW, aN, aS, aP, red_mask, black_mask)
            elif self.method_type == 'standard':
                self._standard_gauss_seidel_step(p_2d, b_2d, aE, aW, aN, aS, aP)
            else:
                self._symmetric_gauss_seidel_step(p_2d, b_2d, aE, aW, aN, aS, aP)
            
            # Track convergence if needed
            if track_residuals:
                # Calculate residual using Ax - b
                p_flat = p_2d.flatten('F')
                r = b_2d.flatten('F') - compute_Ap_product(p_flat, nx, ny, dx, dy, rho, d_u, d_v)
                res_norm = np.linalg.norm(r)
                rel_norm = res_norm / np.linalg.norm(b_2d.flatten('F'))
                self.residual_history.append(res_norm)
                
                # Calculate convergence rate
                if k > 0 and old_res_norm > 0:
                    conv_rate = res_norm / old_res_norm
                    self.convergence_rates.append(conv_rate)
                old_res_norm = res_norm
                
                # Store residual field for return value
                final_residual_field = r.reshape((nx, ny), order='F')
                
                # Check residual-based convergence
                if rel_norm < self.tolerance:
                    print(f"Gauss-Seidel converged in {k+1} iterations, residual: {res_norm:.6e}")
                    break
                #print(f"Gauss-Seidel iteration {k+1}, rel residual: {rel_norm:.6e}, abs residual: {res_norm:.6e}")
        
        # Calculate L2 norm on interior points only
        r_interior = final_residual_field[1:nx-1, 1:ny-1] if final_residual_field is not None else np.zeros((nx-2, ny-2))
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
            'field': final_residual_field  # Absolute residual field
        }
        
        if return_dict:
            return p_2d, residual_info
        else:
            # For backward compatibility
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
    
    def _standard_gauss_seidel_step(self, p, b, aE, aW, aN, aS, aP):
        """
        Perform one standard Gauss-Seidel iteration with SOR.
        Updates grid points sequentially.
        """
        nx, ny = p.shape
        inv_aP = 1.0 / aP

        for j in range(ny):
            for i in range(nx):
                # Skip the reference pressure point
                if i == 0 and j == 0:
                    continue
                
                # Calculate neighbor contributions using the most recent values
                east_contrib = aE[i, j] * p[i + 1, j] if i < nx - 1 else 0
                west_contrib = aW[i, j] * p[i - 1, j] if i > 0 else 0
                north_contrib = aN[i, j] * p[i, j + 1] if j < ny - 1 else 0
                south_contrib = aS[i, j] * p[i, j - 1] if j > 0 else 0
                
                # Calculate the new value without relaxation
                p_new_ij = (b[i, j] + east_contrib + west_contrib + north_contrib + south_contrib) * inv_aP[i, j]
                
                # Apply SOR
                p[i, j] = p[i, j] + self.omega * (p_new_ij - p[i, j])
                
        # Ensure reference point stays fixed after the iteration
        p[0, 0] = 0.0

    def _symmetric_gauss_seidel_step(self, p, b, aE, aW, aN, aS, aP):
        """
        Perform one symmetric Gauss-Seidel iteration with SOR.
        Consists of a forward pass followed by a backward pass.
        """
        nx, ny = p.shape
        inv_aP = 1.0 / aP

        # Forward pass (standard GS)
        for j in range(ny):
            for i in range(nx):
                if i == 0 and j == 0: continue # Skip reference point
                east_contrib = aE[i, j] * p[i + 1, j] if i < nx - 1 else 0
                west_contrib = aW[i, j] * p[i - 1, j] if i > 0 else 0
                north_contrib = aN[i, j] * p[i, j + 1] if j < ny - 1 else 0
                south_contrib = aS[i, j] * p[i, j - 1] if j > 0 else 0
                p_new_ij = (b[i, j] + east_contrib + west_contrib + north_contrib + south_contrib) * inv_aP[i, j]
                p[i, j] = p[i, j] + self.omega * (p_new_ij - p[i, j])

        # Backward pass
        for j in range(ny - 1, -1, -1):
            for i in range(nx - 1, -1, -1):
                if i == 0 and j == 0: continue # Skip reference point
                east_contrib = aE[i, j] * p[i + 1, j] if i < nx - 1 else 0
                west_contrib = aW[i, j] * p[i - 1, j] if i > 0 else 0
                north_contrib = aN[i, j] * p[i, j + 1] if j < ny - 1 else 0
                south_contrib = aS[i, j] * p[i, j - 1] if j > 0 else 0
                p_new_ij = (b[i, j] + east_contrib + west_contrib + north_contrib + south_contrib) * inv_aP[i, j]
                p[i, j] = p[i, j] + self.omega * (p_new_ij - p[i, j])
                
        # Ensure reference point stays fixed after the full iteration
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
            
        method_type = self.method_type.capitalize()
        return {
            'name': 'GaussSeidelSolver',
            'inner_iterations_history': self.inner_iterations_history,
            'total_inner_iterations': self.total_inner_iterations,
            'convergence_rate': avg_rate,
            'omega': self.omega,
            'method': method_type
        } 