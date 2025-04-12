import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product
from .helpers.multigrid_helpers import  restrict_coefficients,restrict_hybrid,restrict_half_weighting,restrict_inject, restrict_full_weighting, interpolate_linear, interpolate_cubic
from .jacobi import JacobiSolver
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .helpers.coeff_matrix import get_coeff_mat

class MultiGridSolver(PressureSolver):
    """
    Matrix-free geometric multigrid solver for pressure correction equation.
    
    This solver uses a geometric multigrid approach with V-cycles to solve
    the pressure correction equation without explicitly forming matrices.
    It requires grid sizes to be 2^k-1 (e.g., 3, 7, 15, 31, 63, 127, etc.)
    to ensure proper coarsening down to a 1x1 grid.
    
    Added debug functionality: when debug is True, the solver stores intermediate 
    arrays (after pre-smoothing, residual computation, restriction, interpolation, 
    correction, and post-smoothing) and outputs a multi-page PDF that plots these arrays 
    in chronological order.
    """
    
    def __init__(self, smoother, max_iterations=100, tolerance=1e-8,
                 pre_smoothing=1, post_smoothing=1,
                 cycle_type='v', cycle_type_buildup='v', cycle_type_final='v',
                 num_cycles_buildup=1,
                 restriction_method='restrict_full_weighting',
                 interpolation_method='interpolate_linear',
                 coarsest_grid_size=7):
        """
        Initialize the multigrid solver.
        
        Parameters:
        -----------
        smoother : PressureSolver
            Smoother to use at each grid level
        max_iterations : int
            Maximum number of V-cycles
        tolerance : float
            Convergence tolerance
        pre_smoothing : int
            Number of pre-smoothing iterations
        post_smoothing : int
            Number of post-smoothing iterations
        cycle_type : str
            Type of cycle to use: 'v', 'w', or 'f'
        cycle_type_buildup : str
            Type of cycle to use during buildup phase of F-cycle: 'v' or 'w'
        cycle_type_final : str
            Type of cycle to use for final iterations: 'v' or 'w'
        num_cycles_buildup : int
            Number of cycles to perform at each level during buildup
        restriction_method : str
            Method for restriction: 'restrict_inject' or 'restrict_full_weighting'
        interpolation_method : str
            Method for interpolation: 'interpolate_linear' or 'interpolate_cubic'
        coarsest_grid_size : int
            Size of the coarsest grid (must be odd and >= 3)
        """
        super().__init__(max_iterations=max_iterations, tolerance=tolerance)
        self.smoother = smoother
        self.pre_smoothing = pre_smoothing
        self.post_smoothing = post_smoothing
        self.cycle_type = cycle_type
        self.cycle_type_buildup = cycle_type_buildup
        self.cycle_type_final = cycle_type_final
        self.num_cycles_buildup = num_cycles_buildup
        self.coarsest_grid_size = coarsest_grid_size
        
        # Get the omega parameter from the smoother if it exists
        self.smoother_omega = getattr(smoother, 'omega', 1.0)
        
        # Validate coarsest grid size
        if coarsest_grid_size < 3:
            raise ValueError("Coarsest grid size must be at least 3")
        if coarsest_grid_size % 2 == 0:
            raise ValueError("Coarsest grid size must be odd")
            
        # Dictionary of restriction operators
        self.restriction_operators = {
            'restrict_inject': restrict_inject,
            'restrict_full_weighting': restrict_full_weighting
        }
        
        # Dictionary of interpolation operators
        self.interpolation_operators = {
            'interpolate_linear': interpolate_linear,
            'interpolate_cubic': interpolate_cubic
        }
        
        # Validate restriction and interpolation methods
        if restriction_method not in self.restriction_operators:
            raise ValueError(f"Restriction method must be one of: {list(self.restriction_operators.keys())}")
        if interpolation_method not in self.interpolation_operators:
            raise ValueError(f"Interpolation method must be one of: {list(self.interpolation_operators.keys())}")
            
        self.restriction_method = restriction_method
        self.interpolation_method = interpolation_method
        
        self.residual_history = []
        self.vcycle_data = []
        self.presmooth_diagnostics = {}
        self.current_iteration = 0

    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star):
        """
        Solve the pressure correction equation using the matrix-free multigrid method.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh.
        u_star, v_star : ndarray
            Intermediate velocity fields.
        d_u, d_v : ndarray
            Momentum equation coefficients.
        p_star : ndarray
            Current pressure field.
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()

        rho = 1.0  # This should come from fluid properties
        
        # Reset residual history and diagnostics
        self.residual_history = []
        self.presmooth_diagnostics = {}
        self.current_iteration = 0
        
        # Get right-hand side of pressure correction equation
        b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        
        # Initial guess
        x = np.zeros_like(b)
        
        # Apply the appropriate cycle based on cycle_type
        for k in range(self.max_iterations):
            self.current_iteration = k + 1
            
            # Apply the appropriate cycle based on cycle_type
            if self.cycle_type == 'v':
                x = self._v_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                self.pre_smoothing, self.post_smoothing, level=0)
            elif self.cycle_type == 'w':
                x = self._w_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                self.pre_smoothing, self.post_smoothing, level=0)
            elif self.cycle_type == 'f':
                # For F-cycle, we only do it once at the beginning
                if k == 0:
                    initial_res = np.linalg.norm(b, 2)
                    x = self._f_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                    self.pre_smoothing, self.post_smoothing, level=0)
                    # Compute residual after F-cycle
                    Ax = compute_Ap_product(x, nx, ny, dx, dy, rho, d_u, d_v)
                    r = b - Ax
                    r_norm = np.linalg.norm(r, 2)
                    res_norm = r_norm / initial_res if initial_res > 0 else r_norm
                    self.fcycle_factor = res_norm
                else:
                    # After the initial F-cycle, use specified cycle type until convergence
                    if self.cycle_type_final == 'v':
                        x = self._v_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                        self.pre_smoothing, self.post_smoothing, level=0)
                    else:  # w-cycle
                        x = self._w_cycle(x, b, mesh, rho, d_u, d_v, self.smoother_omega, 
                                        self.pre_smoothing, self.post_smoothing, level=0)
            
            # Compute residual: r = b - A*x
            Ax = compute_Ap_product(x, nx, ny, dx, dy, rho, d_u, d_v)
            r = b - Ax
            
            # Calculate residual norm
            r_norm = np.linalg.norm(r, 2)
            b_norm = np.linalg.norm(b, 2)
            res_norm = r_norm / b_norm if b_norm > 0 else r_norm
            
            self.residual_history.append(res_norm)
            
            # Check convergence
            if res_norm < self.tolerance:
                # Calculate convergence rate
                if len(self.residual_history) > 1:
                    try:
                        if self.cycle_type == 'f':
                            # For F-cycles, calculate two separate rates
                            # 1. F-cycle rate (first iteration)
                            fcycle_rate = self.fcycle_factor
                            
                            # 2. Rate for subsequent V/W-cycles (only between V/W cycles)
                            vw_conv_rates = []
                            for i in range(3, len(self.residual_history)):  # Start from second W-cycle
                                curr_res = self.residual_history[i]
                                prev_res = self.residual_history[i-1]  # Previous W-cycle
                                if prev_res > 1e-20:  # Avoid division by very small values
                                    rate = curr_res / prev_res
                                    if not np.isnan(rate) and not np.isinf(rate) and abs(rate) < 1.0:
                                        vw_conv_rates.append(rate)
                            
                            if vw_conv_rates:
                                vw_conv_rate = np.power(np.prod(vw_conv_rates), 1.0 / len(vw_conv_rates))
                                print(f"F-cycle rate: {fcycle_rate:.4f}, {self.cycle_type_final}-cycle rate: {vw_conv_rate:.4f}, iterations: {k+1} (1F + {k}{self.cycle_type_final}), residual: {res_norm:.2e}")
                            else:
                                print(f"F-cycle rate: {fcycle_rate:.4f}, no W-cycles, residual: {res_norm:.2e}")
                        else:
                            # For pure V/W-cycles, calculate single rate
                            conv_rates = []
                            for i in range(1, len(self.residual_history)):
                                prev_res = self.residual_history[i-1]
                                if prev_res > 1e-12:  # Avoid division by very small values
                                    rate = self.residual_history[i] / prev_res
                                    if not np.isnan(rate) and not np.isinf(rate) and abs(rate) < 1.0:
                                        conv_rates.append(rate)
                            
                            if conv_rates:
                                overall_conv_rate = np.power(np.prod(conv_rates), 1.0 / len(conv_rates))
                                print(f"Convergence rate: {overall_conv_rate:.4f}, iterations: {k+1} {self.cycle_type}-cycles, residual: {res_norm:.2e}")
                            else:
                                print(f"No convergence rate, iterations: {k+1}, residual: {res_norm:.2e}")
                    except Exception as e:
                        print(f"Error calculating convergence rate: {e}")
                break
        
        # Reshape to 2D with Fortran ordering
        p_prime = x.reshape((nx, ny), order='F')
        
        return p_prime
    
    def _solve_residual_direct(self, mesh, residual, d_u, d_v, rho=1.0):
        """
        Solve the residual equation directly using sparse matrix methods.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh.
        residual : ndarray
            The residual vector to solve for.
        d_u, d_v : ndarray
            Momentum equation coefficients.
        rho : float, optional
            Fluid density.
            
        Returns:
        --------
        p_prime : ndarray
            Solution of the residual equation.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
        
        # Fix reference pressure at P(1,1) - which is (0,0) in 0-based indexing
        # to lil
        A = A.tolil()
        A[0, :] = 0
        A[0, 0] = 1
        eps = 1e-10
        diag_indices = sparse.find(A.diagonal())[0]
        for i in diag_indices:
            A[i, i] += eps
        A = A.tocsr()
        
        # Solve the system
        p_prime_flat = spsolve(A, residual)
        p_prime = p_prime_flat.reshape((nx, ny), order='F')
        
        return p_prime

    def _v_cycle(self, p, rhs, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, level=0):
        """
        Perform one V-cycle of the multigrid method.
        
        Parameters:
        -----------
        p : ndarray
            Current pressure field
        rhs : ndarray
            Right-hand side of the Poisson equation
        mesh : StructuredMesh
            The current grid mesh
        rho : float
            Fluid density
        d_u, d_v : ndarray
            Momentum equation coefficients
        omega : float
            Relaxation factor
        pre_smoothing : int
            Number of pre-smoothing iterations
        post_smoothing : int
            Number of post-smoothing iterations
        level : int
            Current grid level (0 is finest)
            
        Returns:
        --------
        ndarray
            Updated pressure field
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        # If at the coarsest grid, solve directly
        if nx <= self.coarsest_grid_size:
            p = self._solve_residual_direct(mesh, rhs, d_u, d_v, rho)
            return p
            
        # Pre-smoothing steps
        p = self.smoother.solve(mesh=mesh, p=p, b=rhs,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        
        # Compute the residual: r = rhs - A*p
        Ap = compute_Ap_product(p, nx, ny, dx, dy, rho, d_u, d_v)
        r = rhs - Ap
        
        # Reshape residual to 2D for restriction
        r_reshaped = r.reshape((nx, ny), order='F')
        
        # Restrict residual to coarser grid using the selected operator
        r_coarse = self.restriction_operators[self.restriction_method](r_reshaped)
        
        # Determine coarse grid size and create coarse grid mesh
        coarse_grid_size = r_coarse.shape[0]
        mesh_coarse = StructuredMesh(nx=coarse_grid_size, ny=coarse_grid_size, 
                                    length=mesh.length, height=mesh.height)
        
        # Restrict coefficients for the coarse grid
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u, d_v, 
            nx, ny, 
            coarse_grid_size, coarse_grid_size, 
            dx, dy
        )
        
        # Recursive V-cycle on coarse grid
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse = self._v_cycle(
            p=np.zeros_like(r_coarse_flat), 
            rhs=r_coarse_flat,
            mesh=mesh_coarse,
            rho=rho,
            d_u=d_u_coarse,
            d_v=d_v_coarse,
            omega=omega,
            pre_smoothing=pre_smoothing,
            post_smoothing=post_smoothing,
            level=level+1
        )
        
        # Interpolate error correction to fine grid using the selected operator
        e_interpolated = self.interpolation_operators[self.interpolation_method](e_coarse, nx)
        
        # Apply the error correction
        p += e_interpolated.flatten('F').reshape((nx, ny), order='F')
        
        # Post-smoothing steps
        p = self.smoother.solve(mesh=mesh, p=p, b=rhs,
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
        return p
    
    def _w_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                 level=0):
        """
        Performs one W-cycle of the multigrid method.
        
        Parameters:
        -----------
        u : ndarray
            Current approximation (flattened).
        f : ndarray
            Right-hand side (flattened).
        mesh : StructuredMesh
            The current grid mesh.
        rho : float
            Fluid density.
        d_u, d_v : ndarray
            Momentum equation coefficients.
        omega : float
            Relaxation factor.
        pre_smoothing : int
            Number of pre-smoothing iterations.
        post_smoothing : int
            Number of post-smoothing iterations.
        grid_calculations : list
            List to track grid sizes (not used for debugging here).
        level : int
            Current grid level of the multigrid hierarchy.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # If at the coarsest grid, solve directly
        if nx <= self.coarsest_grid_size:
            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            #u = self.smoother.solve(mesh=mesh, p=u, b=f,
            #                    d_u=d_u, d_v=d_v, rho=rho, num_iterations=20, track_residuals=False)
            return u  
  
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        
        # Compute the residual: r = f - A*u
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        # Reshape residual to 2D for restriction
        r_reshaped = r.reshape((nx, ny), order='F')
        
        # Restrict residual to coarser grid
        r_coarse = self.restriction_operators[self.restriction_method](r_reshaped) 
        
        # Determine coarse grid size and create coarse grid mesh
        coarse_grid_size = r_coarse.shape[0]
        mesh_coarse = StructuredMesh(nx=coarse_grid_size, ny=coarse_grid_size, 
                                        length=mesh.length, height=mesh.height)
        
        # Restrict coefficients for the coarse grid
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u, d_v, 
            nx, ny, 
            coarse_grid_size, coarse_grid_size, 
            dx, dy
        )
        
        # First recursive W-cycle on coarse grid
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse1 = self._w_cycle(u=np.zeros_like(r_coarse_flat), f=r_coarse_flat, 
                                 mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                 omega=omega, pre_smoothing=pre_smoothing, 
                                 post_smoothing=post_smoothing, 
                                 level=level+1)
        
        # Second recursive W-cycle on coarse grid (this is what makes it a W-cycle)
        e_coarse2 = self._w_cycle(u=e_coarse1, f=r_coarse_flat, 
                                 mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                 omega=omega, pre_smoothing=pre_smoothing, 
                                 post_smoothing=post_smoothing, 
                                 level=level+1)
        
        # Optionally store the coarse grid error solution
        e_coarse_2D = e_coarse2.reshape((coarse_grid_size, coarse_grid_size), order='F')
       
        # Interpolate error correction to fine grid
        e_interpolated = self.interpolation_operators[self.interpolation_method](e_coarse2, nx)
        
        # Apply the error correction
        u += e_interpolated.flatten('F').reshape((nx, ny), order='F')
        
        # Post-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
        return u
    
    def _f_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, 
                 level=0):
        """
        Performs one F-cycle of the multigrid method.
        Starts from coarsest grid, moves up level by level performing specified number of cycles
        at each level.
        
        Parameters:
        -----------
        u : ndarray
            Current approximation (flattened).
        f : ndarray
            Right-hand side (flattened).
        mesh : StructuredMesh
            The current grid mesh.
        rho : float
            Fluid density.
        d_u, d_v : ndarray
            Momentum equation coefficients.
        omega : float
            Relaxation factor.
        pre_smoothing : int
            Number of pre-smoothing iterations.
        post_smoothing : int
            Number of post-smoothing iterations.
        level : int
            Current grid level of the multigrid hierarchy.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # If at the coarsest grid, solve directly
        if nx <= self.coarsest_grid_size:
            u = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            return u
        
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        
        # Compute the residual: r = f - A*u
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        # Reshape residual to 2D for restriction
        r_reshaped = r.reshape((nx, ny), order='F')
        
        # Restrict residual to coarser grid
        r_coarse = self.restriction_operators[self.restriction_method](r_reshaped) 
        
        # Determine coarse grid size and create coarse grid mesh
        coarse_grid_size = r_coarse.shape[0]
        mesh_coarse = StructuredMesh(nx=coarse_grid_size, ny=coarse_grid_size, 
                                        length=mesh.length, height=mesh.height)
        
        # Restrict coefficients for the coarse grid
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u, d_v, 
            nx, ny, 
            coarse_grid_size, coarse_grid_size, 
            dx, dy
        )
        
        # Recursive F-cycle on coarse grid
        r_coarse_flat = r_coarse.flatten('F')
        e_coarse = self._f_cycle(u=np.zeros_like(r_coarse_flat), f=r_coarse_flat, 
                                 mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse, 
                                 omega=omega, pre_smoothing=pre_smoothing, 
                                 post_smoothing=post_smoothing, 
                                 level=level+1)
        
        # Perform specified number of cycles at this level
        for _ in range(self.num_cycles_buildup):
            if self.cycle_type_buildup == 'v':
                e_coarse = self._v_cycle(e_coarse, r_coarse_flat, mesh_coarse, rho, 
                                       d_u_coarse, d_v_coarse, omega, pre_smoothing, 
                                       post_smoothing, level+1)
            else:  # w-cycle
                e_coarse = self._w_cycle(e_coarse, r_coarse_flat, mesh_coarse, rho, 
                                       d_u_coarse, d_v_coarse, omega, pre_smoothing, 
                                       post_smoothing, level+1)
        
        # Optionally store the coarse grid error solution
        e_coarse_2D = e_coarse.reshape((coarse_grid_size, coarse_grid_size), order='F')
       
        # Interpolate error correction to fine grid
        e_interpolated = self.interpolation_operators[self.interpolation_method](e_coarse, nx)
        
        # Apply the error correction
        u += e_interpolated.flatten('F').reshape((nx, ny), order='F') 
        
        # Post-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u, b=f,
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
        return u
    
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics.
        """
        smoother_info = {}
        if hasattr(self.smoother, 'get_solver_info'):
            smoother_info = self.smoother.get_solver_info()
        
        info = {
            'name': 'MultiGridSolver',
            'inner_iterations_history': [],  # Not tracked directly
            'total_inner_iterations': 0,       # Not tracked directly
            'convergence_rate': None           # Not tracked directly
        }
        
        info['solver_specific'] = {
            'pre_smoothing': self.pre_smoothing,
            'post_smoothing': self.post_smoothing,
            'smoother_type': smoother_info.get('name', 'Unknown'),
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations,
            'cycle_type': self.cycle_type,
        }
        
        return info

    def add_restriction_operator(self, name, operator):
        """
        Add a new restriction operator to the solver.
        
        Parameters:
        -----------
        name : str
            Name of the restriction operator
        operator : callable
            Function that implements the restriction operation
        """
        self.restriction_operators[name] = operator
        
    def add_interpolation_operator(self, name, operator):
        """
        Add a new interpolation operator to the solver.
        
        Parameters:
        -----------
        name : str
            Name of the interpolation operator
        operator : callable
            Function that implements the interpolation operation
        """
        self.interpolation_operators[name] = operator
