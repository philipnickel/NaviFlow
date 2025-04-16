import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product
from .helpers.multigrid_helpers import  restrict_coefficients,restrict_inject, restrict_full_weighting, interpolate_linear, interpolate_cubic
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
                 cycle_type='v', cycle_type_buildup='v', cycle_type_final=None,
                 max_cycles_buildup=1,
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
        max_cycles_buildup : int
            Number of cycles to perform at each level during buildup (also max cycles if solving to tolerance)
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
        self.cycle_type_buildup = cycle_type_buildup # Used within FMG
        self.cycle_type_final = cycle_type_final
        self.max_cycles_buildup = max_cycles_buildup
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
        self.rho = 1.0 # Assume default, can be updated later if needed

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

        self.rho = 1.0  # This should come from fluid properties
        
        # Reset residual history and diagnostics
        self.residual_history = []
        self.presmooth_diagnostics = {}
        self.current_iteration = 0
        
        # Get right-hand side of pressure correction equation
        b = get_rhs(nx, ny, dx, dy, self.rho, u_star, v_star)
        
        # Initial guess
        x = np.zeros_like(b)
        
        # Apply the appropriate cycle based on cycle_type
        if self.cycle_type == 'fmg':
            # FMG is performed only once as an initialization step
            nx_finest, ny_finest = mesh.get_dimensions()
            dx_finest, dy_finest = mesh.get_cell_sizes()
            x = self._fmg_cycle(b, mesh, d_u, d_v, nx_finest, ny_finest, dx_finest, dy_finest)
            #print(f"scales of solution: min: {np.min(x)}, max: {np.max(x)}, mean: {np.mean(x)}")
            # Optionally, perform final V/W cycles after FMG
            if self.cycle_type_final:
                #print(f"Final {self.cycle_type_final}-cycle iteration {k+1} of {self.max_iterations}")
                if self.cycle_type_final == 'v':
                        x = self._v_cycle(x, b, mesh, self.rho, d_u, d_v, self.smoother_omega,
                                        self.pre_smoothing, self.post_smoothing, level=0)
                elif self.cycle_type_final == 'w':
                        x = self._w_cycle(x, b, mesh, self.rho, d_u, d_v, self.smoother_omega,
                                        self.pre_smoothing, self.post_smoothing, level=0)

                # Compute residual: r = b - A*x
                Ax = compute_Ap_product(x, nx, ny, dx, dy, self.rho, d_u, d_v)
                r = b - Ax
                r_norm = np.linalg.norm(r, 2)
                b_norm = np.linalg.norm(b, 2)
                res_norm = r_norm / b_norm if b_norm > 0 else r_norm
                self.residual_history.append(res_norm)
            else:
                 # Compute final residual after FMG if no final cycles run
                 Ax = compute_Ap_product(x, nx, ny, dx, dy, self.rho, d_u, d_v)
                 r = b - Ax
                 r_norm = np.linalg.norm(r, 2)
                 b_norm = np.linalg.norm(b, 2)
                 res_norm = r_norm / b_norm if b_norm > 0 else r_norm
                 self.residual_history.append(res_norm)
                 print(f"FMG cycle completed. Final relative residual: {res_norm:.2e}")

        else: # V or W cycles
            for k in range(self.max_iterations):
                self.current_iteration = k + 1

                if self.cycle_type == 'v':
                    x = self._v_cycle(x, b, mesh, self.rho, d_u, d_v, self.smoother_omega,
                                    self.pre_smoothing, self.post_smoothing, level=0)
                elif self.cycle_type == 'w':
                    x = self._w_cycle(x, b, mesh, self.rho, d_u, d_v, self.smoother_omega,
                                    self.pre_smoothing, self.post_smoothing, level=0)

                # Compute residual: r = b - A*x
                Ax = compute_Ap_product(x, nx, ny, dx, dy, self.rho, d_u, d_v)
                r = b - Ax

                # Calculate residual norm
                r_norm = np.linalg.norm(r, 2)
                b_norm = np.linalg.norm(b, 2)
                res_norm = r_norm / b_norm if b_norm > 0 else r_norm

                self.residual_history.append(res_norm)

                # Check convergence
                if res_norm < self.tolerance:
                    # Calculate convergence rate
                    conv_rates = []
                    if len(self.residual_history) > 1:
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
            Current pressure field (flattened)
        rhs : ndarray
            Right-hand side of the Poisson equation (flattened)
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
            Updated pressure field (flattened)
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        # If at the coarsest grid, solve directly
        if nx <= self.coarsest_grid_size:
            # Direct solve requires reshaping p and rhs if they came in flat
            # But _solve_residual_direct expects rhs flat, and returns flat result
            p_direct = self._solve_residual_direct(mesh, rhs, d_u, d_v, rho)
            return p_direct.flatten(order='F') # Ensure it returns flat

        # Pre-smoothing steps
        p = self.smoother.solve(mesh=mesh, p=p.reshape((nx, ny), order='F'), b=rhs.reshape((nx, ny), order='F'),
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        p = p.flatten(order='F') # Smoother returns 2D, flatten it back

        # Compute the residual: r = rhs - A*p
        Ap = compute_Ap_product(p, nx, ny, dx, dy, rho, d_u, d_v)
        r = rhs - Ap

        # Reshape residual to 2D for restriction
        r_reshaped = r.reshape((nx, ny), order='F')

        # Restrict residual to coarser grid using the selected operator
        r_coarse_2D = self.restriction_operators[self.restriction_method](r_reshaped) 

        # Determine coarse grid size and create coarse grid mesh
        nx_coarse, ny_coarse = r_coarse_2D.shape
        mesh_coarse = StructuredMesh(nx=nx_coarse, ny=ny_coarse,
                                    length=mesh.length, height=mesh.height)

        # Restrict coefficients for the coarse grid
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u, d_v,
            nx, ny,
            nx_coarse, ny_coarse,
            dx, dy # Pass fine grid dx, dy for restriction calculation
        )

        # Recursive V-cycle on coarse grid
        r_coarse_flat = r_coarse_2D.flatten('F')
        e_coarse_flat = self._v_cycle(
            p=np.zeros_like(r_coarse_flat), # Solve for error e: Ae = r
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

        # Reshape coarse error for interpolation
        e_coarse_2D = e_coarse_flat.reshape((nx_coarse, ny_coarse), order='F')

        # Interpolate error correction to fine grid using the selected operator
        e_interpolated_2D = self.interpolation_operators[self.interpolation_method](e_coarse_2D, nx)

        # Apply the error correction (p = p + e)
        p += e_interpolated_2D.flatten('F') # Add flat error to flat p

        # Post-smoothing steps
        p = self.smoother.solve(mesh=mesh, p=p.reshape((nx, ny), order='F'), b=rhs.reshape((nx, ny), order='F'),
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        p = p.flatten(order='F') # Smoother returns 2D, flatten it back

        return p
    
    def _w_cycle(self, u, f, mesh, rho, d_u, d_v, omega, pre_smoothing, post_smoothing, level=0):
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
        level : int
            Current grid level of the multigrid hierarchy.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
    
        # If at the coarsest grid, solve directly
        if nx <= self.coarsest_grid_size:
            u_direct = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            return u_direct.flatten(order='F') # Ensure it returns flat
  
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u.reshape((nx, ny), order='F'), b=f.reshape((nx, ny), order='F'),
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
        u = u.flatten(order='F') # Flatten back
        
        # Compute the residual: r = f - A*u
        Au = compute_Ap_product(u, nx, ny, dx, dy, rho, d_u, d_v)
        r = f - Au
        
        # Reshape residual to 2D for restriction
        r_reshaped = r.reshape((nx, ny), order='F')
        
        # Restrict residual to coarser grid
        r_coarse_2D = self.restriction_operators[self.restriction_method](r_reshaped) 
        
        # Determine coarse grid size and create coarse grid mesh
        nx_coarse, ny_coarse = r_coarse_2D.shape
        mesh_coarse = StructuredMesh(nx=nx_coarse, ny=ny_coarse, 
                                        length=mesh.length, height=mesh.height)
        
        # Restrict coefficients for the coarse grid
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u, d_v, 
            nx, ny, 
            nx_coarse, ny_coarse, 
            dx, dy
        ) 
        
        # Flatten coarse residual
        r_coarse_flat = r_coarse_2D.flatten('F')
        
        # Initialize coarse grid error correction
        e_coarse_flat = np.zeros_like(r_coarse_flat)
        
        # Perform two recursive calls for W-cycle
        for _ in range(2):
             e_coarse_flat = self._w_cycle(u=e_coarse_flat, f=r_coarse_flat,
                                      mesh=mesh_coarse, rho=rho, d_u=d_u_coarse, d_v=d_v_coarse,
                                      omega=omega, pre_smoothing=pre_smoothing,
                                      post_smoothing=post_smoothing,
                                      level=level+1)
        
        # Reshape coarse error for interpolation
        e_coarse_2D = e_coarse_flat.reshape((nx_coarse, ny_coarse), order='F')
        
        # Interpolate error correction to fine grid
        e_interpolated_2D = self.interpolation_operators[self.interpolation_method](e_coarse_2D, nx)
        
        # Apply the error correction: u = u + e
        u += e_interpolated_2D.flatten('F')
        
        # Post-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u.reshape((nx, ny), order='F'), b=f.reshape((nx, ny), order='F'),
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        u = u.flatten(order='F') # Flatten back
        
        return u

    def _fmg_cycle(self, rhs_fine, mesh_fine, d_u_fine, d_v_fine, nx_finest, ny_finest, dx_finest, dy_finest):
        """
        Perform a recursive Full Multigrid (FMG) cycle.
        
        Parameters:
        -----------
        rhs_fine : ndarray
            Right-hand side vector on the current fine grid (flattened).
        mesh_fine : StructuredMesh
            The current fine computational mesh.
        d_u_fine, d_v_fine : ndarray
            Momentum equation coefficients on the current fine grid.
        nx_finest, ny_finest : int
            Dimensions of the original finest grid.
        dx_finest, dy_finest : float
             Grid spacings of the original finest grid.

        Returns:
        --------
        ndarray
            Solution on the current fine grid (flattened).
        """
        nx_fine, ny_fine = mesh_fine.get_dimensions()
        dx_fine, dy_fine = mesh_fine.get_cell_sizes()

        # Base case: If the current grid is the coarsest grid, solve directly
        if nx_fine <= self.coarsest_grid_size:
            #print(f"FMG Base Case: Solving on {nx_fine}x{ny_fine} grid directly.")
            #solution_coarse_2D = self._solve_residual_direct(
            #    mesh_fine, rhs_fine, d_u_fine, d_v_fine, self.rho
            #)
            # solve using smoother
            solution_coarse_2D = self.smoother.solve(mesh=mesh_fine, p=rhs_fine.reshape((nx_fine, ny_fine), order='F'), b=rhs_fine.reshape((nx_fine, ny_fine), order='F'),
                              d_u=d_u_fine, d_v=d_v_fine, rho=self.rho, num_iterations=100, track_residuals=False)
            return solution_coarse_2D.flatten(order='F')
        
        grid_scale = (dx_finest / dx_fine)

        # Recursive step:
        # 1. Restrict the problem to the coarser grid
        rhs_fine_2D = rhs_fine.reshape((nx_fine, ny_fine), order='F')
        # For a Poisson equation, when restricting the right-hand side, a factor of 4 is needed 
        # to maintain consistency with the differential operator scaling
        rhs_coarse_2D = self.restriction_operators['restrict_full_weighting'](rhs_fine_2D) * 4
        nx_coarse, ny_coarse = rhs_coarse_2D.shape

        mesh_coarse = StructuredMesh(nx=nx_coarse, ny=ny_coarse,
                                     length=mesh_fine.length, height=mesh_fine.height)

        # Restrict coefficients from fine to coarse grid
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u_fine, d_v_fine,
            nx_fine, ny_fine,
            nx_coarse, ny_coarse,
            dx_fine, dy_fine
        )

        # 2. Solve the coarse grid problem recursively using FMG
        solution_coarse_flat = self._fmg_cycle(
            rhs_coarse_2D.flatten(order='F'), mesh_coarse, d_u_coarse, d_v_coarse,
            nx_finest, ny_finest, dx_finest, dy_finest # Pass finest grid info down
        )
        solution_coarse_2D = solution_coarse_flat.reshape((nx_coarse, ny_coarse), order='F')
        # 3. Interpolate the coarse grid solution to the fine grid
        solution_fine_initial_2D = self.interpolation_operators[self.interpolation_method](
            solution_coarse_2D, nx_fine
        ) #/ grid_scale
        solution_fine_flat = solution_fine_initial_2D.flatten(order='F')
        # print min max and mean of interpolated solution
        #print(f"solution_fine_flat min: {np.min(solution_fine_flat)}, max: {np.max(solution_fine_flat)}, mean: {np.mean(solution_fine_flat)}")

        # 4. Refine the solution on the fine grid using V/W cycles
        # Calculate discretization error tolerance for this level
        #h_factor_sq = (dx_fine * dy_fine) / (dx_finest * dy_finest) # Approximation of (h/h_finest)^2
        #level_tolerance = self.tolerance * h_factor_sq # Target relative residual for this level (e.g., 0.1 * ||tau||)
        current_h = max(dx_fine, dy_fine)
        level_tolerance = current_h#**self.disc_order # Target relative residual for this level (e.g., 0.1 * ||tau||)
        #print(f"FMG Refinement: Level {nx_fine}x{ny_fine}, Target Tolerance: {level_tolerance:.2e}")
        #level_tolerance = 0.1 * h_factor_sq # Target relative residual for this level (e.g., 0.1 * ||tau||)

        #print(f"FMG Refinement: Level {nx_fine}x{ny_fine}, Target Tolerance: {level_tolerance:.2e}")

        # Use specified buildup cycle type (V or W)
        cycle_method = self._v_cycle if self.cycle_type_buildup == 'v' else self._w_cycle
        cycle_arg_name = 'p' if self.cycle_type_buildup == 'v' else 'u' # Argument name for solution differs
        # if current grid is the finest grid return

        for cycle_num in range(self.max_cycles_buildup):
             # Prepare arguments for the cycle method
             cycle_args = {
                 cycle_arg_name: solution_fine_flat,
                 'f' if cycle_arg_name == 'u' else 'rhs': rhs_fine, # RHS argument name differs
                 'mesh': mesh_fine,
                 'rho': self.rho,
                 'd_u': d_u_fine,
                 'd_v': d_v_fine,
                 'omega': self.smoother_omega,
                 'pre_smoothing': self.pre_smoothing,
                 'post_smoothing': self.post_smoothing,
                 'level': 0 # Level argument might not be strictly needed here but pass for consistency
             }
             solution_fine_flat = cycle_method(**cycle_args)

             # Check residual against level tolerance
             Ax = compute_Ap_product(solution_fine_flat, nx_fine, ny_fine, dx_fine, dy_fine, self.rho, d_u_fine, d_v_fine)
             residual = rhs_fine - Ax
             r_norm = np.linalg.norm(residual, 2)
             b_norm = np.linalg.norm(rhs_fine, 2)
             rel_res = r_norm / b_norm if b_norm > 0 else r_norm

             #print(f"  FMG Refinement Cycle {cycle_num+1}/{self.max_cycles_buildup}: Relative Residual = {rel_res:.2e}")

             if rel_res < level_tolerance:
                 #print(f"  FMG Refinement converged in {cycle_num+1} cycles.")
                 #break
                 pass
        else:
            pass
             #print(f"  FMG Refinement finished {self.max_cycles_buildup} cycles without reaching tolerance {level_tolerance:.2e}.")


        return solution_fine_flat

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
            'max_iterations': self.max_iterations, # Max iterations for final cycles if used
            'cycle_type': self.cycle_type,
            'cycle_type_buildup': self.cycle_type_buildup,
            'max_cycles_buildup': self.max_cycles_buildup,
            'restriction': self.restriction_method,
            'interpolation': self.interpolation_method,
            'coarsest_grid_size': self.coarsest_grid_size
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
