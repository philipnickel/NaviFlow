import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .base_pressure_solver import PressureSolver
from .helpers.rhs_construction import get_rhs
from .helpers.matrix_free import compute_Ap_product
from .helpers.multigrid_helpers import  restrict_coefficients,restrict_inject, restrict_full_weighting, interpolate_linear, interpolate_cubic
from .helpers.boundary_conditions import enforce_zero_gradient_bc
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
           # Initialize coefficient matrices for pressure correction equation
        self.p_a_e = None
        self.p_a_w = None
        self.p_a_n = None
        self.p_a_s = None
        self.p_a_p = None
        self.p_source = None

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
        use_zero_pressure_bc : bool, optional
            If True, applies zero pressure (Dirichlet) boundary conditions
            instead of the default zero gradient (Neumann) conditions.
        zero_pressure_boundaries : list or str, optional
            Specifies which boundaries to apply zero pressure BC.
            Can include 'west', 'east', 'south', 'north', or 'all'.
            Only used when use_zero_pressure_bc is True.
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        self.rho = 1.0  # This should come from fluid properties
        
        # Store mesh for boundary conditions
        self.mesh = mesh
        
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
                #print(f"multigrid residual: {res_norm:.2e}")
                #print(f"No convergence, iterations: {k+1}, residual: {res_norm:.2e}")
        
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
        
        # Solve the system
        p_prime_flat = spsolve(A, residual)
        p_prime = p_prime_flat.reshape((nx, ny), order='F')

        # enforce zero gradient boundary conditions
        #p_prime = self._enforce_pressure_boundary_conditions(mesh, p_prime)
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
            p_direct_2d = p_direct.reshape((nx, ny), order='F')
            
            # Apply zero gradient boundary conditions to the direct solution
            #p_direct_2d = self._enforce_pressure_boundary_conditions(mesh, p_direct_2d)
                
            return p_direct_2d.flatten(order='F') # Ensure it returns flat

        # Pre-smoothing steps
        p = self.smoother.solve(mesh=mesh, p=p.reshape((nx, ny), order='F'), b=rhs.reshape((nx, ny), order='F'),
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
                              
        # Apply zero gradient boundary conditions after pre-smoothing
        #p = self._enforce_pressure_boundary_conditions(mesh, p)
            
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

        # apply zero gradient boundary conditions to the coarse residual
        #r_coarse_2D = self._enforce_pressure_boundary_conditions(mesh_coarse, r_coarse_2D)

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
            level=level+1,
        )

        # Reshape coarse error for interpolation
        e_coarse_2D = e_coarse_flat.reshape((nx_coarse, ny_coarse), order='F')
        
    

        # Interpolate error correction to fine grid using the selected operator
        e_interpolated_2D = self.interpolation_operators[self.interpolation_method](e_coarse_2D, nx)
        
        # Apply zero gradient boundary conditions to the interpolated error
        #e_interpolated_2D = self._enforce_pressure_boundary_conditions(mesh, e_interpolated_2D)

        # Apply the error correction (p = p + e)
        p_2d = p.reshape((nx, ny), order='F')
        p_2d += e_interpolated_2D #* 0.9
        
        # Apply zero gradient boundary conditions after error correction
        #p_2d = self._enforce_pressure_boundary_conditions(mesh, p_2d)
            
        p = p_2d.flatten('F')

        # Post-smoothing steps
        p = self.smoother.solve(mesh=mesh, p=p.reshape((nx, ny), order='F'), b=rhs.reshape((nx, ny), order='F'),
                              d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
                              
        # Apply zero gradient boundary conditions after post-smoothing
        #p = self._enforce_pressure_boundary_conditions(mesh, p)
            
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
            Specifies which boundaries to apply zero pressure BC.
            Can include 'west', 'east', 'south', 'north', or 'all'.
            Only used when use_zero_pressure_bc is True.
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
    
        # If at the coarsest grid, solve directly
        if nx <= self.coarsest_grid_size:
            u_direct = self._solve_residual_direct(mesh, f, d_u, d_v, rho)
            u_direct_2d = u_direct.reshape((nx, ny), order='F')
            
            # Apply zero gradient boundary conditions to the direct solution
            #u_direct_2d = self._enforce_pressure_boundary_conditions(mesh, u_direct_2d)
                
            return u_direct_2d.flatten(order='F') # Ensure it returns flat
  
        # Pre-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u.reshape((nx, ny), order='F'), b=f.reshape((nx, ny), order='F'),
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=pre_smoothing, track_residuals=False)
            
        # Apply zero gradient boundary conditions after pre-smoothing
        #u = self._enforce_pressure_boundary_conditions(mesh, u)
            
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
             e_coarse_flat = self._w_cycle(
                 u=e_coarse_flat, 
                 f=r_coarse_flat,
                 mesh=mesh_coarse,
                 rho=rho,
                 d_u=d_u_coarse,
                 d_v=d_v_coarse,
                 omega=omega,
                 pre_smoothing=pre_smoothing,
                 post_smoothing=post_smoothing,
                 level=level+1,
             )
        
        # Reshape coarse error for interpolation
        e_coarse_2D = e_coarse_flat.reshape((nx_coarse, ny_coarse), order='F')
        
        
        # Interpolate error correction to fine grid using the selected operator
        e_interpolated_2D = self.interpolation_operators[self.interpolation_method](e_coarse_2D, nx)
        
        # Apply zero gradient boundary conditions to the interpolated error
        #e_interpolated_2D = self._enforce_pressure_boundary_conditions(mesh, e_interpolated_2D)
        
        # Apply the error correction (u = u + e)
        u_2d = u.reshape((nx, ny), order='F')
        u_2d += e_interpolated_2D
        
        # Apply zero gradient boundary conditions after error correction
        #u_2d = self._enforce_pressure_boundary_conditions(mesh, u_2d)
            
        u = u_2d.flatten('F')
        
        # Post-smoothing steps
        u = self.smoother.solve(mesh=mesh, p=u.reshape((nx, ny), order='F'), b=f.reshape((nx, ny), order='F'),
                                d_u=d_u, d_v=d_v, rho=rho, num_iterations=post_smoothing, track_residuals=False)
        
        # Apply zero gradient boundary conditions after post-smoothing
        #u = self._enforce_pressure_boundary_conditions(mesh, u)
       
        u = u.flatten('F') # Flatten back
        
        return u

    def _fmg_cycle(self, rhs_fine, mesh_fine, d_u_fine, d_v_fine, nx_finest, ny_finest, dx_finest, dy_finest, use_zero_pressure_bc=False, zero_pressure_boundaries=None):
        """
        Perform a recursive Full Multigrid (FMG) cycle.
        
        Parameters:
        -----------
        rhs_fine : ndarray
            Right-hand side of the equation on the finest level
        mesh_fine : StructuredMesh
            The computational mesh on the finest level
        d_u_fine, d_v_fine : ndarray
            Momentum equation coefficients on the finest level
        nx_finest, ny_finest : int
            Grid dimensions of the original finest grid.
        dx_finest, dy_finest : float
            Grid spacings of the original finest grid.
              Returns:
        --------
        ndarray
            Solution field interpolated back to the finest level.
        """
        nx_fine, ny_fine = mesh_fine.get_dimensions()
        dx_fine, dy_fine = mesh_fine.get_cell_sizes()


        # If we're at the coarsest level, solve directly
        if nx_fine <= self.coarsest_grid_size:
            rhs_fine_flat = rhs_fine
            if rhs_fine.ndim == 2:
                rhs_fine_flat = rhs_fine.flatten(order='F')

            solution_fine = self._solve_residual_direct(mesh_fine, rhs_fine_flat, d_u_fine, d_v_fine, self.rho)
            solution_fine_2d = solution_fine.reshape((nx_fine, ny_fine), order='F')
            
            # Apply zero gradient boundary conditions to the direct solution
            #solution_fine_2d = self._enforce_pressure_boundary_conditions(mesh_fine, solution_fine_2d)
                
            return solution_fine_2d.flatten(order='F')

        # Reshape RHS to 2D for restriction if it's flattened
        if rhs_fine.ndim == 1:
            rhs_fine_2D = rhs_fine.reshape((nx_fine, ny_fine), order='F')
        else:
            rhs_fine_2D = rhs_fine


        # Restrict RHS to coarser grid and create coarse mesh
        rhs_coarse_2D = self.restriction_operators[self.restriction_method](rhs_fine_2D)
        nx_coarse, ny_coarse = rhs_coarse_2D.shape
        mesh_coarse = StructuredMesh(nx=nx_coarse, ny=ny_coarse, 
                                  length=mesh_fine.length, height=mesh_fine.height)

        # Restrict coefficients
        d_u_coarse, d_v_coarse = restrict_coefficients(
            d_u_fine, d_v_fine, 
            nx_fine, ny_fine, 
            nx_coarse, ny_coarse, 
            dx_fine, dy_fine
        )

        # Recursive FMG on coarser grid
        solution_coarse_flat = self._fmg_cycle(
            rhs_coarse_2D.flatten(order='F'), mesh_coarse, d_u_coarse, d_v_coarse,
            nx_finest, ny_finest, dx_finest, dy_finest, # Pass finest grid info down
        )
        solution_coarse_2D = solution_coarse_flat.reshape((nx_coarse, ny_coarse), order='F')


        # Interpolate to fine grid with boundary preservation
        solution_fine_2D = self.interpolation_operators[self.interpolation_method](solution_coarse_2D, nx_fine)
        
        # Apply zero gradient boundary conditions after interpolation
        #solution_fine_2D = self._enforce_pressure_boundary_conditions(mesh_fine, solution_fine_2D)

        # Perform additional V/W cycles if desired
        # Outer loop performs at most max_cycles_buildup cycles
        # or stops if tolerance is reached
        if self.max_cycles_buildup > 0:
            for k in range(self.max_cycles_buildup):
                # Determine type of cycle to use during buildup
                if self.cycle_type_buildup == 'v':
                    solution_fine_2D = self._v_cycle(
                        solution_fine_2D.flatten(order='F'),
                        rhs_fine_2D.flatten(order='F'),
                        mesh_fine, 
                        self.rho, 
                        d_u_fine, 
                        d_v_fine, 
                        self.smoother_omega, 
                        self.pre_smoothing, 
                        self.post_smoothing,
                        level=0,
                    )
                elif self.cycle_type_buildup == 'w':
                    solution_fine_2D = self._w_cycle(
                        solution_fine_2D.flatten(order='F'), 
                        rhs_fine_2D.flatten(order='F'), 
                        mesh_fine, 
                        self.rho, 
                        d_u_fine, 
                        d_v_fine, 
                        self.smoother_omega, 
                        self.pre_smoothing, 
                        self.post_smoothing,
                        level=0,
                    )
                
                solution_fine_2D = solution_fine_2D.reshape((nx_fine, ny_fine), order='F')
                
                # Apply zero gradient boundary conditions after each cycle
                #solution_fine_2D = self._enforce_pressure_boundary_conditions(mesh_fine, solution_fine_2D)
                
                # Check convergence if requested
                if self.tolerance < 1.0:
                    solution_flat = solution_fine_2D.flatten(order='F')
                    Ax = compute_Ap_product(solution_flat, nx_fine, ny_fine, dx_fine, dy_fine, 
                                         self.rho, d_u_fine, d_v_fine)
                    
                    r = rhs_fine_2D.flatten(order='F') - Ax
                    r_norm = np.linalg.norm(r)
                    rhs_norm = np.linalg.norm(rhs_fine_2D)
                    rel_res = r_norm / rhs_norm if rhs_norm > 0 else r_norm
                    
                    if rel_res < self.tolerance:
                        break

        return solution_fine_2D.flatten(order='F')

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
