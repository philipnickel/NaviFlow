"""
SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm implementation.
""" 

import numpy as np
import os
import matplotlib.pyplot as plt
from .base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult
from ...postprocessing.validation.cavity_flow import calculate_infinity_norm_error, calculate_l2_norm_error
from ...postprocessing.visualization import plot_final_residuals, plot_live_residuals
from ..pressure_solver.helpers.rhs_construction import get_rhs

class SimpleSolver(BaseAlgorithm):
    """
    SIMPLE algorithm implementation.
    
    The SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm
    is a widely used method for solving the Navier-Stokes equations for incompressible flows.
    It uses a predictor-corrector approach to handle the pressure-velocity coupling.
    """
    def __init__(self, mesh, fluid, pressure_solver=None, momentum_solver=None, 
                 velocity_updater=None, boundary_conditions=None, 
                 alpha_p=0.3, alpha_u=0.7, fix_lid_corners=False):
        """
        Initialize the SIMPLE solver.
        
        Parameters:
        -----------
        mesh : StructuredMesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        pressure_solver : PressureSolver, optional
            Solver for pressure equation
        momentum_solver : MomentumSolver, optional
            Solver for momentum equations
        velocity_updater : VelocityUpdater, optional
            Method to update velocities
        boundary_conditions : dict or BoundaryConditionManager, optional
            Boundary conditions
        alpha_p, alpha_u : float
            Relaxation factors for pressure and velocity
        fix_lid_corners : bool
            Whether to set the corners of the lid to stationary for better stability
        """
        self.alpha_p = alpha_p
        self.alpha_u = alpha_u
        self.fix_lid_corners = fix_lid_corners
        self.mass_residual_history = []  # Initialize mass residual history
        self.u_old = None  # Store old u values
        self.v_old = None  # Store old v values
        self.p_old = None  # Store old p values
        
        # Coefficient storage for relaxed momentum residual calculations
        self.u_coeffs_relaxed = None # Relaxed x-momentum coefficients (a_p, a_nb, source)
        self.v_coeffs_relaxed = None # Relaxed y-momentum coefficients (a_p, a_nb, source)
        # NOTE: p_coeffs_info removed, pressure coeffs calculated on-the-fly for residual
        # self.p_coeffs_info = None 

        # Initialize residual histories
        self.x_momentum_residuals = []  # Track x-momentum residuals
        self.y_momentum_residuals = []  # Track y-momentum residuals
        self.continuity_residuals = []  # Track continuity residuals
        
        super().__init__(mesh, fluid, pressure_solver, momentum_solver, 
                         velocity_updater, boundary_conditions)
    
    def calculate_physical_u_residual(self, u, v, p):
        """
        Compute L2 norm of the physical (unrelaxed) x-momentum residual.
        """
        nx, ny = self.mesh.get_dimensions()
        self.momentum_solver.calculate_coefficients(
            self.mesh, self.fluid, u, v, p, self.bc_manager, relaxation_factor=1.0
        )

        sum_squared_residual = 0.0
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                a_e = self.momentum_solver.u_a_e[i, j]
                a_w = self.momentum_solver.u_a_w[i, j]
                a_n = self.momentum_solver.u_a_n[i, j]
                a_s = self.momentum_solver.u_a_s[i, j]
                a_p = self.momentum_solver.u_a_p[i, j]
                source = self.momentum_solver.u_source[i, j]

                lhs = a_p * u[i, j]
                rhs = (
                    a_e * u[i+1, j] +
                    a_w * u[i-1, j] +
                    a_n * u[i, j+1] +
                    a_s * u[i, j-1] +
                    source
                )
                residual = lhs - rhs
                sum_squared_residual += residual ** 2

        return np.sqrt(sum_squared_residual)


    def calculate_physical_v_residual(self, u, v, p):
        """
        Compute L2 norm of the physical (unrelaxed) y-momentum residual.
        """
        nx, ny = self.mesh.get_dimensions()
        self.momentum_solver.calculate_coefficients(
            self.mesh, self.fluid, u, v, p, self.bc_manager, relaxation_factor=1.0
        )

        sum_squared_residual = 0.0
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                a_e = self.momentum_solver.v_a_e[i, j]
                a_w = self.momentum_solver.v_a_w[i, j]
                a_n = self.momentum_solver.v_a_n[i, j]
                a_s = self.momentum_solver.v_a_s[i, j]
                a_p = self.momentum_solver.v_a_p[i, j]
                source = self.momentum_solver.v_source[i, j]

                lhs = a_p * v[i, j]
                rhs = (
                    a_e * v[i+1, j] +
                    a_w * v[i-1, j] +
                    a_n * v[i, j+1] +
                    a_s * v[i, j-1] +
                    source
                )
                residual = lhs - rhs
                sum_squared_residual += residual ** 2

        return np.sqrt(sum_squared_residual)
    
    def calculate_continuity_residual(self, u, v):
        """
        Calculate the L2 norm of the continuity equation residual.
        
        This implementation calculates the mass imbalance for each cell,
        which for incompressible flow should be zero (divergence-free).
        
        Parameters:
        -----------
        u, v : ndarray
            Velocity fields
            
        Returns:
        --------
        float
            L2 norm of the continuity equation residual
        """
        # Get mesh dimensions
        nx, ny = self.mesh.get_dimensions()
        dx, dy = self.mesh.get_cell_sizes()
        
        # Initialize sum for residual
        sum_residual = 0.0
        
        # Calculate mass imbalance for interior cells
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Calculate velocities at cell faces
                u_e = u[i+1, j]    # East face u-velocity
                u_w = u[i, j]      # West face u-velocity
                v_n = v[i, j+1]    # North face v-velocity 
                v_s = v[i, j]      # South face v-velocity
                
                # Calculate mass imbalance (divergence of velocity field)
                # For incompressible flow, this should be zero (div(u) = 0)
                mdot = (u_e - u_w) * dy + (v_n - v_s) * dx
                
                # Square the residual and add to sum
                sum_residual += mdot**2
        
        # Return L2 norm of residuals
        return np.sqrt(sum_residual)
    
    def calculate_relaxed_u_residual(self, u):
        """
        Compute L2 norm of the relaxed x-momentum residual.
        Uses coefficients stored from the momentum solver step.
        Residual = a_p*u - (sum(a_nb*u_nb) + source)
        """
        if self.u_coeffs_relaxed is None:
            print("Warning: Relaxed U coefficients not available for residual calculation.")
            # Fallback or initial calculation might be needed here
            # For now, return a high value or calculate physical residual as fallback
            return self.calculate_physical_u_residual(u, self.v, self.p) # Fallback

        nx, ny = self.mesh.get_dimensions()
        sum_squared_residual = 0.0
        coeffs = self.u_coeffs_relaxed

        # Ensure coeffs are available
        a_e = coeffs.get('a_e')
        a_w = coeffs.get('a_w')
        a_n = coeffs.get('a_n')
        a_s = coeffs.get('a_s')
        a_p = coeffs.get('a_p')
        source = coeffs.get('source')

        if any(c is None for c in [a_e, a_w, a_n, a_s, a_p, source]):
             raise ValueError("Relaxed U coefficients dictionary is missing expected keys.")


        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # Using stored relaxed coefficients
                lhs = a_p[i, j] * u[i, j]
                rhs = (
                    a_e[i, j] * u[i+1, j] +
                    a_w[i, j] * u[i-1, j] +
                    a_n[i, j] * u[i, j+1] +
                    a_s[i, j] * u[i, j-1] +
                    source[i, j]
                )
                residual = lhs - rhs
                sum_squared_residual += residual ** 2

        return np.sqrt(sum_squared_residual)


    def calculate_relaxed_v_residual(self, v):
        """
        Compute L2 norm of the relaxed y-momentum residual.
        Uses coefficients stored from the momentum solver step.
        Residual = a_p*v - (sum(a_nb*v_nb) + source)
        """
        if self.v_coeffs_relaxed is None:
            print("Warning: Relaxed V coefficients not available for residual calculation.")
            # Fallback or initial calculation might be needed here
            return self.calculate_physical_v_residual(self.u, v, self.p) # Fallback

        nx, ny = self.mesh.get_dimensions()
        sum_squared_residual = 0.0
        coeffs = self.v_coeffs_relaxed

        # Ensure coeffs are available
        a_e = coeffs.get('a_e')
        a_w = coeffs.get('a_w')
        a_n = coeffs.get('a_n')
        a_s = coeffs.get('a_s')
        a_p = coeffs.get('a_p')
        source = coeffs.get('source')

        if any(c is None for c in [a_e, a_w, a_n, a_s, a_p, source]):
             raise ValueError("Relaxed V coefficients dictionary is missing expected keys.")


        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # Using stored relaxed coefficients
                lhs = a_p[i, j] * v[i, j]
                rhs = (
                    a_e[i, j] * v[i+1, j] +
                    a_w[i, j] * v[i-1, j] +
                    a_n[i, j] * v[i, j+1] +
                    a_s[i, j] * v[i, j-1] +
                    source[i, j]
                )
                residual = lhs - rhs
                sum_squared_residual += residual ** 2

        return np.sqrt(sum_squared_residual)

    def calculate_pressure_correction_residual(self):
        """
        Calculate the L2 norm of the pressure correction equation residual (A*pc - b).
        Coefficients (A) and source term (b) are computed on-the-fly using 
        stored intermediate fields (u*, v*, d_u, d_v) and the current p_prime.
        """
        nx, ny = self.mesh.get_dimensions()
        dx, dy = self.mesh.get_cell_sizes()
        rho = self.fluid.get_density()
        
        # Retrieve stored fields needed for calculation
        u_star = self._tmp_u_star
        v_star = self._tmp_v_star
        d_u = self._tmp_d_u
        d_v = self._tmp_d_v
        p_prime = self._tmp_p_prime

        # --- Calculate Pressure Coefficients (A) --- 
        # Adapted from GaussSeidelSolver._precompute_coefficients
        aE = np.zeros((nx, ny))
        aW = np.zeros((nx, ny))
        aN = np.zeros((nx, ny))
        aS = np.zeros((nx, ny))
        aP = np.zeros((nx, ny))

        # Off-diagonal coefficients for interior cells
        aE[:-1, :] = rho * d_u[1:nx, :] * dy
        aW[1:, :] = rho * d_u[1:nx, :] * dy
        aN[:, :-1] = rho * d_v[:, 1:ny] * dx
        aS[:, 1:] = rho * d_v[:, 1:ny] * dx
        
        # Apply zero-gradient boundary conditions implicitly by adding to diagonal
        # West boundary (i=0)
        aP[0, :] += aE[0, :]
        # East boundary (i=nx-1)
        aP[nx-1, :] += aW[nx-1, :]
        # South boundary (j=0)
        aP[:, 0] += aN[:, 0]
        # North boundary (j=ny-1)
        aP[:, ny-1] += aS[:, ny-1]
        
        # Diagonal term is sum of all coefficients (including boundary adjustments)
        aP += aE + aW + aN + aS
        
        # --- Calculate Source Term (b = mdot) --- 
        source_b = get_rhs(nx, ny, dx, dy, rho, u_star, v_star)
        # Reshape the 1D source term to 2D for easier indexing
        source_b_2d = source_b.reshape((nx, ny), order='F')

        # --- Calculate Residual Norm ||A*pc - b||_2 --- 
        sum_squared_residual = 0.0
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                 # Calculate A_p * pc term using calculated coefficients
                 ap_pc = (
                    aE[i, j] * p_prime[i+1, j] +
                    aW[i, j] * p_prime[i-1, j] +
                    aN[i, j] * p_prime[i, j+1] +
                    aS[i, j] * p_prime[i, j-1] +
                    aP[i, j] * p_prime[i, j]
                 )
                 # Calculate residual: A*pc - b
                 # Note: The sign convention might differ from solver implementation depending on
                 # how RHS is defined (e.g., mdot or -mdot). get_rhs returns -mdot.
                 # The residual here is A*p - b, where b = get_rhs = -mdot.
                 # In Rust, source_p is mdot, and residual is A*p - source_p.
                 # Let's match Rust: calculate A*p - mdot = A*p + b
                 # Use the reshaped 2D source term
                 residual = ap_pc + source_b_2d[i, j]
                 sum_squared_residual += residual ** 2

        return np.sqrt(sum_squared_residual)

    def solve(self, max_iterations=1000, tolerance=1e-6, save_profile=True, profile_dir='results/profiles', 
              track_infinity_norm=False, infinity_norm_interval=10, use_l2_norm=False):
        self.profiler.start()
        nx, ny = self.mesh.get_dimensions()
        p_star = self.p.copy()
        p_prime = np.zeros((nx, ny))
        self.residual_history = []
        self.momentum_residual_history = []
        self.pressure_residual_history = []
        self.infinity_norm_history = []

        iteration = 1
        # Initialize absolute residuals for the first iteration check and convergence
        u_res_abs = v_res_abs = p_res_abs = 1e6 
        # Use absolute residuals for the loop condition check and reporting
        total_res_check = max(u_res_abs, v_res_abs, p_res_abs) 
        
        print(f"Using α_p = {self.alpha_p}, α_u = {self.alpha_u} with absolute relaxed residuals.") # Updated print message

        try:
            while iteration <= max_iterations and total_res_check > tolerance:
                self.u_old = self.u.copy()
                self.v_old = self.v.copy()
                self.p_old = self.p.copy()

                u_star, d_u = self.momentum_solver.solve_u_momentum(
                    self.mesh, self.fluid, self.u, self.v, p_star,
                    relaxation_factor=self.alpha_u,
                    boundary_conditions=self.bc_manager
                )
                # Store relaxed coefficients for U residual calculation
                # NOTE: Assumes momentum_solver stores these attributes after solve_u_momentum
                self.u_coeffs_relaxed = {
                    'a_e': self.momentum_solver.u_a_e.copy(),
                    'a_w': self.momentum_solver.u_a_w.copy(),
                    'a_n': self.momentum_solver.u_a_n.copy(),
                    'a_s': self.momentum_solver.u_a_s.copy(),
                    'a_p': self.momentum_solver.u_a_p.copy(),
                    'source': self.momentum_solver.u_source.copy()
                }


                v_star, d_v = self.momentum_solver.solve_v_momentum(
                    self.mesh, self.fluid, self.u, self.v, p_star,
                    relaxation_factor=self.alpha_u,
                    boundary_conditions=self.bc_manager
                )
                # Store relaxed coefficients for V residual calculation
                # NOTE: Assumes momentum_solver stores these attributes after solve_v_momentum
                self.v_coeffs_relaxed = {
                    'a_e': self.momentum_solver.v_a_e.copy(),
                    'a_w': self.momentum_solver.v_a_w.copy(),
                    'a_n': self.momentum_solver.v_a_n.copy(),
                    'a_s': self.momentum_solver.v_a_s.copy(),
                    'a_p': self.momentum_solver.v_a_p.copy(),
                    'source': self.momentum_solver.v_source.copy()
                }

                p_prime = self.pressure_solver.solve(self.mesh, u_star, v_star, d_u, d_v, p_star)

                # Store intermediate fields needed for pressure residual calculation
                self._tmp_u_star = u_star
                self._tmp_v_star = v_star
                self._tmp_d_u = d_u
                self._tmp_d_v = d_v
                self._tmp_p_prime = p_prime # Store current p_prime

                self.p = p_star + self.alpha_p * p_prime
                self._enforce_pressure_boundary_conditions()
                p_star = self.p.copy()

                self.u, self.v = self.velocity_updater.update_velocity(
                    self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
                )

                # Calculate absolute residuals based on relaxed coefficients / pressure correction eq.
                u_res_abs = self.calculate_relaxed_u_residual(self.u)
                v_res_abs = self.calculate_relaxed_v_residual(self.v)
                # Pass p_prime, the variable solved for in the pressure correction equation
                p_res_abs = self.calculate_pressure_correction_residual() # No args needed now

                # Store absolute residuals
                self.x_momentum_residuals.append(u_res_abs)
                self.y_momentum_residuals.append(v_res_abs)
                self.continuity_residuals.append(p_res_abs) # Continuity residual is represented by pressure correction

                # Total absolute residual for convergence check and history
                total_res_abs = max(u_res_abs, v_res_abs, p_res_abs) 
                self.residual_history.append(total_res_abs)
                self.momentum_residual_history.append(max(u_res_abs, v_res_abs))
                self.pressure_residual_history.append(p_res_abs) # Store absolute pressure residual

                # Update the value used for the loop condition check
                total_res_check = total_res_abs

                if track_infinity_norm and (iteration % infinity_norm_interval == 0 or total_res_check < tolerance):
                    try:
                        inf_err = calculate_infinity_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                        l2_err = calculate_l2_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                        self.infinity_norm_history.append(l2_err if use_l2_norm else inf_err)
                        print(f"Iteration {iteration}: ∞-norm error = {inf_err:.3e}, L2 error = {l2_err:.3e}")
                    except Exception as e:
                        print(f"Error calc failed: {e}")

                # Print absolute residuals
                print(f"[{iteration}] Absolute Residuals -> u: {u_res_abs:.3e}, v: {v_res_abs:.3e}, continuity: {p_res_abs:.3e}")
                iteration += 1

        except KeyboardInterrupt:
            print("Interrupted by user.")

        # Ensure total_res_check holds the last calculated absolute residual for final reporting
        final_residual_to_report = total_res_check

        self.profiler.set_iterations(iteration - 1)
        self.profiler.set_convergence_info(
            tolerance=tolerance,
            final_residual=final_residual_to_report, # Store the final absolute residual
            residual_history=self.residual_history, # Already contains absolute residuals
            converged=(final_residual_to_report < tolerance)
        )

        if hasattr(self.pressure_solver, 'get_solver_info'):
            info = self.pressure_solver.get_solver_info()
            self.profiler.set_pressure_solver_info(
                solver_name=info.get('name', 'unknown'),
                inner_iterations=info.get('inner_iterations_history'),
                convergence_rate=info.get('convergence_rate'),
                solver_specific=info.get('solver_specific')
            )

        self.profiler.end()

        result = SimulationResult(
            self.u, self.v, self.p, self.mesh,
            iterations=iteration-1,
            residuals=self.residual_history,
            momentum_residuals=self.momentum_residual_history,
            pressure_residuals=self.pressure_residual_history,
            divergence=self.calculate_divergence(),
            reynolds=self.fluid.get_reynolds_number()
        )

        if save_profile:
            os.makedirs(profile_dir, exist_ok=True)
            filename = os.path.join(profile_dir, f"SIMPLE_Re{int(self.fluid.get_reynolds_number())}_mesh{nx}x{ny}_profile.h5")
            print(f"Saved profile to {self.save_profiling_data(filename)}")

        return result