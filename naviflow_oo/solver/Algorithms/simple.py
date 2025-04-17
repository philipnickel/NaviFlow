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
        
        # Coefficient storage for residual calculations
        self.u_coeffs = None  # x-momentum coefficients
        self.v_coeffs = None  # y-momentum coefficients
        self.p_coeffs = None  # pressure equation coefficients
        
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
        u_res = v_res = p_res = total_res = 1e6
        print(f"Using α_p = {self.alpha_p}, α_u = {self.alpha_u} with physical (α=1) residuals.")

        try:
            while iteration <= max_iterations and max(u_res, v_res, p_res) > tolerance:
                self.u_old = self.u.copy()
                self.v_old = self.v.copy()
                self.p_old = self.p.copy()

                u_star, d_u = self.momentum_solver.solve_u_momentum(
                    self.mesh, self.fluid, self.u, self.v, p_star,
                    relaxation_factor=self.alpha_u,
                    boundary_conditions=self.bc_manager
                )

                v_star, d_v = self.momentum_solver.solve_v_momentum(
                    self.mesh, self.fluid, self.u, self.v, p_star,
                    relaxation_factor=self.alpha_u,
                    boundary_conditions=self.bc_manager
                )

                p_prime = self.pressure_solver.solve(self.mesh, u_star, v_star, d_u, d_v, p_star)
                self.p = p_star + self.alpha_p * p_prime
                self._enforce_pressure_boundary_conditions()
                p_star = self.p.copy()

                self.u, self.v = self.velocity_updater.update_velocity(
                    self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
                )

                # ✅ Use α = 1.0 to compute physical residuals
                u_res = self.calculate_physical_u_residual(self.u, self.v, self.p)
                v_res = self.calculate_physical_v_residual(self.u, self.v, self.p)
                p_res = self.calculate_continuity_residual(self.u, self.v)

                self.x_momentum_residuals.append(u_res)
                self.y_momentum_residuals.append(v_res)
                self.continuity_residuals.append(p_res)

                total_res = max(u_res, v_res, p_res)
                self.residual_history.append(total_res)
                self.momentum_residual_history.append(max(u_res, v_res))
                self.pressure_residual_history.append(p_res)

                if track_infinity_norm and (iteration % infinity_norm_interval == 0 or total_res < tolerance):
                    try:
                        inf_err = calculate_infinity_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                        l2_err = calculate_l2_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                        self.infinity_norm_history.append(l2_err if use_l2_norm else inf_err)
                        print(f"Iteration {iteration}: ∞-norm error = {inf_err:.3e}, L2 error = {l2_err:.3e}")
                    except Exception as e:
                        print(f"Error calc failed: {e}")

                print(f"[{iteration}] Residuals -> u: {u_res:.3e}, v: {v_res:.3e}, continuity: {p_res:.3e}")
                iteration += 1

        except KeyboardInterrupt:
            print("Interrupted by user.")

        self.profiler.set_iterations(iteration - 1)
        self.profiler.set_convergence_info(
            tolerance=tolerance,
            final_residual=total_res,
            residual_history=self.residual_history,
            converged=(total_res < tolerance)
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