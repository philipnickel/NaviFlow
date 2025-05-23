"""
SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm implementation
with residual dictionary handling.
""" 

import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from .base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult
from ...postprocessing.validation.cavity_flow import calculate_infinity_norm_error, calculate_l2_norm_error
from ...postprocessing.visualization import plot_final_residuals
from ..pressure_solver.helpers.rhs_construction import get_rhs

class SimpleSolverDict(BaseAlgorithm):
    """
    SIMPLE algorithm implementation with residual dictionaries.
    
    This version of the SIMPLE algorithm uses dictionaries for residual information
    rather than calculating residuals in multiple places.
    
    .. deprecated:: Use SimpleSolver instead, which now uses dictionary-based 
       residual handling by default.
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
        warnings.warn(
            "SimpleSolverDict is deprecated. Use SimpleSolver instead, which now uses dictionary-based residual handling by default.",
            DeprecationWarning, stacklevel=2
        )
        
        self.alpha_p = alpha_p
        self.alpha_u = alpha_u
        self.fix_lid_corners = fix_lid_corners
        self.mass_residual_history = []  # Initialize mass residual history
        self.u_old = None  # Store old u values
        self.v_old = None  # Store old v values
        self.p_old = None  # Store old p values

        # Initialize residual histories
        self.x_momentum_residuals_relaxed = []
        self.x_momentum_residuals_unrelaxed = []
        self.y_momentum_residuals_relaxed = []
        self.y_momentum_residuals_unrelaxed = []
        self.continuity_residuals = []  # Track continuity residuals (norms)
        
        # Add absolute residual history tracking
        self.x_momentum_abs_relaxed = []
        self.x_momentum_abs_unrelaxed = []
        self.y_momentum_abs_relaxed = []
        self.y_momentum_abs_unrelaxed = []
        self.pressure_abs_residuals = []
        
        # Variables to store final residual fields (should be unrelaxed)
        self._final_u_residual_field = None
        self._final_v_residual_field = None
        self._final_p_residual_field = None
        
        super().__init__(mesh, fluid, pressure_solver, momentum_solver, 
                         velocity_updater, boundary_conditions)
    
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
        
        print(f"Using α_p = {self.alpha_p}, α_u = {self.alpha_u} with normalized algebraic residuals.") # Updated print message

        try:
            while iteration <= max_iterations and total_res_check > tolerance:
                self.u_old = self.u.copy()
                self.v_old = self.v.copy()
                self.p_old = self.p.copy()

                # Solve momentum equations and get comprehensive residual info
                u_star, d_u, u_res_info = self.momentum_solver.solve_u_momentum(
                    self.mesh, self.fluid, self.u, self.v, p_star,
                    relaxation_factor=self.alpha_u,
                    boundary_conditions=self.bc_manager,
                    return_dict=True
                )

                v_star, d_v, v_res_info = self.momentum_solver.solve_v_momentum(
                    self.mesh, self.fluid, self.u, self.v, p_star,
                    relaxation_factor=self.alpha_u,
                    boundary_conditions=self.bc_manager,
                    return_dict=True
                )

                # Store intermediate fields needed for pressure residual calculation
                self._tmp_u_star = u_star
                self._tmp_v_star = v_star
                self._tmp_d_u = d_u
                self._tmp_d_v = d_v

                # Solve pressure correction equation with residual info
                p_prime, p_res_info = self.pressure_solver.solve(self.mesh, u_star, v_star, d_u, d_v, p_star, return_dict=True)
                
                # Update pressure with relaxation
                self.p = p_star + self.alpha_p * p_prime
                self._enforce_pressure_boundary_conditions()
                p_star = self.p.copy()

                # Update velocities
                self.u, self.v = self.velocity_updater.update_velocity(
                    self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
                )
                
                # Extract residual information from dictionaries
                u_res_norm_relaxed = u_res_info['relaxed_norm']
                u_res_norm_unrelaxed = u_res_info['unrelaxed_norm']
                v_res_norm_relaxed = v_res_info['relaxed_norm']
                v_res_norm_unrelaxed = v_res_info['unrelaxed_norm']
                p_res_abs = p_res_info['norm']
                
                # Extract absolute residuals
                u_abs_l2_relaxed = u_res_info['relaxed_abs']
                u_abs_l2_unrelaxed = u_res_info['unrelaxed_abs']
                v_abs_l2_relaxed = v_res_info['relaxed_abs']
                v_abs_l2_unrelaxed = v_res_info['unrelaxed_abs']
                p_abs_l2 = p_res_info['abs']
                
                # Store residual fields
                u_res_field_unrelaxed = u_res_info['unrelaxed_field']
                v_res_field_unrelaxed = v_res_info['unrelaxed_field']
                p_res_field = p_res_info['field']
                
                # Store normalized algebraic residual norms
                self.x_momentum_residuals_relaxed.append(u_res_norm_relaxed)
                self.x_momentum_residuals_unrelaxed.append(u_res_norm_unrelaxed)
                self.y_momentum_residuals_relaxed.append(v_res_norm_relaxed)
                self.y_momentum_residuals_unrelaxed.append(v_res_norm_unrelaxed)
                self.continuity_residuals.append(p_res_abs) # Continuity residual is represented by pressure correction

                # Store absolute L2 residual norms
                self.x_momentum_abs_relaxed.append(u_abs_l2_relaxed)
                self.y_momentum_abs_relaxed.append(v_abs_l2_relaxed)
                self.x_momentum_abs_unrelaxed.append(u_abs_l2_unrelaxed)
                self.y_momentum_abs_unrelaxed.append(v_abs_l2_unrelaxed)
                self.pressure_abs_residuals.append(p_abs_l2)

                # Define convergence criteria based on UNRELAXED residuals
                total_res_unrelaxed = max(u_res_norm_unrelaxed, v_res_norm_unrelaxed, p_res_abs)
                total_res_check = total_res_unrelaxed # Use unrelaxed for convergence check

                # For general history tracking, use unrelaxed (matches convergence check)
                self.residual_history.append(total_res_unrelaxed)
                # Store combined momentum residual norms for simpler plotting/reporting if needed
                self.momentum_residual_history.append(max(u_res_norm_unrelaxed, v_res_norm_unrelaxed))
                self.pressure_residual_history.append(p_res_abs)

                if track_infinity_norm and (iteration % infinity_norm_interval == 0 or total_res_check < tolerance):
                    try:
                        inf_err = calculate_infinity_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                        l2_err = calculate_l2_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                        self.infinity_norm_history.append(l2_err if use_l2_norm else inf_err)
                        print(f"Iteration {iteration}: ∞-norm error = {inf_err:.3e}, L2 error = {l2_err:.3e}")
                    except Exception as e:
                        print(f"Error calc failed: {e}")

                # Print both normalized and absolute L2 norms
                print(f"[{iteration}] Rel Alg Res -> u: {u_res_norm_relaxed:.3e}/{u_res_norm_unrelaxed:.3e}, "
                      f"v: {v_res_norm_relaxed:.3e}/{v_res_norm_unrelaxed:.3e}, "
                      f"p: {p_res_abs:.3e}")
                print(f"[{iteration}] Abs L2 Norms -> u: {u_abs_l2_relaxed:.3e}/{u_abs_l2_unrelaxed:.3e}, "
                      f"v: {v_abs_l2_relaxed:.3e}/{v_abs_l2_unrelaxed:.3e}, "
                      f"p: {p_abs_l2:.3e}")
                
                iteration += 1

        except KeyboardInterrupt:
            print("Interrupted by user.")

        # After the loop, store the UNRELAXED residual fields from the last completed iteration
        self._final_u_residual_field = u_res_field_unrelaxed
        self._final_v_residual_field = v_res_field_unrelaxed
        self._final_p_residual_field = p_res_field

        # Ensure total_res_check holds the final UNRELAXED residual for reporting
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
            reynolds=self.fluid.get_reynolds_number(),
            # Pass final residual fields
            u_residual_field=self._final_u_residual_field,
            v_residual_field=self._final_v_residual_field,
            p_residual_field=self._final_p_residual_field
        )

        # Add relaxed/unrelaxed histories to the result object for plotting
        result.add_history('u_momentum_relaxed', self.x_momentum_residuals_relaxed)
        result.add_history('u_momentum_unrelaxed', self.x_momentum_residuals_unrelaxed)
        result.add_history('v_momentum_relaxed', self.y_momentum_residuals_relaxed)
        result.add_history('v_momentum_unrelaxed', self.y_momentum_residuals_unrelaxed)
        result.add_history('continuity', self.continuity_residuals)
        result.add_history('total_unrelaxed', self.residual_history)
        
        # Add absolute residual histories
        result.add_history('u_abs_relaxed', self.x_momentum_abs_relaxed)
        result.add_history('v_abs_relaxed', self.y_momentum_abs_relaxed)
        result.add_history('u_abs_unrelaxed', self.x_momentum_abs_unrelaxed)
        result.add_history('v_abs_unrelaxed', self.y_momentum_abs_unrelaxed)
        result.add_history('p_abs', self.pressure_abs_residuals)

        self.profiler.start_section() # Start timing finalization
        if save_profile:
            os.makedirs(profile_dir, exist_ok=True)
            filename = os.path.join(profile_dir, f"SIMPLE_Re{int(self.fluid.get_reynolds_number())}_mesh{nx}x{ny}_profile.h5")
            print(f"Saved profile to {self.save_profiling_data(filename)}")
        self.profiler.end_section("Finalization") # End timing finalization

        return result 