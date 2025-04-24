"""
SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm implementation.
""" 

import numpy as np
import os
import matplotlib.pyplot as plt
from .base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult
from ...postprocessing.validation.cavity_flow import calculate_infinity_norm_error, calculate_l2_norm_error
from ...postprocessing.visualization import plot_final_residuals
from ..pressure_solver.helpers.rhs_construction import get_rhs

class SimpleSolver(BaseAlgorithm):
    """
    SIMPLE algorithm implementation.
    
    The SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm
    is a widely used method for solving the Navier-Stokes equations for incompressible flows.
    It uses a predictor-corrector approach to handle the pressure-velocity coupling.
    Uses dictionary-based residual information for improved code structure.
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
        
        # Initialize/reset residual histories - only store what's needed
        self.residual_history = []  # Overall convergence 
        
        # Store only relative residual norms for each equation
        self.x_momentum_rel_norms = []
        self.y_momentum_rel_norms = []
        self.pressure_rel_norms = []
        
        # For infinity norm tracking if needed
        self.infinity_norm_history = []
        
        # Final residual fields
        self._final_u_residual_field = None
        self._final_v_residual_field = None
        self._final_p_residual_field = None
        
        iteration = 1
        # Initialize residuals for convergence check
        u_rel_norm = v_rel_norm = p_rel_norm = 1.0
        total_res_check = max(u_rel_norm, v_rel_norm, p_rel_norm)
        
        print(f"Using α_p = {self.alpha_p}, α_u = {self.alpha_u}")

        stall_check_window = 20
        stall_threshold = 1e-8
        recent_total_residuals = []


        try:
            while iteration <= max_iterations and total_res_check > tolerance:
                # Store previous solution
                self.u_old = self.u.copy()
                self.v_old = self.v.copy()
                self.p_old = self.p.copy()

                # Solve momentum equations
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

                # Save intermediate fields for pressure equation
                self._tmp_u_star = u_star
                self._tmp_v_star = v_star
                self._tmp_d_u = d_u
                self._tmp_d_v = d_v

                # Solve pressure correction equation
                p_prime, p_res_info = self.pressure_solver.solve(
                    self.mesh, u_star, v_star, d_u , d_v , p_star, 
                    return_dict=True
                )
                
                # Update pressure with relaxation
                self.p = p_star + self.alpha_p * p_prime
                self._enforce_pressure_boundary_conditions()
                p_star = self.p.copy()

                # Update velocities
                self.u, self.v = self.velocity_updater.update_velocity(
                    self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
                )
   

                # Extract relative norms for convergence check
                u_rel_norm = u_res_info['rel_norm']
                v_rel_norm = v_res_info['rel_norm']
                p_rel_norm = p_res_info['rel_norm']
                
                # Save residual fields for final visualization
                u_res_field = u_res_info['field']
                v_res_field = v_res_info['field']
                p_res_field = p_res_info['field']
                
                # Store relative norms
                self.x_momentum_rel_norms.append(u_rel_norm)
                self.y_momentum_rel_norms.append(v_rel_norm)
                self.pressure_rel_norms.append(p_rel_norm)

                # Define convergence criteria using relative norms
                total_res_check = max(u_rel_norm, v_rel_norm)#, p_rel_norm)

                # Store total residual for history tracking
                self.residual_history.append(total_res_check)

                # Track infinity norm if requested
                if track_infinity_norm and (iteration % infinity_norm_interval == 0 or total_res_check < tolerance):
                    try:
                        inf_err = calculate_infinity_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                        l2_err = calculate_l2_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                        self.infinity_norm_history.append(l2_err if use_l2_norm else inf_err)
                        print(f"Iteration {iteration}: ∞-norm error = {inf_err:.3e}, L2 error = {l2_err:.3e}")
                    except Exception as e:
                        print(f"Error calc failed: {e}")

                # Print relative norms
                print(f"[{iteration}] Relative L2 norms: u: {u_rel_norm:.3e}, "
                      f"v: {v_rel_norm:.3e}, p: {p_rel_norm:.3e}")


                # Stall check
                                # Store total residual for history tracking
                self.residual_history.append(total_res_check)

                # Update rolling residual history for stall detection
                recent_total_residuals.append(total_res_check)
                if len(recent_total_residuals) > stall_check_window:
                    recent_total_residuals.pop(0)
                    res_change = max(recent_total_residuals) - min(recent_total_residuals)
                    avg_res = np.mean(recent_total_residuals)
                    if avg_res > 0:  # Avoid divide-by-zero
                        rel_change = res_change / avg_res
                        if rel_change < 0.001:  # 0.1% relative change
                            print(f"Residuals have stalled (<0.1% change) over the last {stall_check_window} iterations. Stopping early.")
                            break


                
                iteration += 1

        except KeyboardInterrupt:
            print("Interrupted by user.")

        # Store the final residual fields
        self._final_u_residual_field = u_res_field
        self._final_v_residual_field = v_res_field
        self._final_p_residual_field = p_res_field

        # For reporting
        final_residual = total_res_check

        self.profiler.set_iterations(iteration - 1)
        self.profiler.set_convergence_info(
            tolerance=tolerance,
            final_residual=final_residual,
            residual_history=self.residual_history,
            converged=(final_residual < tolerance)
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

        # Create simulation result with the relevant residual histories
        result = SimulationResult(
            self.u, self.v, self.p, self.mesh,
            iterations=iteration-1,
            residuals=self.residual_history,
            reynolds=self.fluid.get_reynolds_number(),
            # Pass final residual fields
            u_residual_field=self._final_u_residual_field,
            v_residual_field=self._final_v_residual_field,
            p_residual_field=self._final_p_residual_field
        )

        # Add only the necessary residual histories to the result
        result.add_history('u_rel_norm', self.x_momentum_rel_norms)
        result.add_history('v_rel_norm', self.y_momentum_rel_norms)
        result.add_history('p_rel_norm', self.pressure_rel_norms)
        result.add_history('total_rel_norm', self.residual_history)

        self.profiler.start_section() # Start timing finalization
        if save_profile:
            os.makedirs(profile_dir, exist_ok=True)
            filename = os.path.join(profile_dir, f"SIMPLE_Re{int(self.fluid.get_reynolds_number())}_mesh{nx}x{ny}_profile.h5")
            print(f"Saved profile to {self.save_profiling_data(filename)}")
        self.profiler.end_section("Finalization") # End timing finalization

        return result