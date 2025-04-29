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
    
    def initialize_fields(self):
        """Initialize velocity and pressure fields as 1D arrays."""
        # u_shape, v_shape, p_shape = self.mesh.get_field_shapes() # Old
        n_cells = self.mesh.n_cells
        
        # Initialize fields as 1D arrays
        self.u = np.zeros(n_cells)
        self.v = np.zeros(n_cells)
        self.p = np.zeros(n_cells)
        
        # Apply boundary conditions
        # self._enforce_velocity_boundary_conditions()
        # self._enforce_pressure_boundary_conditions()
        self.apply_boundary_conditions() # Call the base class method
    
    def solve(self, max_iterations=1000, tolerance=1e-6, save_profile=True, profile_dir='results/profiles', 
              track_infinity_norm=False, infinity_norm_interval=10, use_l2_norm=False):
        self.profiler.start()
        # nx, ny = self.mesh.get_dimensions() # Not needed if fields are 1D
        p_star = self.p.copy() # p is now 1D
        # p_prime = np.zeros((nx, ny)) # Initialize p_prime as 1D later
        p_prime = np.zeros_like(p_star) # Initialize as 1D
        
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

        stall_check_window = 50
        stall_threshold = 1e-8
        recent_total_residuals = []


        try:
            while iteration <= max_iterations: #and total_res_check > tolerance:
                # Store previous solution
                self.u_old = self.u.copy()
                self.v_old = self.v.copy()
                self.p_old = self.p.copy()

                # --- Pre-momentum solve sanity ---
                if np.isnan(self.p).any() or np.isinf(self.p).any():
                    raise ValueError(f"[{iteration}] NaN/Inf detected in pressure field p before momentum solve")

                # --- New debug print to catch bad input early ---
                print(f"[{iteration}] Checking fields before u-momentum:")
                print(f"  u min/max/mean: {self.u.min():.2e} / {self.u.max():.2e} / {self.u.mean():.2e}")
                print(f"  v min/max/mean: {self.v.min():.2e} / {self.v.max():.2e} / {self.v.mean():.2e}")
                print(f"  p min/max/mean: {p_star.min():.2e} / {p_star.max():.2e} / {p_star.mean():.2e}")
                if np.isnan(self.u).any() or np.isinf(self.u).any():
                    raise ValueError(f"[{iteration}] NaN/Inf detected in input u field BEFORE solving u-momentum")
                if np.isnan(self.v).any() or np.isinf(self.v).any():
                    raise ValueError(f"[{iteration}] NaN/Inf detected in input v field BEFORE solving u-momentum")
                if np.isnan(p_star).any() or np.isinf(p_star).any():
                    raise ValueError(f"[{iteration}] NaN/Inf detected in input p_star field BEFORE solving u-momentum")
                # --------------------------------------------------

                # Solve momentum equations
                u_star, d_u, u_res_info = self.momentum_solver.solve_u_momentum(
                    self.mesh, self.fluid, self.u, self.v, p_star,
                    relaxation_factor=self.alpha_u,
                    return_dict=True
                )

                # --- Post-u momentum sanity ---
                if np.isnan(u_star).any() or np.isinf(u_star).any() or np.isnan(d_u).any() or np.isinf(d_u).any():
                    raise ValueError(f"[{iteration}] NaN/Inf detected in u_star or d_u after u momentum solve")
                if (d_u <= 0).any():
                    raise ValueError(f"[{iteration}] Negative or zero diagonal entry detected in d_u")

                v_star, d_v, v_res_info = self.momentum_solver.solve_v_momentum(
                    self.mesh, self.fluid, self.u, self.v, p_star,
                    relaxation_factor=self.alpha_u,
                    return_dict=True
                )

                # --- Post-v momentum sanity ---
                if np.isnan(v_star).any() or np.isinf(v_star).any() or np.isnan(d_v).any() or np.isinf(d_v).any():
                    raise ValueError(f"[{iteration}] NaN/Inf detected in v_star or d_v after v momentum solve")
                if (d_v <= 0).any():
                    raise ValueError(f"[{iteration}] Negative or zero diagonal entry detected in d_v")

                # Save intermediate fields for pressure equation
                self._tmp_u_star = u_star
                self._tmp_v_star = v_star
                self._tmp_d_u = d_u
                self._tmp_d_v = d_v

                # Solve pressure correction equation using solve_pressure_correction

                if hasattr(self.pressure_solver, 'solve_pressure_correction'):
                    p_prime, p_res_info = self.pressure_solver.solve_pressure_correction(
                        self.mesh, self.fluid, u_star, v_star, d_u, d_v, 
                        relaxation_factor=self.alpha_p, boundary_conditions=self.bc_manager
                    )
                else:
                    # Fall back to original solve method - pass bc_manager here too
                    p_prime, p_res_info = self.pressure_solver.solve(
                        self.mesh, u_star, v_star, d_u, d_v, p_star, 
                        bc_manager=self.bc_manager, # Add bc_manager argument
                        return_dict=True
                    )
                
                # --- Post-pressure correction sanity ---
                if np.isnan(p_prime).any() or np.isinf(p_prime).any():
                    raise ValueError(f"[{iteration}] NaN/Inf detected in pressure correction p_prime")
                
                # --- Sanity Checks for p' --- 
                if iteration < 4: # Check first few iterations
                    if np.isnan(p_prime).any() or np.isinf(p_prime).any():
                        print(f"Iteration {iteration}: !!! NaN/Inf detected in p_prime !!!")
                    else: 
                        print(f"Iteration {iteration}: p_prime min/max/mean = {p_prime.min():.2e} / {p_prime.max():.2e} / {p_prime.mean():.2e}")
                # ----------------------------- 
                
                # Update pressure with relaxation (all 1D arrays)
                self.p = p_star + self.alpha_p * p_prime 
                # self._enforce_pressure_boundary_conditions() # Commented out: Assumes BCs handled by pressure solver matrix/pinning
                p_star = self.p.copy() # p_star remains 1D
                
                # Update velocities
                self.u, self.v = self.velocity_updater.update_velocity(
                     self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
                )
                
                # --- Post-velocity update sanity ---
                if np.isnan(self.u).any() or np.isinf(self.u).any() or \
                   np.isnan(self.v).any() or np.isinf(self.v).any():
                    raise ValueError(f"[{iteration}] NaN/Inf detected in updated velocities u or v")
                
                # --- Mass balance check (optional) ---
                try:
                    rhs = get_rhs(self.mesh, self.fluid.get_density(), self.u, self.v, p=self.p, d_u=self._tmp_d_u, d_v=self._tmp_d_v) 
                    mass_residual = np.linalg.norm(rhs, ord=2)
                    if iteration < 4:
                        print(f"Iteration {iteration}: Mass residual norm (RHS) = {mass_residual:.3e}")
                except Exception as e:
                    print(f"[{iteration}] Mass residual check failed: {e}")
   
                # --- Sanity Checks for u, v --- 
                if iteration < 4: # Check first few iterations
                    if np.isnan(self.u).any() or np.isinf(self.u).any() or \
                       np.isnan(self.v).any() or np.isinf(self.v).any():
                        print(f"Iteration {iteration}: !!! NaN/Inf detected in updated u/v !!!")
                    else:
                        print(f"Iteration {iteration}: u min/max/mean = {self.u.min():.2e} / {self.u.max():.2e} / {self.u.mean():.2e}")
                        print(f"Iteration {iteration}: v min/max/mean = {self.v.min():.2e} / {self.v.max():.2e} / {self.v.mean():.2e}")
                # ------------------------------

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
                            #break


                
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
            filename = os.path.join(profile_dir, f"SIMPLE_Re{int(self.fluid.get_reynolds_number())}_mesh{self.mesh.n_cells}_profile.h5")
            print(f"Saved profile to {self.save_profiling_data(filename)}")
        self.profiler.end_section("Finalization") # End timing finalization

        return result