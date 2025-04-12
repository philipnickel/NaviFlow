"""
SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm implementation.
""" 

import numpy as np
import os
import matplotlib.pyplot as plt
from .base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult
from ...postprocessing.validation.cavity_flow import calculate_infinity_norm_error
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
        super().__init__(mesh, fluid, pressure_solver, momentum_solver, 
                         velocity_updater, boundary_conditions)
 
    def solve(self, max_iterations=1000, tolerance=1e-6, save_profile=False, profile_dir=None, 
              track_infinity_norm=False, infinity_norm_interval=10, should_plot_final_residuals=False):
        """
        Solve the Navier-Stokes equations using the SIMPLE algorithm.
        
        Args:
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            save_profile (bool): Whether to save profiling data
            profile_dir (str): Directory to save profiling data
            track_infinity_norm (bool): Whether to track infinity norm
            infinity_norm_interval (int): Interval for tracking infinity norm
            should_plot_final_residuals (bool): Whether to plot final residuals
        """
        # Start profiling
        self.profiler.start()
        
        # Get mesh dimensions and fluid properties
        nx, ny = self.mesh.get_dimensions()
        dx, dy = self.mesh.get_cell_sizes()
        rho = self.fluid.get_density()
        mu = self.fluid.get_viscosity()
        
        # Initialize variables
        p_star = self.p.copy()
        p_prime = np.zeros((nx, ny))
        self.residual_history = []  # Reset residual history
        self.momentum_residual_history = []  # Track momentum residuals
        self.pressure_residual_history = []  # Track pressure residuals
        self.infinity_norm_history = []  # Track infinity norm errors
        
        # Main iteration loop
        iteration = 1
        total_res = 1000
        momentum_res = 1000
        pressure_res = 1000
        
        # Create directory for debugging arrays
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                'main_scripts', 'geo_multigrid', 'multigrid_debugging', 'arrays_5x5')
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Using relaxation factors: alpha_p = {self.alpha_p}, alpha_u = {self.alpha_u}") 
        
        try:
            while (iteration <= max_iterations) and (total_res > tolerance):
                # Store old values for convergence check
                self.u_old = self.u.copy()
                self.v_old = self.v.copy()
                self.p_old = self.p.copy()
                
                # Solve momentum equations with relaxation factor
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

                # Calculate momentum residual using 2-norm
                u_momentum_res = np.linalg.norm(u_star - self.u, ord=2)
                v_momentum_res = np.linalg.norm(v_star - self.v, ord=2)

                momentum_res =(u_momentum_res + v_momentum_res)
                # Solve pressure correction equation
                p_prime = self.pressure_solver.solve(
                    self.mesh, u_star, v_star, d_u, d_v, p_star
                )
                # Update pressure with relaxation
                self.p = p_star + self.alpha_p * p_prime
                
                # Apply zero-gradient boundary conditions to pressure
                self._enforce_pressure_boundary_conditions()
                
                # Calculate pressure residual using 2-norm
                pressure_res = np.linalg.norm(self.p - self.p_old, ord=2)
                p_star = self.p
                
                # Update velocity
                self.u, self.v = self.velocity_updater.update_velocity(
                    self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
                )
                
                # Calculate scaled residuals
                n_cells = nx * ny

                # Velocity residuals (normalized by number of cells)
                u_res = np.linalg.norm(self.u - self.u_old, ord=2) / np.sqrt(n_cells)
                v_res = np.linalg.norm(self.v - self.v_old, ord=2) / np.sqrt(n_cells)

                # Pressure residual (normalized)
                p_res = np.linalg.norm(self.p - self.p_old, ord=2) / np.sqrt(n_cells)
                # Combined residual as sum of velocity components and pressure only
                total_res = u_res + v_res + p_res

                # Store all residuals separately for monitoring
                self.residual_history.append(total_res)
                self.momentum_residual_history.append(u_res + v_res)
                self.pressure_residual_history.append(p_res)
                
                # Calculate infinity norm error if requested
                infinity_norm_error = None
                if track_infinity_norm and (iteration % infinity_norm_interval == 0 or iteration == max_iterations or total_res <= tolerance):
                    try:
                        infinity_norm_error = calculate_infinity_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                        self.infinity_norm_history.append(infinity_norm_error)
                        print(f"Iteration {iteration}, Infinity Norm Error: {infinity_norm_error:.6e}")
                    except Exception as e:
                        print(f"Warning: Could not calculate infinity norm error: {str(e)}")
                
                # Add detailed residual data to profiler
                self.profiler.add_residual_data(
                    iteration=iteration,
                    total_residual=total_res,
                    momentum_residual=u_res + v_res,
                    pressure_residual=p_res,
                    infinity_norm_error=infinity_norm_error
                )
                
                # Print progress with all residuals
                print(f"Iteration {iteration}, "
                      f"Total Residual: {total_res:.6e}, "
                      f"Velocity Residual: {u_res + v_res:.6e}, "
                      f"Pressure Residual: {p_res:.6e}")
                
                iteration += 1
                
        except KeyboardInterrupt:
            print("\nSimulation stopped by keyboard interrupt (Ctrl+C).")
            # Continue with the rest of the code to save results
        
        # Update profiling data
        self.profiler.set_iterations(iteration - 1)
        
        # Set convergence information
        final_residual = total_res
        converged = total_res <= tolerance
        self.profiler.set_convergence_info(
            tolerance=tolerance,
            final_residual=final_residual,
            residual_history=self.residual_history,
            converged=converged
        )
        
        # Collect pressure solver performance metrics if available
        if hasattr(self.pressure_solver, 'get_solver_info'):
            solver_info = self.pressure_solver.get_solver_info()
            self.profiler.set_pressure_solver_info(
                solver_name=solver_info.get('name', self.pressure_solver.__class__.__name__),
                inner_iterations=solver_info.get('inner_iterations_history'),
                convergence_rate=solver_info.get('convergence_rate'),
                solver_specific=solver_info.get('solver_specific')
            )
        
        # End profiling
        self.profiler.end()
        
        # Calculate divergence for final solution
        divergence = self.calculate_divergence()
        
        # Create result object with the Reynolds number
        reynolds_value = self.fluid.get_reynolds_number()
        
        result = SimulationResult(
            self.u, self.v, self.p, self.mesh, 
            iterations=iteration-1, 
            residuals=self.residual_history,
            momentum_residuals=self.momentum_residual_history,
            pressure_residuals=self.pressure_residual_history,
            divergence=divergence,
            reynolds=reynolds_value
        )
        
        # Calculate final infinity norm error if not already done
        if track_infinity_norm and not self.infinity_norm_history:
            try:
                result.calculate_infinity_norm_error()
                print(f"Final Infinity Norm Error: {result.infinity_norm_error:.6e}")
            except Exception as e:
                print(f"Warning: Could not calculate final infinity norm error: {str(e)}")
        elif self.infinity_norm_history:
            result.infinity_norm_error = self.infinity_norm_history[-1]
        
        # Save profiling data if requested
        if save_profile:
            os.makedirs(profile_dir, exist_ok=True)
            filename = os.path.join(
                profile_dir, 
                f"SIMPLE_Re{int(reynolds_value)}_mesh{nx}x{ny}_profile.h5"
            )
            profile_path = self.save_profiling_data(filename)
            print(f"Profiling data saved to: {profile_path}")
        
        # Plot final residuals if requested
        if should_plot_final_residuals:
            plot_final_residuals(
                self.u, self.v, self.p,
                self.u_old, self.v_old, self.p_old,
                self.mesh,
                title=f'Final Residuals (Re={reynolds_value})',
                filename='final_residuals.pdf',
                show=True
            )
        
        return result