"""
PISO (Pressure Implicit with Splitting of Operators) algorithm implementation.
""" 

import numpy as np
import os
from .base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult
from ...postprocessing.validation.cavity_flow import calculate_infinity_norm_error

class PisoSolver(BaseAlgorithm):
    """
    PISO algorithm implementation.
    
    The PISO (Pressure Implicit with Splitting of Operators) algorithm
    is a widely used method for solving the Navier-Stokes equations for incompressible flows.
    It uses a predictor-corrector approach with multiple pressure corrections per iteration.
    """
    def __init__(self, mesh, fluid, pressure_solver=None, momentum_solver=None, 
                 velocity_updater=None, boundary_conditions=None, 
                 alpha_p=0.3, alpha_u=0.7, n_corrections=2):
        """
        Initialize the PISO solver.
        
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
        n_corrections : int
            Number of pressure corrections per iteration
        """
        super().__init__(mesh, fluid, pressure_solver, momentum_solver, 
                         velocity_updater, boundary_conditions)
        self.alpha_p = alpha_p
        self.alpha_u = alpha_u
        self.n_corrections = n_corrections
        # Store old values for plotting final residuals
        self.u_old = None
        self.v_old = None
        self.p_old = None
        
    def solve(self, max_iterations=1000, tolerance=1e-6, save_profile=True, profile_dir='results/profiles', 
              track_infinity_norm=False, infinity_norm_interval=10):
        """
        Solve using the PISO algorithm.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        save_profile : bool, optional
            Whether to save profiling data to a file
        profile_dir : str, optional
            Directory to save profiling data
        track_infinity_norm : bool, optional
            Whether to track infinity norm error against Ghia data
        infinity_norm_interval : int, optional
            Interval (in iterations) at which to calculate infinity norm error
            
        Returns:
        --------
        SimulationResult
            Object containing the solution fields and convergence history
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
        self.residual_history = []
        self.momentum_residual_history = []
        self.pressure_residual_history = []
        self.infinity_norm_history = []
        
        # Main iteration loop
        iteration = 1
        max_res = 1000
        momentum_res = 1000
        pressure_res = 1000
        
        while (iteration <= max_iterations) and (max_res > tolerance):
            # Store old values for convergence check
            u_old = self.u.copy()
            v_old = self.v.copy()
            p_old = self.p.copy()
            
            # Predictor step: Solve momentum equations
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
            
            # Calculate momentum residual
            u_momentum_res = np.abs(u_star - self.u)
            v_momentum_res = np.abs(v_star - self.v)
            momentum_res = max(np.max(u_momentum_res), np.max(v_momentum_res))
            
            # Multiple pressure corrections (PISO steps)
            for correction in range(self.n_corrections):
                # Solve pressure correction equation
                p_prime = self.pressure_solver.solve(
                    self.mesh, u_star, v_star, d_u, d_v, p_star
                )
                
                # Update pressure
                self.p = p_star + self.alpha_p * p_prime
                
                # Apply pressure boundary conditions
                self._enforce_pressure_boundary_conditions()
                
                # Update velocity
                self.u, self.v = self.velocity_updater.update_velocity(
                    self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
                )
                
                # Update p_star for next correction
                p_star = self.p.copy()
                
                # Update u_star and v_star for next correction
                u_star = self.u.copy()
                v_star = self.v.copy()
                
                # Add this block to recalculate momentum equation coefficients
                if correction < self.n_corrections - 1:  # Only if not the last correction
                    # Recalculate momentum coefficients for next correction step
                    u_star, d_u = self.momentum_solver.solve_u_momentum(
                        self.mesh, self.fluid, u_star, v_star, p_star, 
                        relaxation_factor=1.0,  # No relaxation for intermediate corrections
                        boundary_conditions=self.bc_manager
                    )
                    
                    v_star, d_v = self.momentum_solver.solve_v_momentum(
                        self.mesh, self.fluid, u_star, v_star, p_star, 
                        relaxation_factor=1.0,  # No relaxation for intermediate corrections
                        boundary_conditions=self.bc_manager
                    )
            
            # Calculate pressure residual
            pressure_res = np.linalg.norm(self.p - p_old, ord=2) / np.sqrt(nx * ny)
            
            # Calculate total residual
            u_res = np.linalg.norm(self.u - u_old, ord=2) / np.sqrt(nx * ny)
            v_res = np.linalg.norm(self.v - v_old, ord=2) / np.sqrt(nx * ny)
            max_res = u_res + v_res + pressure_res
            
            # Store residual history
            self.residual_history.append(max_res)
            self.momentum_residual_history.append(u_res + v_res)
            self.pressure_residual_history.append(pressure_res)
            
            # Calculate infinity norm error if requested
            infinity_norm_error = None
            if track_infinity_norm and (iteration % infinity_norm_interval == 0 or iteration == max_iterations or max_res <= tolerance):
                try:
                    infinity_norm_error = calculate_infinity_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                    self.infinity_norm_history.append(infinity_norm_error)
                    print(f"Iteration {iteration}, Infinity Norm Error: {infinity_norm_error:.6e}")
                except Exception as e:
                    print(f"Warning: Could not calculate infinity norm error: {str(e)}")
            
            # Add detailed residual data to profiler
            self.profiler.add_residual_data(
                iteration=iteration,
                total_residual=max_res,
                momentum_residual=u_res + v_res,
                pressure_residual=pressure_res,
                infinity_norm_error=infinity_norm_error
            )
            
            # Print progress with all residuals
            print(f"Iteration {iteration}, "
                  f"Total Residual: {max_res:.6e}, "
                  f"Momentum Residual: {u_res + v_res:.6e}, "
                  f"Pressure Residual: {pressure_res:.6e}")
            
            # Store old fields for final residual plotting before proceeding to next iteration
            self.u_old = u_old
            self.v_old = v_old
            self.p_old = p_old
            
            iteration += 1
        
        # Update profiling data
        self.profiler.set_iterations(iteration - 1)
        
        # Set convergence information
        final_residual = max_res
        converged = max_res <= tolerance
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
                f"PISO_Re{int(reynolds_value)}_mesh{nx}x{ny}_profile.h5"
            )
            profile_path = self.save_profiling_data(filename)
            print(f"Profiling data saved to: {profile_path}")
        
        return result
