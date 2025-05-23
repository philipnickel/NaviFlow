"""
SIMPLEC (SIMPLE-Consistent) algorithm implementation.
"""

import numpy as np
import os
from .base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult
from ...postprocessing.validation.cavity_flow import calculate_infinity_norm_error, calculate_l2_norm_error

class SimplecSolver(BaseAlgorithm):
    """
    SIMPLEC algorithm implementation.
    
    The SIMPLEC (SIMPLE-Consistent) algorithm is a variant of SIMPLE that uses
    a modified pressure correction equation. This can lead to better convergence
    behavior in many cases, especially for complex flows.
    """
    def __init__(self, mesh, fluid, pressure_solver=None, momentum_solver=None, 
                 velocity_updater=None, boundary_conditions=None, 
                 alpha_p=0.2, alpha_u=0.7):
        """
        Initialize the SIMPLEC solver.
        
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
        """
        super().__init__(mesh, fluid, pressure_solver, momentum_solver, 
                         velocity_updater, boundary_conditions)
        self.alpha_p = alpha_p
        self.alpha_u = alpha_u
       
    def solve(self, max_iterations=1000, tolerance=1e-6, save_profile=True, profile_dir='results/profiles', 
              track_infinity_norm=False, infinity_norm_interval=10, use_l2_norm=False):
        """
        Solve using the SIMPLEC algorithm with vectorized operations.
        
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
        use_l2_norm : bool, optional
            Whether to use L2 norm instead of infinity norm for error calculation
            
        Returns:
        --------
        SimulationResult
            Object containing the solution fields and convergence history
        """
        # Save use_l2_norm parameter for later use
        self.use_l2_norm = use_l2_norm
        
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
            
            # Calculate momentum residual
            u_momentum_res = np.abs(u_star - self.u)
            v_momentum_res = np.abs(v_star - self.v)
            momentum_res = max(np.max(u_momentum_res), np.max(v_momentum_res))
            
            # Add SIMPLEC modification to d_u and d_v
            d_u_simplec = d_u / (1 - (1 - self.alpha_u))
            d_v_simplec = d_v / (1 - (1 - self.alpha_u))
            
            # Use modified coefficients in pressure correction
            p_prime = self.pressure_solver.solve(
                mesh=self.mesh,
                u_star=u_star,
                v_star=v_star,
                d_u=d_u_simplec,  # Use SIMPLEC modified coefficient
                d_v=d_v_simplec,  # Use SIMPLEC modified coefficient
                p_star=p_star
            )

            self._enforce_pressure_boundary_conditions()
            
            # Apply pressure correction smoothing
            p_prime_smooth = np.zeros_like(p_prime)
            p_prime_smooth[1:-1, 1:-1] = (
                0.6 * p_prime[1:-1, 1:-1] +
                0.1 * (p_prime[2:, 1:-1] + p_prime[:-2, 1:-1] +
                      p_prime[1:-1, 2:] + p_prime[1:-1, :-2])
            )
            p_prime = p_prime_smooth
            
            # Update pressure with dynamic relaxation
            if iteration > 1:
                prev_res = self.residual_history[-1]
                if max_res > prev_res:
                    self.alpha_p *= 0.95  # Reduce relaxation if residual increased
            self.p = p_star + self.alpha_p * p_prime
            
            # Calculate pressure residual
            pressure_res = np.max(np.abs(self.p - p_old))
            
            p_star = self.p.copy()
            
            # Use modified coefficients in velocity correction
            self.u, self.v = self.velocity_updater.update_velocity(
                self.mesh, u_star, v_star, p_prime, 
                d_u_simplec, d_v_simplec,  # Use SIMPLEC modified coefficients
                self.bc_manager
            )
            
            # Calculate total residual
            u_res = np.abs(self.u - u_old)
            v_res = np.abs(self.v - v_old)
            max_res = max(np.max(u_res), np.max(v_res))
            
            # Store residual history
            self.residual_history.append(max_res)
            self.momentum_residual_history.append(momentum_res)
            self.pressure_residual_history.append(pressure_res)
            
            # Calculate infinity norm error if requested
            infinity_norm_error = None
            l2_norm_error = None
            if track_infinity_norm and (iteration % infinity_norm_interval == 0 or iteration == max_iterations or max_res <= tolerance):
                try:
                    # Calculate both error metrics
                    infinity_norm_error = calculate_infinity_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                    l2_norm_error = calculate_l2_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                    
                    # Store the one requested (or infinity norm by default)
                    error_to_store = l2_norm_error if use_l2_norm else infinity_norm_error
                    self.infinity_norm_history.append(error_to_store)
                    
                    # Print both errors on the same line
                    print(f"Iteration {iteration}, Infinity Norm Error: {infinity_norm_error:.6e}, L2 Norm Error: {l2_norm_error:.6e}")
                except Exception as e:
                    print(f"Warning: Could not calculate error: {str(e)}")
            
            # Add detailed residual data to profiler
            self.profiler.add_residual_data(
                iteration=iteration,
                total_residual=max_res,
                momentum_residual=momentum_res,
                pressure_residual=pressure_res,
                infinity_norm_error=infinity_norm_error
            )
            
            # Print progress with all residuals
            print(f"Iteration {iteration}, "
                  f"Total Residual: {max_res:.6e}, "
                  f"Momentum Residual: {momentum_res:.6e}, "
                  f"Pressure Residual: {pressure_res:.6e}")
            
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
                # Calculate both error metrics for the final result
                inf_error = result.calculate_infinity_norm_error()
                l2_error = result.calculate_l2_norm_error()
                
                # Print both on same line
                print(f"Final Errors - Infinity Norm: {inf_error:.6e}, L2 Norm: {l2_error:.6e}")
                
                # Set the right one for backward compatibility based on use_l2_norm if available
                if use_l2_norm:
                    result.infinity_norm_error = l2_error
            except Exception as e:
                print(f"Warning: Could not calculate final error: {str(e)}")
        elif self.infinity_norm_history:
            result.infinity_norm_error = self.infinity_norm_history[-1]
        
        # Save profiling data if requested
        if save_profile:
            os.makedirs(profile_dir, exist_ok=True)
            filename = os.path.join(
                profile_dir, 
                f"SIMPLEC_Re{int(reynolds_value)}_mesh{nx}x{ny}_profile.h5"
            )
            profile_path = self.save_profiling_data(filename)
            print(f"Profiling data saved to: {profile_path}")
        
        return result 