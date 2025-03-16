"""
SIMPLER (SIMPLE Revised) algorithm implementation.
""" 

import numpy as np
import os
from .base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult

class SimplerSolver(BaseAlgorithm):
    """
    SIMPLER (SIMPLE Revised) algorithm implementation.
    
    The SIMPLER algorithm is an improved version of SIMPLE that solves an additional
    pressure equation to obtain a better pressure field. This often leads to faster
    convergence compared to the standard SIMPLE algorithm.
    """
    def __init__(self, mesh, fluid, pressure_solver=None, momentum_solver=None, 
                 velocity_updater=None, boundary_conditions=None, 
                 alpha_u=0.7):
        """
        Initialize the SIMPLER solver.
        
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
        alpha_u : float
            Relaxation factor for velocity
        """
        super().__init__(mesh, fluid, pressure_solver, momentum_solver, 
                         velocity_updater, boundary_conditions)
        self.alpha_u = alpha_u
        
        # Set default solvers if not provided
        if self.pressure_solver is None:
            from ..pressure_solver.direct import DirectPressureSolver
            self.pressure_solver = DirectPressureSolver()
            
        if self.momentum_solver is None:
            from ..momentum_solver.standard import StandardMomentumSolver
            self.momentum_solver = StandardMomentumSolver()
            
        if self.velocity_updater is None:
            from ..velocity_solver.standard import StandardVelocityUpdater
            self.velocity_updater = StandardVelocityUpdater()
    
    def solve(self, max_iterations=1000, tolerance=1e-6, save_profile=True, profile_dir='results/profiles'):
        """
        Solve using the SIMPLER algorithm.
        
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
        
        print(f"DEBUG: Fluid object in SimplerSolver.solve: {self.fluid}")
        print(f"DEBUG: Reynolds number at start of SimplerSolver.solve: {self.fluid.get_reynolds_number()}")
        
        # Initialize variables
        p_star = self.p.copy()
        p_prime = np.zeros((nx, ny))
        self.residual_history = []  # Reset residual history
        self.momentum_residual_history = []  # Track momentum residuals
        self.pressure_residual_history = []  # Track pressure residuals
        
        # Main iteration loop
        iteration = 1
        max_res = 1000
        momentum_res = 1000
        pressure_res = 1000
        
        self.profiler.start_section()  # Start timing other operations
        
        while (iteration <= max_iterations) and (max_res > tolerance):
            # Store old values for convergence check
            u_old = self.u.copy()
            v_old = self.v.copy()
            p_old = self.p.copy()
            
            self.profiler.end_section('other_time')  # End timing other operations
            self.profiler.start_section()  # Start timing first momentum solve
            
            # Step 1: Solve momentum equations with relaxation factor to get u* and v*
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
            
            # Calculate first momentum residual
            u_momentum_res1 = np.abs(u_star - self.u)
            v_momentum_res1 = np.abs(v_star - self.v)
            momentum_res1 = max(np.max(u_momentum_res1), np.max(v_momentum_res1))
            
            self.profiler.end_section('momentum_solve_time')  # End timing first momentum solve
            self.profiler.start_section()  # Start timing pressure solve
            
            # Step 2: Solve pressure equation (not pressure correction) to get p
            # This is the key difference between SIMPLE and SIMPLER
            # In SIMPLER, we solve for p directly using u* and v*
            p_old_before_update = self.p.copy()
            self.p = self.pressure_solver.solve(
                self.mesh, u_star, v_star, d_u, d_v, np.zeros_like(p_star)
            )
            
            # Calculate pressure equation residual
            pressure_eq_res = np.max(np.abs(self.p - p_old_before_update))
            
            p_star = self.p.copy()
            
            self.profiler.end_section('pressure_solve_time')  # End timing pressure solve
            self.profiler.start_section()  # Start timing second momentum solve
            
            # Step 3: Solve momentum equations again with the new pressure field
            # This step can be skipped to save computation time, but it improves accuracy
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
            
            # Calculate second momentum residual
            u_momentum_res2 = np.abs(u_star - self.u)
            v_momentum_res2 = np.abs(v_star - self.v)
            momentum_res2 = max(np.max(u_momentum_res2), np.max(v_momentum_res2))
            
            # Use the maximum of both momentum residuals
            momentum_res = max(momentum_res1, momentum_res2)
            
            self.profiler.end_section('momentum_solve_time')  # End timing second momentum solve
            self.profiler.start_section()  # Start timing pressure correction solve
            
            # Step 4: Solve pressure correction equation to get p'
            p_prime = self.pressure_solver.solve(
                self.mesh, u_star, v_star, d_u, d_v, p_star
            )
            
            # Calculate pressure correction residual
            pressure_corr_res = np.max(np.abs(p_prime))
            
            # Use the maximum of both pressure residuals
            pressure_res = max(pressure_eq_res, pressure_corr_res)
            
            self.profiler.end_section('pressure_correction_time')  # End timing pressure correction solve
            self.profiler.start_section()  # Start timing velocity update
            
            # Step 5: Update velocity (but not pressure, which was already updated)
            self.u, self.v = self.velocity_updater.update_velocity(
                self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
            )
            
            self.profiler.end_section('velocity_update_time')  # End timing velocity update
            self.profiler.start_section()  # Start timing other operations
            
            # Calculate total residual
            u_res = np.abs(self.u - u_old)
            v_res = np.abs(self.v - v_old)
            max_res = max(np.max(u_res), np.max(v_res))
            
            # Store residual history
            self.residual_history.append(max_res)
            self.momentum_residual_history.append(momentum_res)
            self.pressure_residual_history.append(pressure_res)
            
            # Add detailed residual data to profiler
            self.profiler.add_residual_data(
                iteration=iteration,
                total_residual=max_res,
                momentum_residual=momentum_res,
                pressure_residual=pressure_res
            )
            
            # Print progress with all residuals
            print(f"Iteration {iteration}, "
                  f"Total Residual: {max_res:.6e}, "
                  f"Momentum Residual: {momentum_res:.6e}, "
                  f"Pressure Residual: {pressure_res:.6e}")
            
            # Update memory usage every 10 iterations
            if iteration % 10 == 0:
                self.profiler.update_memory_usage()
                
            iteration += 1
            
        self.profiler.end_section('other_time')  # End timing other operations
        
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
        print(f"DEBUG: Reynolds number in SimplerSolver.solve: {reynolds_value}")
        
        result = SimulationResult(
            self.u, self.v, self.p, self.mesh, 
            iterations=iteration-1, 
            residuals=self.residual_history,
            divergence=divergence,
            reynolds=reynolds_value
        )
        
        # Save profiling data if requested
        if save_profile:
            os.makedirs(profile_dir, exist_ok=True)
            filename = os.path.join(
                profile_dir, 
                f"SIMPLER_Re{int(reynolds_value)}_mesh{nx}x{ny}_profile.txt"
            )
            profile_path = self.save_profiling_data(filename)
            print(f"Profiling data saved to: {profile_path}")
        
        return result 