"""
SIMPLER (SIMPLE Revised) algorithm implementation.
""" 

import numpy as np
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
    
    def solve(self, max_iterations=1000, tolerance=1e-6):
        """
        Solve using the SIMPLER algorithm.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        SimulationResult
            Object containing the solution fields and convergence history
        """
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
        
        # Main iteration loop
        iteration = 1
        max_res = 1000
        
        while (iteration <= max_iterations) and (max_res > tolerance):
            # Store old values for convergence check
            u_old = self.u.copy()
            v_old = self.v.copy()
            
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
            
            # Step 2: Solve pressure equation (not pressure correction) to get p
            # This is the key difference between SIMPLE and SIMPLER
            # In SIMPLER, we solve for p directly using u* and v*
            self.p = self.pressure_solver.solve(
                self.mesh, u_star, v_star, d_u, d_v, np.zeros_like(p_star)
            )
            p_star = self.p.copy()
            
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
            
            # Step 4: Solve pressure correction equation to get p'
            p_prime = self.pressure_solver.solve(
                self.mesh, u_star, v_star, d_u, d_v, p_star
            )
            
            # Step 5: Update velocity (but not pressure, which was already updated)
            self.u, self.v = self.velocity_updater.update_velocity(
                self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
            )
            
            # Calculate residuals
            u_res = np.abs(self.u - u_old)
            v_res = np.abs(self.v - v_old)
            max_res = max(np.max(u_res), np.max(v_res))
            self.residual_history.append(max_res)
            
            # Print progress
            print(f"Iteration {iteration}, Residual: {max_res:.6e}")
            
            iteration += 1
        
        # Calculate divergence for final solution
        divergence = self.calculate_divergence()
        
        # Create and return result object with the Reynolds number
        reynolds_value = self.fluid.get_reynolds_number()
        print(f"DEBUG: Reynolds number in SimplerSolver.solve: {reynolds_value}")
        
        result = SimulationResult(
            self.u, self.v, self.p, self.mesh, 
            iterations=iteration-1, 
            residuals=self.residual_history,
            divergence=divergence,
            reynolds=reynolds_value
        )
        
        return result 