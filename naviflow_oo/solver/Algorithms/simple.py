"""
SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm implementation.
""" 

import numpy as np
from . import Algorithm
from ...postprocessing.simulation_result import SimulationResult

class SimpleSolver(Algorithm):
    """
    SIMPLE algorithm implementation.
    """
    def __init__(self, mesh, fluid, pressure_solver=None, momentum_solver=None, 
                 velocity_updater=None, boundary_conditions=None, 
                 alpha_p=0.3, alpha_u=0.7):
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
        """
        super().__init__(mesh, fluid, pressure_solver, momentum_solver, 
                         velocity_updater, boundary_conditions)
        self.alpha_p = alpha_p
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
        Solve using the SIMPLE algorithm.
        
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
        
        print(f"DEBUG: Fluid object in SimpleSolver.solve: {self.fluid}")
        print(f"DEBUG: Reynolds number at start of SimpleSolver.solve: {self.fluid.get_reynolds_number()}")
        
        # Initialize variables
        p_star = self.p.copy()
        p_prime = np.zeros((nx, ny))
        residuals = []
        
        # Main iteration loop
        iteration = 1
        max_res = 1000
        
        while (iteration <= max_iterations) and (max_res > tolerance):
            # Store old values for convergence check
            u_old = self.u.copy()
            v_old = self.v.copy()
            
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
            
            # Solve pressure correction equation
            p_prime = self.pressure_solver.solve(
                self.mesh, u_star, v_star, d_u, d_v, p_star
            )
            
            # Update pressure with relaxation
            self.p = p_star + self.alpha_p * p_prime
            p_star = self.p.copy()  # Update p_star for next iteration
            
            # Update velocity
            self.u, self.v = self.velocity_updater.update_velocity(
                self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
            )
            
            # Calculate residuals
            u_res = np.abs(self.u - u_old)
            v_res = np.abs(self.v - v_old)
            max_res = max(np.max(u_res), np.max(v_res))
            residuals.append(max_res)
            
            # Print progress
            print(f"Iteration {iteration}, Residual: {max_res:.6e}")
            
            iteration += 1
        
        # Calculate divergence for final solution
        from ...postprocessing.validation.cavity_flow import calculate_divergence
        divergence = calculate_divergence(self.u, self.v, dx, dy)
        
        # Create and return result object with the Reynolds number
        reynolds_value = self.fluid.get_reynolds_number()
        print(f"DEBUG: Reynolds number in SimpleSolver.solve: {reynolds_value}")
        
        result = SimulationResult(
            self.u, self.v, self.p, self.mesh, 
            iterations=iteration-1, 
            residuals=residuals,
            divergence=divergence,
            reynolds=reynolds_value
        )
        
        return result
