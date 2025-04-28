"""
Mesh-agnostic SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm implementation.
Designed to work with arbitrary mesh topologies.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from ..Algorithms.base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult
from ...postprocessing.visualization import plot_final_residuals
from ..pressure_solver.helpers.rhs_construction import get_rhs
from .profiler import SimpleProfiler

class SimpleVectorField:
    """
    Simple wrapper for vector field data stored at cell centers.
    Used for collocated grid storage.
    """
    def __init__(self, u, v):
        """
        Initialize with u and v components.
        
        Parameters:
        -----------
        u, v : ndarray
            Velocity components at cell centers
        """
        self.u = u
        self.v = v
    
    def get_u_at_cells(self):
        """Return the u-component at cell centers."""
        return self.u
    
    def get_v_at_cells(self):
        """Return the v-component at cell centers."""
        return self.v
    
    def get_values_at_cells(self):
        """Return both components as a tuple."""
        return self.u, self.v

class SimpleScalarField:
    """
    Simple wrapper for scalar field data stored at cell centers.
    Used for collocated grid storage.
    """
    def __init__(self, values):
        """
        Initialize with scalar values.
        
        Parameters:
        -----------
        values : ndarray
            Scalar values at cell centers
        """
        self.values = values
    
    def get_values_at_cells(self):
        """Return the scalar values at cell centers."""
        return self.values

class MeshAgnosticSimpleSolver(BaseAlgorithm):
    """
    Mesh-agnostic SIMPLE algorithm implementation.
    
    The SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm
    is a widely used method for solving the Navier-Stokes equations for incompressible flows.
    This implementation works with arbitrary mesh topologies.
    """
    def __init__(self, mesh, fluid, pressure_solver=None, momentum_solver=None, 
                 velocity_updater=None, boundary_conditions=None, 
                 alpha_p=0.3, alpha_u=0.7):
        """
        Initialize the mesh-agnostic SIMPLE solver.
        
        Parameters:
        -----------
        mesh : Mesh
            The computational mesh (structured or unstructured)
        fluid : FluidProperties
            Fluid properties
        pressure_solver : PressureSolver, optional
            Solver for pressure equation (should be mesh-agnostic)
        momentum_solver : MomentumSolver, optional
            Solver for momentum equations (should be mesh-agnostic)
        velocity_updater : VelocityUpdater, optional
            Method to update velocities
        boundary_conditions : dict or BoundaryConditionManager, optional
            Boundary conditions
        alpha_p, alpha_u : float
            Relaxation factors for pressure and velocity
        """
        self.alpha_p = alpha_p
        self.alpha_u = alpha_u
        self.u_old = None  # Store old u values
        self.v_old = None  # Store old v values
        self.p_old = None  # Store old p values

        # Initialize residual histories
        self.x_momentum_residuals = []
        self.y_momentum_residuals = []
        self.continuity_residuals = []
        
        # Variables to store final residual fields
        self._final_u_residual_field = None
        self._final_v_residual_field = None
        self._final_p_residual_field = None
        
        # Initialize mesh and fluid
        self.mesh = mesh
        self.fluid = fluid
        
        # Initialize fields
        self.initialize_fields()
        
        # Initialize solvers
        self.momentum_solver = momentum_solver
        self.pressure_solver = pressure_solver
        self.velocity_updater = velocity_updater
        
        # Initialize profiler
        self.profiler = SimpleProfiler()
        
        # Set up boundary conditions
        from ...constructor.boundary_conditions import BoundaryConditionManager
        if isinstance(boundary_conditions, BoundaryConditionManager):
            self.bc_manager = boundary_conditions
        else:
            self.bc_manager = BoundaryConditionManager()
            if boundary_conditions:
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        self.bc_manager.set_condition(boundary, field_type, values)
    
    def initialize_fields(self):
        """
        Initialize velocity and pressure fields.
        Using mesh.n_cells instead of get_dimensions for compatibility with all mesh types.
        """
        # Get the number of cells
        n_cells = self.mesh.n_cells
        
        # Initialize fields if needed
        # For a general mesh, we store velocity components at cell centers
        # and pressure at cell centers
        if not hasattr(self, 'u') or self.u is None:
            self.u = np.zeros(n_cells)
        if not hasattr(self, 'v') or self.v is None:
            self.v = np.zeros(n_cells)
        if not hasattr(self, 'p') or self.p is None:
            self.p = np.zeros(n_cells)
    
    def solve(self, max_iterations=1000, tolerance=1e-6, save_profile=True, profile_dir='results/profiles'):
        """
        Solve the Navier-Stokes equations using the SIMPLE algorithm with Rhie-Chow interpolation.
        This implementation uses a collocated grid arrangement where all variables are stored at cell centers.
        
        Parameters:
        -----------
        max_iterations : int, optional
            Maximum number of iterations (default: 1000)
        tolerance : float, optional
            Convergence tolerance (default: 1e-6)
        save_profile : bool, optional
            Whether to save profiling data (default: True)
        profile_dir : str, optional
            Directory to save profiling data (default: 'results/profiles')
            
        Returns:
        --------
        SimulationResult
            Object containing simulation results and statistics
        """
        self.profiler.start()
        
        # Get properties of the mesh
        n_cells = self.mesh.n_cells
        n_faces = self.mesh.n_faces
        
        # Intermediate fields
        p_star = self.p.copy()
        p_prime = np.zeros(n_cells)
        
        # Initialize/reset residual histories
        self.residual_history = []  # Overall convergence 
        self.x_momentum_rel_norms = []
        self.y_momentum_rel_norms = []
        self.pressure_rel_norms = []
        
        # Final residual fields
        self._final_u_residual_field = None
        self._final_v_residual_field = None
        self._final_p_residual_field = None
        
        iteration = 1
        # Initialize residuals for convergence check
        u_rel_norm = v_rel_norm = p_rel_norm = 1.0
        total_res_check = max(u_rel_norm, v_rel_norm, p_rel_norm)
        
        print(f"Using α_p = {self.alpha_p}, α_u = {self.alpha_u}")
        print(f"Using collocated grid with Rhie-Chow interpolation")

        # For detecting stalled convergence
        stall_check_window = 50
        stall_threshold = 1e-8
        recent_total_residuals = []

        try:
            while iteration <= max_iterations and total_res_check > tolerance:
                # Store previous solution
                self.u_old = self.u.copy()
                self.v_old = self.v.copy()
                self.p_old = self.p.copy()

                # Create simple field wrappers for the solver
                velocity_field = SimpleVectorField(self.u, self.v)
                pressure_field = SimpleScalarField(p_star)

                # Start momentum solution timing
                self.profiler.start_section("Momentum")
                
                # Solve momentum equations
                u_res_info = None
                v_res_info = None
                
                try:
                    # Solve x-momentum equation
                    u_star, d_u, u_res_info = self.momentum_solver.solve_u_momentum(
                        self.mesh, self.fluid, velocity_field, pressure_field,
                        relaxation_factor=self.alpha_u,
                        boundary_conditions=self.bc_manager,
                        return_dict=True
                    )

                    # Solve y-momentum equation
                    v_star, d_v, v_res_info = self.momentum_solver.solve_v_momentum(
                        self.mesh, self.fluid, velocity_field, pressure_field,
                        relaxation_factor=self.alpha_u,
                        boundary_conditions=self.bc_manager,
                        return_dict=True
                    )
                    
                except Exception as e:
                    print(f"Error solving momentum equations: {e}")
                    raise
                
                self.profiler.end_section("Momentum")
                
                # Save intermediate fields for pressure equation
                self._tmp_u_star = u_star
                self._tmp_v_star = v_star
                self._tmp_d_u = d_u
                self._tmp_d_v = d_v

                # Start pressure solution timing
                self.profiler.start_section("Pressure")
                
                # Solve pressure correction equation with Rhie-Chow interpolation
                try:
                    p_prime, p_res_info = self.pressure_solver.solve(
                        self.mesh, u_star, v_star, d_u, d_v, p_star, 
                        return_dict=True
                    )
                except Exception as e:
                    print(f"Error solving pressure equation: {e}")
                    raise
                
                self.profiler.end_section("Pressure")
                
                # Update pressure with relaxation
                self.p = p_star + self.alpha_p * p_prime
                p_star = self.p.copy()

                # Start velocity correction timing
                self.profiler.start_section("Velocity Correction")
                
                # Update velocities
                if hasattr(self.velocity_updater, 'update_velocity'):
                    # Use the provided velocity updater if compatible
                    self.u, self.v = self.velocity_updater.update_velocity(
                        self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
                    )
                else:
                    # Use a simple implementation for testing
                    self._update_velocity(u_star, v_star, p_prime, d_u, d_v)
                
                self.profiler.end_section("Velocity Correction")

                # Extract relative norms for convergence check
                if u_res_info and v_res_info and p_res_info:
                    u_rel_norm = u_res_info.get('rel_norm', 1.0)
                    v_rel_norm = v_res_info.get('rel_norm', 1.0)
                    p_rel_norm = p_res_info.get('rel_norm', 1.0)
                    
                    # Save residual fields for final visualization
                    u_res_field = u_res_info.get('field')
                    v_res_field = v_res_info.get('field')
                    p_res_field = p_res_info.get('field')
                    
                    # Store relative norms
                    self.x_momentum_rel_norms.append(u_rel_norm)
                    self.y_momentum_rel_norms.append(v_rel_norm)
                    self.pressure_rel_norms.append(p_rel_norm)

                    # Define convergence criteria using relative norms
                    total_res_check = max(u_rel_norm, v_rel_norm, p_rel_norm)

                    # Store total residual for history tracking
                    self.residual_history.append(total_res_check)

                    # Print relative norms
                    print(f"[{iteration}] Relative L2 norms: u: {u_rel_norm:.3e}, "
                          f"v: {v_rel_norm:.3e}, p: {p_rel_norm:.3e}")

                    # Stall check
                    recent_total_residuals.append(total_res_check)
                    if len(recent_total_residuals) > stall_check_window:
                        recent_total_residuals.pop(0)
                        res_change = max(recent_total_residuals) - min(recent_total_residuals)
                        avg_res = np.mean(recent_total_residuals)
                        if avg_res > 0:  # Avoid divide-by-zero
                            rel_change = res_change / avg_res
                            if rel_change < stall_threshold:
                                print(f"Residuals have stalled (<{stall_threshold:.1e} relative change) over the last {stall_check_window} iterations. Stopping early.")
                                break
                
                iteration += 1

        except KeyboardInterrupt:
            print("Interrupted by user.")
        except Exception as e:
            print(f"Error during SIMPLE algorithm: {e}")
            # Still try to return partial results

        # Store the final residual fields
        if 'u_res_field' in locals() and 'v_res_field' in locals() and 'p_res_field' in locals():
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

        self.profiler.start_section("Finalization") # Start timing finalization
        if save_profile:
            os.makedirs(profile_dir, exist_ok=True)
            filename = os.path.join(profile_dir, f"MeshAgnosticSIMPLE_Re{int(self.fluid.get_reynolds_number())}_mesh{n_cells}_profile.h5")
            print(f"Saved profile to {self.save_profiling_data(filename)}")
        self.profiler.end_section("Finalization") # End timing finalization

        return result
        
    def _update_velocity(self, u_star, v_star, p_prime, d_u, d_v):
        """
        Simple velocity update method for arbitrary meshes.
        
        Parameters:
        -----------
        u_star, v_star : ndarray
            Intermediate velocity fields
        p_prime : ndarray
            Pressure correction field
        d_u, d_v : ndarray
            Momentum equation coefficients
        """
        # Get mesh topology
        n_cells = self.mesh.n_cells
        n_faces = self.mesh.n_faces
        owner_cells, neighbor_cells = self.mesh.get_owner_neighbor()
        face_centers = self.mesh.get_face_centers()
        cell_centers = self.mesh.get_cell_centers()
        
        # Initialize updated velocity fields
        self.u = u_star.copy()
        self.v = v_star.copy()
        
        # For each cell, correct velocity using pressure gradient
        for cell_idx in range(n_cells):
            # Get all faces of this cell
            cell_faces = []
            face_directions = []
            
            # Find faces connected to this cell
            for face_idx in range(n_faces):
                owner = owner_cells[face_idx]
                neighbor = neighbor_cells[face_idx]
                
                if owner == cell_idx or neighbor == cell_idx:
                    cell_faces.append(face_idx)
                    # Determine if this cell is the owner or neighbor
                    face_directions.append(1 if owner == cell_idx else -1)
            
            # Calculate pressure gradient for this cell
            # This is a simple approximation - more sophisticated methods would be used
            # in a real implementation
            dp_dx = 0.0
            dp_dy = 0.0
            
            # Use face values to approximate pressure gradient
            for i, face_idx in enumerate(cell_faces):
                direction = face_directions[i]
                if neighbor_cells[face_idx] >= 0:  # Internal face
                    neighbor_idx = owner_cells[face_idx] if owner_cells[face_idx] != cell_idx else neighbor_cells[face_idx]
                    
                    # Vector from cell center to neighbor center
                    dr = cell_centers[neighbor_idx] - cell_centers[cell_idx]
                    dr_mag = np.linalg.norm(dr)
                    
                    # Pressure difference
                    dp = p_prime[neighbor_idx] - p_prime[cell_idx]
                    
                    # Contribute to gradient components
                    if dr_mag > 1e-10:
                        dp_dx += dp * dr[0] / (dr_mag * dr_mag)
                        dp_dy += dp * dr[1] / (dr_mag * dr_mag)
            
            # Apply velocity corrections
            self.u[cell_idx] = u_star[cell_idx] - d_u[cell_idx] * dp_dx
            self.v[cell_idx] = v_star[cell_idx] - d_v[cell_idx] * dp_dy
        
        # Apply boundary conditions if available
        if self.bc_manager:
            # This would call the boundary condition manager to apply BCs
            # to the updated velocity fields
            pass
    
    def _enforce_pressure_boundary_conditions(self):
        """
        Apply boundary conditions to the pressure field.
        This is typically zero gradient for walls and inlets.
        """
        # For pressure, we typically apply zero-gradient BCs at all boundaries
        # This method could implement this if needed
        pass
    
    def set_boundary_condition(self, boundary, condition_type, values=None):
        """
        Set a boundary condition.
        
        Parameters:
        -----------
        boundary : str
            Boundary name ('top', 'bottom', 'left', 'right')
        condition_type : str
            Type of boundary condition ('velocity', 'pressure', 'wall')
        values : dict, optional
            Values for the boundary condition, e.g., {'u': 1.0, 'v': 0.0}
        """
        self.bc_manager.set_condition(boundary, condition_type, values)
        
    def save_profiling_data(self, filename):
        """
        Save profiling data to file.
        
        Parameters:
        -----------
        filename : str
            Path to the output file
            
        Returns:
        --------
        str
            Path to the saved file
        """
        return self.profiler.save_to_file(filename) 