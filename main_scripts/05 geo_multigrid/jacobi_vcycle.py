"""
Lid-driven cavity flow simulation using the object-oriented framework with multigrid solver
that uses JacobiSolver as the smoother.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.preprocessing.fields.scalar_field import ScalarField
from naviflow_oo.preprocessing.fields.vector_field import VectorField
from naviflow_oo.solver.Algorithms.simple import SimpleSolver
from naviflow_oo.solver.pressure_solver.multigrid import MultiGridSolver
from naviflow_oo.solver.pressure_solver.jacobi import JacobiSolver
from naviflow_oo.solver.momentum_solver.power_law import PowerLawMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.postprocessing.visualization import plot_final_residuals

# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = 2**6-1, 2**6-1 # Grid size
reynolds = 100             # Reynolds number
alpha_p = 0.3              # Pressure relaxation factor
alpha_u = 0.7              # Velocity relaxation factor
max_iterations = 10000     # Maximum number of iterations

h = 1/nx 
disc_order = 1
expected_disc_error = h**(disc_order)
tolerance = expected_disc_error * 1e-3
pressure_tolerance = expected_disc_error 
print(f"Tolerance: {tolerance}")
print(f"Pressure tolerance: {pressure_tolerance}")

# Create mesh
print(f"Creating mesh with {nx}x{ny} cells...")
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
dx, dy = mesh.get_cell_sizes()
print(f"Cell sizes: dx={dx}, dy={dy}")

# Create initial conditions
Re = 100
print(f"Reynolds number: {Re}")
viscosity = 0.01
print(f"Calculated viscosity: {viscosity}")

# 4. Create solvers
# Create a Jacobi smoother for the multigrid solver
smoother = JacobiSolver(omega=0.79)  # Increased from 0.5 for better convergence

# Create multigrid solver with improved parameters
multigrid_solver = MultiGridSolver(
    smoother=smoother,
    max_iterations=1000,        # Increased from 1000
    tolerance=tolerance,             # Tighter tolerance (was 1e-4)
    pre_smoothing=20,            # Using default values (was 10)
    post_smoothing=20,           # Using default values (was 10)
    cycle_type='w',
    coarsest_grid_size=7,
    restriction_method='restrict_inject',
    interpolation_method='interpolate_cubic'
)
momentum_solver = PowerLawMomentumSolver()
velocity_updater = StandardVelocityUpdater()

# 5. Create algorithm
algorithm = SimpleSolver(
    mesh=mesh,
    fluid=FluidProperties(
        density=1.0,
        reynolds_number=Re,
        characteristic_velocity=1.0
    ),
    pressure_solver=multigrid_solver,
    momentum_solver=momentum_solver,
    velocity_updater=velocity_updater,
    alpha_p=alpha_p,
    alpha_u=alpha_u
)

# 6. Set boundary conditions
algorithm.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
algorithm.set_boundary_condition('bottom', 'wall')
algorithm.set_boundary_condition('left', 'wall')
algorithm.set_boundary_condition('right', 'wall')

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# 7. Solve the problem
print("Starting simulation...")
result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, 
                         track_infinity_norm=True, infinity_norm_interval=5, save_profile=True, profile_dir=results_dir)

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# 8. Print results
print(f"Simulation completed in {elapsed_time:.2f} seconds")
print(f"Total Iterations = {result.iterations}")

# 9. Check mass conservation
max_div = result.get_max_divergence()
print(f"Maximum absolute divergence: {max_div:.6e}")

# 10. Visualize results
result.plot_combined_results(
    title=f'Multigrid with Jacobi Smoother Cavity Flow Results (Re={Re})',
    filename=os.path.join(results_dir, f'cavity_Re{Re}_multigrid_jacobi_results.pdf'),
    show=False
)
# 11. Visualize the final residuals
# 11. Visualize final residuals
plot_final_residuals(
    result.u, result.v, result.p,
    algorithm.u_old, algorithm.v_old, algorithm.p_old,
    mesh,
    title=f'Final Residuals (Re={Re})',
    filename=os.path.join(results_dir, f'final_residuals_Re{Re}_multigrid_jacobi.pdf'),
    show=False
)

