"""
Lid-driven cavity flow simulation using the object-oriented framework with Preconditioned CG solver.

This script tests the SIMPLER algorithm with a Preconditioned Conjugate Gradient solver for the lid-driven cavity problem.
The solver uses PyAMG as a preconditioner to accelerate convergence of the conjugate gradient method.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the parent directory to the path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.solver.Algorithms.simpler import SimplerSolver
from naviflow_oo.solver.pressure_solver.preconditioned_cg_solver import PreconditionedCGSolver
from naviflow_oo.solver.momentum_solver.jacobi_solver import JacobiMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.postprocessing.visualization import plot_final_residuals

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = 63, 63         # Grid size
reynolds = 100           # Reynolds number
alpha_p = 0.1             # Pressure relaxation factor
alpha_u = 0.8             # Velocity relaxation factor
max_iterations = 300     # Maximum iterations
tolerance = 1e-4          # Convergence tolerance
h = 1/nx 
disc_order = 1
expected_disc_error = h**(disc_order)
pressure_tolerance = expected_disc_error 
print(f"Pressure tolerance: {pressure_tolerance}")

# 2. Create mesh
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
print(f"Created mesh with {nx}x{ny} cells")
print(f"Cell sizes: dx={mesh.dx:.6f}, dy={mesh.dy:.6f}")

# 3. Define fluid properties
fluid = FluidProperties(
    density=1.0,
    reynolds_number=reynolds,
    characteristic_velocity=1.0
)
print(f"Reynolds number: {fluid.get_reynolds_number()}")
print(f"Calculated viscosity: {fluid.get_viscosity()}")

# 4. Create solvers
# Use Preconditioned CG solver for pressure correction
pressure_solver = PreconditionedCGSolver(
    tolerance=pressure_tolerance,
    max_iterations=10000,
    smoother='gauss_seidel',
    presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
    postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
    cycle_type='V'
)
momentum_solver = JacobiMomentumSolver(n_jacobi_sweeps=5)
velocity_updater = StandardVelocityUpdater()

# 5. Create algorithm
algorithm = SimplerSolver(
    mesh=mesh,
    fluid=fluid,
    pressure_solver=pressure_solver,
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

# 7. Solve the problem
print("Starting simulation with SIMPLER algorithm and Preconditioned CG solver...")
result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, save_profile=True, profile_dir=results_dir, track_infinity_norm=True, infinity_norm_interval=10)

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
    title=f'Preconditioned CG Cavity Flow with SIMPLER Results (Re={reynolds})',
    filename=os.path.join(results_dir, f'SIMPLER_cavity_Re{reynolds}_preconditioned_cg_results.pdf'),
    show=False
)

# 11. Visualize final residuals
result.plot_residuals(
    title=f'Final Residuals for Preconditioned CG Cavity Flow (Re={reynolds})',
    filename=os.path.join(results_dir, f'SIMPLER_Re={reynolds}_final_residuals.pdf'),
    show=False
)
