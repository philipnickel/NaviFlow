"""
Lid-driven cavity flow simulation using the object-oriented framework with PyAMG solver.

This script tests the SIMPLE algorithm with PyAMG solver for the lid-driven cavity problem.
PyAMG provides efficient algebraic multigrid methods that can significantly accelerate convergence.
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
from naviflow_oo.solver.Algorithms.simple import SimpleSolver
from naviflow_oo.solver.pressure_solver.pyamg_solver import PyAMGSolver
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.postprocessing.visualization import plot_final_residuals
# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = 2**10-1, 2**10-1          # Grid size (63x63 to match MATLAB example)
reynolds = 10000           # Reynolds number
alpha_p = 0.3            # Pressure relaxation factor
alpha_u = 0.7            # Velocity relaxation factor
max_iterations = 10# Maximum number of iterations
tolerance = 1e-4         # Convergence tolerance

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
# Use PyAMG solver for pressure correction
pressure_solver = PyAMGSolver(
    tolerance=1e-5,
    max_iterations=100000,
    smoother='gauss_seidel',
    presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
    postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
    cycle_type='V'
)
momentum_solver = StandardMomentumSolver()
velocity_updater = StandardVelocityUpdater()

# 5. Create algorithm
algorithm = SimpleSolver(
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
print("Starting simulation with SIMPLE algorithm and PyAMG solver...")
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
    title=f'PyAMG Cavity Flow Results (Re={reynolds})',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_pyamg_results.pdf'),
    show=False
)

# 11. Visualize final residuals
plot_final_residuals(
    result.u, result.v, result.p,
    algorithm.u_old, algorithm.v_old, algorithm.p_old,
    mesh,
    title=f'Final Residuals (Re={reynolds})',
    filename=os.path.join(results_dir, f'final_residuals_Re{reynolds}.pdf'),
    show=False
)
