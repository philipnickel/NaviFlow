"""
Lid-driven cavity flow simulation using the object-oriented framework with Gauss-Seidel solver.
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
from naviflow_oo.solver.pressure_solver.gauss_seidel import GaussSeidelSolver
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = 63, 63          # Grid size
reynolds = 100           # Reynolds number
alpha_p = 0.1            # Pressure relaxation factor (lower for stability)
alpha_u = 0.7            # Velocity relaxation factor
max_iterations = 1     # Maximum number of iterations
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
# Use Gauss-Seidel solver for pressure correction
pressure_solver = GaussSeidelSolver(
    tolerance=1e-5,  # Relaxed tolerance for inner iterations
    max_iterations=50000,  # Fewer iterations per SIMPLE iteration
    omega=1.5,  # SOR for better convergence (over-relaxation)
    #use_red_black=True
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
print("Starting simulation...")
result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, save_profile=True, profile_dir=results_dir, track_infinity_norm=True, infinity_norm_interval=5, plot_final_residuals=True)

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
    title=f'Gauss-Seidel Cavity Flow Results (Re={reynolds})',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_gauss_seidel_results.pdf'),
    show=False
)

# Print solver info
solver_info = pressure_solver.get_solver_info()
print("\nSolver Information:")
print(f"Solver name: {solver_info['name']}")
print(f"Total inner iterations: {solver_info['total_inner_iterations']}")
print(f"Convergence rate: {solver_info['convergence_rate']:.6f}" if solver_info['convergence_rate'] is not None else "Convergence rate: N/A")
print(f"Relaxation factor (omega): {solver_info['omega']}")

# Print comparison with iterations needed for Jacobi solver
# This is an estimated comparison based on theory
gs_conv_rate = solver_info['convergence_rate'] if solver_info['convergence_rate'] is not None else 0.9
jacobi_est_rate = gs_conv_rate**0.5  # Theoretical relationship
print(f"\nEstimated improvement over Jacobi solver:")
print(f"Estimated Jacobi convergence rate: {jacobi_est_rate:.6f}")
iter_reduction = np.log(tolerance) / np.log(gs_conv_rate) * np.log(jacobi_est_rate) / np.log(tolerance)
print(f"Estimated iteration reduction: {iter_reduction:.2f}x") 