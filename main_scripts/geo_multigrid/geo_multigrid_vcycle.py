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
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Create debug output directory
debug_dir = 'debug_output'
os.makedirs(debug_dir, exist_ok=True)

# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = 63, 63           # Grid size (2^6-1)
reynolds = 100           # Reynolds number
alpha_p = 0.1            # Pressure relaxation factor
alpha_u = 0.7            # Velocity relaxation factor
max_iterations = 1       # Just 1 iteration for debugging
tolerance = 1e-5         # Convergence tolerance

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
# Create a Jacobi smoother for the multigrid solver
jacobi_smoother = JacobiSolver(tolerance=1e-12, omega=0.9)

# Use multigrid solver with Jacobi smoother for pressure correction
pressure_solver = MultiGridSolver(
    tolerance=1e-8,  # Tighter tolerance
    max_iterations=1,  # Only 1 multigrid iteration for debugging
    pre_smoothing=5,  # 3 pre-smoothing steps
    post_smoothing=5,  # 3 post-smoothing steps
    smoother_omega=0.8,  # Adjusted relaxation factor for better performance
    smoother=jacobi_smoother,  # Use the Jacobi smoother
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
result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, 
                         track_infinity_norm=True, infinity_norm_interval=5)

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
    title=f'Multigrid with Jacobi Smoother Cavity Flow Results (Re={reynolds})',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_multigrid_jacobi_results.pdf'),
    show=False
)

# After the simulation completes, plot the V-cycle results
algorithm.pressure_solver.plot_vcycle_results(os.path.join(debug_dir, 'vcycle_analysis.pdf'))