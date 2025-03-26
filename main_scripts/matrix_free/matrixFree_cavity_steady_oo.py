"""
Lid-driven cavity flow simulation using the object-oriented framework with matrix-free solver.
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
from naviflow_oo.solver.pressure_solver.matrix_free_cg import MatrixFreeCGSolver
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.solver.momentum_solver.discretization.convection_schemes import QuickDiscretization
# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = 127, 127          # Grid size
reynolds = 100             # Reynolds number
alpha_p = 0.1              # Pressure relaxation factor
alpha_u = 0.7              # Velocity relaxation factor
max_iterations = 1    # Maximum number of iterations (reduced for quick test)
tolerance = 1e-5           # Convergence tolerance

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
# Use matrix-free conjugate gradient solver instead of direct solver
pressure_solver = MatrixFreeCGSolver(tolerance=1e-12, max_iterations=1000000)
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

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# 7. Solve the problem
print("Starting simulation...")
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
    title=f'Matrix-Free Cavity Flow Results (Re={reynolds})',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_matrix_free_results.pdf'),
    show=False
)

