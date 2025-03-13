"""
Lid-driven cavity flow simulation using the object-oriented framework with F-cycle multigrid solver.

This script tests the F-cycle multigrid solver for the lid-driven cavity problem.
F-cycles provide better convergence than V-cycles by combining aspects of V-cycles and W-cycles.
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

# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = 511, 511          # Grid size (2^6-1)
reynolds = 10000          # Reynolds number (higher)
alpha_p = 0.3            # Pressure relaxation factor
alpha_u = 0.7            # Velocity relaxation factor
max_iterations = 10      # Maximum number of SIMPLE iterations
tolerance = 1e-8         # Convergence tolerance

# 2. Create mesh
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)

# 3. Define fluid properties
fluid = FluidProperties(
    density=1.0,
    reynolds_number=reynolds,
    characteristic_velocity=1.0
)



# 4. Create solvers
# Create a Jacobi smoother for the multigrid solver
jacobi_smoother = JacobiSolver()

# Use F-cycle multigrid solver with Jacobi smoother for pressure correction
pressure_solver = MultiGridSolver(
    tolerance=1e-4,  # Tighter tolerance for pressure solver
    max_iterations=15,  # More iterations per SIMPLE iteration
    pre_smoothing=3,
    post_smoothing=3,
    smoother_iterations=3,
    smoother_omega=0.8,
    smoother=jacobi_smoother,
    cycle_type='f'  # Use F-cycle
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
print("Starting simulation with F-cycle multigrid solver...")
result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, save_profile=True, profile_dir=results_dir)

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# 8. Print results
print(f"F-cycle simulation completed in {elapsed_time:.2f} seconds")
print(f"Total Iterations = {result.iterations}")

# 9. Check mass conservation
max_div = result.get_max_divergence()
print(f"Maximum absolute divergence: {max_div:.6e}")

# 10. Visualize results
result.plot_combined_results(
    title=f'F-cycle Multigrid Cavity Flow Results (Re={reynolds})',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_fcycle_multigrid_results.pdf'),
    show=False
)
