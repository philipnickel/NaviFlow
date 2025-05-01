"""
Lid-driven cavity flow simulation using the object-oriented framework with matrix-free solver.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from naviflow_staggered.preprocessing.mesh.structured import StructuredMesh
from naviflow_staggered.constructor.properties.fluid import FluidProperties
from naviflow_staggered.solver.Algorithms.simple import SimpleSolver
from naviflow_staggered.solver.pressure_solver.matrix_free_BiCGSTAB import MatrixFreeBiCGSTABSolver
from naviflow_staggered.solver.momentum_solver.jacobi_solver import JacobiMomentumSolver
from naviflow_staggered.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_staggered.solver.momentum_solver.BiCGSTAB_solver import BiCGSTABMomentumSolver
from naviflow_staggered.solver.momentum_solver.matrix_free_momentum import MatrixFreeMomentumSolver
from naviflow_staggered.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_staggered.postprocessing.visualization import plot_final_residuals
# Start timing

start_time = time.time()
# 1. Set up simulation parameters
nx, ny = 2**9-1, 2**9-1 # Grid size
reynolds = 3200             # Reynolds number
alpha_p = 0.3              # Pressure relaxation factor
alpha_u = 0.7         # Velocity relaxation factor
max_iterations = 30000     # Maximum number of iterations
tolerance = 1e-4
h = 1/nx 
disc_order = 1
expected_disc_error = h**(disc_order)
#pressure_tolerance = expected_disc_error 
pressure_tolerance = 1e-6
print(f"Expected disc error: {expected_disc_error}")
print(f"Tolerance: {tolerance}")
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
# Use matrix-free conjugate gradient solver instead of direct solver
pressure_solver = MatrixFreeBiCGSTABSolver(
    tolerance=pressure_tolerance,
    max_iterations=100000,
    use_preconditioner=True,
    preconditioner='multigrid',
    mg_pre_smoothing=1,
    mg_post_smoothing=1,
    mg_cycle_type='v',
    mg_max_cycles_buildup=1,
    mg_cycle_type_buildup='v',
    mg_restriction_method='restrict_full_weighting',
    mg_interpolation_method='interpolate_cubic',
    smoother_relaxation=1.5,
    smoother_method_type='red_black'
)
#momentum_solver = AMGMomentumSolver(tolerance=1e-6, max_iterations=10000)
#momentum_solver = BiCGSTABMomentumSolver(tolerance=1e-6, max_iterations=10000)
momentum_solver = MatrixFreeMomentumSolver(tolerance=1e-6, max_iterations=10000, solver_type='gmres')
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
result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, save_profile=True, profile_dir=results_dir, track_infinity_norm=True, infinity_norm_interval=10, use_l2_norm=True)

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
    title=f'BiCGSTAB Cavity Flow Results (Re={reynolds}) Resolution {nx}x{ny}',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_BiCGSTAB_results.pdf'),
    show=False
)


# 11. Visualize final residuals
plot_final_residuals(
    algorithm._final_u_residual_field, 
    algorithm._final_v_residual_field, 
    algorithm._final_p_residual_field,
    mesh,
    title=f'Final Algebraic Residual Fields (Re={reynolds})',
    filename=os.path.join(results_dir, f'final_algebraic_residual_fields_Re{reynolds}_BiCGSTAB.pdf'),
    show=False,
    u_rel_norms=result.get_history('u_rel_norm'),
    v_rel_norms=result.get_history('v_rel_norm'),
    p_rel_norms=result.get_history('p_rel_norm'),
    history_filename=os.path.join(results_dir, f'unrelaxed_rel_residual_history_Re{reynolds}_BiCGSTAB.pdf')
)
