"""
Lid-driven cavity flow simulation using the object-oriented framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
# from naviflow_oo.preprocessing.mesh.structured import StructuredUniform # REMOVED old import
from naviflow_staggered.preprocessing.mesh.structured_mesh import StructuredMesh # ADDED new import
from naviflow_staggered.constructor.properties.fluid import FluidProperties
from naviflow_staggered.preprocessing.fields.scalar_field import ScalarField
from naviflow_staggered.preprocessing.fields.vector_field import VectorField
from naviflow_staggered.solver.Algorithms.simple import SimpleSolver
from naviflow_staggered.solver.mesh_agnostic.mesh_direct_pressure import MeshAgnosticDirectPressureSolver
from naviflow_staggered.solver.momentum_solver.jacobi_solver import JacobiMomentumSolver
from naviflow_staggered.solver.momentum_solver.jacobi_matrix_solver import JacobiMatrixMomentumSolver
from naviflow_staggered.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_staggered.solver.momentum_solver.matrix_free_momentum import MatrixFreeMomentumSolver
from naviflow_staggered.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_staggered.postprocessing.visualization import plot_final_residuals
from naviflow_staggered.solver.pressure_solver import DirectPressureSolver
# Start timing
start_time = time.time()
# 1. Set up simulation parameters
nx, ny = 2**5-1, 2**5-1 # Grid size
reynolds = 100             # Reynolds number
alpha_p = 0.3              # Pressure relaxation factor (Reduced)
alpha_u = 0.5              # Velocity relaxation factor (Reduced further for stability)
max_iterations = 1     # Maximum number of iterations
tolerance = 1e-10



# 2. Create mesh
mesh = StructuredMesh(n_cells_x=nx, n_cells_y=ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, is_uniform=True)
print(f"Created mesh with {nx}x{ny} cells")
# dx, dy = mesh.get_cell_sizes() # Removed, handled internally
# print(f"Cell sizes: dx={dx:.6f}, dy={dy:.6f}") # Removed

# 3. Define fluid properties
fluid = FluidProperties(
    density=1.0,
    reynolds_number=reynolds,
    characteristic_velocity=1.0
)
print(f"Reynolds number: {fluid.get_reynolds_number()}")
print(f"Calculated viscosity: {fluid.get_viscosity()}")

# 4. Create solvers
pressure_solver = DirectPressureSolver()

# Use the new AMG solver
momentum_solver = AMGMomentumSolver(discretization_scheme='upwind', tolerance=1e-7, max_iterations=10000)
velocity_updater = StandardVelocityUpdater()

# 5. Create algorithm
algorithm = SimpleSolver(
    mesh=mesh,
    fluid=fluid,
    pressure_solver=pressure_solver,
    momentum_solver=momentum_solver,
    velocity_updater=velocity_updater,
    alpha_p=alpha_p,
    alpha_u=alpha_u,
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
result = algorithm.solve(
    max_iterations=max_iterations,
    tolerance=tolerance,
    save_profile=True,
    profile_dir=results_dir,
    track_infinity_norm=True,
    infinity_norm_interval=10,
    #use_l2_norm=True  
)

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
    title=f'Cavity Flow Results (Re={reynolds})',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_results.pdf'),
    show=True
)

# 11. Visualize final residuals
plot_final_residuals(
    algorithm._final_u_residual_field, 
    algorithm._final_v_residual_field, 
    algorithm._final_p_residual_field,
    mesh,
    title=f'Final Algebraic Residual Fields (Re={reynolds})',
    filename=os.path.join(results_dir, f'final_algebraic_residual_fields_Re{reynolds}.pdf'),
    show=False,
    u_rel_norms=result.get_history('u_rel_norm'),
    v_rel_norms=result.get_history('v_rel_norm'),
    p_rel_norms=result.get_history('p_rel_norm'),
    history_filename=os.path.join(results_dir, f'unrelaxed_rel_residual_history_Re{reynolds}.pdf')
)