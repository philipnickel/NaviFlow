"""
Lid-driven cavity flow simulation using the object-oriented framework.
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
from naviflow_oo.solver.pressure_solver.direct import DirectPressureSolver
from naviflow_oo.solver.momentum_solver.jacobi_solver import JacobiMomentumSolver
from naviflow_oo.solver.momentum_solver.jacobi_matrix_solver import JacobiMatrixMomentumSolver
from naviflow_oo.solver.momentum_solver.CG_matrix_solver import CGMatrixMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.postprocessing.visualization import plot_final_residuals, plot_u_v_continuity_residuals

# Start timing
start_time = time.time()
# 1. Set up simulation parameters
nx, ny = 2**6-1, 2**6-1 # Grid size
reynolds = 100             # Reynolds number
alpha_p = 0.1              # Pressure relaxation factor
alpha_u = 0.8              # Velocity relaxation factor
max_iterations = 200     # Maximum number of iterations
tolerance = 1e-4


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
pressure_solver = DirectPressureSolver()

#momentum_solver = JacobiMomentumSolver(n_jacobi_sweeps=5)
momentum_solver = JacobiMatrixMomentumSolver(n_jacobi_sweeps=5) 
#momentum_solver = CGMatrixMomentumSolver(tolerance=1e-1, max_iterations=1000) 

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
    result.u, result.v, result.p,
    algorithm.u_old, algorithm.v_old, algorithm.p_old,
    mesh,
    title=f'Final Residuals (Re={reynolds})',
    filename=os.path.join(results_dir, f'final_residuals_Re{reynolds}.pdf'),
    show=False
)

# 12. Visualize residual history
plot_u_v_continuity_residuals(
    algorithm.x_momentum_residuals, 
    algorithm.y_momentum_residuals, 
    algorithm.continuity_residuals,
    title=f'Residual History (Re={reynolds})',
    filename=os.path.join(results_dir, f'residual_history_Re{reynolds}.pdf'),
    show=False
)



