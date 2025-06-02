"""
Lid-driven cavity flow simulation using the object-oriented framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import yaml
import subprocess
from naviflow_staggered.preprocessing.mesh.structured import StructuredMesh
from naviflow_staggered.constructor.properties.fluid import FluidProperties
from naviflow_staggered.preprocessing.fields.scalar_field import ScalarField
from naviflow_staggered.preprocessing.fields.vector_field import VectorField
from naviflow_staggered.solver.Algorithms.simple import SimpleSolver
from naviflow_staggered.solver.pressure_solver.direct import DirectPressureSolver
from naviflow_staggered.solver.momentum_solver.jacobi_solver import JacobiMomentumSolver
from naviflow_staggered.solver.momentum_solver.jacobi_matrix_solver import JacobiMatrixMomentumSolver
from naviflow_staggered.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_staggered.solver.momentum_solver.matrix_free_momentum import MatrixFreeMomentumSolver
from naviflow_staggered.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_staggered.postprocessing.visualization import plot_final_residuals

# Start timing
start_time = time.time()
# 1. Set up simulation parameters
nx, ny = 2**6-1, 2**6-1 # Grid size
reynolds = 100             # Reynolds number
alpha_p = 0.2              # Pressure relaxation factor
alpha_u = 0.8              # Velocity relaxation factor
max_iterations = 5000     # Maximum number of iterations
tolerance = 1e-5

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

#momentum_solver = JacobiMatrixMomentumSolver(n_jacobi_sweeps=1)
#momentum_solver = CGMatrixMomentumSolver(tolerance=1e-1, max_iterations=1000)
# Use the new AMG solver
#momentum_solver = AMGMomentumSolver(discretization_scheme='power_law', tolerance=1e-7, max_iterations=10000)
momentum_solver = MatrixFreeMomentumSolver(discretization_scheme='power_law', tolerance=1e-8, max_iterations=100000, solver_type='bicgstab')
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


# 12. Save metadata
metadata = {
    'Simulation id': result.simulation_id if hasattr(result, 'simulation_id') else 'unknown',
    'Experiment': 'lidDrivenCavity',
    'Git commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
    'Wall time (s)': f'{elapsed_time:.2f}',
    'Numba threads': 'default',
    'Number of control volumes': nx * ny,
    'Mesh type': 'Structured',
    'Boundary conditions': 'shared_configs/domain/boundaries_lidDrivenCavity.yaml',
    'Reynolds number': str(reynolds),
    'Algorithm': 'SIMPLE',
    'Convection scheme': 'power_law',  # Hardcode this since we know what we're using
    'Limiter': 'None',
    'Convergence tolerance': tolerance,
    'Number of iterations': result.iterations,
    'Final u-residual': f'{result.get_history("u_rel_norm")[-1]:.1e}',
    'Final v-residual': f'{result.get_history("v_rel_norm")[-1]:.1e}',
    'Final continuity-residual': f'{result.get_history("p_rel_norm")[-1]:.1e}',
    'Momentum relaxation': str(alpha_u),
    'Pressure relaxation': str(alpha_p),
    'Pressure solver': pressure_solver.__class__.__name__.lower(),
    'Pressure solver preconditioner': 'none',
    'Pressure solver tolerance': str(pressure_solver.tolerance if hasattr(pressure_solver, 'tolerance') else '1e-8'),
    'Momentum solver': momentum_solver.__class__.__name__.lower(),
    'Momentum solver preconditioner': 'none',
    'Momentum solver tolerance': str(momentum_solver.tolerance if hasattr(momentum_solver, 'tolerance') else '1e-7')
}

# Save metadata to YAML file
with open(os.path.join(results_dir, 'metadata.yaml'), 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False)

# 13. Save solution fields
# For staggered grid, u and v have different dimensions
# Save them separately
# Interpolate staggered u and v to cell centers
u_centers = 0.5 * (result.u[:-1, :] + result.u[1:, :])  # (63, 63)
v_centers = 0.5 * (result.v[:, :-1] + result.v[:, 1:])  # (63, 63)

# Flatten and stack to get (3969, 2) shape
U_final = np.column_stack((u_centers.flatten(), v_centers.flatten()))
np.save(os.path.join(results_dir, 'U_final.npy'), U_final)

# Save original staggered fields as well
np.save(os.path.join(results_dir, 'u_final.npy'), result.u)
np.save(os.path.join(results_dir, 'v_final.npy'), result.v)
# Flatten pressure field for postprocessing compatibility
np.save(os.path.join(results_dir, 'p_final.npy'), result.p.flatten())
np.savez(os.path.join(results_dir, 'residuals.npz'), 
         u=result.get_history('u_rel_norm'),
         v=result.get_history('v_rel_norm'),
         cont=result.get_history('p_rel_norm'))

# Create meshgrid for cell centers
X, Y = np.meshgrid(mesh.x, mesh.y, indexing='ij')
x_flat = X.flatten()
y_flat = Y.flatten()

# Save cell centers as flattened meshgrid arrays
np.savez(os.path.join(results_dir, 'cell_centers.npz'), x=x_flat, y=y_flat)

# Convert staggered arrays to cell-centered velocities
u_centers = 0.5 * (result.u[:-1, :] + result.u[1:, :])  # (nx, ny)
v_centers = 0.5 * (result.v[:, :-1] + result.v[:, 1:])  # (nx, ny)
U_final = np.column_stack((u_centers.flatten(), v_centers.flatten()))  # (nx*ny, 2)
np.save(os.path.join(results_dir, 'U_final.npy'), U_final)

# Save residual fields for postprocessing
np.save(os.path.join(results_dir, 'u_residual.npy'), algorithm._final_u_residual_field.flatten()[:3969])
np.save(os.path.join(results_dir, 'v_residual.npy'), algorithm._final_v_residual_field.flatten()[:3969])
np.save(os.path.join(results_dir, 'continuity_field.npy'), algorithm._final_p_residual_field.flatten())
