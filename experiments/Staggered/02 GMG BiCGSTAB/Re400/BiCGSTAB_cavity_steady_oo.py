"""
Lid-driven cavity flow simulation using the object-oriented framework with matrix-free solver.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import yaml
import subprocess
import hashlib
from naviflow_staggered.preprocessing.mesh.structured import StructuredMesh
from naviflow_staggered.constructor.properties.fluid import FluidProperties
from naviflow_staggered.solver.Algorithms.simple import SimpleSolver
from naviflow_staggered.solver.pressure_solver.matrix_free_BiCGSTAB import MatrixFreeBiCGSTABSolver
from naviflow_staggered.solver.momentum_solver.jacobi_solver import JacobiMomentumSolver
from naviflow_staggered.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_staggered.solver.momentum_solver.BiCGSTAB_solver import MatrixMomentumSolver
from naviflow_staggered.solver.momentum_solver.matrix_free_momentum import MatrixFreeMomentumSolver
from naviflow_staggered.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_staggered.postprocessing.visualization import plot_final_residuals
# Start timing

start_time = time.time()
# 1. Set up simulation parameters
nx, ny = 2**8-1, 2**8-1 # Grid size
reynolds = 400             # Reynolds number
alpha_p = 0.3              # Pressure relaxation factor
alpha_u = 0.6         # Velocity relaxation factor
max_iterations = 2     # Maximum number of iterations
tolerance = 1e-7
h = 1/nx 
disc_order = 1
expected_disc_error = h**(disc_order)
#pressure_tolerance = expected_disc_error 
pressure_tolerance = 1e-10
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
#momentum_solver = MatrixMomentumSolver(tolerance=1e-6, max_iterations=10000)
momentum_solver = MatrixFreeMomentumSolver(tolerance=1e-8, max_iterations=10000, solver_type='bicgstab')
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

# 12. Save data in collocated format for post-processing compatibility
print("Saving data in collocated format...")

# Generate a unique simulation ID with random hash (same approach as collocated)
random_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
sim_id = random_hash

# Save metadata in YAML format
metadata = {
    'Simulation id': sim_id,
    'Experiment': 'lidDrivenCavity',
    'Git commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
    'Wall time (s)': f'{elapsed_time:.2f}',
    'Reynolds number': reynolds,
    'Mesh type': 'Structured',
    'Number of control volumes': nx * ny,
    'Boundary conditions': 'staggered_cavity_boundaries',
    'Algorithm': 'SIMPLE',
    'Convection scheme': 'Power Law',
    'Convergence tolerance': tolerance,
    'Momentum relaxation': f'{alpha_u}',
    'Pressure relaxation': f'{alpha_p}',
    'Momentum solver': 'bcgs',
    'Momentum solver preconditioner': 'ilu',
    'Momentum solver tolerance': '1e-6',
    'Pressure solver': 'bcgs',
    'Pressure solver preconditioner': 'geometric multigrid',
    'Pressure solver tolerance': f'{pressure_tolerance}',
    'Multigrid cycle type': 'v-cycle',
    'Multigrid smoother': 'red-black Gauss-Seidel',
    'Multigrid pre-smoothing steps': 1,
    'Multigrid post-smoothing steps': 1,
    'Multigrid smoother relaxation': 1.5,
    'Number of iterations': result.iterations,
    'Final u-residual': f'{result.get_history("u_rel_norm")[-1]:.2e}',
    'Final v-residual': f'{result.get_history("v_rel_norm")[-1]:.2e}',
    'Final continuity-residual': f'{result.get_history("p_rel_norm")[-1]:.2e}',
}

with open(os.path.join(results_dir, 'metadata.yaml'), 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False)

# Convert staggered grids to cell-centered for post-processing compatibility
# For staggered grid: u is at x-faces, v is at y-faces, p is at cell centers
# Interpolate u and v to cell centers
u_staggered = result.u  # Shape: (nx+1, ny)
v_staggered = result.v  # Shape: (nx, ny+1)
p_centers = result.p    # Shape: (nx, ny)

# Interpolate u from x-faces to cell centers
u_centers = 0.5 * (u_staggered[:-1, :] + u_staggered[1:, :])  # (nx, ny)

# Interpolate v from y-faces to cell centers  
v_centers = 0.5 * (v_staggered[:, :-1] + v_staggered[:, 1:])  # (nx, ny)

# Create velocity field as (N, 2) array where N = nx*ny
U_final = np.column_stack((u_centers.flatten(), v_centers.flatten()))
np.save(os.path.join(results_dir, 'U_final.npy'), U_final)

# Save pressure field (flattened)
np.save(os.path.join(results_dir, 'p_final.npy'), p_centers.flatten())

# Create cell center coordinates
x_centers = np.linspace(mesh.dx/2, 1.0 - mesh.dx/2, nx)
y_centers = np.linspace(mesh.dy/2, 1.0 - mesh.dy/2, ny)
X_centers, Y_centers = np.meshgrid(x_centers, y_centers, indexing='ij')
x_flat = X_centers.flatten()
y_flat = Y_centers.flatten()

# Save cell centers
np.savez(os.path.join(results_dir, 'cell_centers.npz'), x=x_flat, y=y_flat)

# Save residual history
np.savez(os.path.join(results_dir, 'residuals.npz'), 
         u=np.array(result.get_history('u_rel_norm')),
         v=np.array(result.get_history('v_rel_norm')),
         cont=np.array(result.get_history('p_rel_norm')))

# Save final residual fields (interpolated to cell centers)
if hasattr(algorithm, '_final_u_residual_field'):
    u_res_centers = 0.5 * (algorithm._final_u_residual_field[:-1, :] + algorithm._final_u_residual_field[1:, :])
    np.save(os.path.join(results_dir, 'u_residual.npy'), u_res_centers.flatten())
else:
    # Create dummy residual field if not available
    np.save(os.path.join(results_dir, 'u_residual.npy'), np.zeros(nx*ny))

if hasattr(algorithm, '_final_v_residual_field'):
    v_res_centers = 0.5 * (algorithm._final_v_residual_field[:, :-1] + algorithm._final_v_residual_field[:, 1:])
    np.save(os.path.join(results_dir, 'v_residual.npy'), v_res_centers.flatten())
else:
    # Create dummy residual field if not available
    np.save(os.path.join(results_dir, 'v_residual.npy'), np.zeros(nx*ny))

if hasattr(algorithm, '_final_p_residual_field'):
    np.save(os.path.join(results_dir, 'continuity_field.npy'), algorithm._final_p_residual_field.flatten())
else:
    # Create dummy residual field if not available
    np.save(os.path.join(results_dir, 'continuity_field.npy'), np.zeros(nx*ny))

print(f"Data saved in collocated format to {results_dir}")
print(f"You can now run post-processing with:")
print(f"python naviflow_collocated/utils/postprocess/postprocess.py --config {os.path.join(os.path.dirname(__file__), 'pseudo_config.yaml')} --all")

# Create a pseudo config file for post-processing compatibility
pseudo_config = {
    'experiment': 'lidDrivenCavity',
    'physical_properties': {
        'reynolds_number': reynolds
    },
    'algorithm': {
        'convection_discretization': 'Power Law'
    },
    'domain': {
        'mesh': ['structured', f'{nx}x{ny}']
    }
}

with open(os.path.join(os.path.dirname(__file__), 'pseudo_config.yaml'), 'w') as f:
    yaml.dump(pseudo_config, f, default_flow_style=False)

print("Created pseudo_config.yaml for post-processing compatibility")
