"""
Lid-driven cavity flow simulation using the object-oriented framework with multigrid solver
that uses GaussSeidelSolver as the smoother.
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
from naviflow_staggered.preprocessing.fields.scalar_field import ScalarField
from naviflow_staggered.preprocessing.fields.vector_field import VectorField
from naviflow_staggered.solver.Algorithms.simple import SimpleSolver
from naviflow_staggered.solver.pressure_solver.multigrid import MultiGridSolver
from naviflow_staggered.solver.pressure_solver.gauss_seidel import GaussSeidelSolver
from naviflow_staggered.solver.momentum_solver.jacobi_solver import JacobiMomentumSolver
from naviflow_staggered.solver.momentum_solver.jacobi_matrix_solver import JacobiMatrixMomentumSolver
from naviflow_staggered.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_staggered.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_staggered.solver.momentum_solver.matrix_free_momentum import MatrixFreeMomentumSolver
from naviflow_staggered.solver.momentum_solver.matrix_free_momentum_PETSc import MatrixFreeMomentumSolverPETSc
from naviflow_staggered.postprocessing.visualization import plot_final_residuals
# Start timing
start_time = time.time()
# 1. Set up simulation parameters
nx, ny = 2**7-1, 2**7-1 # Grid size
reynolds = 3200            # Reynolds number
alpha_p = 0.1              # Pressure relaxation factor
alpha_u = 0.4              # Velocity relaxation factor
max_iterations = 35000     # Maximum number of iterations

tolerance = 1e-8
#pressure_tolerance = expected_disc_error
pressure_tolerance = 1e-3 # not used

# Create mesh
print(f"Creating mesh with {nx}x{ny} cells...")
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
dx, dy = mesh.get_cell_sizes()
print(f"Cell sizes: dx={dx}, dy={dy}")

# Create initial conditions

# Create solvers
#smoother = GaussSeidelSolver(omega=1.5, method_type='symmetric')
smoother = GaussSeidelSolver(omega=1.5, method_type='red_black') 
#smoother = GaussSeidelSolver(omega=1.8, method_type='standard')
# Create multigrid solver with the Gauss-Seidel smoother
multigrid_solver = MultiGridSolver(
    smoother=smoother,
    max_iterations=100,    # Maximum V-cycles
    tolerance=pressure_tolerance,         # Overall tolerance
    pre_smoothing=3,        # Pre-smoothing steps
    post_smoothing=3,       # Post-smoothing steps
    cycle_type='fmg',         # Use W-cycles
    cycle_type_buildup='v',
    cycle_type_final='v',
    max_cycles_buildup=1,
    #restriction_method='restrict_inject',  # Use direct injection restriction
    restriction_method='restrict_full_weighting',  # Use linear interpolation
    #interpolation_method='interpolate_linear',  # Use cubic interpolation
    interpolation_method='interpolate_cubic',  # Use cubic interpolation
    coarsest_grid_size= 7,    # Size of the coarsest grid
)

# Configure Matrix-Free PETSc Solver for maximum performance
momentum_solver = MatrixFreeMomentumSolverPETSc(
    tolerance=1e-12, 
    max_iterations=10000, 
    solver_type='bcgs',       # Use BiCGSTAB (fastest for this problem)
    use_preconditioner=False, # Disable preconditioning
    petsc_pc_type='none',     # No preconditioner
    print_its=False,           # Print iteration information to see convergence
    restart=100               # Restart parameter for GMRES (if used)
)

velocity_updater = StandardVelocityUpdater()
# Create algorithm
algorithm = SimpleSolver(
    mesh=mesh,
    fluid=FluidProperties(
        density=1.0,
        reynolds_number=reynolds,
        characteristic_velocity=1.0
    ),
    pressure_solver=multigrid_solver,
    momentum_solver=momentum_solver,
    velocity_updater=velocity_updater,
    alpha_p=alpha_p,
    alpha_u=alpha_u
)

# Set boundary conditions
algorithm.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
algorithm.set_boundary_condition('bottom', 'wall')
algorithm.set_boundary_condition('left', 'wall')
algorithm.set_boundary_condition('right', 'wall')

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Solve the problem
print("Starting simulation...")
result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, 
                        track_infinity_norm=True, infinity_norm_interval=5, 
                        save_profile=True, profile_dir=results_dir, 
                        use_l2_norm=True
                        )  # Plot every iteration

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# Print results
print(f"Simulation completed in {elapsed_time:.2f} seconds")
print(f"Total Iterations = {result.iterations}")

# Check mass conservation
max_div = result.get_max_divergence()
print(f"Maximum absolute divergence: {max_div:.6e}")

# Visualize results
result.plot_combined_results(
    title=f'Multigrid with Gauss-Seidel Smoother Cavity Flow Results (Re={reynolds}, nx={nx}, ny={ny})',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_multigrid_gauss_seidel_results.pdf'),
    show=False
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
    'Momentum solver preconditioner': 'none',
    'Momentum solver tolerance': '1e-12',
    'Pressure solver': 'geometric multigrid',
    'Pressure solver preconditioner': 'gauss-seidel smoother',
    'Pressure solver tolerance': f'{pressure_tolerance}',
    'Multigrid cycle type': f'{multigrid_solver.cycle_type}-cycle',
    'Multigrid smoother': f'{smoother.method_type} Gauss-Seidel',
    'Multigrid pre-smoothing steps': multigrid_solver.pre_smoothing,
    'Multigrid post-smoothing steps': multigrid_solver.post_smoothing,
    'Multigrid smoother relaxation': smoother.omega,
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
