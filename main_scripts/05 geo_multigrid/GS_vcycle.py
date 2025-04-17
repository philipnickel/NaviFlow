"""
Lid-driven cavity flow simulation using the object-oriented framework with multigrid solver
that uses GaussSeidelSolver as the smoother.
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
from naviflow_oo.solver.pressure_solver.gauss_seidel import GaussSeidelSolver
from naviflow_oo.solver.momentum_solver.power_law import PowerLawMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.postprocessing.visualization import plot_final_residuals
from naviflow_oo.postprocessing.visualization import plot_u_v_continuity_residuals
# Start timing
start_time = time.time()
# 1. Set up simulation parameters
nx, ny = 2**9-1, 2**9-1 # Grid size
reynolds = 10000            # Reynolds number
alpha_p = 0.1              # Pressure relaxation factor
alpha_u = 0.8              # Velocity relaxation factor
max_iterations = 25000     # Maximum number of iterations

h = 1/nx 
disc_order = 1
expected_disc_error = h**(disc_order)
#tolerance = expected_disc_error * 1e-3
tolerance = 1e-5
pressure_tolerance = expected_disc_error 
print(f"Tolerance: {tolerance}")
print(f"Pressure tolerance: {pressure_tolerance}")


# Create mesh
print(f"Creating mesh with {nx}x{ny} cells...")
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
dx, dy = mesh.get_cell_sizes()
print(f"Cell sizes: dx={dx}, dy={dy}")

# Create initial conditions

# Create solvers
# Create a Gauss-Seidel smoother for the multigrid solver with SOR
#smoother = GaussSeidelSolver(omega=0.87) # somehow 1.3 is good
smoother = GaussSeidelSolver(omega=0.87) # somehow 1.3 is good
# Create multigrid solver with the Gauss-Seidel smoother
multigrid_solver = MultiGridSolver(
    smoother=smoother,
    max_iterations=100,    # Maximum V-cycles
    tolerance=pressure_tolerance,         # Overall tolerance
    pre_smoothing=3,        # Pre-smoothing steps
    post_smoothing=4,       # Post-smoothing steps
    cycle_type='fmg',         # Use W-cycles
    cycle_type_buildup='v',
    cycle_type_final='v',
    max_cycles_buildup=1,
    restriction_method='restrict_inject',  # Use direct injection restriction
    #restriction_method='restrict_full_weighting',  # Use linear interpolation
    #interpolation_method='interpolate_linear',  # Use cubic interpolation
    interpolation_method='interpolate_cubic',  # Use cubic interpolation
    coarsest_grid_size= 7,    # Size of the coarsest grid
)
momentum_solver = PowerLawMomentumSolver()
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
    title=f'Multigrid with Gauss-Seidel Smoother Cavity Flow Results (Re={reynolds})',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_multigrid_gauss_seidel_results.pdf'),
    show=True
)
# 11. Visualize final residuals
plot_final_residuals(
    result.u, result.v, result.p,
    algorithm.u_old, algorithm.v_old, algorithm.p_old,
    mesh,
    title=f'Final Residuals (Re={reynolds})',
    filename=os.path.join(results_dir, f'final_residuals_Re_GS{reynolds}.pdf'),
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