"""
Lid-driven cavity flow simulation using the object-oriented framework with 
geometric multigrid preconditioned conjugate gradient solver.
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
from naviflow_oo.solver.pressure_solver.geo_multigrid_cg import GeoMultigridPrecondCGSolver
from naviflow_oo.solver.pressure_solver.gauss_seidel import GaussSeidelSolver
from naviflow_oo.solver.momentum_solver.power_law import PowerLawMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.postprocessing.visualization import plot_final_residuals

# Start timing
start_time = time.time()

# Grid size - must be 2^k-1 for multigrid (e.g., 31, 63, 127, 255)
nx, ny = 2**7-1, 2**7-1 

# Relaxation factors and iterations
max_iterations = 10000
convergence_tolerance = 1e-5
alpha_p = 0.3  # Pressure relaxation
alpha_u = 0.7  # Velocity relaxation
Re = 100

h = 1/nx 
disc_order = 1
expected_disc_error = h**(disc_order)
#tolerance = expected_disc_error * 1e-3
tolerance = 1e-6
pressure_tolerance = expected_disc_error * 1e-2



# Create mesh
print(f"Creating mesh with {nx}x{ny} cells...")
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
dx, dy = mesh.get_cell_sizes()
print(f"Cell sizes: dx={dx}, dy={dy}")

# Create initial conditions
print(f"Reynolds number: {Re}")

# Create a Gauss-Seidel smoother for the multigrid solver with SOR
smoother = GaussSeidelSolver(omega=0.87)

# Create the geometric multigrid preconditioned CG solver
geomg_cg_solver = GeoMultigridPrecondCGSolver(
    tolerance=pressure_tolerance,        # CG tolerance
    max_iterations=100000,   # Maximum CG iterations - increased for better convergence
    mg_pre_smoothing=1,    # Multigrid pre-smoothing steps
    mg_post_smoothing=0,   # Multigrid post-smoothing steps
    mg_cycles=1,           # Increased from 1 to 2 cycles per preconditioning step
    mg_cycle_type='w',     # Use W-cycle for better preconditioning
    mg_cycle_type_buildup='v',
    mg_max_cycles_buildup=1,
    mg_coarsest_grid_size=7,
    mg_restriction_method='restrict_inject',
    mg_interpolation_method='interpolate_cubic',
    smoother=smoother      # Smoother to use in the multigrid
)



momentum_solver = PowerLawMomentumSolver()
velocity_updater = StandardVelocityUpdater()

# Create algorithm
algorithm = SimpleSolver(
    mesh=mesh,
    fluid=FluidProperties(
        density=1.0,
        reynolds_number=Re,
        characteristic_velocity=1.0
    ),
    pressure_solver=geomg_cg_solver,
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
result = algorithm.solve(
    max_iterations=max_iterations, 
    tolerance=convergence_tolerance, 
    track_infinity_norm=True, 
    infinity_norm_interval=5, 
    save_profile=True, 
    profile_dir=results_dir
)

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
    title=f'Geo-Multigrid Preconditioned CG Cavity Flow Results (Re={Re})',
    filename=os.path.join(results_dir, f'cavity_Re{Re}_geomg_cg_results.pdf'),
    show=False
)

# Visualize final residuals
plot_final_residuals(
    result.u, result.v, result.p,
    algorithm.u_old, algorithm.v_old, algorithm.p_old,
    mesh,
    title=f'Final Residuals (Re={Re})',
    filename=os.path.join(results_dir, f'final_residuals_Re_geomg_cg_{Re}.pdf'),
    show=False
) 