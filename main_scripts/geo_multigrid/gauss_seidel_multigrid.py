"""
Lid-driven cavity flow simulation using the object-oriented framework with multigrid solver
that uses GaussSeidelSolver as the smoother. Supports V-cycle, W-cycle, and F-cycle multigrid.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.preprocessing.fields.scalar_field import ScalarField
from naviflow_oo.preprocessing.fields.vector_field import VectorField
from naviflow_oo.solver.Algorithms.simple import SimpleSolver
from naviflow_oo.solver.pressure_solver.multigrid import MultiGridSolver
from naviflow_oo.solver.pressure_solver.gauss_seidel import GaussSeidelSolver
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.postprocessing.visualization import plot_final_residuals

# Parse command line arguments
if len(sys.argv) > 1:
    cycle_type = sys.argv[1].lower()
    if cycle_type not in ['v', 'w', 'f']:
        print(f"Invalid cycle type: {cycle_type}. Using default 'v'.")
        cycle_type = 'v'
else:
    cycle_type = 'v'  # Default cycle type

print(f"Using {cycle_type.upper()}-cycle multigrid")

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Create debug output directory
debug_dir = os.path.join(os.path.dirname(__file__), 'debug_output')
os.makedirs(debug_dir, exist_ok=True)

# Start timing
start_time = time.time()

# Grid size - must be 2^k-1 for multigrid (e.g., 31, 63, 127, 255)
nx, ny = 2**6-1, 2**6-1  # 63x63 grid

# Relaxation factors and iterations
max_iterations = 10000
convergence_tolerance = 1e-4
alpha_p = 0.1  # Pressure relaxation
alpha_u = 0.7 # Velocity relaxation

# Create mesh
print(f"Creating mesh with {nx}x{ny} cells...")
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
dx, dy = mesh.get_cell_sizes()
print(f"Cell sizes: dx={dx}, dy={dy}")

# Create initial conditions
Re = 100
print(f"Reynolds number: {Re}")

# Create solvers
# Create a Gauss-Seidel smoother for the multigrid solver with SOR
smoother = GaussSeidelSolver(omega=1.9)

# Create multigrid solver with the Gauss-Seidel smoother
multigrid_solver = MultiGridSolver(
    smoother=smoother,
    max_iterations=100,    # Maximum cycles
    tolerance=1e-4,        # Overall tolerance
    pre_smoothing=10,       # Pre-smoothing steps
    post_smoothing=10,      # Post-smoothing steps
    cycle_type=cycle_type   # Use the specified cycle type
)
momentum_solver = StandardMomentumSolver()
velocity_updater = StandardVelocityUpdater()

# Create algorithm
algorithm = SimpleSolver(
    mesh=mesh,
    fluid=FluidProperties(
        density=1.0,
        reynolds_number=Re,
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

# Print solver settings
print("Starting simulation...")
print(f"Using relaxation factors: alpha_p = {alpha_p}, alpha_u = {alpha_u}")
print(f"Using {cycle_type.upper()}-cycle multigrid with {multigrid_solver.pre_smoothing} pre-smoothing and {multigrid_solver.post_smoothing} post-smoothing steps")

# Solve the problem
result = algorithm.solve(max_iterations=max_iterations, tolerance=convergence_tolerance, 
                        track_infinity_norm=True, infinity_norm_interval=5)

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
    title=f'Multigrid ({cycle_type.upper()}-cycle) with Gauss-Seidel Smoother (Re={Re})',
    filename=os.path.join(results_dir, f'cavity_Re{Re}_multigrid_{cycle_type}cycle_results.pdf'),
    show=True
)
# Visualize final residuals
plot_final_residuals(
    result.u, result.v, result.p,
    algorithm.u_old, algorithm.v_old, algorithm.p_old,
    mesh,
    title=f'Final Residuals (Re={Re}, {cycle_type.upper()}-cycle)',
    filename=os.path.join(results_dir, f'final_residuals_Re{Re}_{cycle_type}cycle.pdf'),
    show=False
)