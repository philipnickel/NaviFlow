"""
Lid-driven cavity flow simulation using the object-oriented framework with Conjugate Gradient solver.

This script tests the SIMPLE algorithm with a Conjugate Gradient solver for the lid-driven cavity problem.
The solver uses the basic conjugate gradient method without preconditioning.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the parent directory to the path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.solver.Algorithms.simple import SimpleSolver
from naviflow_oo.solver.pressure_solver.matrix_BiCGSTAB import BiCGSTABSolver
from naviflow_oo.solver.momentum_solver.jacobi_solver import JacobiMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_oo.postprocessing.visualization import plot_final_residuals
# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = 2**6-1, 2**6-1           # Grid size (smaller for quick testing)
reynolds = 100            # Reynolds number
alpha_p = 0.1            # Even more conservative pressure relaxation
alpha_u = 0.8             # Even more conservative velocity relaxation
max_iterations = 1000    
tolerance = 1e-3          # Convergence tolerance
h = 1/nx 
disc_order = 1
expected_disc_error = h**(disc_order)
#pressure_tolerance = expected_disc_error 
pressure_tolerance = 1e-6



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
# Use basic CG solver for pressure correction
pressure_solver = BiCGSTABSolver(
    tolerance=pressure_tolerance,        # Even tighter tolerance for pressure solver
    max_iterations=100000,   # More iterations allowed
    use_preconditioner=False
)

momentum_solver = AMGMomentumSolver(tolerance=1e-5, max_iterations=10000)
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
print("Starting simulation with SIMPLE algorithm and Conjugate Gradient solver...")
result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, save_profile=True, profile_dir=results_dir, track_infinity_norm=True, infinity_norm_interval=10)

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
    title=f'Conjugate Gradient Cavity Flow Results (Re={reynolds})',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_cg_results.pdf'),
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
