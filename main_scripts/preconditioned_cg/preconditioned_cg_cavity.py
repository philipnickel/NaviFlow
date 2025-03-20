"""
Lid-driven cavity flow simulation using the object-oriented framework with Preconditioned CG solver.

This script tests the SIMPLE algorithm with a Preconditioned Conjugate Gradient solver for the lid-driven cavity problem.
The solver uses PyAMG as a preconditioner to accelerate convergence of the conjugate gradient method.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import argparse

# Add the parent directory to the path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.solver.Algorithms.simple import SimpleSolver
from naviflow_oo.solver.pressure_solver.preconditioned_cg_solver import PreconditionedCGSolver
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.solver.momentum_solver.discretization.convection_schemes import QuickDiscretization
from naviflow_oo.solver.momentum_solver.discretization.convection_schemes import PowerLawDiscretization
from naviflow_oo.solver.momentum_solver.discretization.convection_schemes import UpwindDiscretization

# Parse command line arguments
parser = argparse.ArgumentParser(description='Lid-driven cavity flow simulation with Preconditioned CG solver')
parser.add_argument('--nx', type=int, default=127, help='Number of cells in x direction')
parser.add_argument('--ny', type=int, default=127, help='Number of cells in y direction')
parser.add_argument('--reynolds', type=float, default=100, help='Reynolds number')
parser.add_argument('--scheme', type=str, default='power_law', choices=['power_law', 'upwind', 'quick'], 
                    help='Discretization scheme to use')
parser.add_argument('--max_iter', type=int, default=100000, help='Maximum number of iterations')
parser.add_argument('--tolerance', type=float, default=1e-5, help='Convergence tolerance')
parser.add_argument('--alpha_p', type=float, default=0.1, help='Pressure relaxation factor')
parser.add_argument('--alpha_u', type=float, default=0.7, help='Velocity relaxation factor')
parser.add_argument('--quiet', action='store_true', help='Suppress detailed iteration output')
parser.add_argument('--output', type=str, default=None, help='Custom output filename prefix')
args = parser.parse_args()

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = args.nx, args.ny
reynolds = args.reynolds
alpha_p = args.alpha_p
alpha_u = args.alpha_u
max_iterations = args.max_iter
tolerance = args.tolerance

# 2. Create mesh
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
if not args.quiet:
    print(f"Created mesh with {nx}x{ny} cells")
    print(f"Cell sizes: dx={mesh.dx:.6f}, dy={mesh.dy:.6f}")

# 3. Define fluid properties
fluid = FluidProperties(
    density=1.0,
    reynolds_number=reynolds,
    characteristic_velocity=1.0
)
if not args.quiet:
    print(f"Reynolds number: {fluid.get_reynolds_number()}")
    print(f"Calculated viscosity: {fluid.get_viscosity()}")

# 4. Create solvers
# Choose discretization scheme based on input
if args.scheme == 'power_law':
    discretization_scheme = PowerLawDiscretization()
elif args.scheme == 'upwind':
    discretization_scheme = UpwindDiscretization()
elif args.scheme == 'quick':
    discretization_scheme = QuickDiscretization()
else:
    discretization_scheme = PowerLawDiscretization()

# Use Preconditioned CG solver for pressure correction
pressure_solver = PreconditionedCGSolver(
    tolerance=1e-12,
    max_iterations=100000,
    smoother='gauss_seidel',
    presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 5}),
    postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 5}),
    cycle_type='F'
)
momentum_solver = StandardMomentumSolver()
velocity_updater = StandardVelocityUpdater()

# Print which discretization scheme is being used (for information only)
if not args.quiet:
    print(f"Note: Using default Power Law discretization scheme regardless of --scheme parameter")
    print(f"The --scheme parameter is kept for compatibility but has no effect.")

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
if not args.quiet:
    print(f"Starting simulation with SIMPLE algorithm and Preconditioned CG solver using {args.scheme} discretization...")


result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, save_profile=True, profile_dir=results_dir, track_infinity_norm=True args.quiet, infinity_norm_interval=10)

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# 8. Print results
print(f"Simulation completed in {elapsed_time:.2f} seconds")
print(f"Total Iterations = {result.iterations}")

# 9. Check mass conservation
max_div = result.get_max_divergence()
print(f"Maximum absolute divergence: {max_div:.6e}")

# Generate output filename
if args.output:
    output_prefix = args.output
else:
    output_prefix = f'cavity_Re{reynolds}_{args.scheme}'

# 10. Visualize results
result.plot_combined_results(
    title=f'Cavity Flow Results (Re={reynolds}, {args.scheme}, {nx}x{ny})',
    filename=os.path.join(results_dir, f'{output_prefix}_results.pdf'),
    show=False
)

# Save basic metadata to a simple text file for reference
metadata_file = os.path.join(results_dir, f'{output_prefix}_metadata.txt')
with open(metadata_file, 'w') as f:
    f.write(f"Simulation Parameters:\n")
    f.write(f"Mesh: {nx}x{ny}\n")
    f.write(f"Reynolds: {reynolds}\n")
    f.write(f"Scheme: {args.scheme}\n")
    f.write(f"Iterations: {result.iterations}\n")
    f.write(f"Max Divergence: {max_div:.6e}\n")
    f.write(f"Runtime: {elapsed_time:.2f} seconds\n")

print(f"Results saved to {results_dir}")