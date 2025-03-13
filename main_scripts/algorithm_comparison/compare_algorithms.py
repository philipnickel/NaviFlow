"""
Compare different CFD algorithms on the lid-driven cavity problem.

This script compares the performance of different algorithms:
- SIMPLE (Semi-Implicit Method for Pressure-Linked Equations)
- SIMPLER (SIMPLE Revised)

It measures convergence rate, time, and accuracy for each algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the parent directory to the path so we can import naviflow_oo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.solver.Algorithms import SimpleSolver, SimplerSolver
from naviflow_oo.solver.pressure_solver import MatrixFreeCGSolver
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Problem parameters
nx, ny = 65, 65
reynolds = 100
max_iterations = 50
tolerance = 1e-6

# Create mesh
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
print(f"Created mesh with {nx}x{ny} cells")
print(f"Cell sizes: dx={mesh.dx:.6f}, dy={mesh.dy:.6f}")

# Define fluid properties
fluid = FluidProperties(
    density=1.0,
    reynolds_number=reynolds,
    characteristic_velocity=1.0
)
print(f"Reynolds number: {fluid.get_reynolds_number()}")
print(f"Calculated viscosity: {fluid.get_viscosity()}")

# Create solvers
pressure_solver = MatrixFreeCGSolver(tolerance=1e-6, max_iterations=1000)
momentum_solver = StandardMomentumSolver()
velocity_updater = StandardVelocityUpdater()

# Create algorithms to test
algorithms = [
    {
        'name': 'SIMPLE',
        'solver': SimpleSolver(
            mesh=mesh,
            fluid=fluid,
            pressure_solver=pressure_solver,
            momentum_solver=momentum_solver,
            velocity_updater=velocity_updater,
            alpha_p=0.3,
            alpha_u=0.7
        ),
        'color': 'b',
        'marker': 'o'
    },
    {
        'name': 'SIMPLER',
        'solver': SimplerSolver(
            mesh=mesh,
            fluid=fluid,
            pressure_solver=pressure_solver,
            momentum_solver=momentum_solver,
            velocity_updater=velocity_updater,
            alpha_u=0.7
        ),
        'color': 'r',
        'marker': 's'
    }
]

# Test each algorithm
results = []
for algo_info in algorithms:
    print(f"\n{'='*40}")
    print(f"Testing {algo_info['name']} Algorithm")
    print(f"{'='*40}")
    
    algorithm = algo_info['solver']
    
    # Set boundary conditions
    algorithm.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
    algorithm.set_boundary_condition('bottom', 'wall')
    algorithm.set_boundary_condition('left', 'wall')
    algorithm.set_boundary_condition('right', 'wall')
    
    # Start timing
    start_time = time.time()
    
    # Solve the problem
    result = algorithm.solve(max_iterations=max_iterations, tolerance=tolerance, save_profile=True, profile_dir=results_dir)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print results
    print(f"Algorithm: {algo_info['name']}")
    print(f"Iterations: {result.iterations}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Maximum absolute divergence: {result.get_max_divergence():.6e}")
    
    # Visualize results
    result.plot_combined_results(
        title=f'{algo_info["name"]} Cavity Flow Results (Re={reynolds})',
        filename=os.path.join(results_dir, f'cavity_Re{reynolds}_{algo_info["name"].lower()}_results.pdf'),
        show=False
    )
    
    # Store results for comparison
    results.append({
        'name': algo_info['name'],
        'iterations': result.iterations,
        'time': elapsed_time,
        'divergence': result.get_max_divergence(),
        'residuals': result.residuals,
        'color': algo_info['color'],
        'marker': algo_info['marker']
    })

# Create comparison plots
plt.figure(figsize=(12, 8))

# Plot convergence history for each algorithm
for result in results:
    if 'residuals' in result and len(result['residuals']) > 0:
        plt.semilogy(
            range(1, len(result['residuals']) + 1),
            result['residuals'],
            label=result['name'],
            color=result['color'],
            marker=result['marker'],
            markevery=max(1, len(result['residuals']) // 10)
        )

plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Residual (log scale)')
plt.title(f'Convergence Comparison - Lid-Driven Cavity (Re={reynolds})')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'convergence_comparison_Re{reynolds}.pdf'))

# Create bar chart for time comparison
plt.figure(figsize=(10, 6))
names = [result['name'] for result in results]
times = [result['time'] for result in results]
colors = [result['color'] for result in results]

plt.bar(names, times, color=colors)
plt.grid(True, axis='y')
plt.xlabel('Algorithm')
plt.ylabel('Time (seconds)')
plt.title(f'Time Comparison - Lid-Driven Cavity (Re={reynolds})')

# Add time values on top of bars
for i, time_val in enumerate(times):
    plt.text(i, time_val * 1.05, f'{time_val:.2f}s', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'time_comparison_Re{reynolds}.pdf'))

# Create bar chart for iteration comparison
plt.figure(figsize=(10, 6))
iterations = [result['iterations'] for result in results]

plt.bar(names, iterations, color=colors)
plt.grid(True, axis='y')
plt.xlabel('Algorithm')
plt.ylabel('Iterations')
plt.title(f'Iteration Comparison - Lid-Driven Cavity (Re={reynolds})')

# Add iteration values on top of bars
for i, iter_val in enumerate(iterations):
    plt.text(i, iter_val * 1.05, f'{iter_val}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'iteration_comparison_Re{reynolds}.pdf'))

# Print summary
print("\nResults Summary:")
print(f"{'Algorithm':<10} {'Iterations':<12} {'Time (s)':<12} {'Divergence':<16}")
print("-" * 50)
for result in results:
    print(f"{result['name']:<10} {result['iterations']:<12} {result['time']:<12.2f} {result['divergence']:<16.6e}")

print(f"\nResults saved to {results_dir}")

plt.show() 