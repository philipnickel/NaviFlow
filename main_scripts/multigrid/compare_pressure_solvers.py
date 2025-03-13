"""
Compare different pressure solvers on a test problem.

This script compares the performance of different pressure solvers:
- Direct solver
- Matrix-free CG solver
- Jacobi solver
- Weighted Jacobi solver
- Multigrid solver

It measures convergence rate, time, and accuracy for each solver.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.solver.pressure_solver import (
    DirectPressureSolver,
    MatrixFreeCGSolver,
    JacobiSolver,
    MultiGridSolver
)
from naviflow_oo.utils import test_solver_convergence

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Problem parameters
problem_size = 63  # Changed to 63 (2^6-1) to match the 2^k-1 requirement
problem_type = 'poisson'
max_iterations = 1000
tolerance = 1e-6

# Create solvers to test
solvers = [
    {
        'name': 'Direct',
        'solver': DirectPressureSolver(),
        'color': 'k',
        'marker': 'o'
    },
    {
        'name': 'CG',
        'solver': MatrixFreeCGSolver(tolerance=tolerance, max_iterations=max_iterations),
        'color': 'b',
        'marker': 's'
    },
    {
        'name': 'Jacobi',
        'solver': JacobiSolver(tolerance=tolerance, max_iterations=max_iterations, omega=1.0),
        'color': 'r',
        'marker': '^'
    },
    {
        'name': 'Weighted Jacobi (0.8)',
        'solver': JacobiSolver(tolerance=tolerance, max_iterations=max_iterations, omega=0.8),
        'color': 'g',
        'marker': 'v'
    },
    {
        'name': 'Multigrid',
        'solver': MultiGridSolver(
            tolerance=tolerance,
            max_iterations=50,
            pre_smoothing=2,
            post_smoothing=2,
            smoother_iterations=3,
            smoother_omega=0.8
        ),
        'color': 'm',
        'marker': 'D'
    }
]

# Test each solver
results = []
for solver_info in solvers:
    print(f"\n{'='*40}")
    print(f"Testing {solver_info['name']} Solver")
    print(f"{'='*40}")
    
    solver = solver_info['solver']
    
    # Test solver convergence
    result = test_solver_convergence(
        solver=solver,
        problem_size=problem_size,
        problem_type=problem_type,
        max_iterations=max_iterations,
        tolerance=tolerance,
        title=f"{solver_info['name']} Solver",
        save_plot=True,
        filename=os.path.join(results_dir, f"{solver_info['name'].lower().replace(' ', '_')}_{problem_type}_{problem_size}"),
        save_profile=True,
        profile_dir=results_dir
    )
    
    # Add solver info to result
    result['solver_name'] = solver_info['name']
    result['color'] = solver_info['color']
    result['marker'] = solver_info['marker']
    
    results.append(result)

# Create comparison plots
plt.figure(figsize=(12, 8))

# Plot convergence history for each solver
for result in results:
    if 'convergence_history' in result and len(result['convergence_history']) > 0:
        plt.semilogy(
            range(1, len(result['convergence_history']) + 1),
            result['convergence_history'],
            label=result['solver_name'],
            color=result['color'],
            marker=result['marker'],
            markevery=max(1, len(result['convergence_history']) // 10)
        )

plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Residual (log scale)')
plt.title(f'Convergence Comparison - {problem_type.capitalize()} Problem ({problem_size}x{problem_size})')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'convergence_comparison_{problem_type}_{problem_size}.pdf'))

# Create bar chart for time comparison
plt.figure(figsize=(12, 6))
names = [result['solver_name'] for result in results]
times = [result['time_taken'] for result in results]
colors = [result['color'] for result in results]

plt.bar(names, times, color=colors)
plt.yscale('log')
plt.grid(True, axis='y')
plt.xlabel('Solver')
plt.ylabel('Time (seconds, log scale)')
plt.title(f'Time Comparison - {problem_type.capitalize()} Problem ({problem_size}x{problem_size})')

# Add time values on top of bars
for i, time_val in enumerate(times):
    plt.text(i, time_val * 1.1, f'{time_val:.4f}s', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'time_comparison_{problem_type}_{problem_size}.pdf'))

# Print results
print("\nResults Summary:")
print(f"{'Solver':<20} {'Iterations':<12} {'Time (s)':<12} {'Final Residual':<16} {'Error':<16}")
print("-" * 70)
for result in results:
    iterations = result.get('iterations', 'N/A')
    time_taken = f"{result.get('time_taken', 'N/A'):.4f}"
    final_residual = f"{result.get('final_residual', 'N/A')}"
    error = f"{result.get('error', 'N/A')}"
    print(f"{result['solver_name']:<20} {iterations:<12} {time_taken:<12} {final_residual:<16} {error:<16}")

# Save detailed results to a file
with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
    f.write("Results Summary:\n")
    f.write(f"{'Solver':<20} {'Iterations':<12} {'Time (s)':<12} {'Final Residual':<16} {'Error':<16}\n")
    f.write("-" * 70 + "\n")
    for result in results:
        iterations = result.get('iterations', 'N/A')
        time_taken = f"{result.get('time_taken', 'N/A'):.4f}"
        final_residual = f"{result.get('final_residual', 'N/A')}"
        error = f"{result.get('error', 'N/A')}"
        f.write(f"{result['solver_name']:<20} {iterations:<12} {time_taken:<12} {final_residual:<16} {error:<16}\n")

print(f"\nResults saved to {results_dir}")

plt.show() 