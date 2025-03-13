"""
Sanity check for pressure solvers in naviflow_oo.

This script tests the convergence of different pressure solvers on a standard test problem.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import naviflow_oo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from naviflow_oo.utils import test_solver_convergence
from naviflow_oo.solver.pressure_solver import (
    DirectPressureSolver,
    MatrixFreeCGSolver,
    JacobiSolver
)

# Problem parameters
problem_size = 33
problem_type = 'poisson'
max_iterations = 1000
tolerance = 1e-6

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Test Direct Solver
print(f"\n{'='*40}")
print(f"Testing Direct Solver")
print(f"{'='*40}")
direct_solver = DirectPressureSolver()
test_solver_convergence(
    solver=direct_solver,
    problem_size=problem_size,
    problem_type=problem_type,
    max_iterations=max_iterations,
    tolerance=tolerance,
    title="Direct Solver",
    save_plot=True,
    filename=os.path.join(results_dir, f"direct_{problem_type}_{problem_size}"),
    save_profile=True,
    profile_dir=results_dir
)

# Test CG Solver
print(f"\n{'='*40}")
print(f"Testing CG Solver")
print(f"{'='*40}")
cg_solver = MatrixFreeCGSolver(tolerance=tolerance, max_iterations=max_iterations)
test_solver_convergence(
    solver=cg_solver,
    problem_size=problem_size,
    problem_type=problem_type,
    max_iterations=max_iterations,
    tolerance=tolerance,
    title="CG Solver",
    save_plot=True,
    filename=os.path.join(results_dir, f"cg_{problem_type}_{problem_size}"),
    save_profile=True,
    profile_dir=results_dir
)

# Test Jacobi Solver
print(f"\n{'='*40}")
print(f"Testing Jacobi Solver")
print(f"{'='*40}")
jacobi_solver = JacobiSolver(tolerance=tolerance, max_iterations=max_iterations, omega=1.0)
test_solver_convergence(
    solver=jacobi_solver,
    problem_size=problem_size,
    problem_type=problem_type,
    max_iterations=max_iterations,
    tolerance=tolerance,
    title="Jacobi Solver",
    save_plot=True,
    filename=os.path.join(results_dir, f"jacobi_{problem_type}_{problem_size}"),
    save_profile=True,
    profile_dir=results_dir
)

# Test Weighted Jacobi Solver
print(f"\n{'='*40}")
print(f"Testing Weighted Jacobi Solver (omega=0.8)")
print(f"{'='*40}")
weighted_jacobi_solver = JacobiSolver(tolerance=tolerance, max_iterations=max_iterations, omega=0.8)
test_solver_convergence(
    solver=weighted_jacobi_solver,
    problem_size=problem_size,
    problem_type=problem_type,
    max_iterations=max_iterations,
    tolerance=tolerance,
    title="Weighted Jacobi Solver (omega=0.8)",
    save_plot=True,
    filename=os.path.join(results_dir, f"weighted_jacobi_0.8_{problem_type}_{problem_size}"),
    save_profile=True,
    profile_dir=results_dir
)

# Show all plots
plt.show()
