"""
Main script to run lid-driven cavity flow simulations with different solvers
across multiple Reynolds numbers.

This script compares the performance of various pressure solvers:
- Direct solver
- Jacobi solver
- Matrix-free CG solver
- Multigrid solver with Jacobi smoother (V-cycle)
- Multigrid solver with Jacobi smoother (F-cycle)
- PyAMG solver

For each solver, it runs simulations at Reynolds numbers:
100, 400, 1000, 3200, 5000, 7500, 10000
"""

import numpy as np
import time
import os
import sys
import traceback
from datetime import datetime

# Import required modules
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.solver.Algorithms.simple import SimpleSolver

# Import pressure solvers
from naviflow_oo.solver.pressure_solver.direct import DirectPressureSolver
from naviflow_oo.solver.pressure_solver.jacobi import JacobiSolver
from naviflow_oo.solver.pressure_solver.matrix_free_cg import MatrixFreeCGSolver
from naviflow_oo.solver.pressure_solver.multigrid import MultiGridSolver
from naviflow_oo.solver.pressure_solver.pyamg_solver import PyAMGSolver

# Import other required components
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater

# ============================================================================
# GLOBAL SIMULATION PARAMETERS
# ============================================================================
MESH_SIZE = 127                    # Fixed mesh size for all solvers (2^n-1 for multigrid)
MAX_ITERATIONS = 100000            # Maximum number of SIMPLE iterations
SIMPLE_TOLERANCE = 1e-8           # Convergence tolerance for SIMPLE algorithm
INNER_SOLVER_TOLERANCE = 1e-7     # Tolerance for inner pressure solvers
INNER_SOLVER_MAX_ITER = 1000       # Max iterations for inner pressure solvers
ALPHA_P = 0.3                     # Pressure relaxation factor
ALPHA_U = 0.7                     # Velocity relaxation factor

# Reynolds numbers to test
REYNOLDS_NUMBERS = [100, 400, 1000, 3200, 5000, 7500, 10000]
# ============================================================================

# Create main results directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_results_dir = os.path.join(os.path.dirname(__file__), f'results_{timestamp}')
os.makedirs(main_results_dir, exist_ok=True)

# Create a log file
log_file = os.path.join(main_results_dir, 'simulation_log.txt')

def log_message(message, also_print=True):
    """Write message to log file and optionally print to console"""
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    if also_print:
        print(message)

# Log system information and global parameters
log_message(f"Starting simulations at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_message(f"Results directory: {main_results_dir}")
log_message(f"Global parameters:")
log_message(f"  Mesh size: {MESH_SIZE}x{MESH_SIZE}")
log_message(f"  Max iterations: {MAX_ITERATIONS}")
log_message(f"  SIMPLE tolerance: {SIMPLE_TOLERANCE}")
log_message(f"  Inner solver tolerance: {INNER_SOLVER_TOLERANCE}")
log_message(f"  Inner solver max iterations: {INNER_SOLVER_MAX_ITER}")
log_message(f"  Alpha_p: {ALPHA_P}")
log_message(f"  Alpha_u: {ALPHA_U}")
log_message(f"Reynolds numbers: {REYNOLDS_NUMBERS}")

# Define solver configurations - using fixed parameters
solver_configs = [
    {
        'name': 'direct',
        'class': DirectPressureSolver,
        'params': {},
    },
    {
        'name': 'jacobi',
        'class': JacobiSolver,
        'params': {
            'tolerance': INNER_SOLVER_TOLERANCE, 
            'max_iterations': INNER_SOLVER_MAX_ITER, 
            'omega': 0.5
        },
    },
    {
        'name': 'matrix_free_cg',
        'class': MatrixFreeCGSolver,
        'params': {
            'tolerance': INNER_SOLVER_TOLERANCE, 
            'max_iterations': INNER_SOLVER_MAX_ITER
        },
    },
    {
        'name': 'multigrid_vcycle',
        'class': MultiGridSolver,
        'params': {
            'tolerance': INNER_SOLVER_TOLERANCE,
            'max_iterations': INNER_SOLVER_MAX_ITER,
            'pre_smoothing': 5,
            'post_smoothing': 5,
            'smoother_iterations': 5,
            'smoother_omega': 0.8,
            'smoother': JacobiSolver(
                omega=0.8
            ),
            'cycle_type': 'v'
        },
    },
    {
        'name': 'multigrid_fcycle',
        'class': MultiGridSolver,
        'params': {
            'tolerance': INNER_SOLVER_TOLERANCE,
            'max_iterations': INNER_SOLVER_MAX_ITER,
            'pre_smoothing': 5,
            'post_smoothing': 5,
            'smoother_iterations': 5,
            'smoother_omega': 0.8,
            'smoother': JacobiSolver(
                omega=0.8
            ),
            'cycle_type': 'f'
        },
    },
    {
        'name': 'pyamg',
        'class': PyAMGSolver,
        'params': {
            'tolerance': INNER_SOLVER_TOLERANCE,
            'max_iterations': INNER_SOLVER_MAX_ITER,
            'smoother': 'gauss_seidel',
            'presmoother': ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 5}),
            'postsmoother': ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 5}),
            'cycle_type': 'V'
        },
    }
]

# Create a summary file for results
summary_file = os.path.join(main_results_dir, 'simulation_summary.csv')
with open(summary_file, 'w') as f:
    f.write("Solver,Reynolds,MeshSize,Iterations,Time(s),Converged,MaxDivergence\n")

# Function to run a single simulation
def run_simulation(solver_config, reynolds, results_dir):
    """Run a single simulation with the given solver and Reynolds number"""
    
    solver_name = solver_config['name']
    
    log_message(f"Starting {solver_name} solver simulation for Re={reynolds} on {MESH_SIZE}x{MESH_SIZE} mesh")
    
    # Create mesh
    mesh = StructuredMesh(nx=MESH_SIZE, ny=MESH_SIZE, length=1.0, height=1.0)
    
    # Define fluid properties
    fluid = FluidProperties(
        density=1.0,
        reynolds_number=reynolds,
        characteristic_velocity=1.0
    )
    
    # Create pressure solver
    pressure_solver = solver_config['class'](**solver_config['params'])
    
    # Create other solvers
    momentum_solver = StandardMomentumSolver()
    velocity_updater = StandardVelocityUpdater()
    
    # Create algorithm (SIMPLE only)
    algorithm = SimpleSolver(
        mesh=mesh,
        fluid=fluid,
        pressure_solver=pressure_solver,
        momentum_solver=momentum_solver,
        velocity_updater=velocity_updater,
        alpha_p=ALPHA_P,
        alpha_u=ALPHA_U
    )
    
    # Set boundary conditions
    algorithm.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
    algorithm.set_boundary_condition('bottom', 'wall')
    algorithm.set_boundary_condition('left', 'wall')
    algorithm.set_boundary_condition('right', 'wall')
    
    # Start timing
    start_time = time.time()
    
    # Solve the problem
    try:
        # Create specific profile directory for this run
        profile_dir = os.path.join(results_dir, 'profiles')
        os.makedirs(profile_dir, exist_ok=True)
        
        # Solve with profiling and residual tracking enabled
        result = algorithm.solve(
            max_iterations=MAX_ITERATIONS,
            tolerance=SIMPLE_TOLERANCE,
            save_profile=True,
            profile_dir=profile_dir,
            track_infinity_norm=True,
            infinity_norm_interval=100  # Check against Ghia data every 100 iterations
        )
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Check mass conservation
        max_div = result.get_max_divergence()
        
        # Log results
        log_message(f"Simulation completed in {elapsed_time:.2f} seconds")
        log_message(f"Total Iterations = {result.iterations}")
        log_message(f"Maximum absolute divergence: {max_div:.6e}")
        
        # Save results using the built-in methods
        result_dir = os.path.join(results_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)
        
        # Save solution fields
        result.save_solution(os.path.join(result_dir, f'cavity_Re{reynolds}_{solver_name}_solution.npz'))
        
        # Save visualization using the built-in method
        result.plot_combined_results(
            title=f'{solver_name.replace("_", " ").title()} Cavity Flow Results (Re={reynolds})',
            filename=os.path.join(result_dir, f'cavity_Re{reynolds}_{solver_name}_results.png'),
            show=False
        )
        
        
        
        # Add to summary
        with open(summary_file, 'a') as f:
            f.write(f"{solver_name},{reynolds},{MESH_SIZE},{result.iterations},{elapsed_time:.2f},True,{max_div:.6e}\n")
        
        return True, result.iterations, elapsed_time, max_div
    
    except Exception as e:
        # End timing even if there's an error
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Log error
        error_msg = f"Error in {solver_name} solver for Re={reynolds}: {str(e)}"
        log_message(error_msg)
        log_message(traceback.format_exc())
        
        # Add to summary
        with open(summary_file, 'a') as f:
            f.write(f"{solver_name},{reynolds},{MESH_SIZE},0,{elapsed_time:.2f},False,N/A\n")
        
        return False, 0, elapsed_time, None

# Main execution loop
for re in REYNOLDS_NUMBERS:
    log_message(f"\n{'='*50}")
    log_message(f"Running simulations for Reynolds number {re}")
    log_message(f"{'='*50}")
    
    # Run all solvers for this Reynolds number
    for solver_config in solver_configs:
        solver_name = solver_config['name']
        
        # Create solver-specific results directory
        solver_results_dir = os.path.join(main_results_dir, solver_name)
        os.makedirs(solver_results_dir, exist_ok=True)
        
        # Create Reynolds-specific results directory
        re_results_dir = os.path.join(solver_results_dir, f'Re{re}')
        os.makedirs(re_results_dir, exist_ok=True)
        
        log_message(f"\nRunning {solver_name} solver for Re={re}")
        
        # Run the simulation
        success, iterations, elapsed_time, max_div = run_simulation(solver_config, re, re_results_dir)
        
        if success:
            log_message(f"Completed Re={re} with {solver_name} solver in {elapsed_time:.2f}s ({iterations} iterations)")
        else:
            log_message(f"Failed to complete Re={re} with {solver_name} solver")

log_message(f"\nAll simulations completed. Results saved to {main_results_dir}")
