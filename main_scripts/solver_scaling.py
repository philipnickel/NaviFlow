#!/usr/bin/env python
"""
Benchmark script for testing solver performance with larger meshes.

This script compares different pressure solvers (algebraic multigrid, conjugate gradient, 
and preconditioned multigrid) on a series of increasingly larger mesh sizes 
with varying tolerance levels.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import pandas as pd
from collections import defaultdict

# Add the parent directory to the path so we can import the NaviFlow package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.solver.pressure_solver.jacobi import JacobiSolver
from naviflow_oo.solver.pressure_solver.pyamg_solver import PyAMGSolver
from naviflow_oo.solver.pressure_solver.matrix_free_cg import MatrixFreeCGSolver
from naviflow_oo.solver.pressure_solver.preconditioned_cg_solver import PreconditionedCGSolver
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.solver.Algorithms.simple import SimpleSolver

# Import SciencePlots for styling
try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    print("Warning: scienceplots package not found. Using default matplotlib style.")


def run_benchmark(mesh_sizes, tolerances, iterations=3, re=100):
    """
    Run benchmark comparing different solvers with varying mesh sizes and tolerances.
    
    Parameters:
    -----------
    mesh_sizes : list
        List of mesh sizes to test (nx = ny for square meshes)
    tolerances : list
        List of tolerance values to test
    iterations : int
        Number of iterations to run for each test
    re : float
        Reynolds number for the simulation
    
    Returns:
    --------
    results : dict
        Dictionary containing benchmark results
    """
    # Initialize results dictionary
    results = {
        'mesh_sizes': mesh_sizes,
        'tolerances': tolerances,
        'iterations': iterations,
        'algebraic_multigrid': defaultdict(list),
        'conjugate_gradient': defaultdict(list),
        'preconditioned_cg': defaultdict(list)
    }
    
    # Common parameters
    p_relax = 0.1
    u_relax = 0.7
    max_iter = iterations  # Use only 3 iterations
    
    # Loop through all tolerance values
    for tol in tolerances:
        print(f"\n{'='*80}")
        print(f"Running benchmark with tolerance {tol:.1e}")
        print(f"{'='*80}")
        
        # Loop through all mesh sizes
        for nx in mesh_sizes:
            print(f"\n{'-'*50}")
            print(f"Running benchmark with mesh size {nx}x{nx}, Re={re}, tolerance={tol:.1e}")
            print(f"{'-'*50}")
            
            # Create mesh
            mesh = StructuredMesh(nx=nx, ny=nx, length=1.0, height=1.0)
            print(f"Created mesh with {nx}x{nx} cells")
            
            # Create fluid properties
            fluid = FluidProperties(
                density=1.0,
                reynolds_number=re,
                characteristic_velocity=1.0
            )
            
            # Create momentum solver and velocity updater (shared by all methods)
            momentum_solver = StandardMomentumSolver()
            velocity_updater = StandardVelocityUpdater()
            
            # ----------------- ALGEBRAIC MULTIGRID SOLVER -----------------
            print("\nRunning Algebraic MultiGrid Solver...")
            
            # Create the algebraic multigrid solver (PyAMG)
            amg_solver = PyAMGSolver(
                tolerance=tol,
                max_iterations=1000,
                matrix_free=False,  # Use explicit matrix construction for better performance
                smoother='gauss_seidel',
                presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
                postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
                cycle_type='V'
            )
            
            # Create the SIMPLE algorithm with the algebraic multigrid solver
            algorithm_amg = SimpleSolver(
                mesh=mesh,
                fluid=fluid,
                pressure_solver=amg_solver,
                momentum_solver=momentum_solver,
                velocity_updater=velocity_updater,
                alpha_p=p_relax,
                alpha_u=u_relax
            )
            
            # Set boundary conditions (lid-driven cavity)
            algorithm_amg.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
            algorithm_amg.set_boundary_condition('bottom', 'wall')
            algorithm_amg.set_boundary_condition('left', 'wall')
            algorithm_amg.set_boundary_condition('right', 'wall')
            
            # Run algebraic multigrid solver and time it
            start_time_amg = time.time()
            result_amg = algorithm_amg.solve(
                max_iterations=max_iter,
                tolerance=tol,
                track_infinity_norm=True,
                infinity_norm_interval=1,
                save_profile=False
            )
            end_time_amg = time.time()
            amg_time = end_time_amg - start_time_amg
            
            # Calculate residuals
            amg_residuals = result_amg.residuals[-1] if hasattr(result_amg, 'residuals') else 0.0
            
            print(f"Algebraic MultiGrid completed in {amg_time:.2f} seconds")
            print(f"Final residual: {amg_residuals:.6e}")
            
            # ----------------- CONJUGATE GRADIENT SOLVER -----------------
            print("\nRunning Conjugate Gradient Solver...")
            
            # Create the conjugate gradient solver
            cg_solver = MatrixFreeCGSolver(
                tolerance=tol,
                max_iterations=5000  # May need more iterations for CG
            )
            
            # Create the SIMPLE algorithm with the CG solver
            algorithm_cg = SimpleSolver(
                mesh=mesh,
                fluid=fluid,
                pressure_solver=cg_solver,
                momentum_solver=momentum_solver,
                velocity_updater=velocity_updater,
                alpha_p=p_relax,
                alpha_u=u_relax
            )
            
            # Set boundary conditions
            algorithm_cg.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
            algorithm_cg.set_boundary_condition('bottom', 'wall')
            algorithm_cg.set_boundary_condition('left', 'wall')
            algorithm_cg.set_boundary_condition('right', 'wall')
            
            # Run CG solver and time it
            start_time_cg = time.time()
            result_cg = algorithm_cg.solve(
                max_iterations=max_iter,
                tolerance=tol,
                track_infinity_norm=True,
                infinity_norm_interval=1,
                save_profile=False
            )
            end_time_cg = time.time()
            cg_time = end_time_cg - start_time_cg
            
            # Calculate residuals
            cg_residuals = result_cg.residuals[-1] if hasattr(result_cg, 'residuals') else 0.0
            
            print(f"Conjugate Gradient completed in {cg_time:.2f} seconds")
            print(f"Final residual: {cg_residuals:.6e}")
            
            # ----------------- PRECONDITIONED CG SOLVER -----------------
            print("\nRunning Preconditioned CG Solver...")
            
            # Create the preconditioned CG solver
            pcg_solver = PreconditionedCGSolver(
                tolerance=tol,
                max_iterations=2000,  
                smoother='gauss_seidel',
                presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                cycle_type='V'
            )
            
            # Create the SIMPLE algorithm with the PCG solver
            algorithm_pcg = SimpleSolver(
                mesh=mesh,
                fluid=fluid,
                pressure_solver=pcg_solver,
                momentum_solver=momentum_solver,
                velocity_updater=velocity_updater,
                alpha_p=p_relax,
                alpha_u=u_relax
            )
            
            # Set boundary conditions
            algorithm_pcg.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
            algorithm_pcg.set_boundary_condition('bottom', 'wall')
            algorithm_pcg.set_boundary_condition('left', 'wall')
            algorithm_pcg.set_boundary_condition('right', 'wall')
            
            # Run PCG solver and time it
            start_time_pcg = time.time()
            result_pcg = algorithm_pcg.solve(
                max_iterations=max_iter,
                tolerance=tol,
                track_infinity_norm=True,
                infinity_norm_interval=1,
                save_profile=False
            )
            end_time_pcg = time.time()
            pcg_time = end_time_pcg - start_time_pcg
            
            # Calculate residuals
            pcg_residuals = result_pcg.residuals[-1] if hasattr(result_pcg, 'residuals') else 0.0
            
            print(f"Preconditioned CG completed in {pcg_time:.2f} seconds")
            print(f"Final residual: {pcg_residuals:.6e}")
            
            # Store results for this mesh size and tolerance
            results['algebraic_multigrid'][tol].append((nx, amg_time, amg_residuals))
            results['conjugate_gradient'][tol].append((nx, cg_time, cg_residuals))
            results['preconditioned_cg'][tol].append((nx, pcg_time, pcg_residuals))
    
    # Save numerical results to CSV
    save_results_to_csv(results, "results/solver_benchmark_results.csv")
    
    return results


def save_results_to_csv(results, filename):
    """
    Save benchmark results to a CSV file.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing benchmark results
    filename : str
        Path to save the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    mesh_sizes = results['mesh_sizes']
    tolerances = results['tolerances']
    
    with open(filename, 'w') as f:
        f.write('Tolerance,Mesh Size,AMG Time (s),CG Time (s),PCG Time (s),AMG Residual,CG Residual,PCG Residual\n')
        
        for tol in tolerances:
            for i, nx in enumerate(mesh_sizes):
                amg_data = results['algebraic_multigrid'][tol][i]
                cg_data = results['conjugate_gradient'][tol][i]
                pcg_data = results['preconditioned_cg'][tol][i]
                
                f.write(f"{tol:.1e},{nx},{amg_data[1]:.6f},{cg_data[1]:.6f},{pcg_data[1]:.6f},{amg_data[2]:.6e},{cg_data[2]:.6e},{pcg_data[2]:.6e}\n")
    
    print(f"\nResults saved to {filename}")


def plot_results(results_file, output_dir="results"):
    """
    Plot benchmark results from a CSV file.
    
    Parameters:
    -----------
    results_file : str
        Path to the CSV file containing benchmark results
    output_dir : str
        Directory to save output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read results from CSV
    df = pd.read_csv(results_file)
    
    # Get unique mesh sizes and tolerances
    mesh_sizes = sorted(df['Mesh Size'].unique())
    tolerances = sorted(df['Tolerance'].unique())
    
    # Colors and markers for different solvers
    solver_styles = {
        'AMG': {'color': 'blue', 'marker': 'o', 'label': 'Algebraic Multigrid'},
        'CG': {'color': 'red', 'marker': 's', 'label': 'Conjugate Gradient'},
        'PCG': {'color': 'green', 'marker': '^', 'label': 'Preconditioned CG'}
    }
    
    # Line styles for different tolerances
    tolerance_styles = ['-', '--', '-.', ':']
    
    # ----- COMBINED PLOT FOR EACH TOLERANCE -----
    for tol in tolerances:
        plt.figure(figsize=(10, 6))
        
        # Filter data for this tolerance
        tol_data = df[df['Tolerance'] == tol]
        
        # Plot each solver
        for solver_key, style in solver_styles.items():
            time_col = f'{solver_key} Time (s)'
            plt.plot(
                tol_data['Mesh Size'], 
                tol_data[time_col], 
                color=style['color'], 
                marker=style['marker'], 
                label=style['label']
            )
        
        plt.xlabel('Mesh Size (n x n)')
        plt.ylabel('Wall Time (seconds)')
        plt.title(f'Solver Performance Comparison (Tolerance = {tol:.1e})')
        plt.grid(True)
        plt.legend()
        plt.xscale('log', base=2)
        plt.yscale('log')
        
        # Save this tolerance-specific plot
        plt.tight_layout()
        #plt.savefig(os.path.join(output_dir, f'solver_scaling_tol_{tol:.1e}.png'), dpi=300)
    
    # ----- COMBINED PLOT WITH ALL TOLERANCES -----
    plt.figure(figsize=(12, 8))
    
    # Plot wall time vs mesh size for each solver and tolerance
    for i, tol in enumerate(tolerances):
        line_style = tolerance_styles[i % len(tolerance_styles)]
        tol_data = df[df['Tolerance'] == tol]
        
        for solver_key, style in solver_styles.items():
            time_col = f'{solver_key} Time (s)'
            label = f"{style['label']}, tol={tol:.1e}"
            plt.plot(
                tol_data['Mesh Size'], 
                tol_data[time_col], 
                color=style['color'], 
                marker=style['marker'], 
                linestyle=line_style,
                label=label
            )
    
    plt.xlabel('Mesh Size (n x n)')
    plt.ylabel('Wall Time (seconds)')
    plt.title('Solver Performance Comparison for Different Mesh Sizes and Tolerances')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Save the combined plot
    plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, 'solver_scaling_comparison.png'), dpi=300)
    
    # ----- 4-PANEL PLOT SHOWING SCALING FOR EACH TOLERANCE -----
    # Create a 4-panel plot with one panel for each tolerance level
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    axs = axs.flat
    
    # Plot titles for each panel (one tolerance per panel)
    panel_titles = [f"Tolerance = {tol:.1e}" for tol in tolerances[:4]]  # Use up to 4 tolerances
    
    # Create each panel
    for i, tol in enumerate(tolerances[:4]):  # Use up to 4 tolerances
        if i >= len(axs):  # In case we have fewer than 4 tolerances
            break
            
        ax = axs[i]
        # Filter data for this tolerance
        tol_data = df[df['Tolerance'] == tol]
        
        # Add all solvers to each tolerance panel
        for solver_key, style in solver_styles.items():
            time_col = f'{solver_key} Time (s)'
            ax.plot(
                tol_data['Mesh Size'],
                tol_data[time_col],
                color=style['color'],
                marker=style['marker'],
                label=style['label']
            )
        
        ax.set_title(panel_titles[i])
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        
        # Only add x-label to bottom plots
        if i >= 2:
            ax.set_xlabel('Mesh Size (n x n)')
        
        # Only add y-label to leftmost plots
        if i % 2 == 0:
            ax.set_ylabel('Wall Time (seconds)')
    
    # Add overall title
    fig.suptitle(f'Solver Scaling by Tolerance Level (Re=10000, SIMPLE Iterations=1)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the overall title
    plt.savefig(os.path.join(output_dir, 'solver_scaling_by_tolerance.pdf'), dpi=300)
    print(f"Plots saved to {output_dir}/")


def run_and_plot():
    """Run benchmarks and create plots."""
    # Define mesh sizes to test
    mesh_sizes = [63, 127, 255, 511, 1023]
    
    # Define tolerance values to test
    tolerances = [1e-2, 1e-3, 1e-5, 1e-10]
    
    # Run the benchmark
    results = run_benchmark(mesh_sizes, tolerances, iterations=1, re=5000)
    
    # Plot the results
    plot_results("results/solver_benchmark_results.csv")


def plot_only():
    """Plot results from existing CSV file without running benchmarks."""
    # Plot the results from previously saved data
    plot_results("results/solver_benchmark_results.csv")


if __name__ == "__main__":
    
    
    #plot_only()
        
    run_and_plot()






