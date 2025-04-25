"""
Example script that demonstrates the use of different matrix solvers.

This script compares the performance of BiCGSTAB, GMRES, and AMG solvers 
on a simple lid-driven cavity problem.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the solver
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from naviflow_oo.solver.momentum_solver.matrix_momentum_solver import MatrixMomentumSolver
from naviflow_oo.mesh.structured_mesh import StructuredMesh
from naviflow_oo.fluid.fluid_properties import FluidProperties
from naviflow_oo.constructor.boundary_conditions import BoundaryConditionManager


def run_solver_comparison():
    """Run a comparison of different solvers for the momentum equations."""
    # Create a mesh for testing
    nx, ny = 50, 50  # Larger mesh to better see performance differences
    lx, ly = 1.0, 1.0
    mesh = StructuredMesh(nx, ny, lx, ly)
    
    # Set fluid properties
    fluid = FluidProperties(density=1.0, viscosity=0.01)
    
    # Initialize fields
    imax, jmax = nx, ny
    u = np.zeros((imax+1, jmax))
    v = np.zeros((imax, jmax+1))
    p = np.zeros((imax, jmax))
    
    # Set boundary conditions (lid-driven cavity)
    bc_manager = BoundaryConditionManager()
    bc_manager.set_condition('north', 'u', {'type': 'dirichlet', 'value': 1.0})
    bc_manager.set_condition('north', 'v', {'type': 'dirichlet', 'value': 0.0})
    bc_manager.set_condition('south', 'u', {'type': 'dirichlet', 'value': 0.0})
    bc_manager.set_condition('south', 'v', {'type': 'dirichlet', 'value': 0.0})
    bc_manager.set_condition('east', 'u', {'type': 'dirichlet', 'value': 0.0})
    bc_manager.set_condition('east', 'v', {'type': 'dirichlet', 'value': 0.0})
    bc_manager.set_condition('west', 'u', {'type': 'dirichlet', 'value': 0.0})
    bc_manager.set_condition('west', 'v', {'type': 'dirichlet', 'value': 0.0})
    
    # Apply BCs to initial fields
    u, v = bc_manager.apply_velocity_boundary_conditions(u, v, imax, jmax)
    
    # Define solver configurations to test
    solver_configs = [
        {
            'name': 'BiCGSTAB',
            'type': 'bicgstab',
            'precond': False,
            'color': 'blue'
        },
        {
            'name': 'BiCGSTAB + ILU',
            'type': 'bicgstab',
            'precond': True,
            'color': 'green'
        },
        {
            'name': 'GMRES',
            'type': 'gmres',
            'precond': False,
            'color': 'red'
        },
        {
            'name': 'GMRES + ILU',
            'type': 'gmres',
            'precond': True,
            'color': 'purple'
        }
    ]
    
    # Add AMG if available
    try:
        import pyamg
        solver_configs.append({
            'name': 'AMG (V-cycle)',
            'type': 'amg',
            'precond': False,
            'color': 'orange',
            'cycle_type': 'V'
        })
        solver_configs.append({
            'name': 'AMG (W-cycle)',
            'type': 'amg',
            'precond': False,
            'color': 'brown',
            'cycle_type': 'W'
        })
    except ImportError:
        print("PyAMG not installed. AMG solver will not be tested.")
    
    # Store results
    results = []
    
    # Test each solver configuration
    for config in solver_configs:
        print(f"\nTesting {config['name']}...")
        
        # Create solver
        solver_args = {
            'solver_type': config['type'],
            'discretization_scheme': 'power_law',
            'tolerance': 1e-6,
            'max_iterations': 1000,
            'use_preconditioner': config['precond'],
            'print_its': True,
            'restart': 30
        }
        
        # Add AMG-specific parameters if applicable
        if config['type'] == 'amg':
            solver_args['amg_cycle_type'] = config.get('cycle_type', 'V')
        
        solver = MatrixMomentumSolver(**solver_args)
        
        # Time the solver for u-momentum
        start_time = time.time()
        u_star, _, residual_info_u = solver.solve_u_momentum(
            mesh, fluid, u, v, p,
            relaxation_factor=0.8, boundary_conditions=bc_manager
        )
        u_time = time.time() - start_time
        
        # Time the solver for v-momentum
        start_time = time.time()
        v_star, _, residual_info_v = solver.solve_v_momentum(
            mesh, fluid, u, v, p,
            relaxation_factor=0.8, boundary_conditions=bc_manager
        )
        v_time = time.time() - start_time
        
        # Save results
        results.append({
            'name': config['name'],
            'u_time': u_time,
            'v_time': v_time,
            'u_iterations': residual_info_u['iterations'],
            'v_iterations': residual_info_v['iterations'],
            'color': config['color'],
            'u_solution': u_star.copy(),
            'v_solution': v_star.copy()
        })
        
        print(f"  U-momentum: {u_time:.4f}s, {residual_info_u['iterations']} iterations")
        print(f"  V-momentum: {v_time:.4f}s, {residual_info_v['iterations']} iterations")
    
    # Plot the results
    plot_results(results, nx, ny)


def plot_results(results, nx, ny):
    """Plot the comparison of the different solvers."""
    plt.figure(figsize=(12, 8))
    
    # Plot timing comparison
    plt.subplot(2, 1, 1)
    names = [r['name'] for r in results]
    u_times = [r['u_time'] for r in results]
    v_times = [r['v_time'] for r in results]
    colors = [r['color'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, u_times, width, label='U-momentum', color=colors, alpha=0.7)
    plt.bar(x + width/2, v_times, width, label='V-momentum', color=colors, alpha=0.4)
    
    plt.xlabel('Solver')
    plt.ylabel('Time (s)')
    plt.title('Solver Performance Comparison')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot iteration comparison
    plt.subplot(2, 1, 2)
    u_iterations = [r['u_iterations'] for r in results]
    v_iterations = [r['v_iterations'] for r in results]
    
    plt.bar(x - width/2, u_iterations, width, label='U-momentum', color=colors, alpha=0.7)
    plt.bar(x + width/2, v_iterations, width, label='V-momentum', color=colors, alpha=0.4)
    
    plt.xlabel('Solver')
    plt.ylabel('Iterations')
    plt.title('Solver Convergence Comparison')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('solver_comparison.png')
    print("\nResults saved to 'solver_comparison.png'")
    
    # Plot u-velocity contours for the first solver (as reference)
    if results:
        plt.figure(figsize=(8, 6))
        u_solution = results[0]['u_solution']
        
        # Create a grid for plotting
        x = np.linspace(0, 1, nx+1)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Transpose and flip u for plotting
        u_plot = u_solution.T
        
        # Create contour plot
        plt.contourf(X, Y, u_plot, 20, cmap='viridis')
        plt.colorbar(label='U-velocity')
        plt.title(f'U-velocity Contours ({results[0]["name"]})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig('u_velocity_contours.png')
        print("U-velocity contours saved to 'u_velocity_contours.png'")
    
    plt.show()


if __name__ == '__main__':
    run_solver_comparison() 