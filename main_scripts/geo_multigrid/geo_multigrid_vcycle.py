"""
Lid-driven cavity flow simulation using the object-oriented framework with multigrid solver
that uses JacobiSolver as the smoother.
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
from naviflow_oo.solver.pressure_solver.multigrid import MultiGridSolver
from naviflow_oo.solver.pressure_solver.jacobi import JacobiSolver
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater

# Debug mode for testing

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Create debug output directory
debug_dir = 'debug_output'
os.makedirs(debug_dir, exist_ok=True)

# Start timing
start_time = time.time()

# Update parameters
# Grid size
nx, ny = 2**8-1, 2**8-1  # Smaller grid for testing


# Reduced relaxation factors and iterations
max_iterations = 10000
convergence_tolerance = 1e-4
alpha_p = 0.1  # Pressure relaxation (reduced from 0.1)
alpha_u = 0.7   # Velocity relaxation (reduced from 0.7)

# Create mesh
print(f"Creating mesh with {nx}x{ny} cells...")
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
dx, dy = mesh.get_cell_sizes()
print(f"Cell sizes: dx={dx}, dy={dy}")

# Create initial conditions
Re = 100
print(f"Reynolds number: {Re}")
viscosity = 0.01
print(f"Calculated viscosity: {viscosity}")

# 4. Create solvers
# Create a Jacobi smoother for the multigrid solver
smoother = JacobiSolver(omega=2/3)  # Lower omega for stability

# Create multigrid solver with conservative parameters
multigrid_solver = MultiGridSolver(
    smoother=smoother,
    max_iterations=1000,        # Fewer iterations
    tolerance=1e-6,          # Tighter tolerance
    pre_smoothing=20,         # Fewer pre-smoothing steps
    post_smoothing=20,        # Fewer post-smoothing steps 
    smoother_omega=2/3       # Conservative relaxation
)
momentum_solver = StandardMomentumSolver()
velocity_updater = StandardVelocityUpdater()

# 5. Create algorithm
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

# 6. Set boundary conditions
algorithm.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
algorithm.set_boundary_condition('bottom', 'wall')
algorithm.set_boundary_condition('left', 'wall')
algorithm.set_boundary_condition('right', 'wall')

# 7. Solve the problem
print("Starting simulation...")
result = algorithm.solve(max_iterations=max_iterations, tolerance=convergence_tolerance, 
                         track_infinity_norm=True, infinity_norm_interval=5)

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
    title=f'Multigrid with Jacobi Smoother Cavity Flow Results (Re={Re})',
    filename=os.path.join(results_dir, f'cavity_Re{Re}_multigrid_jacobi_results.pdf'),
    show=False
)
"""
# After the simulation completes, plot the V-cycle results
algorithm.pressure_solver.plot_vcycle_results(os.path.join(debug_dir, 'vcycle_analysis.pdf'))

# Update the benchmark comparison section
print("Loading benchmark pressure data...")
benchmark_path = os.path.join(os.path.dirname(__file__), 'multigrid_debugging', 'arrays_cg127x127', 'p_prime.npy')
try:
    benchmark_p = np.load(benchmark_path)
    print(f"Benchmark pressure shape: {benchmark_p.shape}")
    
    # Compare only if grids match
    if result.p.shape == benchmark_p.shape:
        # Direct comparison when shapes match
        diff = result.p - benchmark_p
        
        # Create pressure comparison plot
        plt.figure(figsize=(15, 5))
        
        # Plot current solution
        plt.subplot(1, 3, 1)
        plt.contourf(result.p, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title('Current Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plot benchmark solution
        plt.subplot(1, 3, 2)
        plt.contourf(benchmark_p, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title('Benchmark Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plot relative error
        plt.subplot(1, 3, 3)
        relative_error = np.abs(diff) / (np.abs(benchmark_p) + 1e-10)  # Add small constant to avoid division by zero
        log_relative_error = np.log10(relative_error)
        plt.contourf(log_relative_error, levels=50, cmap='coolwarm')
        plt.colorbar()
        plt.title('Relative Error (log scale)')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, 'pressure_comparison.pdf'))
        plt.close()
        
        # Calculate and print error metrics
        max_error = np.max(np.abs(diff))
        mean_error = np.mean(np.abs(diff))
        print("\nPressure Comparison Results:")
        print(f"Maximum absolute error: {max_error:.6e}")
        print(f"Mean absolute error: {mean_error:.6e}")
        
        # Error analysis
        print("\nError Analysis:")
        print(f"Maximum error magnitude: {np.max(np.abs(diff)):.2e}")
        print(f"Mean error magnitude: {np.mean(np.abs(diff)):.2e}")
        print(f"Error/solution ratio: {np.max(np.abs(diff))/np.max(np.abs(result.p)):.6f}")
        
        # Visualize pressure field and error
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.title("Computed Pressure Field")
        plt.imshow(result.p, cmap='viridis')
        plt.colorbar()
        plt.subplot(132)
        plt.title("Error Distribution")
        plt.imshow(diff, cmap='coolwarm')
        plt.colorbar()
        plt.subplot(133)
        plt.title("Relative Error")
        relative_error = np.abs(diff) / (np.abs(benchmark_p) + 1e-10)
        plt.imshow(np.clip(relative_error, 0, 2), cmap='hot')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "pressure_error_analysis.png"))
    else:
        print("\nSkipping benchmark comparison - grid sizes don't match")
        print(f"Result shape: {result.p.shape}, Benchmark shape: {benchmark_p.shape}")
        
        # Just visualize the current solution
        plt.figure(figsize=(8, 6))
        plt.title(f"Pressure Field (Grid size: {nx}x{ny})")
        plt.imshow(result.p, cmap='viridis')
        plt.colorbar(label='Pressure')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"pressure_field_{nx}x{ny}.png"))
        plt.close()
except Exception as e:
    print(f"Error loading benchmark data: {e}")
    print("Skipping benchmark comparison")

# Add safety check after multigrid solver call
if np.any(np.isnan(result.p)) or np.any(np.isinf(result.p)):
    print("ERROR: NaN or Inf values in pressure result!")
    # Apply fallback to get a stable but less accurate solution
    print("Applying fallback direct solver...")
    algorithm.pressure_solver = JacobiSolver(omega=0.5)
    algorithm.solve()
    # Check if fallback solution is valid
    if np.any(np.isnan(result.p)) or np.any(np.isinf(result.p)):
        print("FATAL: Even fallback solver produced invalid results")
        # Replace with zeros as last resort
        result.p = np.zeros_like(result.p)

# After processing results, add detailed output
print("\nFinal Multigrid Solver Statistics:")
print(f"Maximum absolute value in solution: {np.max(np.abs(result.p)):.6e}")
print(f"Mean absolute value in solution: {np.mean(np.abs(result.p)):.6e}")
print(f"Grid size used: {nx}x{ny}")
print(f"Final residual: {algorithm.pressure_solver.final_residual if hasattr(algorithm.pressure_solver, 'final_residual') else 'N/A'}")
print(f"Iterations completed: {algorithm.pressure_solver.iterations if hasattr(algorithm.pressure_solver, 'iterations') else 'N/A'}")
"""