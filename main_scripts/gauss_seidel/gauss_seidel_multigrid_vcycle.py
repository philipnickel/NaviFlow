"""
Lid-driven cavity flow simulation using the object-oriented framework with multigrid solver
that uses GaussSeidelSolver as the smoother.
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
from naviflow_oo.solver.pressure_solver.gauss_seidel import GaussSeidelSolver
from naviflow_oo.solver.momentum_solver.standard import StandardMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)

# Create debug output directory
debug_dir = os.path.join(os.path.dirname(__file__), 'debug_output')
os.makedirs(debug_dir, exist_ok=True)

# Start timing
start_time = time.time()

# Grid size - must be 2^k-1 for multigrid (e.g., 31, 63, 127, 255)
nx, ny = 2**8-1, 2**8-1  # 127x127 grid

# Relaxation factors and iterations
max_iterations = 10000
convergence_tolerance = 1e-5
alpha_p = 1  # Pressure relaxation
alpha_u = 1 # Velocity relaxation

# Create mesh
print(f"Creating mesh with {nx}x{ny} cells...")
mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
dx, dy = mesh.get_cell_sizes()
print(f"Cell sizes: dx={dx}, dy={dy}")

# Create initial conditions
Re = 1000
print(f"Reynolds number: {Re}")
viscosity = 0.01
print(f"Calculated viscosity: {viscosity}")

# Create solvers
# Create a Gauss-Seidel smoother for the multigrid solver with SOR
smoother = GaussSeidelSolver()

# Create multigrid solver with the Gauss-Seidel smoother
multigrid_solver = MultiGridSolver(
    smoother=smoother,
    max_iterations=100,    # Maximum V-cycles
    tolerance=1e-5,         # Overall tolerance
    pre_smoothing=10,        # Pre-smoothing steps
    post_smoothing=10,       # Post-smoothing steps
    smoother_omega=1.5      # SOR parameter
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

# Solve the problem
print("Starting simulation...")
result = algorithm.solve(max_iterations=max_iterations, tolerance=convergence_tolerance, 
                        track_infinity_norm=True, infinity_norm_interval=5, plot_final_residuals=False)

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
    title=f'Multigrid with Gauss-Seidel Smoother Cavity Flow Results (Re={Re})',
    filename=os.path.join(results_dir, f'cavity_Re{Re}_multigrid_gauss_seidel_results.pdf'),
    show=False
)
"""
# Plot the V-cycle results if available
if hasattr(algorithm.pressure_solver, 'plot_vcycle_results'):
    algorithm.pressure_solver.plot_vcycle_results(os.path.join(debug_dir, 'vcycle_analysis.pdf'))

# Print detailed convergence information
print("\nFinal MultiGrid Solver Statistics:")
print(f"Grid size used: {nx}x{ny}")

# Print information about the smoother
smoother_info = smoother.get_solver_info()
print("\nSmoother Information:")
print(f"Smoother type: {smoother_info['name']}")
print(f"Relaxation factor (omega): {smoother_info['omega']}")
print(f"Convergence rate: {smoother_info['convergence_rate']:.6f}" if smoother_info['convergence_rate'] is not None else "Convergence rate: N/A")

# Compare with Jacobi-based multigrid (estimation)
print("\nTheoretical Comparison with Jacobi Smoother:")
if smoother_info['convergence_rate'] is not None:
    gs_rate = smoother_info['convergence_rate']
    jacobi_rate = gs_rate**0.5  # Theoretical relationship
    smoothing_speedup = np.log(1e-4) / np.log(gs_rate) * np.log(jacobi_rate) / np.log(1e-4)
    print(f"Estimated speedup in smoothing operations: {smoothing_speedup:.2f}x")
else:
    print("Convergence rate data not available for comparison") 


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


# After processing results, add detailed output
print("\nFinal Multigrid Solver Statistics:")
print(f"Maximum absolute value in solution: {np.max(np.abs(result.p)):.6e}")
print(f"Mean absolute value in solution: {np.mean(np.abs(result.p)):.6e}")
print(f"Grid size used: {nx}x{ny}")
print(f"Final residual: {algorithm.pressure_solver.final_residual if hasattr(algorithm.pressure_solver, 'final_residual') else 'N/A'}")
print(f"Iterations completed: {algorithm.pressure_solver.iterations if hasattr(algorithm.pressure_solver, 'iterations') else 'N/A'}")
"""