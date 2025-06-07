import sys
import os
import numpy as np
from multigridPerformanceAnalysis.plot_style import plt
from scipy.sparse.linalg import bicgstab, LinearOperator

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from naviflow_staggered.preprocessing.mesh.structured import StructuredMesh
from naviflow_staggered.solver.pressure_solver.helpers.matrix_free import compute_Ap_product

def p_manufactured(x, y):
    """Manufactured solution for pressure with zero-gradient boundaries."""
    return (1 - np.cos(2 * np.pi * x)) * (1 - np.cos(2 * np.pi * y))

def run_mms_test():
    """Run MMS test for the staggered grid multigrid solver."""
    grid_sizes = [15, 31, 63, 127]
    errors = []
    dx_values = []

    for n in grid_sizes:
        print(f"Running for grid size: {n}x{n}")
        mesh = StructuredMesh(nx=n, ny=n, length=1.0, height=1.0)
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        dx_values.append(dx)

        # Create manufactured solution on the grid
        p_mms = p_manufactured(mesh.X, mesh.Y)
        p_mms_flat = p_mms.flatten('F')

        # Set fluid properties and coefficients
        rho = 1.0
        d_u = np.ones((nx + 1, ny))
        d_v = np.ones((nx, ny + 1))

        # Compute the manufactured RHS b = A * p_mms
        b_manufactured = compute_Ap_product(p_mms_flat, nx, ny, dx, dy, rho, d_u, d_v, pin_pressure=True)

        # Define the matrix-vector product for the linear operator
        def mv_product(v):
            return compute_Ap_product(v.astype(np.float64), nx, ny, dx, dy, rho, d_u, d_v, pin_pressure=True)

        # Create a LinearOperator
        A_op = LinearOperator((nx * ny, nx * ny), matvec=mv_product, dtype=np.float64)

        # Initial guess
        x0 = np.zeros_like(b_manufactured, dtype=np.float64)

        # Use BiCGSTAB to solve the system
        p_solution, info = bicgstab(
            A_op,
            b_manufactured,
            x0=x0,
            atol=1e-12,
            maxiter=5000
        )

        if info == 0:
            print(f"  BiCGSTAB converged.")
        else:
            print(f"  Warning: BiCGSTAB did not converge, info={info}")

        # Calculate L2 norm of the error
        error_flat = p_solution - p_mms_flat
        l2_error = np.linalg.norm(error_flat) / np.linalg.norm(p_mms_flat)
        errors.append(l2_error)
        print(f"  L2 Error: {l2_error:.4e}\n")

    # Plotting the error convergence
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(dx_values, errors, 'o-', label='L2 Error', linewidth=2.0, markersize=8)
    
    # Fit a line to the log-log plot to find the order of convergence
    if len(dx_values) > 1:
        log_dx = np.log(dx_values)
        log_errors = np.log(errors)
        coeffs = np.polyfit(log_dx, log_errors, 1)
        order = coeffs[0]
        ax.loglog(dx_values, np.exp(coeffs[1] + order * log_dx), 'r--', label=f'Order = {order:.2f}', linewidth=2.0)
    
    ax.set_xlabel('Grid Spacing (dx)')
    ax.set_ylabel('L2 Norm of Error')
    ax.set_title('MMS Convergence Test for Staggered Grid Solver')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.gca().invert_xaxis()
    
    # Save the plot
    results_dir = os.path.join(script_dir, 'results_mms_staggered')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, 'mms_convergence.png'), dpi=300)
    print(f"Convergence plot saved to {os.path.join(results_dir, 'mms_convergence.png')}")

if __name__ == "__main__":
    run_mms_test() 