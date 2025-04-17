"""
Example demonstrating how to use zero pressure boundary conditions with the multigrid solver.

This example shows how to use the MultigridSolver with different boundary condition types
using the unified boundary condition handling approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.solver.pressure_solver.multigrid import MultiGridSolver
from naviflow_oo.solver.pressure_solver.jacobi import JacobiSolver


def channel_flow_example():
    """
    Setup and solve a simple channel flow problem using different boundary conditions.
    """
    # Create a mesh (channel geometry)
    nx, ny = 63, 31  # Grid size
    length, height = 2.0, 1.0  # Channel dimensions
    mesh = StructuredMesh(nx=nx, ny=ny, length=length, height=height)
    
    # Create velocity and coefficient fields
    # For a simple example, these can be initialized with placeholder values
    u_star = np.zeros((nx+1, ny))
    v_star = np.zeros((nx, ny+1))
    p_star = np.zeros((nx, ny))
    
    # Initialize with a parabolic velocity profile at the inlet
    for j in range(ny):
        y = j * height / (ny-1)
        # Parabolic profile: u(y) = 4*u_max*y*(H-y)/H^2 for y ∈ [0,H]
        u_max = 1.0
        u_star[0, j] = 4.0 * u_max * y * (height - y) / (height * height)
    
    # For the pressure equation, we need momentum coefficients
    # These would normally come from the momentum equations
    d_u = np.ones((nx+1, ny)) * 0.1
    d_v = np.ones((nx, ny+1)) * 0.1
    
    # Set up the multigrid solver with a Jacobi smoother
    smoother = JacobiSolver(omega=0.8)
    multigrid_solver = MultiGridSolver(
        smoother=smoother,
        max_iterations=100,
        tolerance=1e-8,
        pre_smoothing=2,
        post_smoothing=2,
        cycle_type='v'
    )
    
    # Solve with different boundary conditions for comparison
    
    # 1. Default zero gradient (Neumann) boundary conditions
    p_prime_neumann = multigrid_solver.solve(
        mesh=mesh,
        u_star=u_star,
        v_star=v_star,
        d_u=d_u,
        d_v=d_v,
        p_star=p_star
    )
    
    # 2. Zero pressure at the outlet (east boundary)
    p_prime_dirichlet = multigrid_solver.solve(
        mesh=mesh,
        u_star=u_star,
        v_star=v_star,
        d_u=d_u,
        d_v=d_v,
        p_star=p_star,
        use_zero_pressure_bc=True,
        zero_pressure_boundaries=['east']  # Set outlet (east) to zero pressure
    )
    
    # Plot the results for comparison
    plot_comparison(p_prime_neumann, p_prime_dirichlet, length, height)


def plot_comparison(p_neumann, p_dirichlet, length, height):
    """Plot and compare the pressure fields with different boundary conditions."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot zero gradient BC result
    im1 = ax1.imshow(p_neumann, origin='lower', aspect='auto', 
                    extent=[0, length, 0, height])
    ax1.set_title('Pressure with Zero Gradient BC')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)
    
    # Plot zero pressure BC result
    im2 = ax2.imshow(p_dirichlet, origin='lower', aspect='auto',
                    extent=[0, length, 0, height])
    ax2.set_title('Pressure with Zero Pressure at Outlet')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)
    
    # Plot centerline pressure comparison
    centerline_idx = p_neumann.shape[0] // 2
    ax3.plot(np.linspace(0, length, p_neumann.shape[1]), 
            p_neumann[centerline_idx, :], 'b-', label='Zero Gradient BC')
    ax3.plot(np.linspace(0, length, p_dirichlet.shape[1]), 
            p_dirichlet[centerline_idx, :], 'r--', label='Zero Pressure at Outlet')
    ax3.set_title('Centerline Pressure Comparison')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Pressure')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('pressure_boundary_condition_comparison.pdf')
    plt.show()


if __name__ == '__main__':
    channel_flow_example() 