import numpy as np
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh
from naviflow_oo.solver.pressure_solver.multigrid import MultiGridSolver

def test_multigrid_solver():
    # Create a simple test mesh (7x7 grid)
    nx, ny = 7, 7
    mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
    
    # Create test velocity fields (simple divergence-free field)
    u_star = np.zeros((nx+1, ny))
    v_star = np.zeros((nx, ny+1))
    
    # Create test momentum equation coefficients
    d_u = np.ones((nx+1, ny)) * 0.1
    d_v = np.ones((nx, ny+1)) * 0.1
    
    # Create test pressure field
    p_star = np.zeros((nx, ny))
    
    # Initialize solver
    solver = MultiGridSolver(
        tolerance=1e-6,
        max_iterations=100,
        pre_smoothing=3,
        post_smoothing=3
    )
    
    # Solve pressure correction equation
    p_prime = solver.solve(mesh, u_star, v_star, d_u, d_v, p_star)
    
    # Print results
    print("Pressure correction field:")
    print(p_prime)
    print("\nResidual history:")
    print(solver.residual_history)
    
    # Plot V-cycle results
    solver.plot_vcycle_results('test_vcycle_results.pdf')

if __name__ == "__main__":
    test_multigrid_solver() 