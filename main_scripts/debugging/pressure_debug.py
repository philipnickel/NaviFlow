import numpy as np
from naviflow_oo.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_oo.solver.pressure_solver.direct import DirectPressureSolver
# from naviflow_oo.solver.pressure_solver.helpers.rhs_construction import get_rhs # REMOVED
# from naviflow_oo.solver.pressure_solver.helpers.coeff_matrix import get_coeff_mat # REMOVED
from naviflow_oo.preprocessing.mesh.structured import StructuredUniform
from naviflow_oo.constructor.properties.fluid import FluidProperties
# from naviflow_oo.boundary_conditions.manager import BoundaryConditionManager # REMOVED
# from scipy.sparse import csc_matrix # REMOVED
# from scipy.sparse.linalg import spsolve # REMOVED (Solver handles this)

def test_pressure_solve_direct():
    # --- Setup similar to momentum debug ---
    # Create structured 3x3 mesh
    mesh = StructuredUniform(n_cells_x=3, n_cells_y=3, xmin=0, xmax=1, ymin=0, ymax=1)
    nx_cells, ny_cells = mesh.get_dimensions()
    n_cells = mesh.n_cells

    # Create fluid properties
    fluid = FluidProperties(density=1.0, viscosity=1.0)

    # Initialize fields
    u_shape, v_shape, p_shape = mesh.get_field_shapes()
    u = np.zeros(u_shape)
    v = np.zeros(v_shape)
    p = np.zeros(p_shape) # Use this as p_star initially

    # --- Get u_star, v_star, d_u, d_v ---
    momentum_solver = AMGMomentumSolver(discretization_scheme="upwind", tolerance=1e-10)
    u_star_flat, d_u_flat, _ = momentum_solver.solve_u_momentum(mesh, fluid, u, v, p, relaxation_factor=1.0)
    v_star_flat, d_v_flat, _ = momentum_solver.solve_v_momentum(mesh, fluid, u_star_flat.reshape(u_shape), v, p, relaxation_factor=1.0)

    # Flatten inputs for pressure solver
    u_star = u_star_flat
    v_star = v_star_flat
    p_star = p.flatten()
    # Calculate average d coefficient
    d_avg = 0.5 * (d_u_flat + d_v_flat) 
    # d_u = d_u_flat # No longer pass d_u/d_v separately
    # d_v = d_v_flat

    # --- Instantiate Pressure Solver ---
    pressure_solver = DirectPressureSolver()

    # --- Solve Pressure Correction using the solver ---
    try:
        p_prime, residual_info = pressure_solver.solve(
            mesh, u_star, v_star, d_avg, p_star, return_dict=True
        )
        print("Solver call successful.")

        # --- Verify Solution ---
        # Note: p_prime returned by solve should already be reshaped if p_star had shape
        # If p_star was flat, p_prime will be flat. Let's check and reshape if needed.
        if p_prime.shape != p_shape:
             if p_prime.size == np.prod(p_shape):
                 print("Reshaping p_prime...")
                 p_prime = p_prime.reshape(p_shape)
             else:
                 print(f"WARNING: p_prime shape {p_prime.shape} cannot be reshaped to {p_shape}")
        
        print("Pressure Correction Field (p_prime):")
        np.set_printoptions(precision=3, suppress=True)
        print(p_prime)
        print(f"p_prime min/max/mean: {p_prime.min():.4g} / {p_prime.max():.4g} / {p_prime.mean():.4g}")

        # Check residual info from solver
        rel_norm = residual_info['rel_norm']
        print(f"Relative residual norm (from solver info): {rel_norm:.4g}")

    except Exception as e:
        print(f"ERROR during solver.solve(): {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_pressure_solve_direct()
