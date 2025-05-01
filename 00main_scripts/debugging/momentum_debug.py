import numpy as np
from naviflow_staggered.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_staggered.preprocessing.mesh.structured_mesh import StructuredMesh
from naviflow_staggered.constructor.properties.fluid import FluidProperties  # Assuming a Fluid class is available


def test_lid_driven_cavity_amg():
    # Create structured 10x10 mesh (10x10 cells -> 11x11 nodes)
    mesh = StructuredMesh(n_cells_x=50, n_cells_y=50, xmin=0, xmax=1, ymin=0, ymax=1, is_uniform=True)
    owners, neighbors = mesh.get_owner_neighbor()
    invalid_owners = [i for i, o in enumerate(owners) if o < 0 or o >= mesh.n_cells]
    if invalid_owners:
        print(f"ERROR: Found invalid owner indices at faces: {invalid_owners}")


    # Create fluid properties (density=1, viscosity=1)
    fluid = FluidProperties(density=1.0, viscosity=1.0)

    # Initialize fields (collocated grid: u, v, p)
    u_shape, v_shape, p_shape = mesh.get_field_shapes()
    u = np.zeros(u_shape)
    v = np.zeros(v_shape)
    p = np.zeros(p_shape)

    # Create AMG solver using upwind discretization
    solver = AMGMomentumSolver(discretization_scheme="upwind", tolerance=1e-10)

    # Solve u-momentum
    u_star, d_u, u_residual = solver.solve_u_momentum(mesh, fluid, u, v, p, relaxation_factor=1.0)

    # Solve v-momentum
    v_star, d_v, v_residual = solver.solve_v_momentum(mesh, fluid, u_star, v, p, relaxation_factor=1.0)

    # Basic sanity checks
    print("--- Test Results ---")
    print("u* min/max:", u_star.min(), u_star.max())
    print("v* min/max:", v_star.min(), v_star.max())
    print("d_u min/max:", d_u.min(), d_u.max())
    print("d_v min/max:", d_v.min(), d_v.max())
    print("Residual u rel norm:", u_residual["rel_norm"])
    print("Residual v rel norm:", v_residual["rel_norm"])

    # Check if top wall velocity applied (should be non-zero u on top)
    top_indices = mesh.get_boundary_cell_indices("top")
    if top_indices is not None:
        print("Top wall u-velocity (mean):", np.mean(u_star[top_indices]))

    # Verify symmetry (optional)
    asymmetry_u = np.linalg.norm(u_star - u_star[::-1]) / (np.linalg.norm(u_star) + 1e-10)
    asymmetry_v = np.linalg.norm(v_star - v_star[::-1]) / (np.linalg.norm(v_star) + 1e-10)
    print("u symmetry deviation:", asymmetry_u)
    print("v symmetry deviation:", asymmetry_v)


if __name__ == '__main__':
    test_lid_driven_cavity_amg()
    # Recreate the mesh to print its dimensions based on the new parameters
    #mesh = StructuredMesh(n_cells_x=10, n_cells_y=10, xmin=0, xmax=1, ymin=0, ymax=1, is_uniform=True)
    #print(f"Mesh dimensions: {mesh.n_cells} cells, {mesh.n_faces} faces, {mesh.n_nodes} nodes")