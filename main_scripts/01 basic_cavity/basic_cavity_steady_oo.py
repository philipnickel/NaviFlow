"""
Lid-driven cavity flow simulation using the object-oriented framework.
This version uses the updated mesh topology classes with non-uniform mesh.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from naviflow_oo.preprocessing.mesh.structured import StructuredMesh as OldStructuredMesh
from naviflow_oo.preprocessing.mesh.mesh import UniformStructuredMesh, NonUniformStructuredMesh
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.preprocessing.fields.scalar_field import ScalarField
from naviflow_oo.preprocessing.fields.vector_field import VectorField
from naviflow_oo.solver.Algorithms.simple import SimpleSolver
from naviflow_oo.solver.pressure_solver.direct import DirectPressureSolver
from naviflow_oo.solver.momentum_solver.jacobi_solver import JacobiMomentumSolver
from naviflow_oo.solver.momentum_solver.jacobi_matrix_solver import JacobiMatrixMomentumSolver
from naviflow_oo.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_oo.solver.momentum_solver.matrix_free_momentum import MatrixFreeMomentumSolver
from naviflow_oo.solver.velocity_solver.standard import StandardVelocityUpdater
from naviflow_oo.postprocessing.visualization import plot_final_residuals

# Start timing
start_time = time.time()

# 1. Set up simulation parameters
nx, ny = 2**6-1, 2**6-1 # Grid size
reynolds = 100             # Reynolds number
alpha_p = 0.3              # Pressure relaxation factor
alpha_u = 0.8              # Velocity relaxation factor
max_iterations = 500       # Maximum number of iterations
tolerance = 1e-10

# Create an adapter class that provides all the necessary interfaces
class MeshAdapter:
    def __init__(self, mesh):
        """
        Adapter for UniformStructuredMesh or NonUniformStructuredMesh to provide the same interface as the old StructuredMesh
        
        Parameters:
        -----------
        mesh : UniformStructuredMesh or NonUniformStructuredMesh
            The structured mesh to adapt
        """
        self.mesh = mesh
        self.nx = mesh.nx - 1  # Cell count in x
        self.ny = mesh.ny - 1  # Cell count in y
        
        # Calculate cell sizes
        self.dx = (mesh.x_nodes[-1] - mesh.x_nodes[0]) / (mesh.nx - 1)
        self.dy = (mesh.y_nodes[-1] - mesh.y_nodes[0]) / (mesh.ny - 1)
        
        # Calculate domain dimensions
        self.length = mesh.x_nodes[-1] - mesh.x_nodes[0]
        self.height = mesh.y_nodes[-1] - mesh.y_nodes[0]
        
        # Additional methods
        self.bc_manager = None  # Will be set by the solver
    
    def get_dimensions(self):
        """Return the dimensions of the mesh."""
        return self.nx, self.ny
    
    def get_cell_sizes(self):
        """Return the cell sizes."""
        return self.dx, self.dy
    
    def get_node_positions(self):
        """Returns all node positions."""
        return self.mesh.get_node_positions()
    
    def get_cell_centers(self):
        """Returns all cell centers."""
        return self.mesh.get_cell_centers()
    
    def get_face_centers(self):
        """Returns all face centers."""
        return self.mesh.get_face_centers()
    
    def get_face_normals(self):
        """Returns all face normals."""
        return self.mesh.get_face_normals()
    
    def get_face_areas(self):
        """Returns all face areas."""
        return self.mesh.get_face_areas()
    
    def get_owner_neighbor(self):
        """Returns owner and neighbor cell indices for all faces."""
        return self.mesh.get_owner_neighbor()
    
    @property
    def n_cells(self):
        """Returns the number of cells in the mesh."""
        return self.mesh.n_cells
    
    @property
    def n_faces(self):
        """Returns the number of faces in the mesh."""
        return self.mesh.n_faces
    
    @property
    def n_nodes(self):
        """Returns the number of nodes in the mesh."""
        return self.mesh.n_nodes

# 2. Create mesh
# Create a stretching function for non-uniform mesh
def stretch_function(n, beta=1.5):
    """Create a stretched distribution of n points with bias toward endpoints."""
    xi = np.linspace(0, 1, n)
    return 0.5 * (1 + np.tanh(beta * (2 * xi - 1)) / np.tanh(beta))

# Create non-uniform node distributions with clustering near boundaries
x = stretch_function(nx, beta=1.8)  # Stronger clustering near walls
y = stretch_function(ny, beta=1.8)

# Option 1: Use the old structured mesh (uncomment to use)
# old_mesh = OldStructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)

# Option 2: Use non-uniform mesh with boundary layer refinement
nonuniform_mesh = NonUniformStructuredMesh(
    x_nodes=x,
    y_nodes=y
)

# Create an adapter for the non-uniform mesh that works with both old and new solvers
mesh = MeshAdapter(nonuniform_mesh)

print(f"Created non-uniform mesh with {nx}x{ny} nodes")
print(f"Average cell sizes: dx={mesh.dx:.6f}, dy={mesh.dy:.6f}")
print(f"Number of faces: {mesh.n_faces}")
print(f"Number of cells: {mesh.n_cells}")

# 3. Define fluid properties
fluid = FluidProperties(
    density=1.0,
    reynolds_number=reynolds,
    characteristic_velocity=1.0
)
print(f"Reynolds number: {fluid.get_reynolds_number()}")
print(f"Calculated viscosity: {fluid.get_viscosity()}")

# 4. Create solvers
# Use the updated direct pressure solver with mesh topology support
pressure_solver = DirectPressureSolver()

# Uncomment one momentum solver to use
# momentum_solver = JacobiMatrixMomentumSolver(n_jacobi_sweeps=1)
# momentum_solver = CGMatrixMomentumSolver(tolerance=1e-1, max_iterations=1000)
# Use the new AMG solver
momentum_solver = AMGMomentumSolver(discretization_scheme='power_law', tolerance=1e-7, max_iterations=10000)
# momentum_solver = MatrixFreeMomentumSolver(discretization_scheme='power_law', tolerance=1e-7, max_iterations=100000, solver_type='bicgstab')
velocity_updater = StandardVelocityUpdater()

# 5. Create algorithm
algorithm = SimpleSolver(
    mesh=mesh,
    fluid=fluid,
    pressure_solver=pressure_solver,
    momentum_solver=momentum_solver,
    velocity_updater=velocity_updater,
    alpha_p=alpha_p,
    alpha_u=alpha_u,
)

# 6. Set boundary conditions
algorithm.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
algorithm.set_boundary_condition('bottom', 'wall')
algorithm.set_boundary_condition('left', 'wall')
algorithm.set_boundary_condition('right', 'wall')

# Create results directory
results_dir = os.path.join(os.path.dirname(__file__), 'results_nonuniform_mesh')
os.makedirs(results_dir, exist_ok=True)

# 7. Solve the problem
print("Starting simulation with non-uniform mesh...")
result = algorithm.solve(
    max_iterations=max_iterations,
    tolerance=tolerance,
    save_profile=True,
    profile_dir=results_dir,
    track_infinity_norm=True,
    infinity_norm_interval=10,
    #use_l2_norm=True  
)

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
    title=f'Cavity Flow Results (Re={reynolds}, Non-Uniform Mesh)',
    filename=os.path.join(results_dir, f'cavity_Re{reynolds}_nonuniform_mesh_results.pdf'),
    show=True
)

# 11. Visualize final residuals
plot_final_residuals(
    algorithm._final_u_residual_field, 
    algorithm._final_v_residual_field, 
    algorithm._final_p_residual_field,
    mesh,
    title=f'Final Algebraic Residual Fields (Re={reynolds}, Non-Uniform Mesh)',
    filename=os.path.join(results_dir, f'final_algebraic_residual_fields_Re{reynolds}_nonuniform_mesh.pdf'),
    show=False,
    u_rel_norms=result.get_history('u_rel_norm'),
    v_rel_norms=result.get_history('v_rel_norm'),
    p_rel_norms=result.get_history('p_rel_norm'),
    history_filename=os.path.join(results_dir, f'unrelaxed_rel_residual_history_Re{reynolds}_nonuniform_mesh.pdf')
)
