"""
Lid-driven cavity flow simulation using mesh-agnostic solvers.
This script demonstrates using the same solver code with different mesh types.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from naviflow_staggered.preprocessing.mesh.mesh import UniformStructuredMesh, NonUniformStructuredMesh, UnstructuredMesh
from naviflow_staggered.preprocessing.mesh_generators import StructuredMeshGenerator, UnstructuredMeshGenerator
from naviflow_staggered.constructor.properties.fluid import FluidProperties
from naviflow_staggered.solver.mesh_agnostic.mesh_simple import MeshAgnosticSimpleSolver
from naviflow_staggered.solver.mesh_agnostic.mesh_direct_pressure import MeshAgnosticDirectPressureSolver
from naviflow_staggered.solver.mesh_agnostic.mesh_amg_solver import MeshAgnosticAMGSolver
from naviflow_staggered.solver.velocity_solver.standard import StandardVelocityUpdater

# Configuration
# Choose mesh type: 'structured_uniform', 'structured_nonuniform', or 'unstructured'
MESH_TYPE = 'unstructured'
RESOLUTION = 21  # Approximate resolution for unstructured grid
REYNOLDS = 100   # Reynolds number
MAX_ITERATIONS = 100
TOLERANCE = 1e-6
ALPHA_P = 0.3  # Pressure relaxation
ALPHA_U = 0.7  # Velocity relaxation

def create_tanh_clustered_nodes(min_val, max_val, n_points, alpha=2.0):
    """Creates non-uniform spacing clustered towards both ends using tanh."""
    if n_points <= 1:
        return np.array([min_val]) if n_points == 1 else np.array([])
    
    # Create uniform points in [0, 1]
    x_uniform = np.linspace(0.0, 1.0, n_points)
    
    # Apply tanh stretching function
    # y = 0.5 * (1 + tanh(alpha * (2*x - 1)) / tanh(alpha))
    tanh_alpha = np.tanh(alpha)
    if tanh_alpha == 0: # Avoid division by zero if alpha is extremely small
        y_stretched = x_uniform
    else:
        y_stretched = 0.5 * (1.0 + np.tanh(alpha * (2.0 * x_uniform - 1.0)) / tanh_alpha)
        
    # Scale and shift to the desired range [min_val, max_val]
    nodes = min_val + (max_val - min_val) * y_stretched
    return nodes

def create_mesh(mesh_type, resolution):
    """
    Create a mesh of the specified type and resolution.
    
    Parameters:
    -----------
    mesh_type : str
        Type of mesh to create
    resolution : int
        Resolution of the mesh
        
    Returns:
    --------
    mesh : Mesh
        The created mesh
    """
    # Domain dimensions (unit square)
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    
    if mesh_type == 'structured_uniform':
        # Create uniform structured mesh
        mesh = StructuredMeshGenerator.generate_uniform(
            xmin, xmax, ymin, ymax, resolution, resolution
        )
        print(f"Created uniform structured mesh with {mesh.n_cells} cells")
        
    elif mesh_type == 'structured_nonuniform':
        # Create non-uniform structured mesh with clustering near boundaries
        x_nodes = create_tanh_clustered_nodes(xmin, xmax, resolution, alpha=2.0)
        y_nodes = create_tanh_clustered_nodes(ymin, ymax, resolution, alpha=2.0)
        mesh = StructuredMeshGenerator.generate_nonuniform(
            x_nodes, y_nodes
        )
        print(f"Created non-uniform structured mesh with {mesh.n_cells} cells")
        
    elif mesh_type == 'unstructured':
        # Create unstructured mesh using gmsh
        import pygmsh
        import gmsh
        
        mesh_size = 1.0 / (resolution - 1)  # Approximate cell size
        
        with pygmsh.geo.Geometry() as geom:
            # Define the square domain
            p1 = geom.add_point([xmin, ymin, 0], mesh_size=mesh_size)
            p2 = geom.add_point([xmax, ymin, 0], mesh_size=mesh_size)
            p3 = geom.add_point([xmax, ymax, 0], mesh_size=mesh_size)
            p4 = geom.add_point([xmin, ymax, 0], mesh_size=mesh_size)
            
            # Define boundary
            l1 = geom.add_line(p1, p2)
            l2 = geom.add_line(p2, p3)
            l3 = geom.add_line(p3, p4)
            l4 = geom.add_line(p4, p1)
            
            # Create surface
            ll = geom.add_curve_loop([l1, l2, l3, l4])
            surface = geom.add_plane_surface(ll)
            
            # Generate mesh
            msh = geom.generate_mesh(dim=2)
        
        # Convert to our mesh format
        mesh = UnstructuredMeshGenerator.from_meshio(msh)
        print(f"Created unstructured mesh with {mesh.n_cells} cells")
    
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")
    
    return mesh

# Simple wrapper classes for vector and scalar fields
class SimpleVectorField:
    def __init__(self, u, v):
        self.u = u
        self.v = v
    
    def get_u_at_cells(self):
        return self.u
    
    def get_v_at_cells(self):
        return self.v

class SimpleScalarField:
    def __init__(self, values):
        self.values = values
    
    def get_values_at_cells(self):
        return self.values

def main():
    # Start timing
    start_time = time.time()
    
    # 1. Create mesh
    mesh = create_mesh(MESH_TYPE, RESOLUTION)
    
    # 2. Define fluid properties
    fluid = FluidProperties(
        density=1.0,
        reynolds_number=REYNOLDS,
        characteristic_velocity=1.0
    )
    print(f"Reynolds number: {fluid.get_reynolds_number()}")
    print(f"Calculated viscosity: {fluid.get_viscosity()}")
    
    # 3. Create solvers
    pressure_solver = MeshAgnosticDirectPressureSolver()
    momentum_solver = MeshAgnosticAMGSolver(discretization_scheme='power_law', tolerance=1e-6, max_iterations=1000)
    
    # 4. Create algorithm
    algorithm = MeshAgnosticSimpleSolver(
        mesh=mesh,
        fluid=fluid,
        pressure_solver=pressure_solver,
        momentum_solver=momentum_solver,
        velocity_updater=None,  # Use the default _update_velocity method
        alpha_p=ALPHA_P,
        alpha_u=ALPHA_U
    )
    
    # 5. Set boundary conditions
    # For a structured mesh, we can use the standard approach
    algorithm.set_boundary_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
    algorithm.set_boundary_condition('bottom', 'wall')
    algorithm.set_boundary_condition('left', 'wall')
    algorithm.set_boundary_condition('right', 'wall')
    
    # 6. Initialize solution fields manually
    n_cells = mesh.n_cells
    n_faces = mesh.n_faces
    
    # For u and v, create fields that are compatible with the solver's expectations
    # The solver expects a field for each velocity component
    u = np.zeros(n_cells)  # Cell-centered u velocity
    v = np.zeros(n_cells)  # Cell-centered v velocity
    p = np.zeros(n_cells)  # Cell-centered pressure
    
    # Set the lid velocity for cells at the top boundary
    # This is a simplified approach - proper boundary conditions are applied by the solver
    cell_centers = mesh.get_cell_centers()
    y_coords = cell_centers[:, 1]
    lid_height = 1.0
    for i in range(n_cells):
        if y_coords[i] > 0.95:  # Approximate for cells near the top
            u[i] = 1.0
    
    # Set the fields in the algorithm
    algorithm.u = u
    algorithm.v = v
    algorithm.p = p
    
    # Create results directory
    results_dir = os.path.join(script_dir, f'results_{MESH_TYPE}_Re{REYNOLDS}')
    os.makedirs(results_dir, exist_ok=True)
    
    # 7. Solve the problem
    print(f"Starting simulation with {MESH_TYPE} mesh, Re = {REYNOLDS}, resolution = {RESOLUTION}...")
    result = algorithm.solve(
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        save_profile=True,
        profile_dir=results_dir
    )
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 8. Print results
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    print(f"Total Iterations = {result.iterations}")
    
    # 9. Visualize the final velocity field
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # For structured mesh, we can reshape the velocities
    if MESH_TYPE.startswith('structured'):
        n = int(np.sqrt(n_cells))  # Assuming square mesh
        u_reshaped = result.u.reshape(n, n)
        v_reshaped = result.v.reshape(n, n)
        
        # Create a grid for plotting
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)
        
        # Plot the velocity field with streamlines
        ax.quiver(X, Y, u_reshaped, v_reshaped, scale=10)
        ax.streamplot(X, Y, u_reshaped, v_reshaped, color='blue', linewidth=1, density=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Velocity Field (Re={REYNOLDS}, {MESH_TYPE}, {RESOLUTION}x{RESOLUTION})')
    else:
        # For unstructured mesh, plot at cell centers
        ax.quiver(cell_centers[:, 0], cell_centers[:, 1], result.u, result.v, scale=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Velocity Field (Re={REYNOLDS}, {MESH_TYPE})')
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, f'velocity_field_{MESH_TYPE}_Re{REYNOLDS}.png'))
    
    # Also create a visualization of the center velocities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    if MESH_TYPE.startswith('structured'):
        # Plot u along vertical centerline
        centerline_x = n // 2
        u_centerline = u_reshaped[:, centerline_x]
        ax1.plot(u_centerline, y, 'r-')
        ax1.set_xlabel('U velocity')
        ax1.set_ylabel('Y coordinate')
        ax1.set_title('U velocity along vertical centerline')
        ax1.grid(True)
        
        # Plot v along horizontal centerline
        centerline_y = n // 2
        v_centerline = v_reshaped[centerline_y, :]
        ax2.plot(x, v_centerline, 'b-')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('V velocity')
        ax2.set_title('V velocity along horizontal centerline')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'centerline_velocities_Re{REYNOLDS}.png'))
    
    print(f"Visualizations saved to {results_dir}")

if __name__ == "__main__":
    main() 