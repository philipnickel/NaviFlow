import numpy as np
import matplotlib.pyplot as plt
import pygmsh
import meshio
import os
import gmsh
import scienceplots # Import scienceplots

# Apply scienceplots style
plt.style.use(['science'])

# Ensure the main package can be imported
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from naviflow_oo.preprocessing.mesh_generators import StructuredMeshGenerator, UnstructuredMeshGenerator
from naviflow_oo.preprocessing.mesh import Mesh  # Import base class for type hints

# --- Configuration ---
# Domain dimensions (unit square)
XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0

# Mesh parameters
N_POINTS = 50  # Number of points for structured grid -> (21-1)^2 = 400 cells
MESH_SIZE_CENTER_UNIFORM = 0.03 # Target for uniform unstructured (~400 cells?)
# Refined unstructured parameters (target ~400 cells, using fields)
MESH_SIZE_WALLS = 0.02  # Moderate refinement near stationary walls/corners
MESH_SIZE_LID = 0.02    # Strong refinement near moving lid/corners
MESH_SIZE_CENTER = 0.07  # Coarse size in the center (for interpolation/background)
REFINEMENT_FACTOR = 0.2 # For structured non-uniform grid bias

# --- Function for Tanh Clustered Nodes ---
def create_tanh_clustered_nodes(min_val, max_val, n_points, alpha=3.0):
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

# --- Mesh Generation ---

# 1. Structured Uniform Mesh
print("Generating structured uniform mesh...")
mesh_struct_uni = StructuredMeshGenerator.generate_uniform(
    XMIN, XMAX, YMIN, YMAX, N_POINTS, N_POINTS
)
n_cells_struct_uni = mesh_struct_uni.n_cells
print(f"  Generated {n_cells_struct_uni} cells.")

# 2. Structured Non-Uniform Mesh (Boundary Refinement using Tanh)
print("Generating structured non-uniform mesh (Tanh clustering)...")
x_nodes_nonuni = create_tanh_clustered_nodes(XMIN, XMAX, N_POINTS) # Use new function
y_nodes_nonuni = create_tanh_clustered_nodes(YMIN, YMAX, N_POINTS) # Use new function
mesh_struct_nonuni = StructuredMeshGenerator.generate_nonuniform(
    x_nodes_nonuni, y_nodes_nonuni
)
n_cells_struct_nonuni = mesh_struct_nonuni.n_cells
print(f"  Generated {n_cells_struct_nonuni} cells.")


# 3. Unstructured Uniform Mesh (using pygmsh + meshio)
print("Generating unstructured uniform mesh...")
with pygmsh.geo.Geometry() as geom:
    # Define the square domain points (mesh_size here defines the uniform size)
    p1 = geom.add_point([XMIN, YMIN, 0], mesh_size=MESH_SIZE_CENTER_UNIFORM)
    p2 = geom.add_point([XMAX, YMIN, 0], mesh_size=MESH_SIZE_CENTER_UNIFORM)
    p3 = geom.add_point([XMAX, YMAX, 0], mesh_size=MESH_SIZE_CENTER_UNIFORM)
    p4 = geom.add_point([XMIN, YMAX, 0], mesh_size=MESH_SIZE_CENTER_UNIFORM)
    l1 = geom.add_line(p1, p2)
    l2 = geom.add_line(p2, p3)
    l3 = geom.add_line(p3, p4)
    l4 = geom.add_line(p4, p1)
    ll = geom.add_curve_loop([l1, l2, l3, l4])
    surface = geom.add_plane_surface(ll)
    geom.synchronize() # Synchronize before setting options
    # Ensure point sizes are interpolated; disable curvature
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # Generate mesh
    msh = geom.generate_mesh(dim=2)

# Convert meshio mesh to our format
mesh_unstruct_uni = UnstructuredMeshGenerator.from_meshio(msh)
n_cells_unstruct_uni = mesh_unstruct_uni.n_cells
print(f"  Generated {n_cells_unstruct_uni} cells.")


# 4. Unstructured Non-Uniform Mesh (Lid-Driven Cavity - Gmsh Fields)
print("Generating unstructured non-uniform mesh (Lid Cavity - Gmsh Fields)...")
with pygmsh.geo.Geometry() as geom:
    # Define corners (mesh_size is less critical now, acts as fallback/initial)
    p1 = geom.add_point([XMIN, YMIN, 0], mesh_size=MESH_SIZE_WALLS) # Bottom-left
    p2 = geom.add_point([XMAX, YMIN, 0], mesh_size=MESH_SIZE_WALLS) # Bottom-right
    p3 = geom.add_point([XMAX, YMAX, 0], mesh_size=MESH_SIZE_LID)   # Top-right (fine)
    p4 = geom.add_point([XMIN, YMAX, 0], mesh_size=MESH_SIZE_LID)   # Top-left (fine)

    # Define boundary lines using only boundary points
    l_bottom = geom.add_line(p1, p2)
    l_right = geom.add_line(p2, p3)
    l_top = geom.add_line(p3, p4)
    l_left = geom.add_line(p4, p1)
    ll = geom.add_curve_loop([l_bottom, l_right, l_top, l_left])
    surface = geom.add_plane_surface(ll)

    # Synchronize before using GMSH API directly for fields
    geom.synchronize()
    gmsh_api = gmsh.model.mesh # Alias for brevity

    # --- Define Mesh Size Fields --- 
    # Field 1: Distance to top lid (line 3) and top corners (points 3, 4)
    field_dist_lid = gmsh_api.field.add("Distance")
    gmsh_api.field.setNumbers(field_dist_lid, "PointsList", [p3._id, p4._id])
    gmsh_api.field.setNumbers(field_dist_lid, "CurvesList", [l_top._id])

    # Field 2: Threshold based on distance to lid/corners (fine refinement)
    field_thresh_lid = gmsh_api.field.add("Threshold")
    gmsh_api.field.setNumber(field_thresh_lid, "InField", field_dist_lid)
    gmsh_api.field.setNumber(field_thresh_lid, "SizeMin", MESH_SIZE_LID)
    gmsh_api.field.setNumber(field_thresh_lid, "SizeMax", MESH_SIZE_CENTER)
    gmsh_api.field.setNumber(field_thresh_lid, "DistMin", 0.05) # Start refining very close
    gmsh_api.field.setNumber(field_thresh_lid, "DistMax", 0.25) # Reach coarse size further away

    # Field 3: Distance to other walls (lines 1, 2, 4)
    field_dist_walls = gmsh_api.field.add("Distance")
    gmsh_api.field.setNumbers(field_dist_walls, "CurvesList", [l_bottom._id, l_right._id, l_left._id])

    # Field 4: Threshold based on distance to other walls (moderate refinement)
    field_thresh_walls = gmsh_api.field.add("Threshold")
    gmsh_api.field.setNumber(field_thresh_walls, "InField", field_dist_walls)
    gmsh_api.field.setNumber(field_thresh_walls, "SizeMin", MESH_SIZE_WALLS)
    gmsh_api.field.setNumber(field_thresh_walls, "SizeMax", MESH_SIZE_CENTER)
    gmsh_api.field.setNumber(field_thresh_walls, "DistMin", 0.05)
    gmsh_api.field.setNumber(field_thresh_walls, "DistMax", 0.4)
    
    # Field 5: Minimum of the two threshold fields
    field_min = gmsh_api.field.add("Min")
    gmsh_api.field.setNumbers(field_min, "FieldsList", [field_thresh_lid, field_thresh_walls])

    # Set the minimum field as the background field
    gmsh_api.field.setAsBackgroundMesh(field_min)

    # Important: Prevent mesh size from points/curvature interfering
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Generate mesh using the background field for sizing
    msh_refined = geom.generate_mesh(dim=2)

# Convert meshio mesh to our format
mesh_unstruct_nonuni = UnstructuredMeshGenerator.from_meshio(msh_refined)
n_cells_unstruct_nonuni = mesh_unstruct_nonuni.n_cells
print(f"  Generated {n_cells_unstruct_nonuni} cells.")


# --- Plotting ---
print("Plotting meshes...")
fig, axs = plt.subplots(2, 2, figsize=(10, 10)) # Adjusted figsize for scienceplots

# Use the plot methods directly from the mesh objects
mesh_struct_uni.plot(axs[0, 0], title=f"Structured Uniform ({n_cells_struct_uni} cells)")
mesh_struct_nonuni.plot(axs[0, 1], title=f"Structured Non-Uniform ({n_cells_struct_nonuni} cells)")
mesh_unstruct_uni.plot(axs[1, 0], title=f"Unstructured Uniform ({n_cells_unstruct_uni} cells)")
mesh_unstruct_nonuni.plot(axs[1, 1], title=f"Unstructured Cavity Refined ({n_cells_unstruct_nonuni} cells)")

# Set axis limits for all plots
for ax in axs.flat:
    ax.set_xlim(XMIN - 0.05, XMAX + 0.05)
    ax.set_ylim(YMIN - 0.05, YMAX + 0.05)

plt.tight_layout()
output_filename = os.path.join(script_dir, "mesh_comparison.pdf") # Use script directory
plt.savefig(output_filename, dpi=150, format='pdf', bbox_inches='tight') # Specify format pdf
print(f"Mesh visualization saved to: {output_filename}")

plt.show()

print("Script finished.") 