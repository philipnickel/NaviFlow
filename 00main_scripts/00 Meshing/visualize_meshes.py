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

from naviflow_staggered.preprocessing.mesh import (
    StructuredUniform,
    StructuredNonUniform,
    UnstructuredUniform,
    UnstructuredRefined
)

# --- Configuration ---
# Domain dimensions (unit square)
XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0

# Mesh parameters
N_POINTS = 5  # Number of points for structured grid 
MESH_SIZE_CENTER_UNIFORM = 0.032 # Target for uniform unstructured 
# Refined unstructured parameters
MESH_SIZE_WALLS = 0.02  # Moderate refinement near stationary walls/corners
MESH_SIZE_LID = 0.02    # Strong refinement near moving lid/corners
MESH_SIZE_CENTER_REFINED = 0.1  # Coarse size in the center 
REFINEMENT_FACTOR = 5.0 # For structured non-uniform grid bias

# --- Mesh Generation ---

# 1. Structured Uniform Mesh
print("Generating structured uniform mesh...")
mesh_struct_uni = StructuredUniform(
    nx=N_POINTS,
    ny=N_POINTS,
    xmin=XMIN,
    xmax=XMAX,
    ymin=YMIN,
    ymax=YMAX
)
n_cells_struct_uni = mesh_struct_uni.n_cells
print(f"  Generated {n_cells_struct_uni} cells.")

# 2. Structured Non-Uniform Mesh (Tanh clustering)
print("Generating structured non-uniform mesh (Tanh clustering)...")
mesh_struct_nonuni = StructuredNonUniform(
    nx=N_POINTS,
    ny=N_POINTS,
    xmin=XMIN,
    xmax=XMAX,
    ymin=YMIN,
    ymax=YMAX,
    clustering_factor=REFINEMENT_FACTOR
)
n_cells_struct_nonuni = mesh_struct_nonuni.n_cells
print(f"  Generated {n_cells_struct_nonuni} cells.")

# 3. Unstructured Uniform Mesh
print("Generating unstructured uniform mesh...")
mesh_unstruct_uni = UnstructuredUniform(
    mesh_size=MESH_SIZE_CENTER_UNIFORM,
    xmin=XMIN,
    xmax=XMAX,
    ymin=YMIN,
    ymax=YMAX
)
n_cells_unstruct_uni = mesh_unstruct_uni.n_cells
print(f"  Generated {n_cells_unstruct_uni} cells.")

# 4. Unstructured Refined Mesh (Lid-driven cavity)
print("Generating unstructured refined mesh (Lid-driven cavity)...")
mesh_unstruct_ref = UnstructuredRefined(
    mesh_size_walls=MESH_SIZE_WALLS,
    mesh_size_lid=MESH_SIZE_LID, 
    mesh_size_center=MESH_SIZE_CENTER_REFINED,
    xmin=XMIN,
    xmax=XMAX,
    ymin=YMIN,
    ymax=YMAX
)
n_cells_unstruct_ref = mesh_unstruct_ref.n_cells
print(f"  Generated {n_cells_unstruct_ref} cells.")

# --- Plotting ---
print("Plotting meshes...")
fig, axs = plt.subplots(2, 2, figsize=(10, 10)) # Adjusted figsize for scienceplots

# Use the direct plot methods from the mesh objects
mesh_struct_uni.plot(axs[0, 0], title=f"Structured Uniform ({n_cells_struct_uni} cells)")
mesh_struct_nonuni.plot(axs[0, 1], title=f"Structured Non-Uniform ({n_cells_struct_nonuni} cells)")
mesh_unstruct_uni.plot(axs[1, 0], title=f"Unstructured Uniform ({n_cells_unstruct_uni} cells)")
mesh_unstruct_ref.plot(axs[1, 1], title=f"Unstructured Refined ({n_cells_unstruct_ref} cells)")

# Set axis limits for all plots
for ax in axs.flat:
    ax.set_xlim(XMIN - 0.05, XMAX + 0.05)
    ax.set_ylim(YMIN - 0.05, YMAX + 0.05)

plt.tight_layout()
output_filename = os.path.join(script_dir, "mesh_comparison.pdf") # Use script directory
plt.savefig(output_filename, dpi=150, format='pdf', bbox_inches='tight') # Specify format pdf
print(f"Mesh visualization saved to: {output_filename}")

# Example of using the direct savePlot method
individual_dir = os.path.join(script_dir, "individual_meshes")
os.makedirs(individual_dir, exist_ok=True)

# Save individual meshes
mesh_struct_uni.savePlot(
    os.path.join(individual_dir, "structured_uniform.pdf"), 
    title=f"Structured Uniform ({n_cells_struct_uni} cells)"
)
mesh_struct_nonuni.savePlot(
    os.path.join(individual_dir, "structured_nonuniform.pdf"), 
    title=f"Structured Non-Uniform ({n_cells_struct_nonuni} cells)"
)
mesh_unstruct_uni.savePlot(
    os.path.join(individual_dir, "unstructured_uniform.pdf"), 
    title=f"Unstructured Uniform ({n_cells_unstruct_uni} cells)"
)
mesh_unstruct_ref.savePlot(
    os.path.join(individual_dir, "unstructured_refined.pdf"), 
    title=f"Unstructured Refined ({n_cells_unstruct_ref} cells)"
)
print("Saved individual mesh plots to:", individual_dir)

plt.show()

print("Script finished.") 