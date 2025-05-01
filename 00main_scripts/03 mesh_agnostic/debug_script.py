"""
Debug script for testing mesh creation and basic operations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from naviflow_staggered.preprocessing.mesh.mesh import UniformStructuredMesh, NonUniformStructuredMesh
from naviflow_staggered.preprocessing.mesh_generators import StructuredMeshGenerator

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

def main():
    # Create a structured uniform mesh
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    resolution = 11  # Small for testing
    
    # Create uniform structured mesh
    mesh = StructuredMeshGenerator.generate_uniform(
        xmin, xmax, ymin, ymax, resolution, resolution
    )
    
    # Print mesh information
    print(f"Created mesh with {mesh.n_cells} cells and {mesh.n_faces} faces")
    
    # Get mesh topology
    owner_cells, neighbor_cells = mesh.get_owner_neighbor()
    face_areas = mesh.get_face_areas()
    face_normals = mesh.get_face_normals()
    
    # Print first few entries
    print("\nFirst 10 faces:")
    print("Face\tOwner\tNeighbor\tArea\tNormal")
    for i in range(min(10, mesh.n_faces)):
        print(f"{i}\t{owner_cells[i]}\t{neighbor_cells[i]}\t{face_areas[i]:.5f}\t{face_normals[i]}")
    
    # Test visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    mesh.plot(ax, title="Test Mesh")
    plt.savefig(os.path.join(script_dir, "debug_mesh.png"))
    print("\nSaved mesh visualization to debug_mesh.png")

if __name__ == "__main__":
    main() 