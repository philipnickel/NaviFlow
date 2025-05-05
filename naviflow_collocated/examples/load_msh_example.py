"""
Example showing how to load a .msh file into NaviFlow's MeshData2D format.

This script demonstrates loading a .msh file and performing basic operations with it.
"""

import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project root to the path if running as a script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from naviflow_collocated.mesh import load_msh_file


def main():
    """Load a .msh file and display basic information about the mesh."""

    # Find a .msh file in the meshing/experiments directory
    base_dir = Path(__file__).parent.parent.parent
    mesh_paths = list(base_dir.glob("meshing/experiments/**/*.msh"))

    if not mesh_paths:
        print("No .msh files found in the meshing/experiments directory.")
        print("Please provide a path to a .msh file as an argument.")

        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            mesh_file = sys.argv[1]
        else:
            print("No valid .msh file provided.")
            return
    else:
        mesh_file = str(mesh_paths[0])
        print(f"Found mesh file: {mesh_file}")

    # Load the mesh
    print(f"Loading mesh from {mesh_file}...")
    mesh = load_msh_file(mesh_file)

    # Display basic mesh information
    print("\nMesh loaded successfully!")
    print(f"Number of cells: {len(mesh.cell_volumes)}")
    print(f"Number of faces: {len(mesh.face_areas)}")
    print(f"Number of boundary faces: {len(mesh.boundary_faces)}")
    print(f"Structured mesh: {mesh.is_structured}")
    print(f"Orthogonal mesh: {mesh.is_orthogonal}")
    print(f"Conforming mesh: {mesh.is_conforming}")

    # Calculate mesh quality metrics
    cell_volumes = mesh.cell_volumes
    face_areas = mesh.face_areas

    print("\nMesh quality metrics:")
    print(f"  Min cell volume: {np.min(cell_volumes):.6e}")
    print(f"  Max cell volume: {np.max(cell_volumes):.6e}")
    print(
        f"  Volume ratio (max/min): {np.max(cell_volumes) / np.min(cell_volumes):.2f}"
    )
    print(f"  Min face area: {np.min(face_areas):.6e}")
    print(f"  Max face area: {np.max(face_areas):.6e}")

    # Simple visualization
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot cell centers
        centers = mesh.cell_centers
        ax.scatter(
            centers[:, 0], centers[:, 1], s=2, color="blue", label="Cell Centers"
        )

        # Plot boundary faces
        boundary_faces = mesh.boundary_faces
        face_centers = mesh.face_centers
        bface_centers = face_centers[boundary_faces]
        ax.scatter(
            bface_centers[:, 0],
            bface_centers[:, 1],
            s=4,
            color="red",
            label="Boundary Faces",
        )

        # Add axes labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Mesh Visualization: {Path(mesh_file).name}")
        ax.legend()
        ax.set_aspect("equal")

        # Save the plot
        plot_file = "mesh_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"\nMesh visualization saved to {plot_file}")

        plt.show()
    except Exception as e:
        print(f"Could not create visualization: {e}")

    print("\nMesh loaded and ready for use in NaviFlow simulations.")


if __name__ == "__main__":
    main()
