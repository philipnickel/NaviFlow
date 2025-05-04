"""
Academic Mesh Visualization Script for 2D CFD Grids

Generates high-quality vector PDF visualizations from Gmsh-generated VTK meshes.
Only PDF export is performed (no PNG). Designed for clean academic figures.

Usage:
    pvpython visualize_meshes.py [path]

Arguments:
    path: Optional. Can be a .vtk file, directory of .vtk files, or an experiment folder.

Usage (Philips Mac from naviflow directory): 
run_with_paraview meshing/visualize_meshes.py
"""

import os
import sys
import glob
import argparse
from paraview.simple import *

# Style configuration
STYLE_CONFIG = {
    "background": [1, 1, 1],              # White
    "edge_color": [0.1, 0.1, 0.1],        # Dark gray
    "resolution": [1600, 1200],           # High-res
    "font": "Arial",
    "font_size": 14,
}

def configure_2d_view(view):
    """Set up consistent 2D academic rendering style"""
    view.ViewSize = STYLE_CONFIG["resolution"]
    view.Background = STYLE_CONFIG["background"]
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1  # 2D projection
    view.CameraViewUp = [0, 1, 0]
    view.CameraPosition = [0, 0, 100]  # Zoomed out of XY plane
    view.CameraFocalPoint = [0, 0, 0]
    Render()

def color_by_physical_groups(display, view):
    """Try coloring by 'gmsh:physical' with legend"""
    try:
        ColorBy(display, ('CELLS', 'gmsh:physical'))
        display.SetScalarBarVisibility(view, True)
        print("Colored by gmsh:physical")
    except Exception:
        print("Warning: 'gmsh:physical' not found, using surface coloring.")
        ColorBy(display, None)
        display.DiffuseColor = [0.85, 0.85, 0.85]
def visualize_mesh(vtk_file):
    """Visualize a single .vtk mesh and export as PDF"""
    print(f"Visualizing: {vtk_file}")
    base = os.path.splitext(os.path.basename(vtk_file))[0]
    output_dir = os.path.dirname(vtk_file)
    output_pdf = os.path.join(output_dir, f"{base}.pdf")

    try:
        # Load mesh
        mesh = OpenDataFile(vtk_file)
        view = GetActiveViewOrCreate('RenderView')
        display = Show(mesh, view)

        # Style
        display.Representation = 'Surface With Edges'
        display.EdgeColor = STYLE_CONFIG["edge_color"]
        display.LineWidth = 1.2

        configure_2d_view(view)
        color_by_physical_groups(display, view)

        view.ResetCamera()
        Render()

        # Save as PDF vector output
        ExportView(output_pdf, view)
        print(f"Exported: {os.path.basename(output_pdf)}")

    except Exception as e:
        print(f"Visualization error: {e}")

    finally:
        # Full cleanup: remove mesh, display, view
        try:
            Delete(display)
        except: pass
        try:
            Delete(mesh)
        except: pass
        try:
            Delete(view)
        except: pass
        ResetSession()

def visualize_experiment_dir(experiment_dir):
    print(f"\nProcessing experiment: {os.path.basename(experiment_dir)}")
    mesh_types = ["structuredUniform", "structuredRefined", "unstructured"]
    for mesh_type in mesh_types:
        mesh_dir = os.path.join(experiment_dir, mesh_type)
        if not os.path.isdir(mesh_dir):
            continue
        mesh_files = glob.glob(os.path.join(mesh_dir, "*.vtk"))
        for mesh_file in mesh_files:
            visualize_mesh(mesh_file)

def process_path(path):
    if os.path.isfile(path) and path.endswith('.vtk'):
        visualize_mesh(path)
    elif os.path.isdir(path):
        if any(os.path.isdir(os.path.join(path, t)) for t in ["structuredUniform", "structuredRefined", "unstructured"]):
            visualize_experiment_dir(path)
        elif os.path.basename(path) == "experiments":
            print(f"Processing all experiments in: {path}")
            for d in sorted(glob.glob(os.path.join(path, "*"))):
                if os.path.isdir(d):
                    visualize_experiment_dir(d)
        else:
            vtk_files = glob.glob(os.path.join(path, "*.vtk"))
            for f in sorted(vtk_files):
                visualize_mesh(f)
    else:
        print(f"Error: Invalid path '{path}'")

def main():
    parser = argparse.ArgumentParser(description="Generate PDF mesh visualizations")
    parser.add_argument("path", nargs="?", help="VTK file or directory")
    args = parser.parse_args()
    target = args.path or os.path.join(os.path.dirname(__file__), "experiments")
    Connect()
    process_path(target)
    print("\nVisualization complete")

if __name__ == "__main__":
    main()