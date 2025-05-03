#!/usr/bin/env pvpython
"""
Mesh Visualization Script for NaviFlow-Collocated

This script uses ParaView's Python interface to load and visualize .vtk files.
It can visualize individual meshes or entire experiment directories.

Usage:
    pvpython visualize_meshes.py [path]

The path can be:
1. A specific .vtk file
2. A directory containing .vtk files
3. The experiments directory (default) to process all experiment meshes
"""

import os
import sys
import glob
import argparse
from paraview.simple import *

def visualize_mesh(vtk_filename):
    """Load and visualize a mesh file, saving a screenshot."""
    base_name = os.path.splitext(os.path.basename(vtk_filename))[0]
    output_dir = os.path.dirname(vtk_filename)
    png_filename = os.path.join(output_dir, f"{base_name}.png")

    print(f"Visualizing: {os.path.basename(vtk_filename)}")

    try:
        active = GetActiveSource()
        if active:
            Delete(active)
    except:
        pass

    try:
        # Load the mesh (auto-detect reader)
        reader = OpenDataFile(vtk_filename)

        renderView = GetActiveViewOrCreate('RenderView')
        display = Show(reader, renderView)

        display.Representation = 'Surface With Edges'
        display.EdgeColor = [0.0, 0.0, 0.0]

        # Reset and adjust view
        renderView.ResetCamera()
        renderView.OrientationAxesVisibility = 1
        renderView.Background = [1.0, 1.0, 1.0]
        renderView.CameraParallelProjection = 1

        Render()

        SaveScreenshot(png_filename, renderView, ImageResolution=[1200, 800])
        print(f"Screenshot saved to {png_filename}")

    except Exception as e:
        print(f"Error processing {vtk_filename}: {e}")
    finally:
        if 'reader' in locals():
            Delete(reader)
            del reader
        if 'display' in locals():
            Delete(display)
            del display
        if 'renderView' in locals():
            Delete(renderView)
            del renderView

def visualize_experiment_dir(experiment_dir):
    """Visualize all mesh types in an experiment directory."""
    experiment_name = os.path.basename(experiment_dir)
    print(f"\nProcessing experiment: {experiment_name}")

    mesh_types = ["structuredUniform", "structuredRefined", "unstructured"]

    for mesh_type in mesh_types:
        mesh_dir = os.path.join(experiment_dir, mesh_type)
        if not os.path.isdir(mesh_dir):
            continue

        mesh_files = glob.glob(os.path.join(mesh_dir, "*.vtk"))
        if mesh_files:
            print(f"  Processing {mesh_type} meshes...")
            for mesh_file in mesh_files:
                visualize_mesh(mesh_file)

def process_path(path):
    """Process a file, experiment dir, or mesh directory."""
    if os.path.isfile(path) and path.endswith('.vtk'):
        visualize_mesh(path)
    elif os.path.isdir(path):
        if any(os.path.isdir(os.path.join(path, mesh_type)) 
               for mesh_type in ["structuredUniform", "structuredRefined", "unstructured"]):
            visualize_experiment_dir(path)
        elif os.path.basename(path) == "experiments":
            print(f"Processing all experiments in: {path}")
            experiment_dirs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)]
            for exp_dir in sorted(experiment_dirs):
                visualize_experiment_dir(exp_dir)
        else:
            mesh_files = glob.glob(os.path.join(path, "*.vtk"))
            if mesh_files:
                print(f"Processing {len(mesh_files)} mesh files in directory: {path}")
                for mesh_file in sorted(mesh_files):
                    visualize_mesh(mesh_file)
            else:
                print(f"No .vtk files found in {path}")
    else:
        print(f"Error: Path not found or not a .vtk file: {path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize CFD meshes with ParaView")
    parser.add_argument("path", nargs="?", help="Path to .vtk file or directory")
    args = parser.parse_args()

    if args.path:
        target_path = args.path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(script_dir, "experiments")

    Connect()
    process_path(target_path)
    print("\nVisualization complete")

if __name__ == "__main__":
    main()