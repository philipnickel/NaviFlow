#!/usr/bin/env python3
"""
Academic Mesh Visualization Script for 2D CFD Grids (VTU only)

Generates high-quality vector PDF visualizations from .vtu files.
Designed for clean academic figures.

Usage:
    run_with_paraview meshing/visualize_meshes.py 
"""

import os
import sys
import glob
import argparse
from paraview.simple import *

# Style configuration
STYLE_CONFIG = {
    "background": [1, 1, 1],
    "edge_color": [0.1, 0.1, 0.1],
    "boundary_color": [0.8, 0.1, 0.1],
    "resolution": [1600, 1200],
    "font_size": 16,
}

# Use names directly from the MSH file's PhysicalNames section
BOUNDARY_LABELS = {
    1: "bottom_boundary",
    2: "right_boundary",
    3: "top_boundary",
    4: "left_boundary",
    5: "obstacle_boundary",  # Added obstacle boundary
    # 10: "fluid_domain" # Domain is usually not labeled as a boundary
}

BOUNDARY_COLORS = {
    1: [0.0, 0.0, 0.8],
    2: [0.0, 0.8, 0.0],
    3: [0.8, 0.0, 0.0],
    4: [0.8, 0.8, 0.0],
    5: [0.8, 0.0, 0.8],
}

# Updated professional colors (e.g., Tableau10-like)
PROFESSIONAL_COLORS = {
    1: [0.121, 0.466, 0.705], # Blue (Bottom)
    2: [1.000, 0.498, 0.055], # Orange (Right - Now unique)
    3: [0.172, 0.627, 0.172], # Green (Top)
    4: [0.839, 0.152, 0.156], # Red (Left)
    5: [0.580, 0.403, 0.741], # Purple (for obstacle)
}

# Specific Cool-to-Warm Approximations
def hex_to_rgb(hex_color): # Helper function
    hex_color = hex_color.lstrip('#')
    hlen = len(hex_color)
    return tuple(int(hex_color[i:i + hlen // 3], 16) / 255.0 for i in range(0, hlen, hlen // 3))

COOL_TO_WARM_MANUAL = {
    1: hex_to_rgb("#3b4cc0"), # Coolest Blue
    2: hex_to_rgb("#8db0fe"), # Lighter Blue
    3: hex_to_rgb("#f4987a"), # Lighter Red/Orange
    4: hex_to_rgb("#b40426"), # Warmest Red
    5: hex_to_rgb("#762a83"), # Purple (for obstacle)
}

def visualize_mesh(file_path):
    print(f"\nVisualizing mesh file: {file_path}")
    base = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)
    output_pdf = os.path.join(output_dir, f"{base}.pdf")

    reader = XMLUnstructuredGridReader(FileName=[file_path])
    reader.UpdatePipeline()

    view = GetActiveViewOrCreate('RenderView')
    view.Background = STYLE_CONFIG["background"]
    view.OrientationAxesVisibility = 0

    display = Show(reader, view)
    display.Representation = 'Wireframe'
    display.ColorArrayName = ['POINTS', '']
    display.AmbientColor = STYLE_CONFIG["edge_color"]
    display.DiffuseColor = STYLE_CONFIG["edge_color"]
    display.LineWidth = 2.0

    scalar_fields = []
    try:
        cell_data_info = reader.GetCellDataInformation()
        for i in range(cell_data_info.GetNumberOfArrays()):
            name = cell_data_info.GetArray(i).GetName()
            scalar_fields.append(('CELLS', name))
        print(f"  Available cell arrays: {[name for _, name in scalar_fields]}")
    except Exception as e:
        print(f"  Warning: Failed to get cell array names. {e}")

    preferred_names = ["physicalgroup", "physical", "entity", "boundary"]
    color_field = next(
        ((ftype, name) for ftype, name in scalar_fields
         if any(key in name.lower() for key in preferred_names)),
        None
    )

    if color_field:
        print(f"  Using field '{color_field[1]}' for boundary identification and legend.")
        lut = GetColorTransferFunction(color_field[1])
        # lut.ApplyPreset('Cool to Warm Extended', True) # REMOVED

        # Manually set LUT colors from COOL_TO_WARM_MANUAL
        rgb_points = []
        # Use only the boundary IDs present in the specific map
        sorted_bids = sorted([bid for bid in COOL_TO_WARM_MANUAL.keys() if bid in BOUNDARY_LABELS])
        for bid in sorted_bids:
            color = COOL_TO_WARM_MANUAL[bid]
            rgb_points.extend([float(bid), color[0], color[1], color[2]])
        if rgb_points:
            lut.RGBPoints = rgb_points

        lut.InterpretValuesAsCategories = 1
        # lut.RescaleTransferFunction(1.0, 4.0) # REMOVED - Not needed with manual points

        # Setup annotations for legend
        annotations = []
        for bid, label in BOUNDARY_LABELS.items():
            annotations.extend([str(bid), label])
        if annotations:
            lut.Annotations = annotations

        # Show scalar bar (legend)
        scalar_bar = GetScalarBar(lut, view)
        scalar_bar.Visibility = 1
        scalar_bar.Title = "Boundary"
        scalar_bar.ComponentTitle = ''
        scalar_bar.LabelFontSize = STYLE_CONFIG["font_size"]
        scalar_bar.TitleFontSize = 18
        scalar_bar.Orientation = 'Horizontal'
        scalar_bar.WindowLocation = 'Upper Right Corner'
        scalar_bar.DrawAnnotations = 1

    else:
        print("  No suitable scalar field found for legend.")

    if color_field:
        for bid, label in BOUNDARY_LABELS.items():
            filt = Threshold(Input=reader)
            filt.LowerThreshold = bid - 0.5
            filt.UpperThreshold = bid + 0.5
            filt.ThresholdMethod = 'Between'
            if hasattr(filt, 'SelectInputScalars'):
                filt.SelectInputScalars = color_field
            elif hasattr(filt, 'Scalars'):
                filt.Scalars = color_field
            else:
                print(f"Error: Could not set scalars for Threshold filter (bid={bid}). Skipping boundary.")
                continue

            surf = ExtractSurface(Input=filt)

            # Show boundary surface, colored via LUT SCALAR MAPPING
            d = Show(surf, view)
            d.Representation = 'Surface'
            d.ColorArrayName = color_field
            d.LookupTable = lut # Use the manually configured LUT
            d.MapScalars = 1

    # Add cell count text (count all triangle cells from the mesh)
    try:
        # Get the total cell count directly from the reader
        reader.UpdatePipeline()
        num_cells = reader.GetDataInformation().GetNumberOfCells()
        cell_text = Text()
        cell_text.Text = f"Cells: {num_cells:,}"
        text_display = Show(cell_text, view)
        text_display.Color = [0.1, 0.1, 0.1]
        text_display.FontSize = STYLE_CONFIG["font_size"]
        text_display.WindowLocation = 'Upper Center'
    except Exception as e:
        print(f"  Warning: Could not add cell count text. {e}")

    # Auto-fit camera based on bounds
    try:
        print("  Auto-fitting camera...")
        view.ResetCamera(False)
        camera = GetActiveCamera()
        bounds = reader.GetDataInformation().GetBounds()
        pad_factor = 0.15 # Increase padding further (15%)
        x_center = (bounds[0] + bounds[1]) / 2.0
        y_center = (bounds[2] + bounds[3]) / 2.0
        x_range = (bounds[1] - bounds[0])
        y_range = (bounds[3] - bounds[2])
        max_range = max(x_range, y_range)
        # Set focal point and position for 2D view
        camera.SetFocalPoint(x_center, y_center, 0)
        camera.SetPosition(x_center, y_center, 1) # Z distance doesn't matter much for parallel
        # Adjust parallel scale based on bounds and padding
        view.CameraParallelProjection = 1 # Ensure parallel projection
        view.CameraParallelScale = (max_range / 2.0) * (1 + pad_factor)
        print(f"    Bounds: {bounds}")
        print(f"    Parallel Scale set to: {view.CameraParallelScale}")
    except Exception as e:
        print(f"  Warning: Auto-fitting camera failed. Using default ResetCamera(). {e}")
        view.ResetCamera() # Fallback

    view.ViewSize = STYLE_CONFIG["resolution"]

    try:
        ExportView(output_pdf, view=view)
        print(f"  Saved visualization to: {output_pdf}")
    except Exception as e:
        print(f"  ❌ Error exporting PDF: {e}")

    try:
        Delete(GetActiveView())
        for src in list(GetSources().values()):
            Delete(src)
    except Exception as e:
        print(f"  Cleanup error: {e}")


def process_path(path):
    if os.path.isfile(path) and path.endswith('.vtu'):
        visualize_mesh(path)
    elif os.path.isdir(path):
        mesh_files = []
        for ext in ['*.vtu']:
            mesh_files.extend(glob.glob(os.path.join(path, ext)))
        for subdir in ['structuredUniform', 'structuredRefined', 'unstructured']:
            mesh_files.extend(glob.glob(os.path.join(path, subdir, '*.vtu')))
        for exp in glob.glob(os.path.join(path, '*')):
            if os.path.isdir(exp):
                for subdir in ['structuredUniform', 'structuredRefined', 'unstructured']:
                    mesh_files.extend(glob.glob(os.path.join(exp, subdir, '*.vtu')))
        if mesh_files:
            print(f"Found {len(mesh_files)} mesh files to visualize")
            for f in mesh_files:
                visualize_mesh(f)
        else:
            print(f"No .vtu files found in {path}")
    else:
        print(f"Invalid path: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CFD mesh files (.vtu only)")
    parser.add_argument('path', nargs='?', default='meshing/experiments',
                        help='Path to a .vtu file, a directory of .vtu files, or an experiment folder')
    args = parser.parse_args()
    process_path(args.path)