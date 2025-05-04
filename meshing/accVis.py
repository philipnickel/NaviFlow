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
    "resolution": [1600, 1200],
    "font_size": 18,
}

# Boundary labels and visual names
BOUNDARY_LABELS = {
    1: "bottom_boundary",
    2: "right_boundary",
    3: "top_boundary",
    4: "left_boundary",
}

# Clean scientific colormap
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    hlen = len(hex_color)
    return tuple(int(hex_color[i:i + hlen // 3], 16) / 255.0 for i in range(0, hlen, hlen // 3))

SCIENTIFIC_COLORS = {
    1: hex_to_rgb("#1f77b4"),  # Blue
    2: hex_to_rgb("#ff7f0e"),  # Orange
    3: hex_to_rgb("#2ca02c"),  # Green
    4: hex_to_rgb("#d62728"),  # Red
}

def visualize_mesh(file_path):
    print(f"\nVisualizing mesh file: {file_path}")
    base = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)
    output_pdf = os.path.join(output_dir, f"{base}.pdf")

    reader = XMLUnstructuredGridReader(FileName=[file_path])
    reader.UpdatePipeline()

    paraview.simple._DisableFirstRenderCameraReset()
    view = GetActiveViewOrCreate('RenderView')
    view.Background = STYLE_CONFIG["background"]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0

    display = Show(reader, view)
    display.Representation = 'Wireframe'
    display.ColorArrayName = ['POINTS', '']
    display.AmbientColor = STYLE_CONFIG["edge_color"]
    display.DiffuseColor = STYLE_CONFIG["edge_color"]
    display.LineWidth = 3.0

    # Find scalar fields
    scalar_fields = []
    try:
        cell_data_info = reader.GetCellDataInformation()
        for i in range(cell_data_info.GetNumberOfArrays()):
            name = cell_data_info.GetArray(i).GetName()
            scalar_fields.append(('CELLS', name))
    except Exception as e:
        print(f"  Warning: Failed to get cell array names. {e}")

    preferred_names = ["physicalgroup", "physical", "entity", "boundary"]
    color_field = next(
        ((ftype, name) for ftype, name in scalar_fields
         if any(key in name.lower() for key in preferred_names)),
        None
    )

    if color_field:
        print(f"  Using field '{color_field[1]}' for boundary coloring.")
        lut = GetColorTransferFunction(color_field[1])

        rgb_points = []
        for bid in sorted(SCIENTIFIC_COLORS):
            color = SCIENTIFIC_COLORS[bid]
            rgb_points.extend([float(bid), *color])
        lut.RGBPoints = rgb_points
        lut.InterpretValuesAsCategories = 1

        annotations = []
        for bid, label in BOUNDARY_LABELS.items():
            annotations.extend([str(bid), label])
        lut.Annotations = annotations

        scalar_bar = GetScalarBar(lut, view)
        scalar_bar.Visibility = 1
        scalar_bar.Title = "Boundary"
        scalar_bar.ComponentTitle = ''
        scalar_bar.LabelFontSize = STYLE_CONFIG["font_size"]
        scalar_bar.TitleFontSize = 18
        scalar_bar.Orientation = 'Horizontal'
        scalar_bar.WindowLocation = 'Lower Right Corner'
        scalar_bar.ScalarBarLength = 0.6
        scalar_bar.ScalarBarThickness = 24
        scalar_bar.Position = [0.3, 0.02]

        for bid in BOUNDARY_LABELS:
            filt = Threshold(Input=reader)
            filt.LowerThreshold = bid - 0.5
            filt.UpperThreshold = bid + 0.5
            filt.ThresholdMethod = 'Between'
            if hasattr(filt, 'SelectInputScalars'):
                filt.SelectInputScalars = color_field
            elif hasattr(filt, 'Scalars'):
                filt.Scalars = color_field
            surf = ExtractSurface(Input=filt)
            d = Show(surf, view)
            d.Representation = 'Surface With Edges'
            d.EdgeColor = [0, 0, 0]
            d.ColorArrayName = color_field
            d.LookupTable = lut
            d.MapScalars = 1
            d.Opacity = 0.6

    # Add cell count
    try:
        fluid_thresh = Threshold(Input=reader)
        fluid_thresh.LowerThreshold = 4.5
        fluid_thresh.UpperThreshold = 5.5
        fluid_thresh.ThresholdMethod = 'Between'
        if hasattr(fluid_thresh, 'SelectInputScalars'):
            fluid_thresh.SelectInputScalars = ('CELLS', 'gmsh:physical')
        elif hasattr(fluid_thresh, 'Scalars'):
            fluid_thresh.Scalars = ('CELLS', 'gmsh:physical')
        fluid_thresh.UpdatePipeline()
        num_cells = fluid_thresh.GetDataInformation().GetNumberOfCells()
        cell_text = Text()
        cell_text.Text = f"Cells: {num_cells:,}"
        text_display = Show(cell_text, view)
        text_display.Color = [0.1, 0.1, 0.1]
        text_display.FontSize = STYLE_CONFIG["font_size"]
        text_display.WindowLocation = 'Upper Center'

        # Optional: Mesh resolution label
        label_text = Text()
        label_text.Text = f"Mesh: {int(num_cells**0.5)}×{int(num_cells**0.5)}"
        label_disp = Show(label_text, view)
        label_disp.Color = [0.1, 0.1, 0.1]
        label_disp.FontSize = 14
        label_disp.WindowLocation = 'Lower Center'
    except Exception as e:
        print(f"  Warning: Could not add cell count text. {e}")

    try:
        view.ResetCamera(False)
        camera = GetActiveCamera()
        bounds = reader.GetDataInformation().GetBounds()
        pad = 0.05
        x_center = (bounds[0] + bounds[1]) / 2.0
        y_center = (bounds[2] + bounds[3]) / 2.0
        max_range = max(bounds[1]-bounds[0], bounds[3]-bounds[2])
        camera.SetFocalPoint(x_center, y_center, 0)
        camera.SetPosition(x_center, y_center, 1)
        view.CameraParallelProjection = 1
        view.CameraParallelScale = (max_range / 2.0) * (1 + pad)
    except Exception as e:
        print(f"  Warning: Camera fitting failed. {e}")
        view.ResetCamera()

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
