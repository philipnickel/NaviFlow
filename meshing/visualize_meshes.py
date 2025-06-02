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
    "background": [0.5, 0.5, 0.5],
    "edge_color": [0.1, 0.1, 0.1],
    "boundary_color": [0.5, 0.5, 0.5],
    "resolution": [1920, 1080],
    "font_size": 100,
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

    # --- Obstacle metadata will be estimated from mesh, not loaded from experiments.yaml ---

    reader = XMLUnstructuredGridReader(FileName=[file_path])
    reader.UpdatePipeline()

    obstacle_center = (0.2, 0.205)
    obstacle_radius = 0.05
    print(f"  Using hardcoded obstacle: center={obstacle_center}, radius={obstacle_radius}")

    # Reset color palette and ensure gradient is applied

    view = GetActiveViewOrCreate('RenderView')
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = 'Gradient'
    view.Background = [1.0, 1.0, 1.0]  # Bottom (white)
    view.Background2 = [0.55, 0.70, 0.90]  # Top (light blue)
    view.OrientationAxesVisibility = 0

    # Base layer: light surface fill (use separate surface filter)
    surface = ExtractSurface(Input=reader)
    surface_display = Show(surface, view)
    surface_display.Representation = 'Surface'
    surface_display.DiffuseColor = [0.85, 0.85, 0.85]  # Light gray fill
    surface_display.Opacity = 1.0

    # Top layer: wireframe overlay
    wireframe_display = Show(reader, view)
    wireframe_display.Representation = 'Wireframe'
    wireframe_display.ColorArrayName = ['POINTS', '']
    wireframe_display.AmbientColor = [0.0, 0.0, 0.0]
    wireframe_display.DiffuseColor = [0.0, 0.0, 0.0]
    wireframe_display.LineWidth = 0.2

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
        
        # Get available boundary IDs from the data by checking which ones actually have elements
        reader.UpdatePipeline()
        data_info = reader.GetCellDataInformation().GetArray(color_field[1])
        _min_val, _max_val = int(data_info.GetRange()[0]), int(data_info.GetRange()[1])
        
        # Check if boundaries actually have elements
        valid_boundaries = set()
        for bid in range(1, 6):  # Check boundary tags 1-5
            filt = Threshold(Input=reader)
            filt.LowerThreshold = bid - 0.5
            filt.UpperThreshold = bid + 0.5
            filt.ThresholdMethod = 'Between'
            if hasattr(filt, 'SelectInputScalars'):
                filt.SelectInputScalars = color_field
            elif hasattr(filt, 'Scalars'):
                filt.Scalars = color_field
            filt.UpdatePipeline()
            if filt.GetDataInformation().GetNumberOfCells() > 0:
                valid_boundaries.add(bid)
        
        print(f"  Valid boundary tags with cells: {sorted(valid_boundaries)}")
        
        # Check the mesh type and handle boundary detection accordingly
        is_structured = "structuredUniform" in file_path or "structuredRefined" in file_path
      
        has_obstacle = 5 in valid_boundaries
        
        active_boundaries = {
            bid: label for bid, label in BOUNDARY_LABELS.items() 
            if bid in valid_boundaries and (bid != 5 or has_obstacle)
        }
        
        print(f"  Active boundaries for display: {', '.join([f'{k}:{v}' for k, v in active_boundaries.items()])}")
        
        # Set up colors only for boundaries that exist in this mesh
        rgb_points = []
        for bid in sorted(active_boundaries.keys()):
            color = COOL_TO_WARM_MANUAL[bid]
            rgb_points.extend([float(bid), color[0], color[1], color[2]])
        
        if rgb_points:
            lut.RGBPoints = rgb_points
            
        lut.InterpretValuesAsCategories = 1
        
        # Setup annotations for legend - only for active boundaries
        annotations = []
        for bid, label in active_boundaries.items():
            annotations.extend([str(bid), label])
        if annotations:
            lut.Annotations = annotations

        # Do not show scalar bar (legend) or add boundary overlays

    else:
        print("  No suitable scalar field found for legend.")
            
    # Add cell count text (using a more robust method)
    try:
        reader.UpdatePipeline()
        
        # Method 1: Use Physical field to distinguish between domain and boundary cells
        if color_field and color_field[1] == 'gmsh:physical':
            # Most reliable approach: Fluid domain cells usually have tag 10
            fluid_cells_filter = Threshold(Input=reader)
            fluid_cells_filter.ThresholdMethod = 'Between'
            fluid_cells_filter.LowerThreshold = 9.5  # Just below 10
            fluid_cells_filter.UpperThreshold = 10.5  # Just above 10
            
            # Assign the correct field for filtering
            if hasattr(fluid_cells_filter, 'SelectInputScalars'):
                fluid_cells_filter.SelectInputScalars = color_field
            elif hasattr(fluid_cells_filter, 'Scalars'):
                fluid_cells_filter.Scalars = color_field
                
            fluid_cells_filter.UpdatePipeline()
            num_cells_2d = fluid_cells_filter.GetDataInformation().GetNumberOfCells()
            
            # Get the total cells for comparison
            total_cells = reader.GetDataInformation().GetNumberOfCells()
            
            # If we don't find any fluid cells, fall back to calculating by subtraction
            if num_cells_2d == 0:
                # Count boundary cells (tags 1-4 typically)
                boundary_cells = 0
                for bid in range(1, 5):  # Common boundary tags
                    boundary_filter = Threshold(Input=reader)
                    boundary_filter.ThresholdMethod = 'Between'
                    boundary_filter.LowerThreshold = bid - 0.5
                    boundary_filter.UpperThreshold = bid + 0.5
                    
                    if hasattr(boundary_filter, 'SelectInputScalars'):
                        boundary_filter.SelectInputScalars = color_field
                    elif hasattr(boundary_filter, 'Scalars'):
                        boundary_filter.Scalars = color_field
                        
                    boundary_filter.UpdatePipeline()
                    boundary_cells += boundary_filter.GetDataInformation().GetNumberOfCells()
                
                # Calculate 2D cells by subtracting boundary cells from total
                num_cells_2d = total_cells - boundary_cells
                print(f"  Using method 1 fallback: {total_cells} total - {boundary_cells} boundary = {num_cells_2d} fluid cells")
            else:
                print(f"  Using method 1: Found {num_cells_2d} fluid domain cells (tag 10)")
        else:
            # Method 2: If we don't have physical tags, estimate based on cell type
            # This is less reliable but better than nothing
            total_cells = reader.GetDataInformation().GetNumberOfCells()
            
            # For structured meshes, we can calculate based on the boundary lines
            # A 2D structured n×m mesh has n*m cells and 2*(n+m) boundary lines
            is_structured = "structuredUniform" in file_path or "structuredRefined" in file_path
            if is_structured:
                # Count vertex points
                points_info = reader.GetPointDataInformation()
                points_info.GetNumberOfArrays()
                
                # Estimate n and m based on total cells and assumption of an n×m grid
                # For a square domain (n=m): total_cells = n² + 4n
                # Let's use a heuristic approach 
                n = int((-4 + (16 + 4*total_cells)**0.5) / 2)
                num_cells_2d = n*n
                print(f"  Using method 2: Estimated {n}×{n} grid = {num_cells_2d} cells")
            else:
                # For unstructured, rough estimate is total minus boundary
                # Typical boundary count is a small fraction of total cells
                # Linear meshes usually have approximately total_cells*0.05 boundary elements
                num_boundary_estimate = int(0.05 * total_cells)
                num_cells_2d = total_cells - num_boundary_estimate
                print(f"  Using method 2: Estimated {num_cells_2d} 2D cells (approx. {num_boundary_estimate} boundary elements)")
        
        # Create text display
        cell_text = Text()
        cell_text.Text = f"Cells: {num_cells_2d:,}"
        text_display = Show(cell_text, view)
        text_display.Color = [0.1, 0.1, 0.1]
        text_display.FontSize = 50
        text_display.WindowLocation = 'Lower Center'
        text_display.Position = [0.4, 0.1]
        
        print(f"  Visualization will show {num_cells_2d} cells")
        
    except Exception as e:
        print(f"  Warning: Cell counting failed with error: {e}")
        # Final fallback - just show total elements count
        try:
            total_cells = reader.GetDataInformation().GetNumberOfCells()
            cell_text = Text()
            cell_text.Text = f"Elements: {total_cells:,}"
            text_display = Show(cell_text, view)
            text_display.Color = [0.1, 0.1, 0.1]
            text_display.FontSize = 20
            text_display.WindowLocation = 'Lower Center'
            text_display.Position = [0.4, 0.05]
            print(f"  Fallback: Showing total element count ({total_cells})")
        except Exception as nested_e:
            print(f"  Warning: Even fallback counting failed. {nested_e}")

    # Auto-fit camera based on bounds
    try:
        print("  Auto-fitting camera...")
        view.ResetCamera(True)
        view.CameraParallelProjection = 1
        #view.CameraParallelScale *= 0.4  # Zoom in to fill more of the screen
    except Exception as e:
        print(f"  Warning: Auto-fitting camera failed. {e}")
        view.ResetCamera()

    view.ViewSize = [3840, 2160]


    # PDF export block
    try:
        ExportView(output_pdf, view=view)
        print(f"  Saved vector PDF to: {output_pdf}")
    except Exception as e:
        print(f"  ❌ Error exporting PDF: {e}")


    try:
        for src in list(GetSources().values()):
            Hide(src)
            Delete(src)
        print("  Cleanup completed.")
    except Exception as e:
        print(f"  Warning during cleanup: {e}")

def process_path(path):
    """Process all VTU files in a given path."""
    files = glob.glob(os.path.join(path, "*.vtu"))
    if not files:
        print(f"No VTU files found in {path}")
        return False
    
    for file_path in sorted(files):
        visualize_mesh(file_path)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Visualize meshes from VTU files")
    parser.add_argument("paths", nargs="*", help="Paths to VTU files or directories")
    args = parser.parse_args()
    
    # If no arguments provided, search for VTU files in experiments directory
    if not args.paths:
        print("No paths specified, searching for mesh files in standard locations...")
        base_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        experiments_dir = os.path.join(base_dir, "meshing", "experiments")
        
        if not os.path.exists(experiments_dir):
            print(f"Error: Experiments directory not found at {experiments_dir}")
            sys.exit(1)
        
        found_files = False
        for exp_dir in glob.glob(os.path.join(experiments_dir, "*")):
            exp_name = os.path.basename(exp_dir)
            print(f"\nSearching for mesh files in experiment: {exp_name}")
            
            # experiments/{experiment}/structuredUniform/{resolution}/*.vtu
            # experiments/{experiment}/unstructured/{resolution}/*.vtu
            for mesh_type in ['structuredUniform', 'structuredRefined', 'unstructured']:
                mesh_dir = os.path.join(exp_dir, mesh_type)
                if not os.path.exists(mesh_dir):
                    continue
                
                for res_dir in glob.glob(os.path.join(mesh_dir, "*")):
                    if os.path.isdir(res_dir):
                        res_name = os.path.basename(res_dir)
                        print(f"  • Checking {mesh_type} / {res_name}")
                        if process_path(res_dir):
                            found_files = True
        
        if not found_files:
            print("\nNo mesh files found in standard locations.")
            print("Generate meshes first or specify a path to VTU files.")
    else:
        # Process user-provided paths
        for path in args.paths:
            if os.path.isdir(path):
                process_path(path)
            elif os.path.isfile(path) and path.lower().endswith(".vtu"):
                visualize_mesh(path)
            else:
                print(f"Error: {path} is not a valid VTU file or directory")

if __name__ == "__main__":
    main()