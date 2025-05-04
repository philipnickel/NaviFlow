#!/usr/bin/env python3
"""
Academic Mesh Visualization Script for 2D CFD Grids

Generates high-quality vector PDF visualizations from GMSH (.msh) files.
Only PDF export is performed (no PNG). Designed for clean academic figures.

Usage:
    pvpython visualize_meshes.py [path]

Arguments:
    path: Optional. Can be a mesh file (.msh), directory of mesh files, or an experiment folder.

Requirements:
    - ParaView with the meshio plugin installed
    - To load the plugin: Tools > Manage Plugins > Load New > [path to meshio plugin]
    - Installation guide: https://github.com/nschloe/meshio

Usage (from naviflow directory): 
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
    "boundary_color": [0.8, 0.1, 0.1],    # Red for boundaries
    "resolution": [1600, 1200],           # High-res
    "font_size": 14,                      # Default font size
}

# Define boundary names and colors for visualization
BOUNDARY_LABELS = {
    1: "bottom",
    2: "right",
    3: "top",
    4: "left",
    5: "obstacle"
}

BOUNDARY_COLORS = {
    1: [0.0, 0.0, 0.8],  # Blue for bottom
    2: [0.0, 0.8, 0.0],  # Green for right
    3: [0.8, 0.0, 0.0],  # Red for top
    4: [0.8, 0.8, 0.0],  # Yellow for left
    5: [0.8, 0.0, 0.8]   # Purple for obstacle
}

def visualize_msh(msh_file):
    """Visualize a GMSH mesh file with separate boundary visualization."""
    print(f"Visualizing MSH file: {msh_file}")
    base = os.path.splitext(os.path.basename(msh_file))[0]
    output_dir = os.path.dirname(msh_file)
    output_pdf = os.path.join(output_dir, f"{base}.pdf")

    try:
        # Load the MSH file using the meshio reader
        # This requires the ParaView meshio plugin to be installed
        reader = None
        
        # Check which reader to use based on ParaView version
        try:
            # Try using the meshio reader plugin first
            reader = MeshioReader(FileName=msh_file)
        except NameError:
            # Fallback to built-in GMSH reader if available
            try:
                reader = GmshReader(FileName=msh_file)
            except NameError:
                print("  Error: Neither meshio nor GMSH reader is available.")
                print("  Please install the meshio plugin for ParaView:")
                print("  https://github.com/nschloe/meshio/blob/main/tools/paraview-meshio-plugin.py")
                return
        
        if not reader:
            print("  Error: Failed to create a reader for MSH files.")
            return
        
        reader.UpdatePipeline()
        
        # Create view
        view = GetActiveViewOrCreate('RenderView')
        view.Background = STYLE_CONFIG["background"]
        view.OrientationAxesVisibility = 0  # Hide axes
        
        # Display the mesh
        display = Show(reader, view)
        display.Representation = 'Surface With Edges'
        display.EdgeColor = STYLE_CONFIG["edge_color"]
        
        # Try to color by physical entity or entity tag if available
        # Property names may vary based on the reader
        scalar_fields = []
        
        # Get available arrays
        try:
            cell_arrays = reader.CellData.GetArrayNames()
            for array_name in cell_arrays:
                scalar_fields.append(('CELLS', array_name))
        except:
            print("  Warning: Could not get cell array names.")
        
        # Choose an appropriate field for coloring
        color_field = None
        for field_type, field_name in scalar_fields:
            if "physical" in field_name.lower() or "entity" in field_name.lower() or "boundary" in field_name.lower():
                color_field = (field_type, field_name)
                break
        
        if color_field:
            # Color by the identified field
            ColorBy(display, color_field)
            lut = GetColorTransferFunction(color_field[1])
            lut.ApplyPreset('Jet', True)
            
            # Set up custom boundaries display
            # Extract physical entities for boundary visualization
            threshold = Threshold(Input=reader)
            threshold.ThresholdRange = [0.5, 5.5]  # Boundary IDs are typically 1-5
            
            try:
                # Different versions of ParaView may have different property names
                if hasattr(threshold, 'Scalars'):
                    threshold.Scalars = color_field
                elif hasattr(threshold, 'SelectInputScalars'):
                    threshold.SelectInputScalars = color_field
            except:
                print("  Warning: Could not set threshold scalar field.")
                
            # Create boundary representation
            boundaryDisplay = Show(threshold, view)
            boundaryDisplay.Representation = 'Surface'
            boundaryDisplay.LineWidth = 3.0
            
            # Add boundary labels
            for boundary_id, label in BOUNDARY_LABELS.items():
                # Extract specific boundary
                boundaryFilter = Threshold(Input=reader)
                boundaryFilter.ThresholdRange = [boundary_id - 0.5, boundary_id + 0.5]
                
                try:
                    # Different versions of ParaView may have different property names
                    if hasattr(boundaryFilter, 'Scalars'):
                        boundaryFilter.Scalars = color_field
                    elif hasattr(boundaryFilter, 'SelectInputScalars'):
                        boundaryFilter.SelectInputScalars = color_field
                except:
                    print(f"  Warning: Could not set threshold scalar field for boundary {boundary_id}.")
                
                # Custom display for this boundary
                filterDisplay = Show(boundaryFilter, view)
                filterDisplay.Representation = 'Surface'
                filterDisplay.LineWidth = 3.0
                filterDisplay.AmbientColor = BOUNDARY_COLORS.get(boundary_id, [0.8, 0.1, 0.1])
                
                # Add text label
                text = Text()
                text.Text = label
                textDisplay = Show(text, view)
                textDisplay.FontSize = STYLE_CONFIG["font_size"]
                textDisplay.Color = BOUNDARY_COLORS.get(boundary_id, [0.8, 0.1, 0.1])
                
                # Position text near a boundary element
                positions = {
                    1: [0.5, 0.1],  # bottom
                    2: [0.9, 0.5],  # right
                    3: [0.5, 0.9],  # top
                    4: [0.1, 0.5],  # left
                    5: [0.7, 0.7]   # obstacle
                }
                textDisplay.WindowLocation = positions.get(boundary_id, [0.5, 0.5])
        
        # Adjust camera for nice view
        ResetCamera()
        camera = GetActiveCamera()
        camera.Zoom(1.1)  # Zoom in slightly for better use of space
        
        # Export visualization
        view.ViewSize = STYLE_CONFIG["resolution"]
        
        # Use ExportView for PDF export
        try:
            # Modern API
            ExportView(output_pdf, view=view)
            print(f"Saved visualization to: {output_pdf}")
        except (NameError, AttributeError):
            try:
                # Older API
                from paraview.simple import SaveScreenshot
                SaveScreenshot(output_pdf, view, ImageResolution=STYLE_CONFIG["resolution"])
                print(f"Saved visualization to: {output_pdf} (using SaveScreenshot)")
            except Exception as e:
                print(f"Failed to export: {e}")
        
    except Exception as e:
        print(f"Error visualizing {msh_file}: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up to avoid memory issues
    try:
        Delete(GetActiveView())
        sources = GetSources()
        for source_key in list(sources.keys()):
            source = sources[source_key]
            Delete(source)
    except Exception as e:
        print(f"Cleanup error: {e}")

def process_path(path):
    """Process a path, which can be a mesh file, directory, or experiment folder"""
    if os.path.isfile(path):
        # Single file
        if path.endswith('.msh'):
            visualize_msh(path)
        else:
            print(f"Unsupported file format: {path}")
    elif os.path.isdir(path):
        # Directory - check if it's a specific experiment structure
        mesh_files = []
        
        # Look for MSH files
        msh_files = glob.glob(os.path.join(path, '*.msh'))
        mesh_files.extend(msh_files)
        
        # Look in subdirectories (structured/unstructured)
        for subdir in ['structuredUniform', 'structuredRefined', 'unstructured']:
            subdir_path = os.path.join(path, subdir)
            if os.path.isdir(subdir_path):
                # Look for MSH files
                subdir_msh = glob.glob(os.path.join(subdir_path, '*.msh'))
                mesh_files.extend(subdir_msh)
        
        # Look in experiment subdirectories too
        for exp_dir in glob.glob(os.path.join(path, '*')):
            if os.path.isdir(exp_dir):
                for subdir in ['structuredUniform', 'structuredRefined', 'unstructured']:
                    subdir_path = os.path.join(exp_dir, subdir)
                    if os.path.isdir(subdir_path):
                        # Look for MSH files
                        subdir_msh = glob.glob(os.path.join(subdir_path, '*.msh'))
                        mesh_files.extend(subdir_msh)
        
        # Process all found mesh files
        if mesh_files:
            print(f"Found {len(mesh_files)} mesh files to visualize")
            for mesh_file in mesh_files:
                if mesh_file.endswith('.msh'):
                    visualize_msh(mesh_file)
        else:
            print(f"No supported mesh files found in {path}")
    else:
        print(f"Invalid path: {path}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate visualizations from mesh files")
    parser.add_argument('path', nargs='?', default='meshing/experiments', 
                        help='Path to mesh file (.msh), directory, or experiment folder')
    
    args = parser.parse_args()
    process_path(args.path)