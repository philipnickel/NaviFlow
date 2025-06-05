#!/usr/bin/env python3
"""
Generate a single-page CFD results overview PDF suitable for thesis figure inclusion.

This script takes a config path, loads existing PDF files from results/plots,
and arranges them in a professional grid layout on a single page.

Usage:
    python generate_appendix.py --config path/to/config.yaml [--output output.pdf]
"""

import os
import argparse
import yaml
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import tempfile
import subprocess

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Import matplotlib with proper styling from plot_style module
from naviflow_collocated.utils.postprocess.plot_style import plt

def find_mesh_visualization_pdf(config):
    """Find the mesh visualization PDF based on experiment configuration."""
    experiment = config.get('experiment', 'unknown')
    domain_info = config.get('domain', {})
    mesh_info = domain_info.get('mesh', ['unknown', 'unknown'])
    
    if len(mesh_info) >= 2:
        mesh_type = mesh_info[0]  # e.g., 'structured'
        mesh_resolution = mesh_info[1]  # e.g., '63x63'
        
        # Map resolution to mesh size category
        if 'x' in str(mesh_resolution):
            resolution_parts = str(mesh_resolution).split('x')
            if len(resolution_parts) == 2:
                n_cells = int(resolution_parts[0])
                if n_cells <= 30:
                    size_category = 'coarse'
                elif n_cells <= 80:
                    size_category = 'medium'
                else:
                    size_category = 'fine'
            else:
                size_category = 'medium'  # default
        else:
            size_category = 'medium'  # default
        
        # Construct potential mesh PDF paths
        mesh_paths = [
            f"meshing/experiments/{experiment}/structuredUniform/{size_category}/{experiment}_uniform_{size_category}.pdf",
            f"meshing/experiments/{experiment}/unstructured/{size_category}/{experiment}_unstructured_{size_category}.pdf",
            f"meshing/experiments/{experiment}/{experiment}_mesh.pdf"
        ]
        
        # Check which path exists
        for mesh_path in mesh_paths:
            if os.path.exists(mesh_path):
                return mesh_path
    
    return None

def collect_pdf_files(config_path):
    """Collect relevant PDF files for the single-page overview."""
    base_dir = os.path.dirname(config_path)
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load metadata for simulation ID
    metadata_path = os.path.join(results_dir, "metadata.yaml")
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    # All 11 files requested for thesis figure
    desired_files = {
        "mesh": ("mesh_visualization", None),  # Will be found separately
        "validation": ("ghia_comparison", "ghia_comparison.pdf"),
        "metadata": ("metadata", "metadata.pdf"),
        "velocity_magnitude": ("velocity_magnitude", "velocity_magnitude.pdf"),
        "u_velocity": ("u_velocity", "u_velocity.pdf"),
        "v_velocity": ("v_velocity", "v_velocity.pdf"),
        "streamlines": ("streamlines", "streamlines.pdf"),
        "residual_history": ("residual_history", "residual_history.pdf"),
        "u_residual": ("u_residual", "u_residual.pdf"),
        "v_residual": ("v_residual", "v_residual.pdf"),
        "continuity_residual": ("continuity_residual", "continuity_residual.pdf"),
    }
    
    pdf_files = {}
    
    # Collect existing PDF files
    for key, (internal_key, filename) in desired_files.items():
        if filename:  # Regular plot files
            filepath = os.path.join(plots_dir, filename)
            if os.path.exists(filepath):
                pdf_files[key] = filepath
            else:
                print(f"Warning: {filename} not found in {plots_dir}")
    
    # Add mesh visualization if available
    mesh_pdf = find_mesh_visualization_pdf(config)
    if mesh_pdf and os.path.exists(mesh_pdf):
        pdf_files["mesh"] = mesh_pdf
        print(f"Found mesh visualization: {mesh_pdf}")
    else:
        print("Warning: Mesh visualization PDF not found")
    
    return pdf_files, config, metadata

def pdf_to_image(pdf_path, dpi=150):
    """Convert first page of PDF to image array using pdf2image."""
    try:
        from pdf2image import convert_from_path
        
        # Convert first page only
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
        if images:
            return np.array(images[0])
        return None
    except ImportError:
        print("Warning: pdf2image not available, trying alternative method...")
        return pdf_to_image_matplotlib(pdf_path)
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
        return None

def pdf_to_image_matplotlib(pdf_path):
    """Alternative PDF to image conversion using matplotlib."""
    try:
        import matplotlib.image as mpimg
        from matplotlib.backends.backend_pdf import PdfPages
        
        # This is a fallback - create a placeholder
        return np.ones((400, 400, 3), dtype=np.uint8) * 240  # Light gray placeholder
    except Exception as e:
        print(f"Error in matplotlib conversion: {e}")
        return None

def create_thesis_figure_pdf(pdf_files, config, metadata, output_path):
    """Create a single-page vertical thesis figure from existing PDF files."""
    
    # Get experiment info
    experiment_name = config.get('experiment', 'Unknown')
    sim_id = metadata.get('Simulation id', 'unknown')
    Re = config.get('physical_properties', {}).get('reynolds_number', 'Unknown')
    
    # Create vertical figure for full-page thesis inclusion
    fig = plt.figure(figsize=(10, 14))  # Vertical format: wider than tall ratio inverted
    
    # Create main title
    main_title = f"{experiment_name.replace('_', ' ').title()} - {sim_id}"
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.97)
    
    # Create grid layout with tighter spacing
    gs = GridSpec(4, 6, figure=fig, hspace=0.10, wspace=0.05, 
                  left=0.02, right=0.98, top=0.94, bottom=0.02)
    
    # Define plot positions - reorganized with validation and residual_history larger in last row
    plot_positions = [
        # Row 1: 3 plots (2 columns each)
        ("mesh", gs[0, 0:2]),
        ("metadata", gs[0, 2:4]),
        ("velocity_magnitude", gs[0, 4:6]),
        
        # Row 2: 3 plots (2 columns each) 
        ("u_velocity", gs[1, 0:2]),
        ("v_velocity", gs[1, 2:4]),
        ("streamlines", gs[1, 4:6]),
        
        # Row 3: 3 plots (2 columns each)
        ("u_residual", gs[2, 0:2]),
        ("v_residual", gs[2, 2:4]),
        ("continuity_residual", gs[2, 4:6]),
        
        # Row 4: 2 larger centered plots (3 columns each)
        ("validation", gs[3, 0:3]),
        ("residual_history", gs[3, 3:6]),
    ]
    
    for key, grid_slice in plot_positions:
        ax = fig.add_subplot(grid_slice)
        
        if key in pdf_files:
            # Convert PDF to image and display
            print(f"Processing {key}...")
            img = pdf_to_image(pdf_files[key], dpi=120)
            
            if img is not None:
                ax.imshow(img)
                ax.axis('off')
            else:
                # Fallback: show placeholder text
                ax.text(0.5, 0.5, f'{key}\n(PDF conversion failed)', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor="lightgray", alpha=0.8))
                ax.axis('off')
        else:
            # Show "not available" placeholder
            ax.text(0.5, 0.5, f'{key}\n(Not Available)', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor="lightcoral", alpha=0.3))
            ax.axis('off')
    
    # Save as single-page PDF
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300, 
                   facecolor='white', edgecolor='none')
    
    plt.close(fig)
    print(f"Single-page vertical thesis figure saved to: {output_path}")
    return output_path

def generate_appendix_pdf(config_path, output_path=None):
    """Generate single-page thesis figure PDF from existing PDF files."""
    
    # Collect PDF files and metadata
    pdf_files, config, metadata = collect_pdf_files(config_path)
    
    if not pdf_files:
        print("Error: No PDF files found to include")
        return None
    
    # Determine output path
    if output_path is None:
        base_dir = os.path.dirname(config_path)
        sim_id = metadata.get('Simulation id', 'unknown')
        output_path = os.path.join(base_dir, 'results', f'thesis_figure_{sim_id}.pdf')
    
    print(f"Creating single-page thesis figure from {len(pdf_files)} PDF files...")
    for key, path in pdf_files.items():
        print(f"  - {key}: {os.path.basename(path)}")
    
    # Create the single-page figure
    return create_thesis_figure_pdf(pdf_files, config, metadata, output_path)

def main():
    parser = argparse.ArgumentParser(description='Generate single-page thesis figure from existing CFD results')
    parser.add_argument('--config', required=True, help='Path to experiment config.yaml or pseudo_config.yaml')
    parser.add_argument('--output', help='Output PDF path (default: results/thesis_figure_<sim_id>.pdf)')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Generate thesis figure
    generate_appendix_pdf(args.config, args.output)

if __name__ == '__main__':
    main() 