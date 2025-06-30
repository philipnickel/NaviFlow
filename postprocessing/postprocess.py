#!/usr/bin/env python3

import os
import sys
# Add workspace root to Python path
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

import argparse
import numpy as np
import yaml
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
import tempfile
import shutil
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import re
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import imageio
from datetime import datetime
import matplotlib.colors
from scipy.ndimage import map_coordinates
import matplotlib.image as mpimg
from PIL import Image

from naviflow_collocated.utils.postprocess.plotting import (
    plot_fields_single_row,
    plot_residuals,
    plot_residual_fields,
    plot_streamlines,
    plot_force_coefficients,
    save_individual_field_plots,
    save_individual_residual_plots,
    plot_velocity_magnitude
)
from naviflow_collocated.utils.postprocess.verification import ghia_comparison, poiseuille_verification
from naviflow_collocated.utils.postprocess.metadata import yaml_to_latex_pdf
from naviflow_collocated.utils.postprocess.utils import save_pdf, get_obstacle_mask_from_msh, flatten_dict
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.utils.postprocess.forces import calculate_cylinder_forces, calculate_pressure_difference

# ----------------------------
# Plotting Helpers
# ----------------------------
def save_pdf(fig, path, also_save_in_plots=False):
    # Save the plot directly in the plots directory
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    print(f"Saved: {path}")
    plt.close(fig)

def get_obstacle_mask_from_msh(x, y, experiment):
    """
    Generate obstacle mask for cylinderFlow experiment.
    For other experiments, return a mask of all False (no obstacles).
    """
    # Extract base experiment name (remove any path components)
    base_experiment = experiment.split('/')[-1]
    
    if base_experiment == "cylinderFlow" or "cylinderFlow" in experiment:
        # Cylinder center and radius from mesh generation
        center = np.array([0.2, 0.2])
        radius = 0.05
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask = dist < radius
        return mask
    # For all other experiments, return a mask of all False (no obstacles)
    return np.zeros_like(x, dtype=bool)

def plot_fields(x, y, U, velocity_magnitude, p, scheme, mesh_type, Re, n_cells, output_path, sim_id=None, experiment=None):
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # Robust obstacle mask from .msh
    obstacle_mask = get_obstacle_mask_from_msh(x, y, experiment)
    
    # U-velocity
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.tricontourf(x, y, U[:, 0], levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        ax1.tricontourf(x[obstacle_mask], y[obstacle_mask], U[obstacle_mask, 0], levels=1, colors='gray', alpha=0.5)
    fig.colorbar(cf1, ax=ax1)
    ax1.set_title("U-velocity")
    ax1.set_aspect("equal", "box")
    
    # V-velocity
    ax2 = fig.add_subplot(gs[0, 1])
    cf2 = ax2.tricontourf(x, y, U[:, 1], levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        ax2.tricontourf(x[obstacle_mask], y[obstacle_mask], U[obstacle_mask, 1], levels=1, colors='gray', alpha=0.5)
    fig.colorbar(cf2, ax=ax2)
    ax2.set_title("V-velocity")
    ax2.set_aspect("equal", "box")
    
    # Velocity Magnitude
    ax3 = fig.add_subplot(gs[1, 0])
    cf3 = ax3.tricontourf(x, y, velocity_magnitude, levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        ax3.tricontourf(x[obstacle_mask], y[obstacle_mask], velocity_magnitude[obstacle_mask], levels=1, colors='gray', alpha=0.5)
    fig.colorbar(cf3, ax=ax3)
    ax3.set_title("Velocity Magnitude")
    ax3.set_aspect("equal", "box")
    
    # Pressure
    ax4 = fig.add_subplot(gs[1, 1])
    cf4 = ax4.tricontourf(x, y, p, levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        ax4.tricontourf(x[obstacle_mask], y[obstacle_mask], p[obstacle_mask], levels=1, colors='gray', alpha=0.5)
    fig.colorbar(cf4, ax=ax4)
    ax4.set_title("Pressure")
    ax4.set_aspect("equal", "box")
    
    fig.suptitle(f"Flow Field | Re={Re}, {scheme}, {mesh_type}")
    if sim_id is not None:
        fig.text(0.5, 0.01, f"Simulation ID: {sim_id}", ha='center', va='bottom', fontsize=6, color='gray', alpha=0.7)
    save_pdf(fig, output_path)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# ----------------------------
# LIC Implementation
# ----------------------------
def enhance_contrast(image):
    """Enhance the contrast of an image using histogram equalization."""
    # This is a simple implementation of histogram equalization
    image_flat = image.flatten()
    hist, bins = np.histogram(image_flat, 256, [0, 1])
    cdf = hist.cumsum()
    
    # Normalize the CDF
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # Create a lookup table from the normalized CDF
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Apply the lookup table to the image
    image_equalized = cdf_final[ (image * 255).astype('uint8') ]
    
    return image_equalized / 255.0

def lic(vx, vy, texture=None, length=10, n_iter=2):
    """
    Performs Line Integral Convolution.
    """
    if texture is None:
        texture = np.random.rand(*vx.shape).astype(np.float32)
    else:
        texture = texture.astype(np.float32)
        if texture.shape != vx.shape:
            raise ValueError('The texture must have the same shape as the vector field.')

    v = np.hypot(vx, vy)
    v[v == 0] = 1.
    nvx = vx / v
    nvy = vy / v

    result = np.copy(texture)

    for _ in range(n_iter):
        cumulative_texture = np.zeros_like(texture, dtype=np.float32)
        cumulative_weight = np.zeros_like(texture, dtype=np.float32)

        for i in range(length):
            fwd_x = nvx * i
            fwd_y = nvy * i
            bwd_x = -nvx * i
            bwd_y = -nvy * i

            for x, y in [(fwd_x, fwd_y), (bwd_x, bwd_y)]:
                coords = np.array([y + np.arange(texture.shape[0])[:, np.newaxis],
                                   x + np.arange(texture.shape[1])])
                tex_val = map_coordinates(texture, coords, order=1, mode='reflect')
                cumulative_texture += tex_val
                cumulative_weight += 1.

        result = cumulative_texture / cumulative_weight

    return result

def get_obstacle_mask_grid(Xg, Yg, experiment, velocity_mag_g=None):
    """
    Generate obstacle mask on the grid for different experiments.
    Returns a boolean mask where True indicates obstacle locations.
    """
    mask = np.zeros_like(Xg, dtype=bool)
    
    # Extract base experiment name (remove any path components)
    base_experiment = experiment.split('/')[-1] if isinstance(experiment, str) else str(experiment)
    
    if "cylinderFlow" in base_experiment or "cylinderFlow" in experiment:
        if velocity_mag_g is not None:
            # Use actual simulation data to detect obstacle (more robust)
            # Find regions with very low velocity magnitude
            threshold = 0.005  # Adjust this threshold as needed
            low_vel_mask = velocity_mag_g < threshold
            
            # Find connected components and select the largest central one
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(low_vel_mask)
            
            if num_features > 0:
                # Find the component closest to expected cylinder center
                center_x_idx = Xg.shape[1] // 10  # Roughly x=0.2 in domain [0, 2.2]
                center_y_idx = Xg.shape[0] // 2   # Roughly y=0.2 in domain [0, 0.4]
                
                min_dist = float('inf')
                best_label = 0
                
                for label in range(1, num_features + 1):
                    component = (labeled_array == label)
                    if np.sum(component) < 100:  # Skip very small components
                        continue
                    
                    # Find center of mass of this component
                    y_indices, x_indices = np.where(component)
                    com_y = np.mean(y_indices)
                    com_x = np.mean(x_indices)
                    
                    # Distance to expected cylinder center
                    dist = np.sqrt((com_x - center_x_idx)**2 + (com_y - center_y_idx)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_label = label
                
                if best_label > 0:
                    mask = (labeled_array == best_label)
        
        # Fallback to geometric definition if data-based detection fails
        if not np.any(mask):
            center = np.array([0.2, 0.2])
            radius = 0.05
            dist = np.sqrt((Xg - center[0])**2 + (Yg - center[1])**2)
            mask = dist <= radius
    
    return mask

def plot_velocity_with_lic(x, y, U, velocity_magnitude, experiment, Re, output_path, sim_id=None):
    """
    Create a velocity magnitude plot with Surface LIC representation of streamlines,
    with style consistent with other individual plots.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    n_grid_x = 800
    domain_width = np.max(x) - np.min(x)
    domain_height = np.max(y) - np.min(y)
    n_grid_y = int(n_grid_x * domain_height / domain_width)

    xi = np.linspace(np.min(x), np.max(x), n_grid_x)
    yi = np.linspace(np.min(y), np.max(y), n_grid_y)
    Xg, Yg = np.meshgrid(xi, yi)
    
    points = np.column_stack((x, y))
    Ug = griddata(points, U[:, 0], (Xg, Yg), method='cubic', fill_value=0)
    Vg = griddata(points, U[:, 1], (Xg, Yg), method='cubic', fill_value=0)
    velocity_mag_g = griddata(points, velocity_magnitude, (Xg, Yg), method='cubic', fill_value=0)

    # Create obstacle mask and apply it
    obstacle_mask = get_obstacle_mask_grid(Xg, Yg, experiment, velocity_mag_g)
    
    # Set velocity to zero inside obstacles for LIC calculation (using detected obstacles)
    Ug_masked = Ug.copy()
    Vg_masked = Vg.copy()
    velocity_mag_masked = velocity_mag_g.copy()
    
    Ug_masked[obstacle_mask] = 0
    Vg_masked[obstacle_mask] = 0
    velocity_mag_masked[obstacle_mask] = 0

    # Generate LIC texture with masked velocity field (obstacles excluded from LIC)
    lic_texture = lic(Ug_masked, Vg_masked, length=35)  # Medium version
    
    # Enhance the contrast of the LIC texture to make streamlines pop
    lic_enhanced = enhance_contrast(lic_texture)

    # Normalize velocity magnitude for color mapping (use original data for proper scaling)
    vmin = np.min(velocity_mag_g[~obstacle_mask]) if np.any(~obstacle_mask) else np.min(velocity_mag_g)
    vmax = np.max(velocity_mag_g[~obstacle_mask]) if np.any(~obstacle_mask) else np.max(velocity_mag_g)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('coolwarm')
    
    # Create color array from velocity magnitude
    colors = cmap(norm(velocity_mag_g))
    
    # Set obstacle regions to a very distinct color (much darker)
    obstacle_color = np.array([0.1, 0.1, 0.1, 1.0])  # Very dark gray/black
    colors[obstacle_mask] = obstacle_color
    
    # Convert the enhanced LIC texture to a grayscale overlay (medium prominence)
    overlay_alpha = 0.35  # Medium version
    lic_overlay = np.zeros((lic_enhanced.shape[0], lic_enhanced.shape[1], 4))
    lic_overlay[..., 0] = 0.0
    lic_overlay[..., 1] = 0.0
    lic_overlay[..., 2] = 0.0
    lic_overlay[..., 3] = (1.0 - lic_enhanced) * overlay_alpha
    
    # Don't overlay LIC texture on obstacles - keep them solid
    lic_overlay[obstacle_mask, 3] = 0.0

    # Plot the velocity magnitude as the base layer
    ax.imshow(colors, origin='lower', extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)], aspect='auto')
    
    # Overlay the LIC texture (excluding obstacles)
    ax.imshow(lic_overlay, origin='lower', extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)], aspect='auto')

    # Add clean gray circle overlay for cylinder (for visual polish)
    if "cylinderFlow" in experiment:
        center = (0.2, 0.2)
        radius = 0.05375  # About 7.5% larger than original (0.05)
        # Add a clean solid gray circle (no outline) for better visual appearance
        circle = mpatches.Circle(center, radius, facecolor='gray', edgecolor='none',
                               alpha=0.8, zorder=10)
        ax.add_patch(circle)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label('Velocity Magnitude', size=12)

    title_str = f"Velocity Magnitude with LIC (Re={Re})"
    ax.set_title(title_str, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"ID: {sim_id}", fontsize=8, color='gray', loc='right')

    # Remove axis labels and ticks for cleaner appearance
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    
    fig.tight_layout(pad=0.1)
    
    # Save as PDF using the save_pdf function for consistency
    save_pdf(fig, output_path)

def composite_paraview_plot(x, y, velocity_magnitude, experiment, Re, output_path, sim_id, results_dir):
    """
    Create a composite plot using ParaView-generated LIC visualization with matplotlib styling.
    This function looks for a pre-generated ParaView image and composites it into a matplotlib figure.
    """
    import os  # Move import to top
    
    # Look for ParaView-generated image
    paraview_image_path = os.path.join(os.path.dirname(results_dir), "paraview_lic_visualization.png")
    
    if not os.path.exists(paraview_image_path):
        print(f"ParaView LIC image not found at {paraview_image_path}")
        print("Run: pvpython generate_paraview_lic.py <results_dir> first")
        return False
    
    print(f"Found ParaView LIC image: {paraview_image_path}")
    
    # Load and display the ParaView image
    paraview_img = Image.open(paraview_image_path)
    paraview_array = np.array(paraview_img)
    
    # Create matplotlib figure with consistent styling
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate data bounds for proper extent
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Display the ParaView image with correct extent to fill plot area entirely
    im = ax.imshow(paraview_array, extent=[x_min, x_max, y_min, y_max], 
              aspect='auto', origin='lower')
    
    # Add colorbar with consistent styling
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    
    # Create scalar mappable for colorbar
    norm = plt.Normalize(vmin=np.min(velocity_magnitude), vmax=np.max(velocity_magnitude))
    cmap = plt.get_cmap('coolwarm')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label('Velocity Magnitude', size=12)
    
    # Consistent title and labels
    title_str = f"Velocity Magnitude with LIC (Re={Re})"
    ax.set_title(title_str, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"ID: {sim_id}", fontsize=8, color='gray', loc='right')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    
    fig.tight_layout(pad=0.1)
    
    # Save with high DPI
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"Hybrid ParaView-matplotlib LIC plot saved: {output_path}")
    return True

# ----------------------------
# Main Entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Relative path to the config.yaml file")
    parser.add_argument("--all", action="store_true", help="Generate all possible plots and analyses.")
    parser.add_argument("--animate", action="store_true", help="Generate animations from transient data.")
    parser.add_argument("--animate-step", type=int, default=1, help="Process every Nth saved frame for animation.")
    args = parser.parse_args()

    # Get the workspace directory
    workspace_dir = os.getcwd()
    
    # Construct absolute path to config file
    config_path = os.path.join(workspace_dir, args.config)
    
    # Validate config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use the config file's directory as the base directory
    base_dir = os.path.dirname(config_path)
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    
    # Extract experiment name - use config first, then path
    if 'experiment' in config:
        experiment = config['experiment']
    else:
        # Fall back to path-based detection
        config_path_parts = config_path.split(os.sep)
        if 'cylinderFlow' in config_path_parts:
            experiment = 'cylinderFlow'
        elif 'channelFlow' in config_path_parts:
            experiment = 'channelFlow'
        elif 'lidDrivenCavity' in config_path_parts:
            experiment = 'lidDrivenCavity'
        else:
            experiment = os.path.basename(os.path.dirname(config_path))
    
    # Create directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)

    # If --animate flag is used, process each time step
    if args.animate:
        # Load metadata once to get sim_id and dt
        metadata_path = os.path.join(results_dir, "metadata.yaml")
        if not os.path.exists(metadata_path):
            print(f"Error: metadata.yaml not found in {results_dir}")
            exit(1)
            
        with open(metadata_path, 'r') as f:
            meta = yaml.safe_load(f)
        sim_id = meta.get("Simulation id", "N/A")
        
        # Correctly parse nested config to get dt, ensuring it's a float
        config_data = meta.get("Config", {})
        dt_str = config_data.get("algorithm", {}).get("dt", "0.0")
        try:
            dt = float(dt_str)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert dt '{dt_str}' to float. Time in animations will be incorrect.")
            dt = 0.0
        
        if dt == 0.0 and "transient" in config_data.get("algorithm", {}):
            print("Warning: dt is 0.0, time in animations will be incorrect.")

        transient_base_dir = os.path.join(results_dir, "transient_data")
        U_dir = os.path.join(transient_base_dir, "U")
        p_dir = os.path.join(transient_base_dir, "p")

        if not os.path.isdir(U_dir) or not os.path.isdir(p_dir):
            print(f"Error: 'transient_data/U' or 'transient_data/p' directory not found in {results_dir}")
            exit(1)

        u_files = sorted([f for f in os.listdir(U_dir) if f.endswith('.npy') and f.startswith('U_')])
        
        if args.animate_step > 1:
            print(f"Animating every {args.animate_step}-th frame.")
            u_files = u_files[::args.animate_step]
            
        # Pre-scan to find global velocity magnitude range for consistent color scaling
        global_vmin, global_vmax = np.inf, -np.inf
        print("Pre-scanning data to determine global color range...")
        for u_file in u_files:
            u_path = os.path.join(U_dir, u_file)
            U = np.load(u_path)
            velocity_magnitude = np.linalg.norm(U, axis=1)
            global_vmin = min(global_vmin, np.min(velocity_magnitude))
            global_vmax = max(global_vmax, np.max(velocity_magnitude))
        print(f"Global velocity range: [{global_vmin:.4f}, {global_vmax:.4f}]")

        streamline_frames = []
        velocity_magnitude_frames = []
        
        for u_file in u_files:
            time_step_str = u_file.split('_')[-1].split('.')[0]
            p_file = f"p_{time_step_str}.npy"
            p_path = os.path.join(p_dir, p_file)
            u_path = os.path.join(U_dir, u_file)

            print(f"Processing frame for time step: {time_step_str}")

            if not os.path.exists(p_path):
                print(f"  > Corresponding pressure file {p_file} not found, skipping.")
                continue

            U = np.load(u_path)
            p = np.load(p_path)

            velocity_magnitude = np.linalg.norm(U, axis=1)
            cell_data = np.load(os.path.join(results_dir, "cell_centers.npz"))
            x = cell_data["x"]
            y = cell_data["y"]
            time_val = int(time_step_str) * dt
            
            # Generate frame for standalone streamlines
            streamline_frame = plot_streamlines(
                x, y, U,
                output_path=None,
                sim_id=sim_id,
                experiment=experiment,
                return_as_array=True,
                time_val=time_val,
                hide_colorbar=True
            )
            streamline_frames.append(streamline_frame)

            # Generate frame for standalone velocity magnitude
            velocity_magnitude_frame = plot_velocity_magnitude(
                x, y, velocity_magnitude,
                output_path=None,
                sim_id=sim_id,
                experiment=experiment,
                Re=config["physical_properties"]["reynolds_number"],
                return_as_array=True,
                time_val=time_val,
                vmin=global_vmin,
                vmax=global_vmax,
                hide_colorbar=True
            )
            velocity_magnitude_frames.append(velocity_magnitude_frame)

        # Create animations from the in-memory frames
        if streamline_frames:
            print(f"\nCreating streamline animations...")
            gif_path = os.path.join(plots_dir, "streamlines.gif")
            mp4_path = os.path.join(plots_dir, "streamlines.mp4")
            imageio.mimsave(gif_path, streamline_frames, fps=5)
            imageio.mimsave(mp4_path, streamline_frames, fps=5)
            print(f"Saved streamline animations to {plots_dir}")

        if velocity_magnitude_frames:
            print(f"\nCreating velocity magnitude animations...")
            gif_path = os.path.join(plots_dir, "velocity_magnitude.gif")
            mp4_path = os.path.join(plots_dir, "velocity_magnitude.mp4")
            imageio.mimsave(gif_path, velocity_magnitude_frames, fps=5)
            imageio.mimsave(mp4_path, velocity_magnitude_frames, fps=5)
            print(f"Saved velocity magnitude animations to {plots_dir}")

        print("\nFinished processing transient data. Continuing to standard post-processing...")
        # If --animate was passed, we assume --all for the final state
        args.all = True

    U = np.load(os.path.join(results_dir, "U_final.npy"))
    p = np.load(os.path.join(results_dir, "p_final.npy"))
    res = np.load(os.path.join(results_dir, "residuals.npz"))
    cell_data = np.load(os.path.join(results_dir, "cell_centers.npz"))
    
    metadata_path = os.path.join(results_dir, "metadata.yaml")
    with open(metadata_path, 'r') as f:
        meta = yaml.safe_load(f)

    x = cell_data["x"]
    y = cell_data["y"]
    velocity_magnitude = np.linalg.norm(U, axis=1)

    scheme = config["algorithm"]["convection_discretization"]
    mesh_type, resolution = config["domain"]["mesh"]
    Re = config["physical_properties"]["reynolds_number"]
    n_cells = len(U)

    out = lambda name: os.path.join(plots_dir, f"{name}.pdf")

    sim_id = meta["Simulation id"]
    print(sim_id)

    if args.all:
        # Flow fields (combined)
        plot_fields_single_row(x, y, U, velocity_magnitude, p, sim_id=sim_id, output_path=out("flow_fields"), experiment=experiment, Re=Re)
        # Individual field plots
        save_individual_field_plots(x, y, U, velocity_magnitude, p, experiment, Re, sim_id, results_dir)
        
        # Residuals (combined and individual)
        plot_residuals(res, out("residual_history"), sim_id=sim_id)

        try:
            u_res = np.load(os.path.join(results_dir, "u_residual.npy"))
            v_res = np.load(os.path.join(results_dir, "v_residual.npy"))
            cont_res = np.load(os.path.join(results_dir, "continuity_field.npy"))
            plot_residual_fields(x, y, u_res, v_res, cont_res, out("residual_fields"), sim_id=sim_id, experiment=experiment, Re=Re)
            save_individual_residual_plots(x, y, u_res, v_res, cont_res, experiment, Re, sim_id, results_dir)
        except FileNotFoundError:
            print("Residual field files not found, skipping their plots.")

        # Ghia plot: check config for experiment name
        if 'lidDrivenCavity' in experiment:
            ghia_comparison(x, y, U, Re, n_cells, scheme, mesh_type, out("ghia_comparison"), sim_id=sim_id)
        # Poiseuille verification for channel flow
        elif experiment == 'channelFlow':
            poiseuille_verification(x, y, U, p, Re, out("poiseuille_verification"), sim_id=sim_id)
        
        elif 'cylinderFlow' in experiment:
            print("Running force calculation for cylinder flow...")
            # Recreate mesh to get geometric info
            experiment_id = config.get("experiment", "unknown")
            mesh_type, resolution = config["domain"]["mesh"]
            
            # Fix: Use absolute paths consistently
            mesh_file = os.path.abspath(os.path.join(
                workspace_dir,
                "meshing", "experiments", experiment_id,
                "structuredUniform" if "uniform" in mesh_type else "unstructured",
                resolution,
                f"{experiment_id}_{mesh_type}_{resolution}.msh"
            ))
            bc_file = os.path.abspath(os.path.join(workspace_dir, config["domain"]["boundary_conditions"]))
            
            print(f"Loading mesh from: {mesh_file}")
            print(f"Using BC file: {bc_file}")
            
            mesh = load_mesh(mesh_file, bc_file)
            
            # Get physical properties from config
            rho = config["physical_properties"]["rho"]
            # Compute mean inlet velocity directly from converged field to
            # ensure consistency with simulation setup. We integrate the
            # normal velocity over all inlet faces and divide by the total
            # inlet height (sum of face lengths) – this matches the
            # definition in Schäfer & Turek.
            inlet_faces = [int(f) for f in mesh.boundary_faces if mesh.boundary_types[f,0] == 2]  # BC_INLET = 2
            if len(inlet_faces) == 0:
                raise RuntimeError("No inlet faces detected for mean-velocity calculation.")

            face_lengths = np.linalg.norm(mesh.vector_S_f[inlet_faces], axis=1)
            owner_ids = mesh.owner_cells[inlet_faces]
            # Normal component of velocity (dot with unit normal)
            n_vecs = mesh.vector_S_f[inlet_faces] / (face_lengths[:, None] + 1e-14)
            u_vals = U[owner_ids]
            # For the Schäfer benchmark, the inlet velocity profile is:
            # u(y) = 4*Um*y(H-y)/H^2 where Um = characteristic_velocity and H = 0.41
            # The mean velocity is Um, and the max velocity is at y = H/2
            # giving u_max = Um * 4 * (H/2)(H/2)/H^2 = Um
            U_inf = config["physical_properties"]["characteristic_velocity"]  # Mean velocity Um from config
            # Override characteristic velocity from config since we need Um
            config["physical_properties"]["characteristic_velocity"] = U_inf
            D = 0.1  # Cylinder diameter from Schäfer benchmark
            # Override characteristic length from config
            config["physical_properties"]["characteristic_length"] = D
            Re_cf = config["physical_properties"]["reynolds_number"]
            # Dynamic viscosity based on Reynolds number definition: Re = rho * U * D / mu
            mu = (rho * U_inf * D) / Re_cf
            
            # Calculate forces
            cd, cl = calculate_cylinder_forces(mesh, p, U, mu, rho, U_inf, D)
            p_diff = calculate_pressure_difference(mesh, p)
            
            print(f"  Drag Coefficient (Cd): {cd:.6f}")
            print(f"  Lift Coefficient (Cl): {cl:.6f}")
            print(f"  Pressure Difference (p_diff): {p_diff:.6f}")
            
            # Save to a simple text file
            force_data = {"cd": float(cd), "cl": float(cl), "p_diff": float(p_diff)}
            with open(os.path.join(results_dir, "force_coefficients.yaml"), "w") as f:
                yaml.dump(force_data, f)
            print(f"  Saved force coefficients to force_coefficients.yaml")
            
            # Also update metadata
            if 'results' not in meta:
                meta['results'] = {}
            meta['results']['drag_coefficient'] = float(cd)
            meta['results']['lift_coefficient'] = float(cl)
            meta['results']['pressure_difference'] = float(p_diff)
            with open(metadata_path, 'w') as f:
                yaml.dump(meta, f, sort_keys=False)
            print(f"  Updated metadata.yaml with force coefficients")

        yaml_to_latex_pdf(os.path.join(results_dir, "metadata.yaml"), out("metadata"))

        # The matplotlib-only LIC plot
        lic_plot_path = out("streamlines")
        plot_velocity_with_lic(x, y, U, velocity_magnitude, experiment, Re, lic_plot_path, sim_id)

        # Note: Removed ParaView hybrid workflow - focusing on matplotlib only