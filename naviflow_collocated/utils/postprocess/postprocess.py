import os
import argparse
import numpy as np
import yaml
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
import tempfile
import shutil
from utils.plot_style import plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import re
import matplotlib.patches as mpatches


# ----------------------------
# Plotting Helpers
# ----------------------------
def save_pdf(fig, path, also_save_in_plots=False):
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(path), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # Save the combined plot
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    print(f"Saved: {path}")
    # Optionally also save in plots directory
    if also_save_in_plots:
        plots_path = os.path.join(plots_dir, os.path.basename(path))
        with PdfPages(plots_path) as pdf:
            pdf.savefig(fig)
        print(f"Saved: {plots_path}")
    plt.close(fig)

def get_obstacle_mask_from_msh(x, y, experiment):
    """
    Use the original mesh tagging from the .msh file to assign obstacle tags to solution cells.
    For 'cylinderFlow', use a geometric mask based on known center and radius.
    Returns a boolean mask where True means obstacle cell (physical tag 5 or inside obstacle geometry).
    """
    if experiment == "cylinderFlow":
        # Cylinder center and radius from mesh generation
        center = np.array([0.2, 0.2])
        radius = 0.05
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask = dist < radius
        return mask
    # Fallback to original .msh-based logic for other experiments
    msh_file = os.path.join("meshing", "experiments", experiment, "unstructured", "medium", f"{experiment}_unstructured_medium.msh")
    try:
        with open(msh_file, 'r') as f:
            lines = f.readlines()
        # Parse $Nodes section
        node_section = lines.index('$Nodes\n')
        n_nodes = int(lines[node_section+1])
        node_lines = lines[node_section+2:node_section+2+n_nodes]
        node_coords = {}
        for line in node_lines:
            parts = line.strip().split()
            idx = int(parts[0])
            coord = tuple(map(float, parts[1:4]))
            node_coords[idx] = coord
        # Parse $Elements section
        elem_section = lines.index('$Elements\n')
        n_elems = int(lines[elem_section+1])
        elem_lines = lines[elem_section+2:elem_section+2+n_elems]
        centroids = []
        tags = []
        for line in elem_lines:
            parts = line.strip().split()
            elem_type = int(parts[1])
            if elem_type == 2:  # triangle (2D cell)
                num_tags = int(parts[2])
                physical_tag = int(parts[3])
                # Node indices are at the end
                node_ids = list(map(int, parts[3+num_tags:]))
                coords = [node_coords[nid] for nid in node_ids]
                centroid = tuple(np.mean(coords, axis=0))
                centroids.append(centroid)
                tags.append(physical_tag)
        centroids = np.array(centroids)
        tags = np.array(tags)
        # Only use x, y for centroid matching
        centroids_2d = centroids[:, :2]
        # Build KDTree for triangle centroids
        tree = cKDTree(centroids_2d)
        sol_xy = np.column_stack((x, y))
        _, idx = tree.query(sol_xy)
        tags_for_solution = tags[idx]
        return tags_for_solution == 5
    except Exception as e:
        print(f"Warning: Could not robustly detect obstacle cells from .msh: {e}")
        return np.zeros_like(x, dtype=bool)

def plot_fields(x, y, U, velocity_magnitude, p, scheme, mesh_type, Re, n_cells, output_path, sim_id=None):
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

def plot_fields_single_row(x, y, U, velocity_magnitude, p, sim_id=None, output_path=None, experiment=None, Re=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Velocity Magnitude", "Pressure", "Streamlines"]

    # Robust obstacle mask from .msh
    obstacle_mask = get_obstacle_mask_from_msh(x, y, experiment)

    # 1. velocity magnitude
    cf1 = axes[0].tricontourf(x, y, velocity_magnitude, levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        axes[0].tricontourf(x[obstacle_mask], y[obstacle_mask], velocity_magnitude[obstacle_mask], levels=1, colors='gray', alpha=0.5)
    fig.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.1)
    axes[0].set_title(titles[0])
    axes[0].set_aspect('equal', 'box')

    # 2. pressure
    cf2 = axes[1].tricontourf(x, y, p, levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        axes[1].tricontourf(x[obstacle_mask], y[obstacle_mask], p[obstacle_mask], levels=1, colors='gray', alpha=0.5)
    fig.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.1)
    axes[1].set_title(titles[1])
    axes[1].set_aspect('equal', 'box')

    # 3. streamlines only
    try:
        n_grid = 100
        xi = np.linspace(np.min(x), np.max(x), n_grid)
        yi = np.linspace(np.min(y), np.max(y), n_grid)
        Xg, Yg = np.meshgrid(xi, yi)
        Ug = griddata((x, y), U[:, 0], (Xg, Yg), method='linear')
        Vg = griddata((x, y), U[:, 1], (Xg, Yg), method='linear')
        # Mask obstacle region in streamline generation
        if experiment == "cylinderFlow":
            center = np.array([0.2, 0.2])
            radius = 0.05
            dist = np.sqrt((Xg - center[0])**2 + (Yg - center[1])**2)
            mask = dist < radius
            Ug[mask] = np.nan
            Vg[mask] = np.nan
        axes[2].streamplot(xi, yi, Ug, Vg, color='tab:blue', density=4.0, linewidth=0.2, arrowsize=0.2)
    except Exception as e:
        print(f"Streamline plotting failed: {e}")
    axes[2].set_title(titles[2])
    axes[2].set_aspect('equal', 'box')

    # Ensure all subplots have the same x and y limits
    xlim = (np.min(x), np.max(x))
    ylim = (np.min(y), np.max(y))
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Overlay and fill obstacle boundary for cylinderFlow
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        for ax in axes:
            circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
            ax.add_patch(circle)

    # Add experiment name and Reynolds number to suptitle
    suptitle = "Flow Fields"
    if experiment is not None:
        suptitle += f" | {experiment}"
    if Re is not None:
        suptitle += f" | Re={Re}"
    fig.suptitle(suptitle)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if sim_id is not None:
        fig.text(0.5, 0.01, f"Simulation ID: {sim_id}", ha='center', va='bottom', fontsize=6, color='gray', alpha=0.7)
    if output_path:
        save_pdf(fig, output_path, also_save_in_plots=True)
    else:
        plt.show()

def plot_residuals(res, output_path, sim_id=None):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    # Slice arrays to exclude first and last iteration
    ax.semilogy(res["u"][3:-3], label="u-momentum", color='tab:blue', linewidth=2)
    ax.semilogy(res["v"][3:-3], label="v-momentum", color='tab:orange', linewidth=2)
    ax.semilogy(res["cont"][3:-3], label="continuity", color='tab:green', linewidth=2)
    title = "Residual History"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Residual", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=12, loc='upper right', frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    fig.tight_layout(pad=0.1)
    save_pdf(fig, output_path, also_save_in_plots=True)

def plot_residual_fields(x, y, u_res, v_res, cont_res, output_path, sim_id=None, experiment=None, Re=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["U Residual", "V Residual", "Continuity Residual"]
    colormap = "viridis"  # perceptually uniform, good for magnitude fields

    # 1. U residual
    cf1 = axes[0].tricontourf(x, y, np.abs(u_res), levels=50, cmap=colormap)
    fig.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.1)
    axes[0].set_title(titles[0])
    axes[0].set_aspect('equal', 'box')

    # 2. V residual
    cf2 = axes[1].tricontourf(x, y, np.abs(v_res), levels=50, cmap=colormap)
    fig.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.1)
    axes[1].set_title(titles[1])
    axes[1].set_aspect('equal', 'box')

    # 3. Continuity residual
    cf3 = axes[2].tricontourf(x, y, np.abs(cont_res), levels=50, cmap=colormap)
    fig.colorbar(cf3, ax=axes[2], orientation='horizontal', pad=0.1)
    axes[2].set_title(titles[2])
    axes[2].set_aspect('equal', 'box')

    # Overlay and fill obstacle boundary for cylinderFlow
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        for ax in axes:
            circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
            ax.add_patch(circle)

    suptitle = "Residual Fields"
    if experiment is not None:
        suptitle += f" | {experiment}"
    if Re is not None:
        suptitle += f" | Re={Re}"
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if sim_id is not None:
        fig.text(0.5, 0.01, f"Simulation ID: {sim_id}", ha='center', va='bottom', fontsize=6, color='gray', alpha=0.7)
    save_pdf(fig, output_path, also_save_in_plots=True)

def ghia_comparison(x, y, U, Re, n_cells, scheme, mesh_type, output_path, sim_id=None):
    if Re != 100:
        print("Ghia comparison only supported for Re=100")
        return

    GHIA_RE_100 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                      0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, 
                      -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 
                      0.10091, 0.09233, 0.00000]),
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                      0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
        'u': np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, 
                      -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 
                      0.78871, 1.00000])
    }

    # centerline extraction using griddata
    points = np.column_stack((x, y))
    unique_y = np.unique(y)
    unique_x = np.unique(x)
    
    # u-velocity at x=0.5 (vertical centerline)
    u_centerline = griddata(
        points=points,
        values=U[:, 0],
        xi=np.column_stack((np.full_like(unique_y, 0.5), unique_y)),
        method='linear'
    )
    # v-velocity at y=0.5 (horizontal centerline)
    v_centerline = griddata(
        points=points,
        values=U[:, 1],
        xi=np.column_stack((unique_x, np.full_like(unique_x, 0.5))),
        method='linear'
    )

    fig = plt.figure(figsize=(10, 6))
    plt.plot(unique_y, u_centerline, '-', color='tab:blue', label="u-velocity (x=0.5)")
    plt.plot(GHIA_RE_100["y"], GHIA_RE_100["u"], 'o', color='tab:blue', label="Ghia u-velocity")
    plt.plot(unique_x, v_centerline, '-', color='tab:red', label="v-velocity (y=0.5)")
    plt.plot(GHIA_RE_100["x"], GHIA_RE_100["v"], 'o', color='tab:red', label="Ghia v-velocity")
    plt.title(f"Ghia Comparison (Re={Re})")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.grid(True)
    plt.legend()
    if sim_id is not None:
        fig = plt.gcf()
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

def yaml_to_latex_pdf(yaml_path, output_pdf_path):
    # Load YAML and flatten
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    flat_data = flatten_dict(data)

    # Create DataFrame and convert to markdown
    df = pd.DataFrame(flat_data.items(), columns=["Parameter", "Value/Setting"])
    md_content = df.to_markdown(index=False)
    
    # Create temporary markdown file
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w', delete=False) as tmp:
        tmp.write(md_content)
        tmp_path = tmp.name

    # Create temporary PDF path for initial generation
    temp_pdf = output_pdf_path + ".temp.pdf"

    # Run pandoc with options to:
    # - Make table fill page width
    # - Remove page numbers
    # - Use full page width
    subprocess.run([
        "pandoc",
        tmp_path,
        "-o", temp_pdf,
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=0.2in",
        "-V", "pagenumbers=false",
        "-V", "tables:width=1.0\\textwidth"
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Crop the PDF to remove excess whitespace but keep some margin
    subprocess.run([
        "pdfcrop",
        "--margins", "25 25 25 25",  # left top right bottom margins in points
        temp_pdf,
        output_pdf_path
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Clean up temporary files
    os.unlink(tmp_path)
    os.unlink(temp_pdf)
    
    print(f"PDF saved: {output_pdf_path}")

def poiseuille_verification(x, y, U, p, Re, output_path, sim_id=None):
    """
    Verify channel flow solution against analytical solution for constant inlet velocity.
    
    Parameters:
    -----------
    x, y : ndarray
        Cell center coordinates
    U : ndarray
        Velocity field (n_cells, 2)
    p : ndarray
        Pressure field
    Re : float
        Reynolds number
    output_path : str
        Path to save the verification plot
    sim_id : str, optional
        Simulation ID for plot annotation
    """
    # Get channel parameters from config
    with open(os.path.join("experiments", "channelFlow", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    u_inlet = config["physical_properties"]["characteristic_velocity"]  # Inlet velocity
    rho = config["physical_properties"]["rho"]
    
    # Calculate channel height from numerical data
    H = np.max(y) - np.min(y)  # Total channel height
    h = H/2  # Half height

    # Calculate channel length from domain coordinates
    L = 5.0
    
    # Create figure with 1x2 subplots
    fig = plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(1, 2)
    
    # 1. Velocity Profile Plot (at x=L/2)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract numerical solution at x=L/2 using griddata
    points = np.column_stack((x, y))
    unique_y = np.unique(y)
    x_center = L/2
    
    # Get u-velocity at x=L/2 using griddata
    xi = np.column_stack((np.full_like(unique_y, x_center), unique_y))
    u_numerical = griddata(
        points=points,
        values=U[:, 0],
        xi=xi,
        method='linear'
    )
    
    # Calculate mean inlet velocity at the leftmost x-coordinate
    x_inlet = np.min(x)
    inlet_mask = np.isclose(x, x_inlet, atol=1e-6)
    u_mean_inlet = np.mean(U[inlet_mask, 0])
    
    # Analytical solution for fully developed flow
    # u(y) = (3/2) * u_inlet * (1 - (y/h)^2)
    r = np.linspace(-0.5, 0.5, len(u_numerical))
    u_analytical = 1.5 * u_mean_inlet * (1 - (r**2/0.5**2))  # Use actual measured inlet velocity
    y = np.linspace(0, 1, len(u_analytical))
    
    
    # Plot both solutions
    ax1.plot(y, u_numerical, 'o-', color='tab:blue', label="Numerical", markersize=2, alpha=0.6)
    ax1.plot(y, u_analytical, '--', color='tab:orange', label="Analytical", linewidth=2)
    
    ax1.set_title(f"Velocity Profile at x=L/2", fontsize=12, pad=10)
    ax1.set_xlabel("y/h", fontsize=10)
    ax1.set_ylabel("u/u_inlet", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.tick_params(axis='both', which='major', labelsize=9)
    
    # 2. Pressure Drop Plot
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get unique x locations and sort them
    unique_x = np.sort(np.unique(x))
    
    # Calculate average pressure at each x location
    p_avg = np.array([np.mean(p[np.isclose(x, x_loc)]) for x_loc in unique_x])
    
    # Normalize x coordinates to [0,1]
    x_norm = (unique_x - np.min(x)) / (np.max(x) - np.min(x))
    
    mu = rho * u_mean_inlet * 1.0 / Re
    # Analytical pressure drop (dp/dx = -8μu_inlet/h²)
    dp_dx_analytical = -8 * mu * u_mean_inlet / (h**2)
    p_analytical = dp_dx_analytical * (unique_x - np.min(x))
    
    # Calculate pressure gradient error
    p_grad_numerical = np.polyfit(unique_x, p_avg, 1)[0]
    p_grad_error = np.abs((p_grad_numerical - dp_dx_analytical) / dp_dx_analytical) * 100
    
    # Plot pressure drop
    ax2.plot(x_norm, p_avg, 'o-', color='tab:blue', label="Numerical", markersize=2, alpha=0.6)
    ax2.plot(x_norm, p_analytical, '--', color='tab:orange', label="Analytical", linewidth=2)
    
    ax2.set_title("Pressure Drop Along Channel", fontsize=12, pad=10)
    ax2.set_xlabel("x/L", fontsize=10)
    ax2.set_ylabel("Pressure", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.tick_params(axis='both', which='major', labelsize=9)
    
    # Add flow parameters to plot
    param_text = (
        f"Channel Parameters:\n"
        f"Height (H): {H:.3f} m\n"
        f"Length (L): {L:.3f} m\n"
        f"Inlet velocity: {u_inlet:.3f} m/s\n"
        f"Reynolds number: {Re}"
    )
    fig.text(0.02, 0.02, param_text, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
             fontsize=9)
    
    # Add overall title with simulation ID
    title = "Hagen–Poiseuille Verification"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_pdf(fig, output_path)
    print(f"Channel flow verification saved to {output_path}")
    
    # Print error metrics
    print("\nChannel Flow Verification Results:")
    print(f"Pressure Gradient Error: {p_grad_error:.2f}%")

def plot_streamlines(x, y, U, output_path, experiment=None, Re=None, sim_id=None):
    """Create a standalone streamlines plot."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Create grid for streamlines
    n_grid = 100
    xi = np.linspace(np.min(x), np.max(x), n_grid)
    yi = np.linspace(np.min(y), np.max(y), n_grid)
    Xg, Yg = np.meshgrid(xi, yi)
    Ug = griddata((x, y), U[:, 0], (Xg, Yg), method='linear')
    Vg = griddata((x, y), U[:, 1], (Xg, Yg), method='linear')
    
    # Mask obstacle region in streamline generation for cylinderFlow
    if experiment == "cylinderFlow":
        center = np.array([0.2, 0.2])
        radius = 0.05
        dist = np.sqrt((Xg - center[0])**2 + (Yg - center[1])**2)
        mask = dist < radius
        Ug[mask] = np.nan
        Vg[mask] = np.nan
        
        # Add cylinder
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    
    # Plot streamlines
    ax.streamplot(xi, yi, Ug, Vg, color='tab:blue', density=4.0, linewidth=0.2, arrowsize=0.2)
    
    # Set title and labels
    title = "Streamlines"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', 'box')
    
    fig.tight_layout(pad=0.1)
    save_pdf(fig, output_path)

def save_individual_field_plots(x, y, U, velocity_magnitude, p, experiment, Re, sim_id, results_dir):
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    obstacle_mask = get_obstacle_mask_from_msh(x, y, experiment)

    # u-velocity
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.tricontourf(x, y, U[:, 0], levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        ax.tricontourf(x[obstacle_mask], y[obstacle_mask], U[obstacle_mask, 0], levels=1, colors='gray', alpha=0.5)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
    title = "u-velocity"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    fig.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "u_velocity.pdf"))
    plt.close(fig)

    # v-velocity
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.tricontourf(x, y, U[:, 1], levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        ax.tricontourf(x[obstacle_mask], y[obstacle_mask], U[obstacle_mask, 1], levels=1, colors='gray', alpha=0.5)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
    title = "v-velocity"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    fig.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "v_velocity.pdf"))
    plt.close(fig)

    # velocity magnitude
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.tricontourf(x, y, velocity_magnitude, levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        ax.tricontourf(x[obstacle_mask], y[obstacle_mask], velocity_magnitude[obstacle_mask], levels=1, colors='gray', alpha=0.5)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
    title = "Velocity Magnitude"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    fig.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "velocity_magnitude.pdf"))
    plt.close(fig)

    # pressure
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.tricontourf(x, y, p, levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        ax.tricontourf(x[obstacle_mask], y[obstacle_mask], p[obstacle_mask], levels=1, colors='gray', alpha=0.5)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
    title = "Pressure"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    fig.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "pressure.pdf"))
    plt.close(fig)

    # streamlines only
    plot_streamlines(x, y, U, os.path.join(plots_dir, "streamlines.pdf"), experiment=experiment, Re=Re, sim_id=sim_id)

    # Overlay obstacle mask on all individual plots for cylinderFlow
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        for ax in [fig.axes[0] for fig in plt.get_fignums()]:
            circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
            ax.add_patch(circle)

def save_individual_residual_plots(x, y, u_res, v_res, cont_res, experiment, Re, sim_id, results_dir):
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    obstacle_mask = get_obstacle_mask_from_msh(x, y, experiment)

    # U residual
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.tricontourf(x, y, np.abs(u_res), levels=50, cmap='viridis')
    if np.any(obstacle_mask):
        ax.tricontourf(x[obstacle_mask], y[obstacle_mask], np.abs(u_res[obstacle_mask]), levels=1, colors='gray', alpha=0.5)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
    title = "U Residual"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    fig.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "u_residual.pdf"))
    plt.close(fig)

    # V residual
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.tricontourf(x, y, np.abs(v_res), levels=50, cmap='viridis')
    if np.any(obstacle_mask):
        ax.tricontourf(x[obstacle_mask], y[obstacle_mask], np.abs(v_res[obstacle_mask]), levels=1, colors='gray', alpha=0.5)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
    title = "V Residual"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    fig.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "v_residual.pdf"))
    plt.close(fig)

    # Continuity residual
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.tricontourf(x, y, np.abs(cont_res), levels=50, cmap='viridis')
    if np.any(obstacle_mask):
        ax.tricontourf(x[obstacle_mask], y[obstacle_mask], np.abs(cont_res[obstacle_mask]), levels=1, colors='gray', alpha=0.5)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
    title = "Continuity Residual"
    if sim_id is not None:
        title += f" | Simulation ID: {sim_id}"
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    if experiment == "cylinderFlow":
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    fig.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "continuity_residual.pdf"))
    plt.close(fig)

def plot_force_coefficients(cd_history, cl_history, output_path, sim_id=None, experiment=None, Re=None):
    """Create plots for drag and lift coefficient history."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot drag coefficient
    ax1.plot(cd_history, 'b-', linewidth=2)
    ax1.set_title("Drag Coefficient (Cd) History")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cd")
    ax1.grid(True, alpha=0.3)
    
    # Plot lift coefficient
    ax2.plot(cl_history, 'r-', linewidth=2)
    ax2.set_title("Lift Coefficient (Cl) History")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cl")
    ax2.grid(True, alpha=0.3)
    
    # Add experiment info to suptitle
    suptitle = "Force Coefficients History"
    if experiment is not None:
        suptitle += f" | {experiment}"
    if Re is not None:
        suptitle += f" | Re={Re}"
    fig.suptitle(suptitle)
    
    if sim_id is not None:
        fig.text(0.5, 0.01, f"Simulation ID: {sim_id}", ha='center', va='bottom', fontsize=6, color='gray', alpha=0.7)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_pdf(fig, output_path, also_save_in_plots=True)

# ----------------------------
# Main Entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    experiment = args.experiment
    experiment_path = os.path.join("experiments", experiment)
    results_dir = os.path.join(experiment_path, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    U = np.load(os.path.join(results_dir, "U_final.npy"))
    p = np.load(os.path.join(results_dir, "p_final.npy"))
    res = np.load(os.path.join(results_dir, "residuals.npz"))
    cell_data = np.load(os.path.join(results_dir, "cell_centers.npz"))
    meta = yaml.safe_load(open(os.path.join(results_dir, "metadata.yaml")))
    config = yaml.safe_load(open(os.path.join(experiment_path, "config.yaml")))

    x = cell_data["x"]
    y = cell_data["y"]
    velocity_magnitude = np.linalg.norm(U, axis=1)

    scheme = config["algorithm"]["convection_discretization"]
    mesh_type, resolution = config["domain"]["mesh"]
    Re = config["physical_properties"]["reynolds_number"]
    n_cells = len(U)

    out = lambda name: os.path.join(results_dir, f"{name}.pdf")

    sim_id = meta["Simulation id"]
    print(sim_id)

    if args.all:
        # Flow fields (combined)
        plot_fields_single_row(x, y, U, velocity_magnitude, p, sim_id=sim_id, output_path=out("flow_fields"), experiment=experiment, Re=Re)
        # Individual field plots
        save_individual_field_plots(x, y, U, velocity_magnitude, p, experiment, Re, sim_id, results_dir)
        
        # Residuals (combined and individual)
        plot_residuals(res, out("residual_history"), sim_id=sim_id)

        u_res = np.load(os.path.join(results_dir, "u_residual.npy"))
        v_res = np.load(os.path.join(results_dir, "v_residual.npy"))
        cont_res = np.load(os.path.join(results_dir, "continuity_field.npy"))
        plot_residual_fields(x, y, u_res, v_res, cont_res, out("residual_fields"), sim_id=sim_id, experiment=experiment, Re=Re)
        save_individual_residual_plots(x, y, u_res, v_res, cont_res, experiment, Re, sim_id, results_dir)

        # Ghia plot: check config, not just directory name
        if config.get('experiment', None) == 'lidDrivenCavity':
            ghia_comparison(x, y, U, Re, n_cells, scheme, mesh_type, out("ghia_comparison"), sim_id=sim_id)
        # Poiseuille verification for channel flow
        elif config.get('experiment', None) == 'channelFlow':
            poiseuille_verification(x, y, U, p, Re, out("poiseuille_verification"), sim_id=sim_id)

        # Plot force coefficients if available
        try:
            force_coeffs = np.load(os.path.join(results_dir, "force_coefficients.npz"))
            plot_force_coefficients(
                force_coeffs["cd"], 
                force_coeffs["cl"], 
                out("force_coefficients"),
                sim_id=sim_id,
                experiment=experiment,
                Re=Re
            )
        except FileNotFoundError:
            print("Force coefficients history not found, skipping plot.")

        yaml_to_latex_pdf(os.path.join(results_dir, "metadata.yaml"), out("metadata"))