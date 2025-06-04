import os
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

from naviflow_collocated.utils.postprocess.plotting import (
    plot_fields_single_row,
    plot_residuals,
    plot_residual_fields,
    plot_streamlines,
    plot_force_coefficients,
    save_individual_field_plots,
    save_individual_residual_plots
)
from naviflow_collocated.utils.postprocess.verification import ghia_comparison, poiseuille_verification
from naviflow_collocated.utils.postprocess.metadata import yaml_to_latex_pdf
from naviflow_collocated.utils.postprocess.utils import save_pdf, get_obstacle_mask_from_msh, flatten_dict

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
    
    if base_experiment == "cylinderFlow":
        # Cylinder center and radius from mesh generation
        center = np.array([0.2, 0.2])
        radius = 0.05
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask = dist < radius
        return mask
    # For all other experiments, return a mask of all False (no obstacles)
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
# Main Entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    # Get the workspace directory
    workspace_dir = os.getcwd()
    
    # Get the experiment path and normalize it
    experiment = args.experiment
    # Remove experiments/ prefix if it exists
    if experiment.startswith("experiments/"):
        experiment = experiment[11:]  # Remove "experiments/"
    # Remove /debugging suffix if it exists
    if experiment.endswith("/debugging"):
        experiment = experiment[:-10]  # Remove "/debugging"
    
    # Construct absolute paths
    experiment_path = os.path.join(workspace_dir, "experiments", experiment)
    config_path = os.path.join(experiment_path, "config.yaml")
    
    # Load config
    config = yaml.safe_load(open(config_path))
    
    # Use the config file's directory as the base directory
    base_dir = os.path.dirname(config_path)
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    
    # Create directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)

    U = np.load(os.path.join(results_dir, "U_final.npy"))
    p = np.load(os.path.join(results_dir, "p_final.npy"))
    res = np.load(os.path.join(results_dir, "residuals.npz"))
    cell_data = np.load(os.path.join(results_dir, "cell_centers.npz"))
    meta = yaml.safe_load(open(os.path.join(results_dir, "metadata.yaml")))

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