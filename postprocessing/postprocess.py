#!/usr/bin/env python3

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
# Main Entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Relative path to the config.yaml file")
    parser.add_argument("--all", action="store_true")
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
    
    # Extract experiment name from config path for compatibility
    # Assuming config is in experiments/{experiment_name}/...
    try:
        path_parts = args.config.split(os.sep)
        exp_index = path_parts.index("experiments")
        experiment = path_parts[exp_index + 2]
    except (ValueError, IndexError):
        experiment = "unknown"
    
    # Create directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)

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

        # Ghia plot: check config, not just directory name
        if experiment == 'lidDrivenCavity':
            ghia_comparison(x, y, U, Re, n_cells, scheme, mesh_type, out("ghia_comparison"), sim_id=sim_id)
        # Poiseuille verification for channel flow
        elif experiment == 'channelFlow':
            poiseuille_verification(x, y, U, p, Re, out("poiseuille_verification"), sim_id=sim_id)
        
        elif experiment == 'cylinderFlow':
            print("Running force calculation for cylinder flow...")
            # Recreate mesh to get geometric info
            experiment_id = config["tags"][0]
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