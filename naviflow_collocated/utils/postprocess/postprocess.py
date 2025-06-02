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


# ----------------------------
# Plotting Helpers
# ----------------------------
def save_pdf(fig, path):
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {path}")

def plot_fields(x, y, U, velocity_magnitude, p, scheme, mesh_type, Re, n_cells, output_path, sim_id=None):
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # U-velocity
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.tricontourf(x, y, U[:, 0], levels=50, cmap='coolwarm')
    fig.colorbar(cf1, ax=ax1)
    ax1.set_title("U-velocity")
    ax1.set_aspect("equal", "box")
    
    # V-velocity
    ax2 = fig.add_subplot(gs[0, 1])
    cf2 = ax2.tricontourf(x, y, U[:, 1], levels=50, cmap='coolwarm')
    fig.colorbar(cf2, ax=ax2)
    ax2.set_title("V-velocity")
    ax2.set_aspect("equal", "box")
    
    # Velocity Magnitude
    ax3 = fig.add_subplot(gs[1, 0])
    cf3 = ax3.tricontourf(x, y, velocity_magnitude, levels=50, cmap='coolwarm')
    fig.colorbar(cf3, ax=ax3)
    ax3.set_title("Velocity Magnitude")
    ax3.set_aspect("equal", "box")
    
    # Pressure
    ax4 = fig.add_subplot(gs[1, 1])
    cf4 = ax4.tricontourf(x, y, p, levels=50, cmap='coolwarm')
    fig.colorbar(cf4, ax=ax4)
    ax4.set_title("Pressure")
    ax4.set_aspect("equal", "box")
    
    fig.suptitle(f"Flow Field | Re={Re}, {scheme}, {mesh_type}")
    if sim_id is not None:
        fig.text(0.5, 0.01, f"Simulation ID: {sim_id}", ha='center', va='bottom', fontsize=6, color='gray', alpha=0.7)
    save_pdf(fig, output_path)

def plot_fields_single_row(x, y, U, velocity_magnitude, p, sim_id=None, output_path=None, experiment=None, Re=None):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    titles = ["u-velocity", "v-velocity", "Velocity Magnitude", "Pressure + Streamlines"]

    # 1. u-velocity
    cf1 = axes[0].tricontourf(x, y, U[:, 0], levels=50, cmap='coolwarm')
    fig.colorbar(cf1, ax=axes[0])
    axes[0].set_title(titles[0])
    axes[0].set_aspect('equal', 'box')

    # 2. v-velocity
    cf2 = axes[1].tricontourf(x, y, U[:, 1], levels=50, cmap='coolwarm')
    fig.colorbar(cf2, ax=axes[1])
    axes[1].set_title(titles[1])
    axes[1].set_aspect('equal', 'box')

    # 3. velocity magnitude
    cf3 = axes[2].tricontourf(x, y, velocity_magnitude, levels=50, cmap='coolwarm')
    fig.colorbar(cf3, ax=axes[2])
    axes[2].set_title(titles[2])
    axes[2].set_aspect('equal', 'box')

    # 4. pressure with streamlines
    cf4 = axes[3].tricontourf(x, y, p, levels=50, cmap='coolwarm')
    fig.colorbar(cf4, ax=axes[3])
    axes[3].set_title(titles[3])
    axes[3].set_aspect('equal', 'box')

    # Streamlines overlay
    try:
        n_grid = 100
        xi = np.linspace(np.min(x), np.max(x), n_grid)
        yi = np.linspace(np.min(y), np.max(y), n_grid)
        Xg, Yg = np.meshgrid(xi, yi)
        Ug = griddata((x, y), U[:, 0], (Xg, Yg), method='linear')
        Vg = griddata((x, y), U[:, 1], (Xg, Yg), method='linear')
        axes[3].streamplot(xi, yi, Ug, Vg, color='gray', density=1.2, linewidth=0.5)
    except Exception as e:
        print(f"Streamline plotting failed: {e}")

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
        save_pdf(fig, output_path)
    else:
        plt.show()

def plot_residuals(res, output_path, sim_id=None):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(res["u"], label="u-momentum", color='tab:blue', linewidth=2)
    ax.semilogy(res["v"], label="v-momentum", color='tab:orange', linewidth=2)
    ax.semilogy(res["cont"], label="continuity", color='tab:green', linewidth=2)
    ax.set_title("Residual History", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Residual", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=12, loc='upper right', frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if sim_id is not None:
        fig.text(0.5, 0.01, f"Simulation ID: {sim_id}", ha='center', va='bottom', fontsize=6, color='gray', alpha=0.7)
    save_pdf(fig, output_path)

def plot_residual_fields(x, y, u_res, v_res, cont_res, output_path, sim_id=None, experiment=None, Re=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["U Residual", "V Residual", "Continuity Residual"]
    colormap = "viridis"  # perceptually uniform, good for magnitude fields

    # 1. U residual
    cf1 = axes[0].tricontourf(x, y, np.abs(u_res), levels=50, cmap=colormap)
    fig.colorbar(cf1, ax=axes[0])
    axes[0].set_title(titles[0])
    axes[0].set_aspect('equal', 'box')

    # 2. V residual
    cf2 = axes[1].tricontourf(x, y, np.abs(v_res), levels=50, cmap=colormap)
    fig.colorbar(cf2, ax=axes[1])
    axes[1].set_title(titles[1])
    axes[1].set_aspect('equal', 'box')

    # 3. Continuity residual
    cf3 = axes[2].tricontourf(x, y, np.abs(cont_res), levels=50, cmap=colormap)
    fig.colorbar(cf3, ax=axes[2])
    axes[2].set_title(titles[2])
    axes[2].set_aspect('equal', 'box')

    suptitle = "Residual Fields"
    if experiment is not None:
        suptitle += f" | {experiment}"
    if Re is not None:
        suptitle += f" | Re={Re}"
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if sim_id is not None:
        fig.text(0.5, 0.01, f"Simulation ID: {sim_id}", ha='center', va='bottom', fontsize=6, color='gray', alpha=0.7)
    save_pdf(fig, output_path)

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
    H = config["physical_properties"]["characteristic_length"]  # Channel height
    h = H/2  # Half height
    rho = config["physical_properties"]["rho"]
    mu = rho * u_inlet * h / Re

    # Calculate channel length from domain coordinates
    L = 5.0
    
    # Create figure with 1x2 subplots
    fig = plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(1, 2)
    
    # 1. Velocity Profile Plot (at x=L/2)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract numerical solution at x=L/2
    points = np.column_stack((x, y))
    unique_y = np.unique(y)
    x_center = (np.max(x) + np.min(x)) / 2
    
    u_numerical = griddata(
        points=points,
        values=U[:, 0],
        xi=np.column_stack((np.full_like(unique_y, x_center), unique_y)),
        method='linear'
    )
    
    # Remove any NaN values
    mask = ~np.isnan(u_numerical)
    y_valid = unique_y[mask]
    u_numerical = u_numerical[mask]
    
    # Normalize y coordinates to [-1,1]
    y_norm = (y_valid)/h -1
    
    # Analytical solution for fully developed flow with constant inlet
    u_analytical = 1.5 * u_inlet * (1 - y_norm**2)  # Parabolic profile with inlet velocity
    
    # Plot both solutions
    ax1.plot(y_norm, u_numerical, 'o-', color='tab:blue', label="Numerical", markersize=4, alpha=0.6)
    ax1.plot(y_norm, u_analytical, '--', color='tab:red', label="Analytical", linewidth=2)
    
    ax1.set_title(f"Velocity Profile at x=L/2 (Re={Re})")
    ax1.set_xlabel("y/h [-]")
    ax1.set_ylabel("u/u_inlet [-]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Pressure Drop Plot
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get unique x locations and sort them
    unique_x = np.sort(np.unique(x))
    
    # Calculate average pressure at each x location
    p_avg = np.array([np.mean(p[np.isclose(x, x_loc)]) for x_loc in unique_x])
    
    # Normalize x coordinates to [0,1]
    x_norm = (unique_x - np.min(x)) / (np.max(x) - np.min(x))
    
    # Analytical pressure drop (dp/dx = -8μu_inlet/h²)
    dp_dx_analytical = -8 * mu * u_inlet / (h**2)
    p_analytical = dp_dx_analytical * (unique_x - np.min(x))
    
    # Plot pressure drop
    ax2.plot(x_norm, p_avg, 'o-', color='tab:blue', label="Numerical", markersize=4, alpha=0.6)
    ax2.plot(x_norm, p_analytical, '--', color='tab:red', label="Analytical", linewidth=2)
    
    ax2.set_title("Pressure Drop Along Channel")
    ax2.set_xlabel("x/L [-]")
    ax2.set_ylabel("Pressure [Pa]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add flow parameters to plot
    param_text = (
        f"Channel height (H): {H:.3f} m\n"
        f"Channel length (L): {L:.3f} m\n"
        f"Inlet velocity: {u_inlet:.3f} m/s\n"
        f"Reynolds number: {Re}\n"
    )
    fig.text(0.02, 0.02, param_text, bbox=dict(facecolor='white', alpha=0.8))
    
    if sim_id is not None:
        fig.text(0.5, 0.01, f"Simulation ID: {sim_id}", ha='center', va='bottom', fontsize=6, color='gray', alpha=0.7)
    
    plt.tight_layout()
    save_pdf(fig, output_path)
    print(f"Channel flow verification saved to {output_path}")
    
    # Print error metrics
    u_error = np.abs(u_numerical - u_analytical)
    l2_error = np.sqrt(np.mean(u_error**2))
    linf_error = np.max(u_error)
    p_grad_numerical = np.polyfit(unique_x, p_avg, 1)[0]
    p_grad_error = np.abs((p_grad_numerical - dp_dx_analytical) / dp_dx_analytical) * 100
    
    print("\nChannel Flow Verification Results:")
    print(f"L2 Error: {l2_error:.2e}")
    print(f"Linf Error: {linf_error:.2e}")
    print(f"Pressure Gradient Error: {p_grad_error:.2f}%")

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
        plot_fields_single_row(x, y, U, velocity_magnitude, p, sim_id=sim_id, output_path=out("flow_fields"), experiment=experiment, Re=Re)
        plot_residuals(res, out("residual_history"), sim_id=sim_id)

        u_res = np.load(os.path.join(results_dir, "u_residual.npy"))
        v_res = np.load(os.path.join(results_dir, "v_residual.npy"))
        cont_res = np.load(os.path.join(results_dir, "continuity_field.npy"))
        plot_residual_fields(x, y, u_res, v_res, cont_res, out("residual_fields"), sim_id=sim_id, experiment=experiment, Re=Re)

        # Ghia plot: check config, not just directory name
        if config.get('experiment', None) == 'lidDrivenCavity':
            ghia_comparison(x, y, U, Re, n_cells, scheme, mesh_type, out("ghia_comparison"), sim_id=sim_id)
        # Poiseuille verification for channel flow
        elif config.get('experiment', None) == 'channelFlow':
            poiseuille_verification(x, y, U, p, Re, out("poiseuille_verification"), sim_id=sim_id)

        yaml_to_latex_pdf(os.path.join(results_dir, "metadata.yaml"), out("metadata"))