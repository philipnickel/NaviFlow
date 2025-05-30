import os
import argparse
import numpy as np
import yaml
import pandas as pd
#import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
import tempfile
import shutil
from utils.plot_style import plt


# ----------------------------
# Plotting Helpers
# ----------------------------
def save_pdf(fig, path):
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {path}")

def plot_fields(x, y, U, velocity_magnitude, p, scheme, mesh_type, Re, n_cells, output_path):
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
    save_pdf(fig, output_path)

def plot_residuals(res, output_path):
    fig = plt.figure()
    plt.semilogy(res["u"], label="u")
    plt.semilogy(res["v"], label="v")
    plt.semilogy(res["cont"], label="continuity")
    plt.title("Residuals vs Iteration")
    plt.grid(True)
    plt.legend()
    save_pdf(fig, output_path)

def plot_residual_fields(x, y, u_res, v_res, cont_res, output_path):
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3)
    for idx, (res, title) in enumerate(zip(
        [np.abs(u_res), np.abs(v_res), np.abs(cont_res)],
        ["U-residual", "V-residual", "Mass Flux Imbalance"]
    )):
        ax = fig.add_subplot(gs[0, idx])
        cf = ax.tricontourf(x, y, res, levels=50)
        fig.colorbar(cf, ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    save_pdf(fig, output_path)

def ghia_comparison(x, y, U, Re, n_cells, scheme, mesh_type, output_path):
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

    fig = plt.figure(figsize=(10, 6))
    
    # Get u-velocity at x=0.5
    centerline_mask = np.abs(x - 0.5) < 0.01
    vertical_y = y[centerline_mask]
    vertical_u = U[centerline_mask, 0]
    idx = np.argsort(vertical_y)
    
    # Get v-velocity at y=0.5
    centerline_mask = np.abs(y - 0.5) < 0.01
    horizontal_x = x[centerline_mask]
    horizontal_v = U[centerline_mask, 1]
    idx_v = np.argsort(horizontal_x)
    
    # Plot both velocities
    plt.plot(vertical_y[idx], vertical_u[idx], 'b-', label="u-velocity (x=0.5)")
    plt.plot(GHIA_RE_100["y"], GHIA_RE_100["u"], 'bo', label="Ghia u-velocity")
    plt.plot(horizontal_x[idx_v], horizontal_v[idx_v], 'r-', label="v-velocity (y=0.5)")
    plt.plot(GHIA_RE_100["x"], GHIA_RE_100["v"], 'ro', label="Ghia v-velocity")
    
    plt.title("Ghia Comparison (Re=100)")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.grid(True)
    plt.legend()
    
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

    if args.all:
        plot_fields(x, y, U, velocity_magnitude, p, scheme, mesh_type, Re, n_cells, out("flow_fields"))
        plot_residuals(res, out("residual_history"))

        u_res = np.load(os.path.join(results_dir, "u_residual.npy"))
        v_res = np.load(os.path.join(results_dir, "v_residual.npy"))
        cont_res = np.load(os.path.join(results_dir, "continuity_field.npy"))
        plot_residual_fields(x, y, u_res, v_res, cont_res, out("residual_fields"))

        if experiment == "lidDrivenCavity":
            ghia_comparison(x, y, U, Re, n_cells, scheme, mesh_type, out("ghia_comparison"))

        yaml_to_latex_pdf(os.path.join(results_dir, "metadata.yaml"), out("metadata"))