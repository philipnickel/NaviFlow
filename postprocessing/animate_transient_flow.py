"""
This script generates transient flow animations (GIF and MP4) from simulation results
and plots the history of drag and lift coefficients.

It requires a path to the simulation's config.yaml file. The script will automatically
find the mesh file and transient data associated with the config file.

Example usage from the project root directory:
    python postprocessing/animate_transient_flow.py experiments/Collocated/transientcylinder/higher_RE/coarse/config.yaml --framerate 30 --end-time 10.0
"""
import os
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import glob
import re
from scipy.signal import find_peaks

# Add project root to path to allow importing naviflow_collocated
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from naviflow_collocated.utils.postprocess.plot_style import plt
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.utils.postprocess.forces import calculate_cylinder_forces
import matplotlib.patches as mpatches

# Apply consistent plot style
plt.style.use(['science', 'grid'])

def get_iter_from_filename(filename):
    """Extracts iteration number from a filename."""
    match = re.search(r'_(\d+)\.npy$', os.path.basename(filename))
    return int(match.group(1)) if match else -1

def get_cylinder_obstacle_mask(x, y, experiment_name):
    """Creates a boolean mask for a cylinder obstacle based on experiment name."""
    if "cylinder" in experiment_name.lower():
        center = (0.2, 0.2)
        radius = 0.05
        return (x - center[0])**2 + (y - center[1])**2 < radius**2
    return np.zeros_like(x, dtype=bool)

def plot_force_history(time_history, cd_history, cl_history, output_dir, Re, sim_id=None):
    """Plots drag and lift coefficient history over time, styled like residual plots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    fig.suptitle(f"Force Coefficients History | Re={Re}", fontsize=16)

    # Add sim_id to the top right, similar to other plots
    if sim_id and sim_id != "N/A":
        fig.text(0.98, 0.98, f"Sim ID: {sim_id}", ha='right', va='top', fontsize=8, color='gray')

    # Drag Coefficient Plot
    ax1.plot(time_history, cd_history, color='tab:blue', linewidth=2)
    ax1.set_ylabel("Drag Coefficient ($C_d$)", fontsize=14)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(["Drag"], loc='lower right', fontsize=12, frameon=True)

    # Lift Coefficient Plot
    ax2.plot(time_history, cl_history, color='tab:orange', linewidth=2)
    ax2.set_xlabel("Time (s)", fontsize=14)
    ax2.set_ylabel("Lift Coefficient ($C_l$)", fontsize=14)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(["Lift"], loc='lower right', fontsize=12, frameon=True)
    
    # --- Annotate max values in the last oscillation period ---
    try:
        if len(time_history) > 100: # Ensure there is enough data
            cl_history_np = np.array(cl_history)
            cd_history_np = np.array(cd_history)
            time_history_np = np.array(time_history)

            # Find peaks in the lift coefficient to identify oscillations.
            # Use distance and prominence to filter out noise.
            peaks, _ = find_peaks(cl_history_np, distance=50, prominence=(np.max(cl_history_np) - np.min(cl_history_np)) * 0.1)

            if len(peaks) >= 2:
                # Define the last full oscillation period
                start_idx, end_idx = peaks[-2], peaks[-1]

                # Annotate max Lift
                cl_period = cl_history_np[start_idx:end_idx]
                max_cl_local_idx = np.argmax(cl_period)
                max_cl_global_idx = start_idx + max_cl_local_idx
                max_cl_val = cl_history_np[max_cl_global_idx]
                time_at_max_cl = time_history_np[max_cl_global_idx]
                
                ax2.annotate(f'Max $C_l$: {max_cl_val:.4f}',
                             xy=(time_at_max_cl, max_cl_val),
                             xytext=(0.7, 0.85), textcoords='axes fraction',
                             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=0.2"),
                             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))

                # Annotate max Drag (in the same period)
                cd_period = cd_history_np[start_idx:end_idx]
                max_cd_local_idx = np.argmax(cd_period)
                max_cd_global_idx = start_idx + max_cd_local_idx
                max_cd_val = cd_history_np[max_cd_global_idx]
                time_at_max_cd = time_history_np[max_cd_global_idx]

                ax1.annotate(f'Max $C_d$: {max_cd_val:.4f}',
                             xy=(time_at_max_cd, max_cd_val),
                             xytext=(0.7, 0.85), textcoords='axes fraction',
                             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=0.2"),
                             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))
    except Exception as e:
        print(f"\nWarning: Could not annotate force coefficients. Reason: {e}")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = f"force_history_Re{Re}.png"
    output_filepath = os.path.join(output_dir, output_filename)
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)
    print(f"\nForce coefficient history plot saved to: {output_filepath}")

def animate_transient_flow(config_path, framerate=20, end_time_arg=None):
    """
    Creates an animation and force plots from transient flow simulation results.
    """
    experiment_path = os.path.dirname(config_path)
    results_path = os.path.join(experiment_path, "results")
    transient_data_path = os.path.join(results_path, "transient_data")

    # --- Load Simulation ID from metadata ---
    metadata_path = os.path.join(results_path, "metadata.yaml")
    sim_id = "N/A"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
                if metadata:
                    sim_id = metadata.get("Simulation id", "N/A")
        except Exception as e:
            print(f"Warning: Could not read metadata.yaml: {e}")
    print(f"Using Simulation ID: {sim_id}")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    if not os.path.exists(transient_data_path):
        print(f"Error: Transient data path not found at {transient_data_path}")
        return

    # --- Load Config ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # --- Get Properties from Config ---
    phys_props = config.get("physical_properties", {})
    rho = phys_props.get("rho", 1.0)
    Re_x = phys_props.get("reynolds_number", "NA")
    D = phys_props.get("characteristic_length", 0.1)
    U_ref = phys_props.get("characteristic_velocity", 1.0)
    
    experiment_name = config.get("experiment", "unknown_experiment")
    # For cylinder flow, ensure D is the diameter, not the radius
    if "cylinder" in experiment_name.lower():
        # Check for a common mistake: using radius (e.g., 0.05) instead of diameter (0.1)
        if np.isclose(D, 0.05):
            print("\nWarning: 'characteristic_length' is 0.05, which appears to be the radius.")
            print("         Doubling to 0.1 (diameter) for force calculation normalization.")
            D = D * 2.0

    mu = (rho * U_ref * D) / Re_x if isinstance(Re_x, (int, float)) and Re_x > 0 else 0.01

    mesh_info = config.get("domain", {}).get("mesh", ["unstructured", "medium"])
    mesh_type, mesh_density = mesh_info[0], mesh_info[1]
    bc_config_file = config.get("domain", {}).get("boundary_conditions")
    dt = config.get("algorithm", {}).get("dt", 0.01)

    # --- Load Full Mesh Object ---
    mesh_filename = f"{experiment_name}_{mesh_type}_{mesh_density}.msh"
    mesh_path = os.path.join("meshing", "experiments", experiment_name, mesh_type, mesh_density, mesh_filename)
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return
    
    print(f"Loading mesh from: {mesh_path}")
    mesh = load_mesh(mesh_path, bc_config_file)
    x, y = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1]

    # --- Find Data Files and Time Range ---
    u_files = sorted(glob.glob(os.path.join(transient_data_path, "U", "U_*.npy")), key=get_iter_from_filename)
    p_files_map = {get_iter_from_filename(f): f for f in glob.glob(os.path.join(transient_data_path, "p", "p_*.npy"))}
    
    if not u_files:
        print("No U_*.npy files found. Cannot generate animation.")
        return
        
    available_iterations = [get_iter_from_filename(f) for f in u_files if get_iter_from_filename(f) != -1]
    
    config_final_t = config.get("algorithm", {}).get("end_time") or \
                     config.get("algorithm", {}).get("n_timesteps", 1000) * dt

    max_sim_time = max(available_iterations) * dt if available_iterations else 0
    effective_final_t = min(end_time_arg or config_final_t, max_sim_time)

    # --- Generate Output Filename ---
    output_basename = f"animation_Re{Re_x}_dt{dt}_T{effective_final_t:.2f}_fps{framerate}"
    gif_output_filename = f"{output_basename}.gif"
    mp4_output_filename = f"{output_basename}.mp4"

    # --- Select Frames for Animation ---
    u_file_map = {get_iter_from_filename(f): f for f in u_files}
    target_times = np.arange(0, effective_final_t, 1.0 / framerate)
    
    selected_iterations = sorted(list(set(
        min(available_iterations, key=lambda it: abs(it * dt - t_target))
        for t_target in target_times
    )))
    
    selected_u_files = [u_file_map[it] for it in selected_iterations]
    num_frames = len(selected_u_files)
    print(f"Selected {num_frames} frames for animation up to t={effective_final_t:.2f}s.")

    # --- Calculate Forces over the entire history ---
    cd_history, cl_history, time_history = [], [], []

    print("\nCalculating force coefficients for all available timesteps...")
    # Determine the iterations to calculate forces for
    force_iterations = [it for it in available_iterations if it * dt <= effective_final_t]
    
    for it in tqdm(force_iterations, desc="Calculating forces"):
        if it in p_files_map:
            U = np.load(u_file_map[it])
            p = np.load(p_files_map[it])
            cd, cl = calculate_cylinder_forces(mesh, p, U, mu, rho, U_ref, D)
            cd_history.append(cd)
            cl_history.append(cl)
            time_history.append(it * dt)

    # --- Generate Color Range for Animation ---
    global_v_min, global_v_max = float('inf'), float('-inf')

    print("\nDetermining color range for animation frames...")
    for u_file in tqdm(selected_u_files, desc="Analyzing animation frames"):
        U = np.load(u_file)
        velocity_magnitude = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
        global_v_min = min(global_v_min, velocity_magnitude.min())
        global_v_max = max(global_v_max, velocity_magnitude.max())
    
    print(f"Global velocity range for animation: {global_v_min:.3f} to {global_v_max:.3f}")

    # --- Plot Force History ---
    if time_history:
        plot_force_history(time_history, cd_history, cl_history, experiment_path, Re_x, sim_id=sim_id)
    else:
        print("\nWarning: No force history to plot. This may be due to missing pressure files.")

    # --- Create Animation Frames ---
    frames_dir = os.path.join(experiment_path, "animation_frames")
    os.makedirs(frames_dir, exist_ok=True)
    frame_files = []
    
    print("\nGenerating animation frames...")
    obstacle_mask = get_cylinder_obstacle_mask(x, y, experiment_name)
    
    for i, u_file in enumerate(tqdm(selected_u_files, total=num_frames, desc="Frame generation")):
        iteration = get_iter_from_filename(u_file)
        U = np.load(u_file)
        velocity_magnitude = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.tricontourf(x, y, velocity_magnitude, levels=50, cmap='coolwarm', vmin=global_v_min, vmax=global_v_max)
        if np.any(obstacle_mask):
            ax.tricontourf(x[obstacle_mask], y[obstacle_mask], velocity_magnitude[obstacle_mask], levels=1, colors='gray', alpha=0.5)

        ax.set_title(f"Velocity Magnitude | {experiment_name} | Re={Re_x}", loc='center')
        ax.set_title(f"Time = {iteration * dt:.4f}s", loc='left', fontsize=10)
        if sim_id and sim_id != "N/A":
            ax.set_title(f"Sim ID: {sim_id}", loc='right', fontsize=8, color='gray')
        
        ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)")
        ax.set_aspect('equal', 'box'), ax.set_xlim(np.min(x), np.max(x)), ax.set_ylim(np.min(y), np.max(y))

        if "cylinder" in experiment_name.lower():
            circle = mpatches.Circle((0.2, 0.2), 0.05, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
            ax.add_patch(circle)
        
        plt.tight_layout()
        frame_filename = os.path.join(frames_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_filename, dpi=150), plt.close(fig)
        frame_files.append(frame_filename)

    # --- Create GIF and MP4 ---
    gif_output_filepath = os.path.join(experiment_path, gif_output_filename)
    print(f"\nCreating GIF: {gif_output_filepath}")
    with imageio.get_writer(gif_output_filepath, mode='I', duration=(1.0/framerate)) as writer:
        for frame_filename in tqdm(frame_files, desc="GIF creation"):
            writer.append_data(imageio.imread(frame_filename))

    mp4_output_filepath = os.path.join(experiment_path, mp4_output_filename)
    print(f"Creating MP4: {mp4_output_filepath}")
    try:
        with imageio.get_writer(mp4_output_filepath, fps=framerate, codec='libx264', quality=8) as writer:
            for frame_filename in tqdm(frame_files, desc="MP4 creation"):
                writer.append_data(imageio.imread(frame_filename))
        print(f"MP4 animation saved to: {mp4_output_filepath}")
    except Exception as e:
        print(f"\nCould not create MP4. Error: {e}\n(Please ensure 'imageio-ffmpeg' is installed: `pip install imageio-ffmpeg`)")
            
    # --- Clean up ---
    # print("\nCleaning up temporary frame files...")
    # for frame_filename in frame_files:
    #     os.remove(frame_filename)
    # os.rmdir(frames_dir)
    print(f"\nAnimation frames saved for LaTeX inclusion in: {frames_dir}")
    print(f"Other outputs saved in: {experiment_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate transient flow results and plot forces from a config file.")
    parser.add_argument("config_path", type=str, help="Path to the simulation config file (e.g., experiments/Collocated/transientcylinder/config.yaml)")
    parser.add_argument("--framerate", type=int, default=20, help="Framerate for the output animation in frames per second.")
    parser.add_argument("--end-time", type=float, default=None, help="End time for the animation in seconds. Defaults to the full available duration.")
    
    args = parser.parse_args()
    animate_transient_flow(args.config_path, args.framerate, args.end_time) 