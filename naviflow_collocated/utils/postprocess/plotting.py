import os
import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from naviflow_collocated.utils.postprocess.utils import get_obstacle_mask_from_msh, save_pdf
import matplotlib.pyplot as plt
from naviflow_collocated.utils.postprocess.plot_style import plt  # This will apply the style
import scienceplots
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Set the style for all plots
plt.style.use(['science', 'grid'])

def plot_fields_single_row(x, y, U, velocity_magnitude, p, sim_id=None, output_path=None, experiment=None, Re=None, return_as_array=False, time_val=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Velocity Magnitude", "Pressure", "Streamlines"]

    # Robust obstacle mask from .msh
    obstacle_mask = get_obstacle_mask_from_msh(x, y, experiment)

    # 1. velocity magnitude
    cf1 = axes[0].tricontourf(x, y, velocity_magnitude, levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        axes[0].tricontourf(x[obstacle_mask], y[obstacle_mask], velocity_magnitude[obstacle_mask], levels=1, colors='gray', alpha=0.5)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar1 = fig.colorbar(cf1, cax=cax, orientation='horizontal')
    cbar1.formatter.set_scientific(True)
    cbar1.formatter.set_powerlimits((0, 0))
    axes[0].set_title(titles[0])
    axes[0].set_aspect('equal', 'box')

    # 2. pressure
    cf2 = axes[1].tricontourf(x, y, p, levels=50, cmap='coolwarm')
    if np.any(obstacle_mask):
        axes[1].tricontourf(x[obstacle_mask], y[obstacle_mask], p[obstacle_mask], levels=1, colors='gray', alpha=0.5)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar2 = fig.colorbar(cf2, cax=cax, orientation='horizontal')
    cbar2.formatter.set_scientific(True)
    cbar2.formatter.set_powerlimits((0, 0))
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
    # Remove axis labels for streamlines
    axes[2].set_xticklabels([])
    axes[2].set_yticklabels([])

    # Ensure all subplots have the same x and y limits
    xlim = (np.min(x), np.max(x))
    ylim = (np.min(y), np.max(y))
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Overlay and fill obstacle boundary for cylinderFlow
    if "cylinderFlow" in experiment:
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
    
    # Add consistent, styled annotations
    if time_val is not None:
        fig.text(0.02, 0.98, f"Time = {time_val:.4f}s", ha='left', va='top', fontsize=8, color='gray')
    if sim_id is not None:
        fig.text(0.98, 0.98, f"Sim ID: {sim_id}", ha='right', va='top', fontsize=8, color='gray')

    if return_as_array:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image_array

    if output_path:
        save_pdf(fig, output_path)
    else:
        plt.show()

def detect_stalled_residuals(residual_array, stall_window=500, stall_threshold=1e-2):
    """
    Detect if residuals have stalled (stopped decreasing significantly).
    
    Parameters:
    -----------
    residual_array : array-like
        Array of residual values
    stall_window : int
        Number of iterations to look back for stall detection
    stall_threshold : float
        Relative change threshold below which residuals are considered stalled
        
    Returns:
    --------
    stall_start : int or None
        Index where stalling begins, or None if no stalling detected
    """
    if len(residual_array) < stall_window * 2:
        return None
    
    # Look for stalling in the last part of the simulation
    for i in range(stall_window, len(residual_array) - stall_window):
        # Check if the relative change over stall_window iterations is small
        window_start = residual_array[i]
        window_end = residual_array[i + stall_window]
        
        # Avoid division by zero
        if abs(window_start) < 1e-20:
            continue
            
        relative_change = abs(window_end - window_start) / abs(window_start)
        
        # If change is very small, consider it stalled
        if relative_change < stall_threshold:
            return i
    
    return None

def trim_stalled_residuals(residual_arrays, keep_stalled_iterations=500):
    """
    Trim stalled residuals from multiple residual arrays while keeping some stalled data.
    
    Parameters:
    -----------
    residual_arrays : dict or NpzFile
        Dictionary or NpzFile of residual arrays (e.g., {'u': [...], 'v': [...], 'cont': [...]})
    keep_stalled_iterations : int
        Number of stalled iterations to keep to show stalling behavior
        
    Returns:
    --------
    trimmed_arrays : dict
        Dictionary of trimmed residual arrays
    stall_info : dict
        Information about detected stalling
    """
    # Convert NpzFile to regular dictionary if needed
    if hasattr(residual_arrays, 'files'):  # NpzFile has a 'files' attribute
        res_dict = {key: residual_arrays[key] for key in residual_arrays.files}
    else:
        res_dict = residual_arrays
    
    stall_starts = {}
    
    # Detect stalling for each residual type
    for key, residuals in res_dict.items():
        stall_start = detect_stalled_residuals(residuals)
        if stall_start is not None:
            stall_starts[key] = stall_start
    
    # If any residuals are stalled, find the earliest stall point
    if stall_starts:
        earliest_stall = min(stall_starts.values())
        trim_point = earliest_stall + keep_stalled_iterations
        
        # Ensure we don't go beyond the array length
        min_length = min(len(arr) for arr in res_dict.values())
        trim_point = min(trim_point, min_length)
        
        # Trim all arrays to the same length
        trimmed_arrays = {key: arr[:trim_point] for key, arr in res_dict.items()}
        
        stall_info = {
            'stalled': True,
            'stall_starts': stall_starts,
            'earliest_stall': earliest_stall,
            'trim_point': trim_point,
            'original_length': min_length,
            'trimmed_length': trim_point
        }
    else:
        trimmed_arrays = res_dict.copy()
        stall_info = {'stalled': False}
    
    return trimmed_arrays, stall_info

def plot_residuals(res, output_path, sim_id=None):
    # Use the full residual data without trimming
    # Convert NpzFile to regular dictionary if needed
    if hasattr(res, 'files'):  # NpzFile has a 'files' attribute
        res_dict = {key: res[key] for key in res.files}
    else:
        res_dict = res
    
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    # Use full residual arrays without any trimming
    u_data = res_dict["u"]
    v_data = res_dict["v"]
    cont_data = res_dict["cont"]
    
    ax.semilogy(u_data, label="u-momentum", color='tab:blue', linewidth=2)
    ax.semilogy(v_data, label="v-momentum", color='tab:orange', linewidth=2)
    ax.semilogy(cont_data, label="continuity", color='tab:green', linewidth=2)
    
    # Simple title without trimming information
    title = "Residual History"
    
    ax.set_title(title, fontsize=16, loc='center')
    if sim_id is not None:
        ax.set_title(f"Simulation ID: {sim_id}", fontsize=8, color='gray', loc='right')
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Residual", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right', frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    fig.tight_layout(pad=0.1)
    save_pdf(fig, output_path)

def plot_residual_fields(x, y, u_res, v_res, cont_res, output_path, sim_id=None, experiment=None, Re=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["U Residual", "V Residual", "Continuity Residual"]
    colormap = "viridis"  # perceptually uniform, good for magnitude fields

    # 1. U residual
    cf1 = axes[0].tricontourf(x, y, np.abs(u_res), levels=50, cmap=colormap)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar1 = fig.colorbar(cf1, cax=cax, orientation='horizontal')
    cbar1.formatter.set_scientific(True)
    cbar1.formatter.set_powerlimits((0, 0))
    axes[0].set_title(titles[0])
    axes[0].set_aspect('equal', 'box')

    # 2. V residual
    cf2 = axes[1].tricontourf(x, y, np.abs(v_res), levels=50, cmap=colormap)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar2 = fig.colorbar(cf2, cax=cax, orientation='horizontal')
    cbar2.formatter.set_scientific(True)
    cbar2.formatter.set_powerlimits((0, 0))
    axes[1].set_title(titles[1])
    axes[1].set_aspect('equal', 'box')

    # 3. Continuity residual
    cf3 = axes[2].tricontourf(x, y, np.abs(cont_res), levels=50, cmap=colormap)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar3 = fig.colorbar(cf3, cax=cax, orientation='horizontal')
    cbar3.formatter.set_scientific(True)
    cbar3.formatter.set_powerlimits((0, 0))
    axes[2].set_title(titles[2])
    axes[2].set_aspect('equal', 'box')

    # Add cylinder for cylinderFlow cases
    if "cylinderFlow" in experiment:
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
    save_pdf(fig, output_path)

def plot_streamlines(x, y, U, output_path, experiment=None, Re=None, sim_id=None, return_as_array=False, time_val=None, hide_colorbar=False):
    """Create a standalone streamlines plot."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Create grid for streamlines
    n_grid = 200
    xi = np.linspace(np.min(x), np.max(x), n_grid)
    yi = np.linspace(np.min(y), np.max(y), n_grid)
    Xg, Yg = np.meshgrid(xi, yi)
    
    # Interpolate velocity field
    Ug = griddata((x, y), U[:, 0], (Xg, Yg), method='cubic', fill_value=0)
    Vg = griddata((x, y), U[:, 1], (Xg, Yg), method='cubic', fill_value=0)
    
    # Calculate velocity magnitude for coloring
    velocity_magnitude = np.sqrt(Ug**2 + Vg**2)
    
    # Plot velocity magnitude as background
    im = ax.pcolormesh(Xg, Yg, velocity_magnitude, 
                       cmap='coolwarm',
                       shading='auto',
                       alpha=0.3)  # Semi-transparent background
    
    # Plot streamlines in solid blue
    strm = ax.streamplot(xi, yi, Ug, Vg, 
                        color='tab:blue',
                        density=4.0,
                        linewidth=0.2,
                        arrowsize=0.2)
    
    if not hide_colorbar:
        # Add horizontal colorbar for velocity magnitude
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.3)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label('Velocity Magnitude')
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))
    
    # Set title and labels
    title = "Streamlines"
    ax.set_title(title, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"Sim ID: {sim_id}", fontsize=8, color='gray', loc='right')
    if time_val is not None:
        ax.set_title(f"Time = {time_val:.4f}s", fontsize=8, color='gray', loc='left')
    ax.set_aspect('equal', 'box')
    
    # Remove axis labels and grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    
    # Ensure the plot covers the entire domain
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    
    # Add cylinder for cylinderFlow cases
    if "cylinderFlow" in experiment:
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    
    fig.tight_layout(pad=0.1)

    if return_as_array:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image_array
        
    save_pdf(fig, output_path)

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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    title = "u-velocity"
    ax.set_title(title, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"Simulation ID: {sim_id}", fontsize=8, color='gray', loc='right')
    ax.set_aspect('equal', 'box')
    if "cylinderFlow" in experiment:
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    title = "v-velocity"
    ax.set_title(title, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"Simulation ID: {sim_id}", fontsize=8, color='gray', loc='right')
    ax.set_aspect('equal', 'box')
    if "cylinderFlow" in experiment:
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    title = "Velocity Magnitude"
    ax.set_title(title, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"Simulation ID: {sim_id}", fontsize=8, color='gray', loc='right')
    ax.set_aspect('equal', 'box')
    if "cylinderFlow" in experiment:
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    title = "Pressure"
    ax.set_title(title, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"Simulation ID: {sim_id}", fontsize=8, color='gray', loc='right')
    ax.set_aspect('equal', 'box')
    if "cylinderFlow" in experiment:
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    fig.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "pressure.pdf"))
    plt.close(fig)

    # streamlines only
    plot_streamlines(x, y, U, os.path.join(plots_dir, "streamlines.pdf"), experiment=experiment, Re=Re, sim_id=sim_id)

def save_individual_residual_plots(x, y, u_res, v_res, cont_res, experiment, Re, sim_id, results_dir):
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # U residual
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.tricontourf(x, y, np.abs(u_res), levels=50, cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    title = "U Residual"
    ax.set_title(title, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"Simulation ID: {sim_id}", fontsize=8, color='gray', loc='right')
    ax.set_aspect('equal', 'box')
    if "cylinderFlow" in experiment:
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    title = "V Residual"
    ax.set_title(title, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"Simulation ID: {sim_id}", fontsize=8, color='gray', loc='right')
    ax.set_aspect('equal', 'box')
    if "cylinderFlow" in experiment:
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    title = "Continuity Residual"
    ax.set_title(title, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"Simulation ID: {sim_id}", fontsize=8, color='gray', loc='right')
    ax.set_aspect('equal', 'box')
    if "cylinderFlow" in experiment:
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
    fig.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "continuity_residual.pdf"))
    plt.close(fig)

def plot_velocity_magnitude(x, y, velocity_magnitude, output_path, experiment=None, Re=None, sim_id=None, time_val=None, return_as_array=False, vmin=None, vmax=None, hide_colorbar=False):
    """Create a standalone velocity magnitude plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    obstacle_mask = get_obstacle_mask_from_msh(x, y, experiment)
    
    # Use global vmin/vmax if provided to keep color scale consistent
    cf = ax.tricontourf(x, y, velocity_magnitude, levels=50, cmap='coolwarm', vmin=vmin, vmax=vmax)
    if np.any(obstacle_mask):
        ax.tricontourf(x[obstacle_mask], y[obstacle_mask], velocity_magnitude[obstacle_mask], levels=1, colors='gray', alpha=0.5)
        
    if not hide_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.3)
        cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.set_label('Velocity Magnitude')

    title = "Velocity Magnitude"
    ax.set_title(title, fontsize=14, loc='center')
    if sim_id is not None:
        ax.set_title(f"Simulation ID: {sim_id}", fontsize=8, color='gray', loc='right')
    if time_val is not None:
        ax.set_title(f"Time = {time_val:.4f}s", fontsize=8, color='gray', loc='left')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    
    if "cylinderFlow" in experiment:
        center = (0.2, 0.2)
        radius = 0.05
        circle = mpatches.Circle(center, radius, edgecolor='grey', facecolor='grey', alpha=1.0, linewidth=0, zorder=10)
        ax.add_patch(circle)
        
    fig.tight_layout(pad=0.1)
    
    if return_as_array:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image_array

    save_pdf(fig, output_path) 