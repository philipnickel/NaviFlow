import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from naviflow_collocated.utils.postprocess.utils import save_pdf
from naviflow_collocated.utils.postprocess.plot_style import plt  # This will apply the style

# Set the style for all plots
plt.style.use(['science', 'grid'])

# Professional color scheme
COLORS = {
    'ghia': 'black',  # Ghia's data points
    'numerical': '#1f77b4'  # Professional blue for numerical solution
}

def extract_centerline_unstructured(x, y, U, direction='vertical', n_points=100):
    """Extract velocity profile along centerline using interpolation."""
    points = np.stack([x, y], axis=1)
    if direction == 'vertical':
        y_line = np.linspace(0, 1, n_points)
        x_line = np.full_like(y_line, 0.5)
        query_points = np.stack([x_line, y_line], axis=1)
        u_profile = griddata(points, U[:, 0], query_points, method='linear')
        # Fallback to nearest for NaNs (outside convex hull)
        nan_mask = np.isnan(u_profile)
        if np.any(nan_mask):
            u_profile[nan_mask] = griddata(points, U[:, 0], query_points[nan_mask], method='nearest')
        return y_line, u_profile
    elif direction == 'horizontal':
        x_line = np.linspace(0, 1, n_points)
        y_line = np.full_like(x_line, 0.5)
        query_points = np.stack([x_line, y_line], axis=1)
        v_profile = griddata(points, U[:, 1], query_points, method='linear')
        nan_mask = np.isnan(v_profile)
        if np.any(nan_mask):
            v_profile[nan_mask] = griddata(points, U[:, 1], query_points[nan_mask], method='nearest')
        return x_line, v_profile
    else:
        raise ValueError('direction must be "vertical" or "horizontal"')

def extract_centerline_uniform(x, y, U, direction='vertical'):
    """Extract velocity profile along centerline for uniform meshes."""
    if direction == 'vertical':
        mask = np.abs(x - 0.5) < 1e-10
        y_c = y[mask]
        u_c = U[mask, 0]
        sort_idx = np.argsort(y_c)
        return y_c[sort_idx], u_c[sort_idx]
    elif direction == 'horizontal':
        mask = np.abs(y - 0.5) < 1e-10
        x_c = x[mask]
        v_c = U[mask, 1]
        sort_idx = np.argsort(x_c)
        return x_c[sort_idx], v_c[sort_idx]
    else:
        raise ValueError('direction must be "vertical" or "horizontal"')

def ghia_comparison(x, y, U, Re, n_cells, scheme, mesh_type, output_path, sim_id=None):
    """
    Compare numerical solution with Ghia's benchmark data for lid-driven cavity.
    Extracts velocity profiles along the centerlines and compares with Ghia's data.
    
    Args:
        x, y : ndarray
            Cell center coordinates
        U : ndarray
            Velocity field (n_cells, 2)
        Re : float
            Reynolds number
        n_cells : int
            Number of cells in the mesh
        scheme : str
            Convection discretization scheme used
        mesh_type : str
            Type of mesh used
        output_path : str
            Path to save the verification plot
        sim_id : str, optional
            Simulation ID for plot annotation
    """
    # Ghia's data for different Reynolds numbers
    ghia_data = {
        100: {
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
        },
        400: {
            'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                          0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
            'v': np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, 
                          -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 
                          0.19713, 0.18360, 0.00000]),
            'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                          0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
            'u': np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, 
                          -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 
                          0.68439, 1.00000])
        },
        1000: {
            'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                          0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
            'v': np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, 
                          -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 
                          0.29012, 0.27485, 0.00000]),
            'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                          0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
            'u': np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, 
                          -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 
                          0.57492, 1.00000])
        },
        3200: {
            'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                          0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
            'v': np.array([0.00000, -0.39017, -0.47425, -0.52357, -0.54053, -0.44307, -0.37401, 
                          -0.31184, 0.00999, 0.28188, 0.29030, 0.37119, 0.42768, 0.41906, 
                          0.40917, 0.39560, 0.00000]),
            'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                          0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
            'u': np.array([0.00000, -0.32407, -0.35344, -0.37827, -0.41933, -0.34323, -0.24427, 
                          -0.86636, -0.04272, 0.07156, 0.19791, 0.34682, 0.46101, 0.46547, 
                          0.48296, 1.00000])
        },
        5000: {
            'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                          0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
            'v': np.array([0.00000, -0.41165, -0.52876, -0.55408, -0.55069, -0.41442, -0.36214, 
                          -0.30018, 0.00945, 0.27280, 0.28066, 0.35368, 0.41824, 0.43564, 
                          0.43154, 0.42735, 0.00000]),
            'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                          0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
            'u': np.array([0.00000, -0.41165, -0.42901, -0.43643, -0.40435, -0.33050, -0.22855, 
                          -0.07404, -0.03039, 0.08183, 0.20087, 0.33556, 0.46036, 0.45992, 
                          0.46120, 1.00000])
        }
    }

    # Check if we have data for this Reynolds number
    if Re not in ghia_data:
        print(f"Warning: No Ghia data available for Re={Re}")
        return

    # Get Ghia's data for this Reynolds number
    ghia = ghia_data[Re]

    # Extract centerline data using appropriate method
    if mesh_type == 'unstructured':
        y_vertical, u_vertical = extract_centerline_unstructured(x, y, U, direction='vertical')
        x_horizontal, v_horizontal = extract_centerline_unstructured(x, y, U, direction='horizontal')
    else:
        y_vertical, u_vertical = extract_centerline_uniform(x, y, U, direction='vertical')
        x_horizontal, v_horizontal = extract_centerline_uniform(x, y, U, direction='horizontal')
    
    # Print debug information
    print(f"Found {len(y_vertical)} points along vertical centerline")
    print(f"Found {len(x_horizontal)} points along horizontal centerline")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot vertical centerline (u-velocity)
    ax1.plot(u_vertical, y_vertical, '-', color=COLORS['numerical'], 
            label='Numerical', linewidth=1.5)
    ax1.plot(ghia['u'], ghia['y'], 'o', color=COLORS['ghia'], 
            label='Ghia et al.', markersize=6, markerfacecolor='none')
    ax1.set_xlabel('u-velocity')
    ax1.set_ylabel('y')
    ax1.set_title('Vertical Centerline')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot horizontal centerline (v-velocity)
    ax2.plot(x_horizontal, v_horizontal, '-', color=COLORS['numerical'], 
            label='Numerical', linewidth=1.5)
    ax2.plot(ghia['x'], ghia['v'], 'o', color=COLORS['ghia'], 
            label='Ghia et al.', markersize=6, markerfacecolor='none')
    ax2.set_xlabel('x')
    ax2.set_ylabel('v-velocity')
    ax2.set_title('Horizontal Centerline')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add simulation details
    fig.suptitle(f'Lid-Driven Cavity Validation (Re={Re})\n{scheme}, {mesh_type}, {n_cells} cells')
    if sim_id is not None:
        fig.text(0.5, 0.01, f"Simulation ID: {sim_id}", ha='center', va='bottom', fontsize=6, color='gray', alpha=0.7)
    
    # Save figure
    save_pdf(fig, output_path)

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
    # Construct path to config.yaml in the experiment directory
    experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(output_path)))
    config_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    u_inlet = config["physical_properties"]["characteristic_velocity"]  # Inlet velocity
    rho = config["physical_properties"]["rho"]
    
    # Calculate channel height from numerical data
    H = np.max(y) - np.min(y)  # Total channel height
    h = H/2  # Half height

    # Calculate channel length from domain coordinates
    L = 5.0
    
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    
    # Velocity Profile Plot (at x=L/2)
    ax = fig.add_subplot(111)
    
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
    r = np.linspace(-h, h, len(u_numerical))
    u_analytical = u_inlet * (1 - (r**2/h**2))  # Use actual measured inlet velocity
    y = np.linspace(0, 1, len(u_analytical))
    
    # Plot both solutions
    ax.plot(y, u_numerical, 'o-', color='tab:blue', label="Numerical", markersize=2, alpha=0.6)
    ax.plot(y, u_analytical, '--', color='tab:orange', label="Analytical", linewidth=2)
    
    ax.set_title(f"Velocity Profile at x=L/2", fontsize=12, pad=10)
    ax.set_xlabel("y/h", fontsize=10)
    ax.set_ylabel("u/u_inlet", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=9)
    
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