"""
Compare multiple lid-driven cavity experiments against Ghia's benchmark data.

This script allows for comparing multiple lid-driven cavity simulations against each other
and against Ghia's benchmark data. It creates comparison plots showing velocity profiles
along the vertical and horizontal centerlines for each Reynolds number.

Usage:
    python compare_lid_driven_cavity.py --config-list path/to/config_list.txt [--output-dir dir]

Example:
    # Create a text file with config paths (supports both config.yaml and pseudo_config.yaml)
    echo "experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml" > configs.txt
    echo "experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml" >> configs.txt
    echo "experiments/Staggered/02 GMG BiCGSTAB/pseudo_config.yaml" >> configs.txt
    
    # Compare experiments
    python compare_lid_driven_cavity.py --config-list configs.txt --output-dir comparison_plots

The config_list.txt file should contain one config file path per line (supports both formats):
    experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml
    experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml
    experiments/Staggered/02 GMG BiCGSTAB/pseudo_config.yaml
    experiments/lidDrivenCavity/ForReport/unstructured/medium/Re_100/config.yaml

The script will:
1. Read config file paths from the input text file
2. Convert config paths to experiment directory paths
3. Load data from each experiment directory (supports both config.yaml and pseudo_config.yaml)
4. Group experiments by Reynolds number
5. For each Reynolds number:
   - Create comparison plots showing:
     * u-velocity along vertical centerline (x=0.5)
     * v-velocity along horizontal centerline (y=0.5)
   - Compare with Ghia's benchmark data
   - Include legend showing mesh resolution and scheme
6. Save plots as PDFs in the specified output directory

Required files in each experiment directory:
- results/U_final.npy: Final velocity field
- results/cell_centers.npz: Cell center coordinates
- results/metadata.yaml: Simulation metadata
- config.yaml OR pseudo_config.yaml: Simulation configuration

The plots use a professional color scheme consistent with other plots in the codebase.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import yaml
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from naviflow_collocated.utils.postprocess.plot_style import plt  # This will apply the style
from scipy.spatial import cKDTree
from matplotlib.lines import Line2D

# Set the style for all plots
plt.style.use(['science', 'grid'])

# Professional color scheme
COLORS = {
    'ghia': 'black',  # Ghia's data points
    'schemes': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # Professional color palette
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
}

def read_config_list(config_list_file):
    """Read config file paths from a text file.
    
    Args:
        config_list_file (str): Path to text file containing config paths
        
    Returns:
        list: List of config file paths
    """
    config_paths = []
    
    if not os.path.exists(config_list_file):
        raise FileNotFoundError(f"Config list file not found: {config_list_file}")
    
    with open(config_list_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            # Check if config file exists
            if not os.path.exists(line):
                print(f"Warning: Config file not found (line {line_num}): {line}")
                continue
                
            config_paths.append(line)
    
    return config_paths

def config_path_to_experiment_path(config_path):
    """Convert config file path to experiment directory path.
    
    Args:
        config_path (str): Path to config.yaml or pseudo_config.yaml file
        
    Returns:
        str: Path to experiment directory
    """
    # Config file should be directly in the experiment directory
    return os.path.dirname(config_path)

def load_experiment_data(experiment_path):
    """Load data from a single experiment directory."""
    results_dir = os.path.join(experiment_path, "results")
    
    # Look for either config.yaml or pseudo_config.yaml
    config_path = None
    for config_name in ["config.yaml", "pseudo_config.yaml"]:
        potential_path = os.path.join(experiment_path, config_name)
        if os.path.exists(potential_path):
            config_path = potential_path
            break
    
    if config_path is None:
        raise FileNotFoundError(f"Neither config.yaml nor pseudo_config.yaml found in {experiment_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load numerical data
    U = np.load(os.path.join(results_dir, "U_final.npy"))
    cell_data = np.load(os.path.join(results_dir, "cell_centers.npz"))
    meta = yaml.safe_load(open(os.path.join(results_dir, "metadata.yaml")))
    
    x = cell_data["x"]
    y = cell_data["y"]
    
    # Extract simulation parameters
    Re = config["physical_properties"]["reynolds_number"]
    scheme = config["algorithm"]["convection_discretization"]
    mesh_type, resolution = config["domain"]["mesh"]
    n_cells = len(U)
    
    return {
        'x': x,
        'y': y,
        'U': U,
        'Re': Re,
        'scheme': scheme,
        'mesh_type': mesh_type,
        'resolution': resolution,
        'n_cells': n_cells,
        'sim_id': meta.get("Simulation id", "unknown")
    }

def get_ghia_data(Re):
    """Get Ghia's benchmark data for a given Reynolds number."""
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
    return ghia_data.get(Re)

def create_legend_label(data):
    """Create a meaningful label for the legend."""
    return f"{data['n_cells']} cells, {data['scheme']}, {data['mesh_type']}"

def extract_centerline_uniform(x, y, U, direction='vertical'):
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

def extract_centerline_unstructured(x, y, U, direction='vertical', n_points=100):
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

def plot_comparison(exps, ghia, output_path, title=None):
    """Plot comparison of numerical results with Ghia's benchmark data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define markers for numerical results (avoiding circles since Ghia uses them)
    markers = ['s', '^', 'D']  # Square, triangle, diamond
    
    # Professional darker grey color for Ghia
    ghia_color = '#555555'
    ghia_markeredgewidth = 1.5  # Increased thickness
    
    print("\nPlotting numerical results:")
    # Plot lines for each experiment first
    for i, exp in enumerate(exps):
        if exp['mesh_type'] == 'uniform':
            y_vertical, u_vertical = extract_centerline_uniform(exp['x'], exp['y'], exp['U'], direction='vertical')
            x_horizontal, v_horizontal = extract_centerline_uniform(exp['x'], exp['y'], exp['U'], direction='horizontal')
        else:
            y_vertical, u_vertical = extract_centerline_unstructured(exp['x'], exp['y'], exp['U'], direction='vertical', n_points=100)
            x_horizontal, v_horizontal = extract_centerline_unstructured(exp['x'], exp['y'], exp['U'], direction='horizontal', n_points=100)
        
        print(f"\nDataset {i+1}:")
        print(f"Vertical centerline points: {len(y_vertical)}")
        print(f"Horizontal centerline points: {len(x_horizontal)}")
        
        # Calculate marker spacing and offset to stagger markers
        n_points = len(y_vertical)
        markevery = max(1, n_points // 12)
        offset = i * (markevery // len(exps) + 1)  # Stagger start
        
        color = COLORS['schemes'][i % len(COLORS['schemes'])]
        marker = markers[i % len(markers)]
        label = create_legend_label(exp)
        print(f"Plotting with label: {label}")
        
        # Plot lines only (no markers yet)
        ax1.plot(u_vertical, y_vertical, '-', color=color, label=label, 
                linewidth=0.8)
        ax2.plot(x_horizontal, v_horizontal, '-', color=color, label=label, 
                linewidth=0.8)
        # Store for marker plotting
        exp['_y_vertical'] = y_vertical
        exp['_u_vertical'] = u_vertical
        exp['_x_horizontal'] = x_horizontal
        exp['_v_horizontal'] = v_horizontal
        exp['_color'] = color
        exp['_marker'] = marker
        exp['_markevery'] = (offset, markevery)
    
    # Plot markers for each experiment (on top)
    for exp in exps:
        ax1.plot(exp['_u_vertical'], exp['_y_vertical'], linestyle='None', color=exp['_color'],
                marker=exp['_marker'], markersize=3, markevery=exp['_markevery'],
                markerfacecolor='white', markeredgewidth=1.0)
        ax2.plot(exp['_x_horizontal'], exp['_v_horizontal'], linestyle='None', color=exp['_color'],
                marker=exp['_marker'], markersize=3, markevery=exp['_markevery'],
                markerfacecolor='white', markeredgewidth=1.0)
    
    # Plot Ghia's data points last (on top, darker grey, hollow circle, thicker edge)
    ax1.plot(ghia['u'], ghia['y'], 'o', color=ghia_color, 
            label='Ghia et al.', markersize=7, markerfacecolor='none', 
            markeredgewidth=ghia_markeredgewidth)
    ax2.plot(ghia['x'], ghia['v'], 'o', color=ghia_color, 
            label='Ghia et al.', markersize=7, markerfacecolor='none', 
            markeredgewidth=ghia_markeredgewidth)
    
    # --- Legend with both marker and line ---
    legend_handles = []
    for exp in exps:
        handle = Line2D([], [], color=exp['_color'], marker=exp['_marker'], linestyle='-',
                        markersize=6, markerfacecolor='white', markeredgewidth=1.0, linewidth=1.2,
                        label=create_legend_label(exp))
        legend_handles.append(handle)
    ghia_handle = Line2D([], [], color=ghia_color, marker='o', linestyle='None',
                         markersize=7, markerfacecolor='none', markeredgewidth=ghia_markeredgewidth,
                         label='Ghia et al.')
    legend_handles.append(ghia_handle)
    ax1.legend(handles=legend_handles)
    ax2.legend(handles=legend_handles)

    # --- Centered annotation row between the two plots ---
    sim_id_annotations = []
    for exp in exps:
        sim_id = exp.get('sim_id', None)
        marker = exp.get('_marker', 'o')
        color = exp.get('_color', '#1f77b4')
        if sim_id:
            sim_id_annotations.append((marker, color, sim_id))

    # Use custom title if provided, otherwise use default
    if title:
        fig.suptitle(f"{title} (Re={exps[0]['Re']})", fontsize=14, y=0.98)
    else:
        fig.suptitle(f"Comparison of Lid-Driven Cavity Simulations at Re={exps[0]['Re']}", fontsize=14, y=0.98)

    if sim_id_annotations:
        fig.subplots_adjust(bottom=0.12)
        n = len(sim_id_annotations)
        y_row = 0.03  # Fixed position well below the plots
        dx = 0.018    # Offset for text to the right of marker
        for i, (marker, color, sim_id) in enumerate(sim_id_annotations):
            # Distribute across the full figure width with margin
            x_fig = 0.15 + (i + 1) / (n + 1) * 0.7
            # Draw marker
            ax1.plot([x_fig], [y_row], marker=marker, color=color, markersize=8, markerfacecolor='white',
                     linestyle='None', transform=fig.transFigure, clip_on=False)
            # Draw sim_id text just to the right of the marker
            fig.text(x_fig + dx, y_row, f"Simulation ID: {sim_id}", color='grey', alpha=0.7, fontsize=7,
                     ha='left', va='center', transform=fig.transFigure)
    
    plt.savefig(output_path)
    plt.close()

def compare_experiments(experiment_paths, output_dir, title=None):
    """Compare multiple lid-driven cavity experiments.
    
    Args:
        experiment_paths (list): List of paths to experiment directories
        output_dir (str): Directory to save comparison plots
        title (str, optional): Custom title for the plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiment data
    experiments = []
    for path in experiment_paths:
        try:
            data = load_experiment_data(path)
            experiments.append(data)
        except Exception as e:
            print(f"Error loading experiment {path}: {e}")
    
    # Group experiments by Reynolds number
    experiments_by_re = {}
    for exp in experiments:
        Re = exp['Re']
        if Re not in experiments_by_re:
            experiments_by_re[Re] = []
        experiments_by_re[Re].append(exp)
    
    # Create comparison plots for each Reynolds number
    for Re, exps in experiments_by_re.items():
        # Get Ghia's data
        ghia = get_ghia_data(Re)
        if ghia is None:
            print(f"No Ghia data available for Re={Re}")
            continue
        
        # Create filename using title if provided
        if title:
            # Sanitize title for filename (replace special characters)
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            filename = f'{safe_title}_Re_{Re}.pdf'
        else:
            filename = f'lid_driven_cavity_comparison_Re_{Re}.pdf'
        
        output_path = os.path.join(output_dir, filename)
        plot_comparison(exps, ghia, output_path, title)
        print(f"Saved comparison plot to {output_path}")

def compare_experiments_from_configs(config_list_file, output_dir):
    """Compare multiple lid-driven cavity experiments from config list file.
    
    Args:
        config_list_file (str): Path to text file containing config paths
        output_dir (str): Directory to save comparison plots
    """
    # Read config paths from file
    config_paths = read_config_list(config_list_file)
    
    if not config_paths:
        print("No valid config files found in the list.")
        return
    
    print(f"Found {len(config_paths)} valid config files:")
    for config_path in config_paths:
        print(f"  - {config_path}")
    
    # Convert config paths to experiment paths
    experiment_paths = [config_path_to_experiment_path(config_path) for config_path in config_paths]
    
    # Extract title from config list filename (without extension)
    title = os.path.splitext(os.path.basename(config_list_file))[0]
    
    # Use existing comparison functionality with custom title
    compare_experiments(experiment_paths, output_dir, title)

def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple lid-driven cavity experiments from config list',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a config list file (supports both config.yaml and pseudo_config.yaml)
  echo "experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml" > configs.txt
  echo "experiments/Staggered/02 GMG BiCGSTAB/pseudo_config.yaml" >> configs.txt
  
  # Compare experiments
  python compare_lid_driven_cavity.py --config-list configs.txt --output-dir comparison_plots
  
  # Alternative with explicit paths (legacy mode)
  python compare_lid_driven_cavity.py --experiments path/to/exp1 path/to/exp2 --output-dir comparison_plots

Config list file format:
  Each line should contain a path to a config file (supports both formats):
  experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml
  experiments/Staggered/02 GMG BiCGSTAB/pseudo_config.yaml
  experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml
  # Comments and empty lines are ignored
        """
    )
    
    # Create mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--config-list', 
                           help='Path to text file containing config file paths (one per line)')
    input_group.add_argument('--experiments', nargs='+',
                           help='Paths to experiment directories (legacy mode)')
    
    # Get the directory where this script is located (postprocessing/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'lid_cavity_comparisons')
    
    parser.add_argument('--output-dir', default=default_output_dir,
                      help=f'Directory to save comparison plots (default: {default_output_dir})')
    
    args = parser.parse_args()
    
    if args.config_list:
        # New mode: read from config list file
        compare_experiments_from_configs(args.config_list, args.output_dir)
    else:
        # Legacy mode: direct experiment paths (no custom title)
        compare_experiments(args.experiments, args.output_dir, title=None)

if __name__ == '__main__':
    main() 