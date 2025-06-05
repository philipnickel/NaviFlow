#!/usr/bin/env python3
"""
Grid Convergence Study for Lid-Driven Cavity CFD simulations (u-velocity centerline)

This script performs grid convergence analysis by calculating L2 errors between
numerical solutions and Ghia's benchmark data for u-velocity along the vertical 
centerline only, then plots the errors as a function of grid size to determine 
the order of accuracy.

Usage:
    python grid_convergence_study.py --config-list path/to/config_list.txt [--output-dir dir]

Example:
    # Create a config list with different mesh resolutions for the same scheme
    echo "experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml" > mesh_study.txt
    echo "experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml" >> mesh_study.txt
    echo "experiments/lidDrivenCavity/ForReport/uniform/fine/Re_100/config.yaml" >> mesh_study.txt
    
    # Run grid convergence study
    python grid_convergence_study.py --config-list mesh_study.txt

The script will:
1. Load data from each experiment in the config list
2. Extract u-velocity along vertical centerline (x=0.5)
3. Interpolate to Ghia's reference points
4. Calculate L2 errors for u-velocity against Ghia's data
5. Estimate grid size (h = 1/sqrt(N) where N is number of cells)
6. Create log-log plot showing error vs grid size
7. Fit line to determine order of accuracy
8. Display observed order of accuracy on the plot
9. Save results as PDF plot and CSV data

Requirements:
- All experiments must be at the same Reynolds number
- Ghia benchmark data must be available for that Reynolds number
- Experiments should represent different mesh resolutions of the same scheme
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Import utilities from the comparison script
from compare_lid_driven_cavity import (
    read_config_list, 
    config_path_to_experiment_path, 
    load_experiment_data,
    get_ghia_data,
    extract_centerline_uniform,
    extract_centerline_unstructured
)

# Set matplotlib backend for non-interactive mode
import matplotlib
matplotlib.use('Agg')

# Import plotting style
from naviflow_collocated.utils.postprocess.plot_style import plt
plt.style.use(['science', 'grid'])

def interpolate_to_ghia_points(numerical_coords, numerical_values, ghia_coords):
    """
    Interpolate numerical solution to Ghia's reference points.
    
    Args:
        numerical_coords (array): Numerical coordinate points
        numerical_values (array): Numerical velocity values
        ghia_coords (array): Ghia's coordinate points
        
    Returns:
        array: Interpolated values at Ghia's points
    """
    # Create interpolation function
    f_interp = interp1d(numerical_coords, numerical_values, 
                       kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Interpolate to Ghia's points
    interpolated_values = f_interp(ghia_coords)
    
    return interpolated_values

def calculate_l2_error(numerical_values, reference_values):
    """
    Calculate L2 error between numerical and reference solutions.
    
    Args:
        numerical_values (array): Numerical solution values
        reference_values (array): Reference solution values
        
    Returns:
        float: L2 error
    """
    return np.sqrt(np.mean((numerical_values - reference_values)**2))

def calculate_u_centerline_error(exp, ghia):
    """
    Calculate L2 error for u-velocity along vertical centerline only.
    
    Args:
        exp (dict): Experiment data dictionary
        ghia (dict): Ghia's benchmark data
        
    Returns:
        float: L2 error for u-velocity centerline
    """
    # Extract u-velocity along vertical centerline (x=0.5)
    if exp['mesh_type'] == 'uniform':
        y_coords, u_profile = extract_centerline_uniform(
            exp['x'], exp['y'], exp['U'], direction='vertical'
        )
    else:
        y_coords, u_profile = extract_centerline_unstructured(
            exp['x'], exp['y'], exp['U'], direction='vertical', n_points=100
        )
    
    # Interpolate numerical solution to Ghia's points
    u_interp = interpolate_to_ghia_points(y_coords, u_profile, ghia['y'])
    
    # Calculate L2 error for u-velocity only
    l2_error_u = calculate_l2_error(u_interp, ghia['u'])
    
    return l2_error_u

def estimate_grid_size(experiment):
    """
    Estimate characteristic grid size.
    
    Args:
        experiment (dict): Experiment data dictionary
        
    Returns:
        float: Characteristic grid size
    """
    n_cells = experiment['n_cells']
    
    # For 2D problems, h = 1/sqrt(N) is a reasonable estimate
    # This assumes square domain with uniform spacing
    h = 1.0 / np.sqrt(n_cells)
    
    return h

def power_law(h, A, p):
    """Power law function for fitting: error = A * h^p"""
    return A * (h ** p)

def perform_convergence_study(config_list_file, output_dir):
    """
    Perform grid convergence study.
    
    Args:
        config_list_file (str): Path to config list file
        output_dir (str): Output directory for results
    """
    # Read config paths
    config_paths = read_config_list(config_list_file)
    
    if not config_paths:
        print("No valid config files found in the list.")
        return
    
    print(f"Found {len(config_paths)} experiments for grid convergence study:")
    
    # Load all experiment data
    experiments = []
    for config_path in config_paths:
        try:
            exp_path = config_path_to_experiment_path(config_path)
            data = load_experiment_data(exp_path)
            experiments.append(data)
            print(f"  - {data['n_cells']} cells, {data['scheme']}, {data['mesh_type']}")
        except Exception as e:
            print(f"Error loading experiment {config_path}: {e}")
    
    if not experiments:
        print("No experiments loaded successfully.")
        return
    
    # Check that all experiments have the same Reynolds number
    reynolds_numbers = [exp['Re'] for exp in experiments]
    if len(set(reynolds_numbers)) > 1:
        print(f"Error: All experiments must have the same Reynolds number.")
        print(f"Found Reynolds numbers: {set(reynolds_numbers)}")
        return
    
    Re = reynolds_numbers[0]
    print(f"\nPerforming convergence study for Re = {Re}")
    
    # Get Ghia's benchmark data
    ghia = get_ghia_data(Re)
    if ghia is None:
        print(f"Error: No Ghia benchmark data available for Re = {Re}")
        return
    
    # Sort experiments by number of cells (coarse to fine)
    experiments.sort(key=lambda x: x['n_cells'])
    
    # Storage for results
    results = {
        'n_cells': [],
        'grid_size': [],
        'l2_error_u_centerline': [],
        'scheme': [],
        'mesh_type': [],
        'sim_id': []
    }
    
    print("\nCalculating L2 errors for u-velocity centerline:")
    
    for exp in experiments:
        # Calculate u-velocity centerline error
        l2_error_u = calculate_u_centerline_error(exp, ghia)
        
        # Estimate grid size
        h = estimate_grid_size(exp)
        
        # Store results
        results['n_cells'].append(exp['n_cells'])
        results['grid_size'].append(h)
        results['l2_error_u_centerline'].append(l2_error_u)
        results['scheme'].append(exp['scheme'])
        results['mesh_type'].append(exp['mesh_type'])
        results['sim_id'].append(exp.get('sim_id', 'unknown'))
        
        print(f"  {exp['n_cells']:6d} cells: h={h:.4f}, L2_u_centerline={l2_error_u:.2e}")
    
    # Convert to arrays for fitting
    h_array = np.array(results['grid_size'])
    error_u_array = np.array(results['l2_error_u_centerline'])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract title from config list filename
    title = os.path.splitext(os.path.basename(config_list_file))[0]
    
    # Create convergence plot
    create_convergence_plot(h_array, error_u_array, 
                          results, Re, title, output_dir)
    
    # Results saved as plot only - no CSV output needed

def create_convergence_plot(h_array, error_u_array, 
                          results, Re, title, output_dir):
    """Create and save grid convergence plot."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Professional color scheme (same as compare_lid_driven_cavity.py)
    COLORS = {
        'schemes': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }
    
    # Define markers for data points
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', 'X']
    
    # Fit power law and calculate order of accuracy
    p_u = float('nan')  # Initialize with NaN
    try:
        if len(h_array) >= 2:  # Need at least 2 points for fitting
            # Fit u-velocity using log-log linear regression for better stability
            # log(error) = log(A) + p * log(h) => p is the slope
            log_h = np.log(h_array)
            log_error = np.log(error_u_array)
            
            # Linear fit: log_error = slope * log_h + intercept
            # slope = p (order of accuracy)
            slope, intercept = np.polyfit(log_h, log_error, 1)
            p_u = slope
            
            print(f"\nOrder of accuracy:")
            print(f"  u-velocity centerline: {p_u:.2f}")
            print(f"  (computed using log-log linear regression)")
            
    except Exception as e:
        print(f"Warning: Could not fit power law: {e}")
        p_u = float('nan')
    
    # Group experiments by scheme for consistent coloring and legend
    schemes = list(set(results['scheme']))
    scheme_colors = {}
    scheme_markers = {}
    plot_data = []
    
    for i, scheme in enumerate(schemes):
        scheme_colors[scheme] = COLORS['schemes'][i % len(COLORS['schemes'])]
        scheme_markers[scheme] = markers[i % len(markers)]
    
    # Plot data points grouped by scheme
    plotted_schemes = set()  # Track which schemes we've added to legend
    
    for i, (h, error_u, n_cells, scheme, mesh_type) in enumerate(zip(
        h_array, error_u_array, 
        results['n_cells'], results['scheme'], results['mesh_type'])):
        
        color = scheme_colors[scheme]
        marker = scheme_markers[scheme]
        
        # Only add to legend if this scheme hasn't been plotted yet
        if scheme not in plotted_schemes:
            # Include observed order in legend label
            if not np.isnan(p_u):
                legend_label = f"{scheme} (order: {p_u:.2f})"
            else:
                legend_label = scheme
            plotted_schemes.add(scheme)
        else:
            legend_label = None  # Don't add duplicate entries to legend
        
        # Plot data points
        ax.loglog(h, error_u, marker=marker, color=color, markersize=8, 
                   linewidth=2, label=legend_label, markerfacecolor='white', 
                   markeredgewidth=1.5)
        
        # Store for simulation ID display
        plot_data.append({
            'color': color,
            'marker': marker,
            'label': f"{n_cells} cells, {scheme}, {mesh_type}",
            'sim_id': results.get('sim_id', [None])[i] if 'sim_id' in results else None
        })
    
    # Connect points with lines if we have more than one point
    # Use single color if all same scheme, or neutral color if mixed
    if len(h_array) > 1:
        if len(schemes) == 1:
            # All same scheme - use scheme color
            line_color = scheme_colors[schemes[0]]
        else:
            # Mixed schemes - use neutral color
            line_color = 'k'
        ax.loglog(h_array, error_u_array, '-', color=line_color, 
                 alpha=0.6, linewidth=1.5, zorder=0)
    
    # Add reference slopes using the minimum error as reference
    error_ref_u = np.min(error_u_array)
    
    # Create reference grid size range
    h_ref = np.array([h_array.min() * 0.8, h_array.max() * 1.2])
    
    # First order reference (O(h^1))
    ref_1st_u = error_ref_u * 2 * (h_ref / h_ref[0])**1
    ax.loglog(h_ref, ref_1st_u, 'k:', alpha=0.7, linewidth=1.5, label=r'$\mathcal{O}(h^1)$')
    
    # Second order reference (O(h^2))
    ref_2nd_u = error_ref_u * 2 * (h_ref / h_ref[0])**2
    ax.loglog(h_ref, ref_2nd_u, 'k--', alpha=0.7, linewidth=1.5, label=r'$\mathcal{O}(h^2)$')
    
    # Formatting with proper grid and mathematical notation
    ax.set_xlabel(r"Grid size $h$", fontsize=12)
    ax.set_ylabel(r"L2 Error", fontsize=12)
    ax.set_title(r"U-Velocity Centerline Convergence", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    
    # Main title with proper formatting (requested format)
    fig.suptitle(f'Grid Refinement - Lid Driven Cavity - Re={Re}', 
                fontsize=16, y=0.98)
    
    # Add simulation ID annotations (similar to compare_lid_driven_cavity.py)
    # Group simulation IDs by marker/color combination
    marker_sim_ids = {}
    for data in plot_data:
        sim_id = data['sim_id']
        if sim_id:
            key = (data['marker'], data['color'])
            if key not in marker_sim_ids:
                marker_sim_ids[key] = []
            marker_sim_ids[key].append(sim_id)
    
    if marker_sim_ids:
        fig.subplots_adjust(bottom=0.12)
        y_row = 0.03  # Fixed position well below the plots
        dx = 0.018    # Offset for text to the right of marker
        
        marker_groups = list(marker_sim_ids.items())
        n = len(marker_groups)
        
        for i, ((marker, color), sim_ids) in enumerate(marker_groups):
            # Distribute across the full figure width with margin
            x_fig = 0.15 + (i + 1) / (n + 1) * 0.7
            # Draw marker using ax.plot like in compare_lid_driven_cavity.py
            ax.plot([x_fig], [y_row], marker=marker, color=color, markersize=8, 
                     markerfacecolor='white', linestyle='None', 
                     transform=fig.transFigure, clip_on=False)
            # Draw sim_ids text just to the right of the marker
            if len(sim_ids) == 1:
                id_text = f"Simulation ID: {sim_ids[0]}"
            else:
                id_text = f"Simulation IDs: {', '.join(sim_ids)}"
            fig.text(x_fig + dx, y_row, id_text, color='grey', 
                    alpha=0.7, fontsize=7, ha='left', va='center', 
                    transform=fig.transFigure)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{title}_grid_convergence_Re_{Re}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved convergence plot to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Grid convergence study for lid-driven cavity CFD simulations (u-velocity centerline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a config list with different mesh resolutions
  echo "experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml" > mesh_study.txt
  echo "experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml" >> mesh_study.txt
  echo "experiments/lidDrivenCavity/ForReport/uniform/fine/Re_100/config.yaml" >> mesh_study.txt
  
  # Run grid convergence study
  python grid_convergence_study.py --config-list mesh_study.txt
  
  # With custom output directory
  python grid_convergence_study.py --config-list mesh_study.txt --output-dir convergence_results

Config list file format:
  Each line should contain a path to a config file:
  experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml
  experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml
  experiments/lidDrivenCavity/ForReport/uniform/fine/Re_100/config.yaml
  # Comments and empty lines are ignored
        """
    )
    
    parser.add_argument('--config-list', required=True,
                       help='Path to text file containing config file paths (one per line)')
    
    # Get the directory where this script is located (postprocessing/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'convergence_studies')
    
    parser.add_argument('--output-dir', default=default_output_dir,
                       help=f'Directory to save results (default: {default_output_dir})')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config_list):
        raise FileNotFoundError(f"Config list file not found: {args.config_list}")
    
    # Perform convergence study
    perform_convergence_study(args.config_list, args.output_dir)

if __name__ == '__main__':
    main() 