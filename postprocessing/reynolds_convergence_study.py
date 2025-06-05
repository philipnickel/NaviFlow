#!/usr/bin/env python3
"""
Grid Convergence Study by Reynolds Number for Lid-Driven Cavity

This script automatically scans the lid-driven cavity experiment directory and creates 
order of accuracy plots for each Reynolds number. Each plot shows convergence curves 
for all available discretization schemes and mesh types (uniform/unstructured) on the 
same figure, comparing u-velocity centerline errors against Ghia's benchmark data.

Usage:
    python reynolds_convergence_study.py [--reynolds 100] [--output-dir dir]

Example:
    # Analyze all Reynolds numbers
    python reynolds_convergence_study.py
    
    # Analyze specific Reynolds number
    python reynolds_convergence_study.py --reynolds 100

The script will:
1. Automatically scan experiments/Collocated/lidDrivenCavity/ directory
2. Group experiments by Reynolds number, scheme, and mesh type
3. For each combination, find all available mesh resolutions (coarse, medium, fine)
4. Calculate L2 errors for u-velocity centerline against Ghia's data
5. Create convergence plots showing order of accuracy for each scheme+mesh combination
6. Save one plot per Reynolds number with all schemes and mesh types

Requirements:
- Experiments must have results/U_final.npy, cell_centers.npz, metadata.yaml
- Ghia benchmark data must be available for the Reynolds number
- At least 2 mesh resolutions needed for convergence calculation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path
import glob
from collections import defaultdict

# Import utilities from existing scripts
from compare_lid_driven_cavity import (
    load_experiment_data,
    get_ghia_data,
    extract_centerline_uniform,
    extract_centerline_unstructured
)
from grid_convergence_study import (
    interpolate_to_ghia_points,
    calculate_l2_error,
    estimate_grid_size
)

# Set matplotlib backend for non-interactive mode
import matplotlib
matplotlib.use('Agg')

# Import plotting style
from naviflow_collocated.utils.postprocess.plot_style import plt
plt.style.use(['science', 'grid'])

def scan_lid_driven_cavity_experiments():
    """
    Scan the entire lid-driven cavity experiment directory and organize by Reynolds number.
    
    Returns:
        dict: Nested dictionary organized as:
              {reynolds_number: {scheme: {mesh_type: {resolution: experiment_path}}}}
    """
    base_path = "experiments/Collocated/lidDrivenCavity"
    experiments = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    if not os.path.exists(base_path):
        print(f"Base path not found: {base_path}")
        return experiments
    
    print("Scanning lid-driven cavity experiments...")
    
    # Walk through the directory structure
    for scheme_dir in os.listdir(base_path):
        scheme_path = os.path.join(base_path, scheme_dir)
        if not os.path.isdir(scheme_path) or scheme_dir.startswith('.'):
            continue
            
        scheme = scheme_dir  # Upwind, TVD, QUICK
        
        for mesh_type_dir in os.listdir(scheme_path):
            mesh_type_path = os.path.join(scheme_path, mesh_type_dir)
            if not os.path.isdir(mesh_type_path) or mesh_type_dir.startswith('.'):
                continue
                
            mesh_type = mesh_type_dir  # uniform, unstructured
            
            for re_dir in os.listdir(mesh_type_path):
                re_path = os.path.join(mesh_type_path, re_dir)
                if not os.path.isdir(re_path) or not re_dir.startswith('Re_'):
                    continue
                    
                # Extract Reynolds number
                try:
                    reynolds_number = int(re_dir.split('Re_')[1])
                except (IndexError, ValueError):
                    print(f"Could not parse Reynolds number from: {re_dir}")
                    continue
                
                for resolution_dir in os.listdir(re_path):
                    resolution_path = os.path.join(re_path, resolution_dir)
                    if not os.path.isdir(resolution_path) or resolution_dir.startswith('.'):
                        continue
                        
                    resolution = resolution_dir  # coarse, medium, fine
                    
                    # Check if this is a valid experiment
                    config_path = os.path.join(resolution_path, "config.yaml")
                    results_path = os.path.join(resolution_path, "results")
                    
                    if os.path.exists(config_path) and os.path.exists(results_path):
                        required_files = [
                            os.path.join(results_path, "U_final.npy"),
                            os.path.join(results_path, "cell_centers.npz"),
                            os.path.join(results_path, "metadata.yaml")
                        ]
                        
                        if all(os.path.exists(f) for f in required_files):
                            experiments[reynolds_number][scheme][mesh_type][resolution] = resolution_path
                            print(f"Found: Re={reynolds_number}, {scheme}, {mesh_type}, {resolution}")
    
    return experiments

def calculate_u_centerline_error(exp, ghia):
    """
    Calculate L2 error for u-velocity along vertical centerline.
    
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

def create_convergence_curves_for_reynolds(reynolds_number, experiments_data, output_dir):
    """
    Create convergence plot for all schemes and mesh types for a given Reynolds number.
    
    Args:
        reynolds_number (int): Reynolds number
        experiments_data (dict): Experiment data for this Reynolds number
        output_dir (str): Output directory
    """
    print(f"\nProcessing Reynolds number: {reynolds_number}")
    
    # Get Ghia's benchmark data
    ghia = get_ghia_data(reynolds_number)
    if ghia is None:
        print(f"Error: No Ghia benchmark data available for Re = {reynolds_number}")
        return
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Professional color scheme
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Markers for different combinations
    MARKERS = {
        ('Upwind', 'uniform'): 'o',
        ('Upwind', 'unstructured'): 's',
        ('TVD', 'uniform'): '^',
        ('TVD', 'unstructured'): 'D',
        ('QUICK', 'uniform'): 'v',
        ('QUICK', 'unstructured'): 'p'
    }
    
    color_idx = 0
    convergence_data = []
    
    # Process each scheme and mesh type combination
    for scheme in sorted(experiments_data.keys()):
        for mesh_type in sorted(experiments_data[scheme].keys()):
            resolutions_data = experiments_data[scheme][mesh_type]
            
            if len(resolutions_data) < 2:
                print(f"  Skipping {scheme} {mesh_type}: need at least 2 resolutions for convergence")
                continue
            
            print(f"  Processing {scheme} {mesh_type}...")
            
            # Collect data for all available resolutions
            h_values = []
            error_values = []
            n_cells_values = []
            
            # Process resolutions in order: coarse, medium, fine
            resolution_order = ['coarse', 'medium', 'fine']
            
            for resolution in resolution_order:
                if resolution in resolutions_data:
                    exp_path = resolutions_data[resolution]
                    
                    try:
                        # Load experiment data
                        exp_data = load_experiment_data(exp_path)
                        
                        # Calculate error
                        l2_error = calculate_u_centerline_error(exp_data, ghia)
                        
                        # Calculate grid size
                        h = estimate_grid_size(exp_data)
                        
                        h_values.append(h)
                        error_values.append(l2_error)
                        n_cells_values.append(exp_data['n_cells'])
                        
                        print(f"    {resolution}: {exp_data['n_cells']} cells, h={h:.4f}, error={l2_error:.2e}")
                        
                    except Exception as e:
                        print(f"    Error loading {resolution}: {e}")
            
            if len(h_values) < 2:
                print(f"    Insufficient valid data for {scheme} {mesh_type}")
                continue
            
            # Convert to arrays
            h_array = np.array(h_values)
            error_array = np.array(error_values)
            
            # Calculate order of accuracy using log-log linear regression
            try:
                log_h = np.log(h_array)
                log_error = np.log(error_array)
                slope, intercept = np.polyfit(log_h, log_error, 1)
                order_of_accuracy = slope
            except:
                order_of_accuracy = float('nan')
            
            # Get color and marker
            color = COLORS[color_idx % len(COLORS)]
            marker = MARKERS.get((scheme, mesh_type), 'o')
            color_idx += 1
            
            # Create legend label
            if not np.isnan(order_of_accuracy):
                label = f"{scheme} {mesh_type} (order: {order_of_accuracy:.2f})"
            else:
                label = f"{scheme} {mesh_type}"
            
            # Plot convergence curve
            ax.loglog(h_array, error_array, marker=marker, color=color, 
                     linewidth=2, markersize=8, label=label,
                     markerfacecolor='white', markeredgewidth=1.5)
            
            # Connect points with lines
            ax.loglog(h_array, error_array, '-', color=color, alpha=0.6, linewidth=1.5)
            
            # Store convergence data
            convergence_data.append({
                'scheme': scheme,
                'mesh_type': mesh_type,
                'h_values': h_values,
                'error_values': error_values,
                'n_cells': n_cells_values,
                'order': order_of_accuracy
            })
    
    if not convergence_data:
        print(f"No valid convergence data found for Re = {reynolds_number}")
        return
    
    # Add reference slopes
    if convergence_data:
        # Use minimum error as reference
        all_errors = []
        all_h = []
        for data in convergence_data:
            all_errors.extend(data['error_values'])
            all_h.extend(data['h_values'])
        
        error_ref = np.min(all_errors)
        h_ref = np.array([np.min(all_h) * 0.8, np.max(all_h) * 1.2])
        
        # First order reference
        ref_1st = error_ref * 2 * (h_ref / h_ref[0])**1
        ax.loglog(h_ref, ref_1st, 'k:', alpha=0.7, linewidth=1.5, label=r'$\mathcal{O}(h^1)$')
        
        # Second order reference
        ref_2nd = error_ref * 2 * (h_ref / h_ref[0])**2
        ax.loglog(h_ref, ref_2nd, 'k--', alpha=0.7, linewidth=1.5, label=r'$\mathcal{O}(h^2)$')
    
    # Formatting
    ax.set_xlabel(r"Grid size $h$", fontsize=12)
    ax.set_ylabel(r"L2 Error (U-velocity centerline)", fontsize=12)
    ax.set_title(f'Grid Convergence Study - Re={reynolds_number}', fontsize=16)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"convergence_study_Re{reynolds_number}.pdf"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved convergence plot to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Grid convergence study for all schemes and mesh types by Reynolds number',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all Reynolds numbers
  python reynolds_convergence_study.py
  
  # Analyze specific Reynolds number
  python reynolds_convergence_study.py --reynolds 100

The script automatically scans the lid-driven cavity experiment directory and creates
order of accuracy plots showing convergence curves for all available schemes and mesh types.
        """
    )
    
    parser.add_argument('--reynolds', type=int,
                       help='Specific Reynolds number to analyze (if not provided, analyzes all)')
    
    # Get the directory where this script is located (postprocessing/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'reynolds_convergence_studies')
    
    parser.add_argument('--output-dir', default=default_output_dir,
                       help=f'Directory to save results (default: {default_output_dir})')
    
    args = parser.parse_args()
    
    # Scan all experiments
    all_experiments = scan_lid_driven_cavity_experiments()
    
    if not all_experiments:
        print("No experiments found!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process Reynolds numbers
    if args.reynolds:
        # Process specific Reynolds number
        if args.reynolds in all_experiments:
            create_convergence_curves_for_reynolds(args.reynolds, all_experiments[args.reynolds], args.output_dir)
        else:
            print(f"No experiments found for Re = {args.reynolds}")
            print(f"Available Reynolds numbers: {sorted(all_experiments.keys())}")
    else:
        # Process all Reynolds numbers
        print(f"\nFound experiments for Reynolds numbers: {sorted(all_experiments.keys())}")
        
        for reynolds_number in sorted(all_experiments.keys()):
            create_convergence_curves_for_reynolds(reynolds_number, all_experiments[reynolds_number], args.output_dir)

if __name__ == '__main__':
    main() 