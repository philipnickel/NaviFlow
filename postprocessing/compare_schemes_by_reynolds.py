#!/usr/bin/env python3
"""
Compare Discretization Schemes for Lid-Driven Cavity at Specific Reynolds Numbers

This script compares all available discretization schemes (Upwind, TVD, QUICK) for a 
given Reynolds number, creating separate plots for uniform and unstructured meshes.
Each plot shows u-velocity along the vertical centerline compared to Ghia's benchmark data.

Usage:
    python compare_schemes_by_reynolds.py --reynolds 100 [--mesh-resolution medium] [--output-dir dir]

Example:
    # Compare all schemes for Re=100 using medium mesh resolution
    python compare_schemes_by_reynolds.py --reynolds 100 --mesh-resolution medium
    
    # Compare all schemes for Re=1000 using fine mesh resolution
    python compare_schemes_by_reynolds.py --reynolds 1000 --mesh-resolution fine

The script will:
1. Automatically find all experiments for the specified Reynolds number
2. Group experiments by mesh type (uniform vs unstructured)
3. Create separate comparison plots for each mesh type
4. Show all available discretization schemes on the same plot
5. Compare against Ghia's benchmark data
6. Save plots as PDFs in the output directory

Requirements:
- Experiments must exist for the specified Reynolds number
- Ghia benchmark data must be available for that Reynolds number
- At least one experiment must be found for each mesh type
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import glob
from pathlib import Path

# Import utilities from the comparison script
from compare_lid_driven_cavity import (
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

def find_experiments_for_reynolds(reynolds_number, mesh_resolution='medium'):
    """
    Find all experiments for a given Reynolds number and mesh resolution.
    
    Args:
        reynolds_number (int): Reynolds number to search for
        mesh_resolution (str): Mesh resolution (coarse, medium, fine)
        
    Returns:
        dict: Dictionary with mesh_type as keys and list of experiment paths as values
    """
    base_path = "experiments/Collocated/lidDrivenCavity"
    schemes = ['Upwind', 'TVD', 'QUICK']
    mesh_types = ['uniform', 'unstructured']
    
    experiments = {'uniform': [], 'unstructured': []}
    
    for scheme in schemes:
        for mesh_type in mesh_types:
            # Construct path to experiment
            exp_path = os.path.join(
                base_path, scheme, mesh_type, f"Re_{reynolds_number}", mesh_resolution
            )
            
            config_path = os.path.join(exp_path, "config.yaml")
            
            # Check if experiment exists
            if os.path.exists(config_path):
                # Verify results directory exists
                results_path = os.path.join(exp_path, "results")
                if os.path.exists(results_path):
                    required_files = [
                        os.path.join(results_path, "U_final.npy"),
                        os.path.join(results_path, "cell_centers.npz"),
                        os.path.join(results_path, "metadata.yaml")
                    ]
                    
                    if all(os.path.exists(f) for f in required_files):
                        experiments[mesh_type].append(exp_path)
                        print(f"Found: {scheme} {mesh_type} Re={reynolds_number} {mesh_resolution}")
                    else:
                        print(f"Missing files in: {exp_path}")
                else:
                    print(f"No results directory: {exp_path}")
            else:
                print(f"Not found: {exp_path}")
    
    return experiments

def create_scheme_comparison_plot(experiments, mesh_type, reynolds_number, 
                                mesh_resolution, output_dir):
    """
    Create comparison plot for all schemes of a given mesh type.
    
    Args:
        experiments (list): List of experiment paths
        mesh_type (str): Mesh type (uniform or unstructured)
        reynolds_number (int): Reynolds number
        mesh_resolution (str): Mesh resolution
        output_dir (str): Output directory
    """
    if not experiments:
        print(f"No experiments found for {mesh_type} mesh type")
        return
    
    # Get Ghia's benchmark data
    ghia = get_ghia_data(reynolds_number)
    if ghia is None:
        print(f"Error: No Ghia benchmark data available for Re = {reynolds_number}")
        return
    
    # Load experiment data
    exp_data = []
    for exp_path in experiments:
        try:
            data = load_experiment_data(exp_path)
            exp_data.append(data)
        except Exception as e:
            print(f"Error loading experiment {exp_path}: {e}")
    
    if not exp_data:
        print(f"No valid experiments loaded for {mesh_type} mesh type")
        return
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Professional color scheme
    COLORS = {
        'ghia': 'black',
        'schemes': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }
    
    # Define markers for different schemes
    MARKERS = {
        'Upwind': 'o',
        'TVD': 's', 
        'QUICK': '^'
    }
    
    # Plot Ghia's data first
    ax.plot(ghia['u'], ghia['y'], 'ko-', markersize=6, linewidth=2, 
            label="Ghia et al. (1982)", markerfacecolor='white', 
            markeredgewidth=1.5, zorder=10)
    
    # Plot each experiment
    for i, exp in enumerate(exp_data):
        scheme = exp['scheme']
        color = COLORS['schemes'][i % len(COLORS['schemes'])]
        marker = MARKERS.get(scheme, 'D')
        
        # Extract centerline data
        if exp['mesh_type'] == 'uniform':
            y_coords, u_profile = extract_centerline_uniform(
                exp['x'], exp['y'], exp['U'], direction='vertical'
            )
        else:
            y_coords, u_profile = extract_centerline_unstructured(
                exp['x'], exp['y'], exp['U'], direction='vertical', n_points=100
            )
        
        # Create label with simulation info
        label = f"{scheme} ({exp['n_cells']} cells, {exp.get('sim_id', 'unknown')[:8]})"
        
        # Plot numerical solution
        ax.plot(u_profile, y_coords, marker=marker, color=color, 
                linewidth=2, markersize=6, label=label, 
                markerfacecolor='white', markeredgewidth=1.5,
                linestyle='-', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('U velocity', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(f'U-Velocity Centerline Comparison\n{mesh_type.title()} Mesh, {mesh_resolution.title()} Resolution', 
                fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.6, 1.2)
    ax.set_ylim(0, 1)
    
    # Main title
    fig.suptitle(f'Discretization Scheme Comparison - Re={reynolds_number}', 
                fontsize=16, y=0.95)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"scheme_comparison_Re{reynolds_number}_{mesh_type}_{mesh_resolution}.pdf"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {mesh_type} mesh comparison to: {output_path}")

def compare_schemes_for_reynolds(reynolds_number, mesh_resolution, output_dir):
    """
    Compare all schemes for a given Reynolds number.
    
    Args:
        reynolds_number (int): Reynolds number
        mesh_resolution (str): Mesh resolution (coarse, medium, fine)
        output_dir (str): Output directory
    """
    print(f"\nSearching for experiments: Re={reynolds_number}, resolution={mesh_resolution}")
    
    # Find all experiments for this Reynolds number
    experiments = find_experiments_for_reynolds(reynolds_number, mesh_resolution)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison plots for each mesh type
    for mesh_type in ['uniform', 'unstructured']:
        if experiments[mesh_type]:
            print(f"\nCreating {mesh_type} mesh comparison plot...")
            create_scheme_comparison_plot(
                experiments[mesh_type], mesh_type, reynolds_number, 
                mesh_resolution, output_dir
            )
        else:
            print(f"No experiments found for {mesh_type} mesh type")

def run_all_reynolds_numbers(mesh_resolution, output_dir):
    """
    Run comparison for all specified Reynolds numbers.
    
    Args:
        mesh_resolution (str): Mesh resolution
        output_dir (str): Output directory
    """
    reynolds_numbers = [100, 400, 1000, 3200, 5000]
    
    for re_num in reynolds_numbers:
        print(f"\n{'='*60}")
        print(f"Processing Reynolds number: {re_num}")
        print(f"{'='*60}")
        
        compare_schemes_for_reynolds(re_num, mesh_resolution, output_dir)

def main():
    parser = argparse.ArgumentParser(
        description='Compare discretization schemes for lid-driven cavity at specific Reynolds numbers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all schemes for Re=100 using medium mesh resolution
  python compare_schemes_by_reynolds.py --reynolds 100 --mesh-resolution medium
  
  # Compare all schemes for Re=1000 using fine mesh resolution
  python compare_schemes_by_reynolds.py --reynolds 1000 --mesh-resolution fine
  
  # Run for all Reynolds numbers (100, 400, 1000, 3200, 5000)
  python compare_schemes_by_reynolds.py --all-reynolds --mesh-resolution medium

The script will create separate plots for uniform and unstructured meshes,
each showing all available discretization schemes compared to Ghia's data.
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--reynolds', type=int,
                       help='Reynolds number to analyze')
    group.add_argument('--all-reynolds', action='store_true',
                       help='Run for all Reynolds numbers (100, 400, 1000, 3200, 5000)')
    
    parser.add_argument('--mesh-resolution', default='medium',
                       choices=['coarse', 'medium', 'fine'],
                       help='Mesh resolution to use (default: medium)')
    
    # Get the directory where this script is located (postprocessing/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'scheme_comparisons')
    
    parser.add_argument('--output-dir', default=default_output_dir,
                       help=f'Directory to save results (default: {default_output_dir})')
    
    args = parser.parse_args()
    
    if args.all_reynolds:
        run_all_reynolds_numbers(args.mesh_resolution, args.output_dir)
    else:
        compare_schemes_for_reynolds(args.reynolds, args.mesh_resolution, args.output_dir)

if __name__ == '__main__':
    main() 