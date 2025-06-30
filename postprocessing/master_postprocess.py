#!/usr/bin/env python3
"""
Master postprocessing script for NaviFlow CFD simulations.

This script provides a unified interface for all postprocessing tasks:
- Standard postprocessing (plots, verification, convergence analysis)  
- Appendix generation (thesis figures saved to AppendixPlots)
- LaTeX code generation for appendix

Usage:
    # Process everything in experiments directory
    python master_postprocess.py

    # Process specific experiment directory
    python master_postprocess.py --experiment-dir experiments/Collocated/lidDrivenCavity

    # Postprocess only (no appendix generation)
    python master_postprocess.py --postprocess-only

    # Appendix only (thesis figures + LaTeX, no standard postprocessing)
    python master_postprocess.py --appendix-only

    # Combine options
    python master_postprocess.py --experiment-dir experiments/Collocated/lidDrivenCavity --postprocess-only

    # Control parallel workers
    python master_postprocess.py --max-workers 8

    # Include grid convergence analysis
    python master_postprocess.py --include-convergence
"""

import os
import argparse
import yaml
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import glob
import concurrent.futures
import multiprocessing
from functools import partial


def find_config_files(base_dir: str) -> List[str]:
    """
    Find all config.yaml files in the given directory tree.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of paths to config.yaml files
    """
    config_files = []
    
    # Search for config.yaml files recursively
    pattern = os.path.join(base_dir, "**/config.yaml")
    config_files.extend(glob.glob(pattern, recursive=True))
    
    # Also search for pseudo_config.yaml files (for backward compatibility)
    pattern = os.path.join(base_dir, "**/pseudo_config.yaml")
    config_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(config_files)


def has_results_directory(config_path: str) -> bool:
    """
    Check if the experiment has a results directory with the required files.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        True if results directory exists with required files
    """
    experiment_dir = os.path.dirname(config_path)
    results_dir = os.path.join(experiment_dir, "results")
    
    if not os.path.exists(results_dir):
        return False
    
    # Check for essential files
    required_files = [
        "U_final.npy",
        "p_final.npy", 
        "residuals.npz",
        "cell_centers.npz",
        "metadata.yaml"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(results_dir, file)):
            return False
    
    return True


def run_postprocessing(config_path: str) -> Tuple[bool, str]:
    """
    Run standard postprocessing for a single experiment.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Tuple of (success, config_path)
    """
    try:
        # Run the postprocess script with --all flag
        cmd = [
            sys.executable,
            "postprocessing/postprocess.py",
            "--config", config_path,
            "--all"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"    Error in postprocessing {config_path}: {result.stderr}")
            return False, config_path
        
        return True, config_path
        
    except Exception as e:
        print(f"    Error running postprocessing {config_path}: {e}")
        return False, config_path


def run_appendix_generation_single(config_path: str) -> Tuple[bool, str]:
    """
    Generate appendix plot (thesis figure) for a single experiment.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Tuple of (success, config_path)
    """
    try:
        # Run generate_appendix.py for this config
        cmd = [
            sys.executable,
            "postprocessing/generate_appendix.py",
            "--config", config_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"    Warning: Failed to generate appendix for {config_path}: {result.stderr}")
            return False, config_path
        else:
            return True, config_path
            
    except Exception as e:
        print(f"    Warning: Error generating appendix for {config_path}: {e}")
        return False, config_path


def run_ghia_comparison(config_paths: List[str]) -> bool:
    """
    Run Ghia comparison for lid-driven cavity experiments.
    
    Args:
        config_paths: List of config file paths
        
    Returns:
        True if successful, False otherwise
    """
    # Filter for lid-driven cavity experiments
    ldc_configs = []
    
    for config_path in config_paths:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config.get('experiment') == 'lidDrivenCavity':
                ldc_configs.append(config_path)
        except Exception as e:
            print(f"Warning: Could not read config {config_path}: {e}")
            continue
    
    if not ldc_configs:
        print("  No lid-driven cavity experiments found for Ghia comparison")
        return True
    
    print(f"  Running Ghia comparison for {len(ldc_configs)} lid-driven cavity experiments")
    
    try:
        # Create a temporary config list file
        config_list_file = "temp_ldc_configs.txt"
        with open(config_list_file, 'w') as f:
            for config_path in ldc_configs:
                f.write(f"{config_path}\n")
        
        # Run the comparison script
        cmd = [
            sys.executable,
            "postprocessing/compare_lid_driven_cavity.py",
            "--config-list", config_list_file,
            "--output-dir", "postprocessing/lid_cavity_comparisons"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        # Clean up temp file
        if os.path.exists(config_list_file):
            os.remove(config_list_file)
        
        if result.returncode != 0:
            print(f"    Error in Ghia comparison: {result.stderr}")
            return False
        
        print(f"    ✓ Ghia comparison completed")
        return True
        
    except Exception as e:
        print(f"    Error running Ghia comparison: {e}")
        return False


def run_parallel_postprocessing(config_paths: List[str], max_workers: int) -> int:
    """
    Run postprocessing in parallel for multiple experiments.
    
    Args:
        config_paths: List of config file paths
        max_workers: Maximum number of parallel workers
        
    Returns:
        Number of successfully processed experiments
    """
    print(f"  Running postprocessing in parallel with {max_workers} workers...")
    
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(run_postprocessing, config_path): config_path 
                  for config_path in config_paths}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            config_path = futures[future]
            try:
                success, _ = future.result()
                if success:
                    success_count += 1
                    print(f"    ✓ Completed: {config_path}")
                else:
                    print(f"    ✗ Failed: {config_path}")
            except Exception as e:
                print(f"    ✗ Exception in {config_path}: {e}")
    
    return success_count


def run_parallel_appendix_generation(config_paths: List[str], max_workers: int) -> int:
    """
    Generate appendix plots in parallel for multiple experiments.
    
    Args:
        config_paths: List of config file paths
        max_workers: Maximum number of parallel workers
        
    Returns:
        Number of successfully processed experiments
    """
    print(f"  Generating appendix plots in parallel with {max_workers} workers...")
    
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(run_appendix_generation_single, config_path): config_path 
                  for config_path in config_paths}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            config_path = futures[future]
            try:
                success, _ = future.result()
                if success:
                    success_count += 1
                    print(f"    ✓ Appendix completed: {config_path}")
                else:
                    print(f"    ✗ Appendix failed: {config_path}")
            except Exception as e:
                print(f"    ✗ Appendix exception in {config_path}: {e}")
    
    return success_count


def run_latex_generation() -> bool:
    """
    Generate LaTeX appendix code from AppendixPlots directory.
    
    Returns:
        True if successful, False otherwise
    """
    print("  Generating LaTeX appendix code...")
    
    try:
        cmd = [
            sys.executable,
            "postprocessing/generate_latex_appendix.py",
            "--output", "AppendixPlots/appendix.tex",
            "--plots-dir", "AppendixPlots"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"    Error generating LaTeX: {result.stderr}")
            return False
        
        print(f"    ✓ LaTeX appendix code generated")
        return True
        
    except Exception as e:
        print(f"    Error generating LaTeX: {e}")
        return False


def run_grid_convergence_analysis(max_workers: int) -> bool:
    """
    Run grid convergence analysis for all txt files in lid_cavity_comparisons.
    
    Args:
        max_workers: Maximum number of parallel workers
        
    Returns:
        True if successful, False otherwise
    """
    print("  Running grid convergence analysis...")
    
    try:
        cmd = [
            sys.executable,
            "postprocessing/run_grid_convergence_analysis.py",
            "--max-workers", str(max_workers)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"    Error in grid convergence analysis: {result.stderr}")
            return False
        
        print(f"    ✓ Grid convergence analysis completed")
        return True
        
    except Exception as e:
        print(f"    Error running grid convergence analysis: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Master postprocessing script for NaviFlow CFD simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                           # Process all experiments
  %(prog)s --experiment-dir experiments/Collocated  # Process specific directory
  %(prog)s --postprocess-only                       # Standard postprocessing only
  %(prog)s --appendix-only                          # Appendix generation only
  %(prog)s --max-workers 8                          # Use 8 parallel workers
  %(prog)s --include-convergence                    # Include grid convergence analysis
        """
    )
    
    parser.add_argument(
        "--experiment-dir",
        help="Specific experiment directory to process (default: process all in experiments/)"
    )
    
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Run only standard postprocessing (no appendix generation)"
    )
    
    parser.add_argument(
        "--appendix-only", 
        action="store_true",
        help="Run only appendix generation (thesis figures + LaTeX code)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(8, multiprocessing.cpu_count()),
        help="Maximum number of parallel workers (default: min(8, CPU count))"
    )
    
    parser.add_argument(
        "--include-convergence",
        action="store_true",
        help="Include grid convergence analysis for all txt files in lid_cavity_comparisons"
    )
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.postprocess_only and args.appendix_only:
        print("Error: --postprocess-only and --appendix-only are mutually exclusive")
        sys.exit(1)
    
    # Determine base directory
    if args.experiment_dir:
        base_dir = args.experiment_dir
        if not os.path.exists(base_dir):
            print(f"Error: Experiment directory not found: {base_dir}")
            sys.exit(1)
    else:
        base_dir = "experiments"
        if not os.path.exists(base_dir):
            print(f"Error: Experiments directory not found: {base_dir}")
            sys.exit(1)
    
    print(f"Master postprocessing starting...")
    print(f"Base directory: {base_dir}")
    print(f"Max workers: {args.max_workers}")
    
    # Find all config files
    config_files = find_config_files(base_dir)
    if not config_files:
        print(f"No config files found in {base_dir}")
        sys.exit(1)
    
    print(f"Found {len(config_files)} config files")
    
    # Filter for experiments with results
    valid_configs = []
    for config_path in config_files:
        if has_results_directory(config_path):
            valid_configs.append(config_path)
        else:
            print(f"  Skipping {config_path} (no results directory)")
    
    if not valid_configs:
        print("No experiments with results found")
        sys.exit(1)
    
    print(f"Processing {len(valid_configs)} experiments with results")
    
    success = True
    
    # Standard postprocessing
    if not args.appendix_only:
        print("\n1. Running standard postprocessing...")
        postprocess_success = run_parallel_postprocessing(valid_configs, args.max_workers)
        print(f"   Completed postprocessing for {postprocess_success}/{len(valid_configs)} experiments")
        
        # Ghia comparison (for lid-driven cavity experiments)
        print("\n2. Running Ghia comparison...")
        ghia_success = run_ghia_comparison(valid_configs)
        if not ghia_success:
            success = False
        
        # Grid convergence analysis (if requested)
        if args.include_convergence:
            print("\n3. Running grid convergence analysis...")
            convergence_success = run_grid_convergence_analysis(args.max_workers)
            if not convergence_success:
                success = False
    
    # Appendix generation
    if not args.postprocess_only:
        print("\n4. Generating appendix plots...")
        appendix_success = run_parallel_appendix_generation(valid_configs, args.max_workers)
        print(f"   Generated appendix plots for {appendix_success}/{len(valid_configs)} experiments")
        if appendix_success == 0:
            success = False
        
        print("\n5. Generating LaTeX code...")
        latex_success = run_latex_generation()
        if not latex_success:
            success = False
    
    if success:
        print("\n✓ Master postprocessing completed successfully!")
    else:
        print("\n⚠ Master postprocessing completed with some errors")
        sys.exit(1)


if __name__ == "__main__":
    main() 