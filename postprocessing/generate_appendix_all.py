#!/usr/bin/env python3
"""
Generate thesis figures for all CFD experiments in the experiments directory.

This script automatically discovers all valid experiments (those with config files 
and results directories) and generates comprehensive single-page thesis figures for each.

Usage:
    python generate_appendix_all.py [--experiments-dir experiments] [--output-dir output]
"""

import os
import argparse
import glob
from pathlib import Path
import concurrent.futures
import threading
from datetime import datetime

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multi-threading

# Import the core functionality from the single experiment script
from generate_appendix import generate_appendix_pdf

def find_valid_experiments(experiments_dir):
    """Find all valid experiments with config files and results directories."""
    valid_experiments = []
    
    # Search for all possible config files
    config_patterns = [
        "config.yaml",
        "pseudo_config.yaml",
        "simulation.yaml"
    ]
    
    print(f"Scanning {experiments_dir} for valid experiments...")
    
    # Walk through all directories
    for root, dirs, files in os.walk(experiments_dir):
        # Check if any config file exists
        config_file = None
        for pattern in config_patterns:
            if pattern in files:
                config_file = os.path.join(root, pattern)
                break
        
        if config_file:
            # Check if results directory exists
            results_dir = os.path.join(root, "results")
            if os.path.isdir(results_dir):
                # Check if results directory has the required files
                required_files = ["metadata.yaml", "U_final.npy", "p_final.npy"]
                has_required = all(
                    os.path.exists(os.path.join(results_dir, f)) 
                    for f in required_files
                )
                
                if has_required:
                    # Get relative path from experiments directory
                    rel_path = os.path.relpath(root, experiments_dir)
                    valid_experiments.append({
                        'path': root,
                        'config': config_file,
                        'results': results_dir,
                        'name': rel_path.replace(os.sep, '_'),
                        'relative_path': rel_path
                    })
                    print(f"  ✓ Found: {rel_path}")
                else:
                    print(f"  ⚠ Missing required files: {os.path.relpath(root, experiments_dir)}")
            else:
                print(f"  ⚠ No results directory: {os.path.relpath(root, experiments_dir)}")
    
    return valid_experiments

def generate_experiment_figure(experiment, output_dir=None, thread_id=None):
    """Generate thesis figure for a single experiment."""
    try:
        thread_prefix = f"[Thread {thread_id}] " if thread_id else ""
        print(f"{thread_prefix}Processing: {experiment['relative_path']}")
        
        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"thesis_figure_{experiment['name']}.pdf")
        else:
            # Use default location in results directory
            output_path = None
        
        # Generate the figure
        result_path = generate_appendix_pdf(experiment['config'], output_path)
        
        if result_path:
            print(f"{thread_prefix}✓ Generated: {result_path}")
            return {
                'experiment': experiment['relative_path'],
                'status': 'success',
                'output': result_path
            }
        else:
            print(f"{thread_prefix}✗ Failed: {experiment['relative_path']}")
            return {
                'experiment': experiment['relative_path'],
                'status': 'failed',
                'error': 'generate_appendix_pdf returned None'
            }
            
    except Exception as e:
        print(f"{thread_prefix}✗ Error processing {experiment['relative_path']}: {e}")
        return {
            'experiment': experiment['relative_path'],
            'status': 'error',
            'error': str(e)
        }

def generate_all_figures(experiments_dir, output_dir=None, max_workers=4):
    """Generate thesis figures for all valid experiments."""
    
    # Find all valid experiments
    experiments = find_valid_experiments(experiments_dir)
    
    if not experiments:
        print("No valid experiments found!")
        return []
    
    print(f"\nFound {len(experiments)} valid experiments")
    print(f"Processing with {max_workers} parallel workers...")
    
    results = []
    
    # Process experiments in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_experiment = {
            executor.submit(generate_experiment_figure, exp, output_dir, i+1): exp 
            for i, exp in enumerate(experiments)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_experiment):
            result = future.result()
            results.append(result)
    
    return results

def print_summary(results):
    """Print a summary of the processing results."""
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] in ['failed', 'error']]
    
    print(f"Total experiments processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ SUCCESSFUL EXPERIMENTS ({len(successful)}):")
        for result in successful:
            print(f"  • {result['experiment']}")
            print(f"    → {result['output']}")
    
    if failed:
        print(f"\n✗ FAILED EXPERIMENTS ({len(failed)}):")
        for result in failed:
            print(f"  • {result['experiment']}")
            print(f"    → Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Generate thesis figures for all CFD experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_appendix_all.py
  python generate_appendix_all.py --experiments-dir experiments --output-dir thesis_figures
  python generate_appendix_all.py --max-workers 8
        """
    )
    
    parser.add_argument(
        '--experiments-dir', 
        default='experiments',
        help='Directory containing experiments (default: experiments)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for all figures (default: each experiment\'s results directory)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only find and list experiments, don\'t generate figures'
    )
    
    args = parser.parse_args()
    
    # Validate experiments directory
    if not os.path.isdir(args.experiments_dir):
        print(f"Error: Experiments directory not found: {args.experiments_dir}")
        return 1
    
    print(f"Thesis Figure Generator - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments directory: {os.path.abspath(args.experiments_dir)}")
    if args.output_dir:
        print(f"Output directory: {os.path.abspath(args.output_dir)}")
    else:
        print("Output: Each experiment's results directory")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No figures will be generated ***")
        experiments = find_valid_experiments(args.experiments_dir)
        print(f"\nFound {len(experiments)} valid experiments:")
        for exp in experiments:
            print(f"  • {exp['relative_path']}")
            print(f"    Config: {exp['config']}")
            print(f"    Results: {exp['results']}")
        return 0
    
    # Generate all figures
    results = generate_all_figures(
        args.experiments_dir, 
        args.output_dir, 
        args.max_workers
    )
    
    # Print summary
    print_summary(results)
    
    # Return appropriate exit code
    failed_count = len([r for r in results if r['status'] in ['failed', 'error']])
    return 1 if failed_count > 0 else 0

if __name__ == '__main__':
    exit(main()) 