#!/usr/bin/env python3
"""
Run comprehensive postprocessing for all experiments.

This script can run both individual experiment postprocessing and thesis figure generation
for all experiments found in the specified directory.

Usage:
    python postprocess_all.py [--experiments-dir DIR] [--output-dir DIR] [--postprocess-only] [--appendix-only] [--max-workers N]

Examples:
    # Run both postprocessing and appendix generation
    python postprocess_all.py
    
    # Run only individual postprocessing
    python postprocess_all.py --postprocess-only
    
    # Run only appendix generation
    python postprocess_all.py --appendix-only
    
    # Specify custom directories
    python postprocess_all.py --experiments-dir experiments/lidDrivenCavity --output-dir thesis_figures
    
    # Control parallelism
    python postprocess_all.py --max-workers 4
"""

import os
import subprocess
import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

def find_config_files(root):
    """Find all config.yaml files in the directory tree."""
    config_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "config.yaml" in filenames:
            config_path = os.path.join(dirpath, "config.yaml")
            config_files.append(config_path)
    return config_files

def postprocess_config(config_path):
    """Run postprocessing for a single config file."""
    print(f"Starting postprocessing: {config_path}")
    try:
        subprocess.run([
            "python",
            "postprocessing/postprocess.py",
            "--config", config_path,
            "--all"
        ], check=True)
        print(f"✅ Completed postprocessing: {config_path}")
        return f"SUCCESS: {config_path}"
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed postprocessing: {config_path} - {e}")
        return f"FAILED: {config_path} - {e}"
    except Exception as e:
        print(f"❌ Error postprocessing: {config_path} - {e}")
        return f"ERROR: {config_path} - {e}"

def run_individual_postprocessing(experiments_dir, max_workers):
    """Run individual postprocessing for all config files."""
    print("=" * 60)
    print("RUNNING INDIVIDUAL POSTPROCESSING")
    print("=" * 60)
    
    all_configs = find_config_files(experiments_dir)
    
    if not all_configs:
        print(f"No config.yaml files found in {experiments_dir}")
        return []
    
    print(f"Found {len(all_configs)} config files to process")
    
    # Use all available CPU cores, but cap at 8 to avoid overwhelming the system
    num_processes = min(cpu_count(), 8, len(all_configs), max_workers or cpu_count())
    print(f"Using {num_processes} parallel processes for postprocessing")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(postprocess_config, all_configs)
    
    return results

def run_appendix_generation(experiments_dir, output_dir, max_workers):
    """Run thesis figure generation for all experiments."""
    print("=" * 60)
    print("RUNNING THESIS FIGURE GENERATION")
    print("=" * 60)
    
    script_path = Path(__file__).parent / "generate_appendix_all.py"
    
    if not script_path.exists():
        print(f"Error: generate_appendix_all.py not found at {script_path}")
        return ["ERROR: generate_appendix_all.py script not found"]
    
    try:
        cmd = [
            "python", 
            str(script_path),
            "--experiments-dir", experiments_dir
        ]
        
        if output_dir:
            cmd.extend(["--output-dir", output_dir])
        
        if max_workers:
            cmd.extend(["--max-workers", str(max_workers)])
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Completed appendix generation")
        print("Appendix generation output:")
        print(result.stdout)
        
        return ["SUCCESS: Appendix generation completed"]
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed appendix generation: {e}")
        print("Error output:")
        print(e.stderr)
        return [f"FAILED: Appendix generation - {e}"]
    except Exception as e:
        print(f"❌ Error in appendix generation: {e}")
        return [f"ERROR: Appendix generation - {e}"]

def print_summary(postprocess_results, appendix_results):
    """Print summary of all operations."""
    print("\n" + "="*60)
    print("COMPREHENSIVE SUMMARY")
    print("="*60)
    
    total_operations = len(postprocess_results) + len(appendix_results)
    
    if postprocess_results:
        postprocess_success = len([r for r in postprocess_results if r.startswith("SUCCESS")])
        postprocess_failed = len(postprocess_results) - postprocess_success
        
        print(f"INDIVIDUAL POSTPROCESSING:")
        print(f"  Total configs: {len(postprocess_results)}")
        print(f"  ✅ Successful: {postprocess_success}")
        print(f"  ❌ Failed: {postprocess_failed}")
        
        if postprocess_failed > 0:
            print("  Failed configs:")
            for result in postprocess_results:
                if not result.startswith("SUCCESS"):
                    print(f"    • {result}")
        print()
    
    if appendix_results:
        appendix_success = len([r for r in appendix_results if r.startswith("SUCCESS")])
        appendix_failed = len(appendix_results) - appendix_success
        
        print(f"THESIS FIGURE GENERATION:")
        print(f"  ✅ Successful: {appendix_success}")
        print(f"  ❌ Failed: {appendix_failed}")
        
        if appendix_failed > 0:
            print("  Failures:")
            for result in appendix_results:
                if not result.startswith("SUCCESS"):
                    print(f"    • {result}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive postprocessing for all experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script can perform two types of operations:

1. Individual Postprocessing:
   - Finds all config.yaml files in the experiments directory
   - Runs postprocess.py --config <file> --all for each
   - Processes multiple configs in parallel

2. Thesis Figure Generation:
   - Runs generate_appendix_all.py to create thesis figures
   - Processes all valid experiments found in the directory
   - Generates comprehensive single-page figures

Examples:
  python postprocess_all.py                                    # Run both operations
  python postprocess_all.py --postprocess-only                # Only individual postprocessing
  python postprocess_all.py --appendix-only                   # Only thesis figures
  python postprocess_all.py --experiments-dir experiments/lidDrivenCavity
  python postprocess_all.py --output-dir thesis_figures --max-workers 4
        """
    )
    
    parser.add_argument(
        '--experiments-dir',
        default='experiments',
        help='Root directory containing experiments (default: experiments)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for thesis figures (default: each experiment\'s results directory)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Maximum number of parallel workers (default: CPU count, capped at 8)'
    )
    parser.add_argument(
        '--postprocess-only',
        action='store_true',
        help='Run only individual postprocessing (skip appendix generation)'
    )
    parser.add_argument(
        '--appendix-only',
        action='store_true',
        help='Run only thesis figure generation (skip individual postprocessing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually running anything'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.postprocess_only and args.appendix_only:
        print("Error: Cannot specify both --postprocess-only and --appendix-only")
        sys.exit(1)
    
    if not os.path.isdir(args.experiments_dir):
        print(f"Error: Experiments directory not found: {args.experiments_dir}")
        sys.exit(1)
    
    print("Comprehensive Postprocessing Runner")
    print(f"Experiments directory: {os.path.abspath(args.experiments_dir)}")
    if args.output_dir:
        print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Max workers: {args.max_workers or 'CPU count (capped at 8)'}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE ***")
        
        # Show what configs would be processed
        configs = find_config_files(args.experiments_dir)
        print(f"\nWould process {len(configs)} config files:")
        for config in configs[:10]:  # Show first 10
            print(f"  • {config}")
        if len(configs) > 10:
            print(f"  ... and {len(configs) - 10} more")
        
        # Show what appendix generation would do
        if not args.postprocess_only:
            print(f"\nWould run appendix generation on: {args.experiments_dir}")
            if args.output_dir:
                print(f"Output would go to: {args.output_dir}")
        
        return 0
    
    postprocess_results = []
    appendix_results = []
    
    # Run individual postprocessing
    if not args.appendix_only:
        postprocess_results = run_individual_postprocessing(args.experiments_dir, args.max_workers)
    
    # Run appendix generation
    if not args.postprocess_only:
        appendix_results = run_appendix_generation(args.experiments_dir, args.output_dir, args.max_workers)
    
    # Print comprehensive summary
    print_summary(postprocess_results, appendix_results)
    
    # Return appropriate exit code
    total_failed = (
        len([r for r in postprocess_results if not r.startswith("SUCCESS")]) +
        len([r for r in appendix_results if not r.startswith("SUCCESS")])
    )
    
    return 1 if total_failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main())