#!/usr/bin/env python3
"""
Run grid convergence analysis for all txt files in lid_cavity_comparisons directory.

This script automatically finds all .txt files in the lid_cavity_comparisons directory
and runs compare_lid_driven_cavity.py for each one individually, saving the results
in the same location as the respective txt file.

Usage:
    python postprocessing/run_grid_convergence_analysis.py [--max-workers N]
"""

import os
import argparse
import subprocess
import sys
import glob
from pathlib import Path
import concurrent.futures
import multiprocessing
from typing import List, Tuple


def find_txt_files(base_dir: str) -> List[str]:
    """
    Find all .txt files in the lid_cavity_comparisons directory.
    
    Args:
        base_dir: Base directory to search (lid_cavity_comparisons)
        
    Returns:
        List of paths to .txt files
    """
    txt_files = []
    
    # Search for .txt files recursively
    pattern = os.path.join(base_dir, "**/*.txt")
    txt_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(txt_files)


def run_convergence_analysis(txt_file_path: str) -> Tuple[bool, str]:
    """
    Run compare_lid_driven_cavity.py for a single txt file.
    
    Args:
        txt_file_path: Path to the txt file containing config paths
        
    Returns:
        Tuple of (success, txt_file_path)
    """
    try:
        # Get the directory where the txt file is located
        output_dir = os.path.dirname(txt_file_path)
        
        # Get the filename without extension for informative output
        filename = os.path.splitext(os.path.basename(txt_file_path))[0]
        
        print(f"  Processing: {filename}")
        
        # Run the compare_lid_driven_cavity.py script
        cmd = [
            sys.executable,
            "postprocessing/compare_lid_driven_cavity.py",
            "--config-list", txt_file_path,
            "--output-dir", output_dir
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            print(f"    ✗ Error processing {filename}: {result.stderr}")
            return False, txt_file_path
        
        print(f"    ✓ Completed: {filename}")
        return True, txt_file_path
        
    except Exception as e:
        print(f"    ✗ Exception processing {txt_file_path}: {e}")
        return False, txt_file_path


def run_parallel_analysis(txt_files: List[str], max_workers: int) -> int:
    """
    Run convergence analysis in parallel for multiple txt files.
    
    Args:
        txt_files: List of txt file paths
        max_workers: Maximum number of parallel workers
        
    Returns:
        Number of successfully processed files
    """
    print(f"Running convergence analysis with {max_workers} workers...")
    
    success_count = 0
    total_files = len(txt_files)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(run_convergence_analysis, txt_file): txt_file 
            for txt_file in txt_files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            txt_file = futures[future]
            try:
                success, _ = future.result()
                if success:
                    success_count += 1
            except Exception as e:
                print(f"    ✗ Exception in {txt_file}: {e}")
    
    return success_count


def validate_txt_files(txt_files: List[str]) -> List[str]:
    """
    Validate that txt files exist and contain valid config paths.
    
    Args:
        txt_files: List of txt file paths
        
    Returns:
        List of valid txt file paths
    """
    valid_files = []
    
    for txt_file in txt_files:
        if not os.path.exists(txt_file):
            print(f"Warning: File not found: {txt_file}")
            continue
            
        try:
            with open(txt_file, 'r') as f:
                lines = f.read().strip().split('\n')
                
            # Check if file has content and lines are not empty
            valid_lines = [line.strip() for line in lines if line.strip()]
            if len(valid_lines) < 2:
                print(f"Warning: {txt_file} has fewer than 2 config paths, skipping")
                continue
                
            # Check if at least one config file exists
            config_exists = False
            for line in valid_lines:
                if os.path.exists(line):
                    config_exists = True
                    break
                    
            if not config_exists:
                print(f"Warning: No valid config files found in {txt_file}, skipping")
                continue
                
            valid_files.append(txt_file)
            
        except Exception as e:
            print(f"Warning: Error reading {txt_file}: {e}")
            continue
    
    return valid_files


def main():
    parser = argparse.ArgumentParser(
        description="Run grid convergence analysis for all txt files in lid_cavity_comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Process all txt files with default workers
  %(prog)s --max-workers 6    # Use 6 parallel workers
        """
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(6, multiprocessing.cpu_count()),
        help="Maximum number of parallel workers (default: min(6, CPU count))"
    )
    
    parser.add_argument(
        "--base-dir",
        default="postprocessing/lid_cavity_comparisons",
        help="Base directory to search for txt files (default: postprocessing/lid_cavity_comparisons)"
    )
    
    args = parser.parse_args()
    
    # Check if base directory exists
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        sys.exit(1)
    
    print(f"Grid convergence analysis starting...")
    print(f"Base directory: {args.base_dir}")
    print(f"Max workers: {args.max_workers}")
    
    # Find all txt files
    txt_files = find_txt_files(args.base_dir)
    if not txt_files:
        print(f"No txt files found in {args.base_dir}")
        sys.exit(1)
    
    print(f"Found {len(txt_files)} txt files")
    
    # Validate txt files
    valid_files = validate_txt_files(txt_files)
    if not valid_files:
        print("No valid txt files found")
        sys.exit(1)
    
    print(f"Processing {len(valid_files)} valid txt files")
    
    # Show what will be processed
    for txt_file in valid_files:
        rel_path = os.path.relpath(txt_file, args.base_dir)
        print(f"  - {rel_path}")
    
    print()
    
    # Run convergence analysis
    success_count = run_parallel_analysis(valid_files, args.max_workers)
    
    print(f"\n✓ Grid convergence analysis completed!")
    print(f"Successfully processed: {success_count}/{len(valid_files)} files")
    
    if success_count < len(valid_files):
        print(f"Failed: {len(valid_files) - success_count} files")
        sys.exit(1)


if __name__ == "__main__":
    main() 