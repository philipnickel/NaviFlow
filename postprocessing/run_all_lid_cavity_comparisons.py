#!/usr/bin/env python3
"""
Run lid-driven cavity comparison for all txt files in lid_cavity_comparisons directory in parallel.

This script automatically finds all .txt files in the lid_cavity_comparisons directory
and runs the compare_lid_driven_cavity.py script for each one in parallel using multiprocessing.

Usage:
    python run_all_lid_cavity_comparisons.py [--max-workers N]

Examples:
    # Run with default number of workers (CPU count)
    python run_all_lid_cavity_comparisons.py
    
    # Run with specific number of parallel workers
    python run_all_lid_cavity_comparisons.py --max-workers 4
"""

import os
import sys
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

def run_comparison(txt_file_path, python_script_path, output_dir):
    """Run comparison for a single txt file.
    
    Args:
        txt_file_path (str): Path to the txt config list file
        python_script_path (str): Path to the compare_lid_driven_cavity.py script
        output_dir (str): Base output directory (not used, we create plots dir next to txt file)
        
    Returns:
        tuple: (filename, success, message)
    """
    filename = os.path.basename(txt_file_path)
    
    # Create plots directory in the same folder as the txt file
    txt_dir = os.path.dirname(txt_file_path)
    plots_dir = os.path.join(txt_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Run the comparison script
        cmd = [
            sys.executable,  # Use the same Python interpreter
            python_script_path,
            "--config-list", txt_file_path,
            "--output-dir", plots_dir  # Use the plots directory next to txt file
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per comparison
        )
        
        if result.returncode == 0:
            return (filename, True, f"Success → {plots_dir}")
        else:
            return (filename, False, f"Error: {result.stderr.strip()}")
            
    except subprocess.TimeoutExpired:
        return (filename, False, "Timeout: Comparison took longer than 5 minutes")
    except Exception as e:
        return (filename, False, f"Exception: {str(e)}")

def find_txt_files(comparison_dir):
    """Find all .txt files in the comparison directory and its subdirectories.
    
    Args:
        comparison_dir (str): Path to the comparison directory
        
    Returns:
        list: List of paths to .txt files
    """
    comparison_path = Path(comparison_dir)
    return list(comparison_path.rglob("*.txt"))  # rglob for recursive search

def main():
    parser = argparse.ArgumentParser(
        description="Run lid-driven cavity comparison for all txt files in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Find all .txt files in the lid_cavity_comparisons directory
2. Run compare_lid_driven_cavity.py for each file in parallel
3. Report results for each comparison

The number of parallel workers defaults to the number of CPU cores.
        """
    )
    
    parser.add_argument(
        "--max-workers", 
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: number of CPU cores)"
    )
    
    args = parser.parse_args()
    
    # Get script directory and paths
    script_dir = Path(__file__).parent.absolute()
    comparison_dir = script_dir / "lid_cavity_comparisons"
    python_script = script_dir / "compare_lid_driven_cavity.py"
    
    # Validate paths
    if not comparison_dir.exists():
        print(f"Error: Directory {comparison_dir} does not exist")
        sys.exit(1)
    
    if not python_script.exists():
        print(f"Error: Python script {python_script} does not exist")
        sys.exit(1)
    
    # Find all txt files
    txt_files = find_txt_files(comparison_dir)
    
    if not txt_files:
        print(f"No .txt files found in {comparison_dir}")
        sys.exit(1)
    
    print(f"Found {len(txt_files)} .txt files in {comparison_dir}")
    print(f"Running comparisons with {args.max_workers or os.cpu_count()} parallel workers...")
    print("=" * 60)
    
    # Track timing
    start_time = time.time()
    
    # Run comparisons in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                run_comparison, 
                str(txt_file), 
                str(python_script), 
                str(comparison_dir)
            ): txt_file.name 
            for txt_file in txt_files
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_file):
            filename, success, message = future.result()
            results.append((filename, success, message))
            
            # Print immediate feedback
            status = "✓" if success else "✗"
            print(f"{status} {filename}: {message}")
    
    # Summary
    end_time = time.time()
    elapsed = end_time - start_time
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print("=" * 60)
    print(f"Completed {len(results)} comparisons in {elapsed:.1f} seconds")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Results saved in individual 'plots' directories next to each txt file")
    
    if failed > 0:
        print("\nFailed comparisons:")
        for filename, success, message in results:
            if not success:
                print(f"  {filename}: {message}")
        sys.exit(1)
    else:
        print("\nAll comparisons completed successfully!")

if __name__ == "__main__":
    main() 