#!/usr/bin/env python3
"""
Run grid convergence studies for all txt files in gridRefinement directory in parallel.

This script automatically finds all .txt files in the gridRefinement directory
and runs the grid_convergence_study.py script for each one in parallel using multiprocessing.
It groups experiments by Reynolds number and generates convergence plots for each group.

Usage:
    python run_all_grid_convergence_studies.py [--max-workers N]

Examples:
    # Run with default number of workers (CPU count)
    python run_all_grid_convergence_studies.py
    
    # Run with specific number of parallel workers
    python run_all_grid_convergence_studies.py --max-workers 4
"""

import os
import sys
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import re

def extract_reynolds_from_filename(filename):
    """Extract Reynolds number from filename.
    
    Args:
        filename (str): Filename like "GridRefinementRe100PowerLaw.txt"
        
    Returns:
        int or None: Reynolds number if found, None otherwise
    """
    # Look for pattern like "Re100", "Re1000", etc.
    match = re.search(r'Re(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def extract_scheme_from_filename(filename):
    """Extract scheme name from filename.
    
    Args:
        filename (str): Filename like "GridRefinementRe100PowerLaw.txt"
        
    Returns:
        str: Scheme name if found, filename without extension otherwise
    """
    # Remove extension and common prefixes
    base = filename.replace('.txt', '')
    base = base.replace('GridRefinement', '')
    
    # Remove Reynolds number pattern
    base = re.sub(r'Re\d+', '', base)
    
    # Clean up any remaining underscores or dashes
    base = base.strip('_-')
    
    return base if base else filename.replace('.txt', '')

def run_grid_convergence_study(txt_file_path, python_script_path, output_dir):
    """Run grid convergence study for a single txt file.
    
    Args:
        txt_file_path (str): Path to the txt config list file
        python_script_path (str): Path to the grid_convergence_study.py script
        output_dir (str): Output directory for results
        
    Returns:
        tuple: (filename, success, message, reynolds_number)
    """
    filename = os.path.basename(txt_file_path)
    reynolds_number = extract_reynolds_from_filename(filename)
    
    # Create plots directory in the same folder as the txt file
    txt_dir = os.path.dirname(txt_file_path)
    plots_dir = os.path.join(txt_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Run the grid convergence study script
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
            timeout=600  # 10 minute timeout per convergence study
        )
        
        if result.returncode == 0:
            return (filename, True, f"Success → {plots_dir}", reynolds_number)
        else:
            return (filename, False, f"Error: {result.stderr.strip()}", reynolds_number)
            
    except subprocess.TimeoutExpired:
        return (filename, False, "Timeout: Study took longer than 10 minutes", reynolds_number)
    except Exception as e:
        return (filename, False, f"Exception: {str(e)}", reynolds_number)

def find_grid_refinement_txt_files(comparison_dir):
    """Find all .txt files in the gridRefinement subdirectory.
    
    Args:
        comparison_dir (str): Path to the lid_cavity_comparisons directory
        
    Returns:
        list: List of paths to .txt files in gridRefinement subdirectory
    """
    grid_refinement_path = Path(comparison_dir) / "gridRefinement"
    
    if not grid_refinement_path.exists():
        return []
    
    return list(grid_refinement_path.glob("*.txt"))

def group_files_by_reynolds(txt_files):
    """Group txt files by Reynolds number.
    
    Args:
        txt_files (list): List of txt file paths
        
    Returns:
        dict: Dictionary mapping Reynolds numbers to lists of files
    """
    groups = {}
    
    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        reynolds = extract_reynolds_from_filename(filename)
        
        if reynolds is not None:
            if reynolds not in groups:
                groups[reynolds] = []
            groups[reynolds].append(txt_file)
        else:
            print(f"Warning: Could not extract Reynolds number from {filename}")
    
    return groups

def main():
    parser = argparse.ArgumentParser(
        description="Run grid convergence studies for all txt files in gridRefinement directory in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Find all .txt files in the gridRefinement subdirectory
2. Group them by Reynolds number (extracted from filenames)
3. Run grid_convergence_study.py for each file in parallel
4. Save results in plots subdirectories next to each txt file

Each txt file should contain config paths for different mesh resolutions
of the same scheme and Reynolds number for proper convergence analysis.

The script expects filenames like:
  GridRefinementRe100PowerLaw.txt
  GridRefinementRe100QUICK.txt
  GridRefinementRe1000PowerLaw.txt
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
    python_script = script_dir / "grid_convergence_study.py"
    
    # Validate paths
    if not comparison_dir.exists():
        print(f"Error: Directory {comparison_dir} does not exist")
        sys.exit(1)
    
    if not python_script.exists():
        print(f"Error: Python script {python_script} does not exist")
        sys.exit(1)
    
    # Find all txt files in gridRefinement subdirectory
    txt_files = find_grid_refinement_txt_files(comparison_dir)
    
    if not txt_files:
        grid_refinement_dir = comparison_dir / "gridRefinement"
        print(f"No .txt files found in {grid_refinement_dir}")
        sys.exit(1)
    
    # Group by Reynolds number
    reynolds_groups = group_files_by_reynolds(txt_files)
    
    print(f"Found {len(txt_files)} .txt files in gridRefinement directory")
    print(f"Grouped by Reynolds number:")
    for re_num, files in reynolds_groups.items():
        print(f"  Re={re_num}: {len(files)} files")
        for f in files:
            scheme = extract_scheme_from_filename(os.path.basename(f))
            print(f"    - {os.path.basename(f)} ({scheme})")
    
    print(f"\nRunning grid convergence studies with {args.max_workers or os.cpu_count()} parallel workers...")
    print("=" * 80)
    
    # Track timing
    start_time = time.time()
    
    # Run convergence studies in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                run_grid_convergence_study, 
                str(txt_file), 
                str(python_script), 
                str(comparison_dir)
            ): txt_file.name 
            for txt_file in txt_files
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_file):
            filename, success, message, reynolds_number = future.result()
            results.append((filename, success, message, reynolds_number))
            
            # Print immediate feedback
            status = "✓" if success else "✗"
            re_info = f"(Re={reynolds_number})" if reynolds_number else ""
            print(f"{status} {filename} {re_info}: {message}")
    
    # Summary
    end_time = time.time()
    elapsed = end_time - start_time
    
    successful = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - successful
    
    print("=" * 80)
    print(f"Completed {len(results)} grid convergence studies in {elapsed:.1f} seconds")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Results saved in individual 'plots' directories next to each txt file")
    
    # Group results by Reynolds number for summary
    results_by_re = {}
    for filename, success, message, reynolds_number in results:
        if reynolds_number not in results_by_re:
            results_by_re[reynolds_number] = {'successful': 0, 'failed': 0}
        if success:
            results_by_re[reynolds_number]['successful'] += 1
        else:
            results_by_re[reynolds_number]['failed'] += 1
    
    print(f"\nResults by Reynolds number:")
    for re_num in sorted(results_by_re.keys()):
        if re_num is not None:
            stats = results_by_re[re_num]
            print(f"  Re={re_num}: {stats['successful']} successful, {stats['failed']} failed")
    
    if failed > 0:
        print("\nFailed convergence studies:")
        for filename, success, message, reynolds_number in results:
            if not success:
                re_info = f"(Re={reynolds_number})" if reynolds_number else ""
                print(f"  {filename} {re_info}: {message}")
        sys.exit(1)
    else:
        print("\nAll grid convergence studies completed successfully!")

if __name__ == "__main__":
    main() 