#!/usr/bin/env python3
"""
Run comprehensive analysis (comparisons and grid convergence studies) for all txt files in a directory.

This script automatically finds all .txt files in a specified directory (or 'experiments' by default)
and runs both comparison plots and grid convergence studies in parallel. It's designed to work with
any experiment type, not just lid-driven cavity simulations.

Usage:
    python run_all_analysis.py [--directory DIR] [--max-workers N] [--comparison-only] [--convergence-only]

Examples:
    # Run all analysis on experiments directory
    python run_all_analysis.py
    
    # Run analysis on specific directory
    python run_all_analysis.py --directory experiments/lidDrivenCavity
    
    # Run only comparisons (no grid convergence)
    python run_all_analysis.py --comparison-only
    
    # Run only grid convergence studies
    python run_all_analysis.py --convergence-only
    
    # Run with specific number of workers
    python run_all_analysis.py --max-workers 8
"""

import os
import sys
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import re
from typing import List, Dict, Tuple, Optional

class AnalysisRunner:
    """Unified analysis runner for comparisons and grid convergence studies."""
    
    def __init__(self, base_directory: str, max_workers: Optional[int] = None):
        self.base_directory = Path(base_directory).absolute()
        self.max_workers = max_workers
        self.script_dir = Path(__file__).parent.absolute()
        
        # Define analysis scripts
        self.comparison_script = self.script_dir / "compare_lid_driven_cavity.py"
        self.convergence_script = self.script_dir / "grid_convergence_study.py"
        
        # Validate scripts exist
        self._validate_scripts()
    
    def _validate_scripts(self):
        """Validate that required analysis scripts exist."""
        if not self.comparison_script.exists():
            raise FileNotFoundError(f"Comparison script not found: {self.comparison_script}")
        if not self.convergence_script.exists():
            raise FileNotFoundError(f"Grid convergence script not found: {self.convergence_script}")
    
    def find_all_txt_files(self) -> List[Path]:
        """Find all .txt files recursively in the base directory."""
        if not self.base_directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.base_directory}")
        
        return list(self.base_directory.rglob("*.txt"))
    
    def categorize_txt_files(self, txt_files: List[Path]) -> Dict[str, List[Path]]:
        """Categorize txt files into comparison and convergence types."""
        categories = {
            'comparison': [],
            'convergence': []
        }
        
        for txt_file in txt_files:
            # Check if file is in a directory that suggests grid convergence studies
            parent_name = txt_file.parent.name.lower()
            grandparent_name = txt_file.parent.parent.name.lower() if txt_file.parent.parent != txt_file.parent else ""
            
            # Look for keywords that suggest grid convergence
            convergence_keywords = ['refinement', 'convergence', 'grid', 'mesh']
            is_convergence = any(keyword in parent_name or keyword in grandparent_name 
                               for keyword in convergence_keywords)
            
            if is_convergence:
                categories['convergence'].append(txt_file)
            else:
                categories['comparison'].append(txt_file)
        
        return categories
    
    def run_comparison_analysis(self, txt_file_path: Path) -> Tuple[str, bool, str]:
        """Run comparison analysis for a single txt file."""
        filename = txt_file_path.name
        
        # Create plots directory in the same folder as the txt file
        txt_dir = txt_file_path.parent
        plots_dir = txt_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            cmd = [
                sys.executable,
                str(self.comparison_script),
                "--config-list", str(txt_file_path),
                "--output-dir", str(plots_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return (filename, True, f"Comparison success → {plots_dir}")
            else:
                return (filename, False, f"Comparison error: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            return (filename, False, "Comparison timeout: took longer than 5 minutes")
        except Exception as e:
            return (filename, False, f"Comparison exception: {str(e)}")
    
    def run_convergence_analysis(self, txt_file_path: Path) -> Tuple[str, bool, str]:
        """Run grid convergence analysis for a single txt file."""
        filename = txt_file_path.name
        
        # Create plots directory in the same folder as the txt file
        txt_dir = txt_file_path.parent
        plots_dir = txt_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            cmd = [
                sys.executable,
                str(self.convergence_script),
                "--config-list", str(txt_file_path),
                "--output-dir", str(plots_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                return (filename, True, f"Convergence success → {plots_dir}")
            else:
                return (filename, False, f"Convergence error: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            return (filename, False, "Convergence timeout: took longer than 10 minutes")
        except Exception as e:
            return (filename, False, f"Convergence exception: {str(e)}")
    
    def run_analysis(self, comparison_files: List[Path], convergence_files: List[Path]) -> Dict[str, List[Tuple]]:
        """Run both types of analysis in parallel."""
        all_tasks = []
        results = {'comparison': [], 'convergence': []}
        
        # Submit all tasks to the executor
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit comparison tasks
            comparison_futures = {
                executor.submit(self.run_comparison_analysis, txt_file): ('comparison', txt_file.name)
                for txt_file in comparison_files
            }
            
            # Submit convergence tasks
            convergence_futures = {
                executor.submit(self.run_convergence_analysis, txt_file): ('convergence', txt_file.name)
                for txt_file in convergence_files
            }
            
            # Combine all futures
            all_futures = {**comparison_futures, **convergence_futures}
            
            # Process completed tasks
            for future in as_completed(all_futures):
                analysis_type, filename = all_futures[future]
                result = future.result()
                results[analysis_type].append(result)
                
                # Print immediate feedback
                status = "✓" if result[1] else "✗"
                print(f"{status} [{analysis_type.capitalize()}] {result[0]}: {result[2]}")
        
        return results

def print_analysis_summary(results: Dict[str, List[Tuple]], elapsed_time: float):
    """Print comprehensive summary of analysis results."""
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    total_tasks = len(results['comparison']) + len(results['convergence'])
    comparison_success = sum(1 for _, success, _ in results['comparison'] if success)
    convergence_success = sum(1 for _, success, _ in results['convergence'] if success)
    total_success = comparison_success + convergence_success
    total_failed = total_tasks - total_success
    
    print(f"Total runtime: {elapsed_time:.1f} seconds")
    print(f"Total tasks: {total_tasks}")
    print(f"✓ Successful: {total_success}")
    print(f"✗ Failed: {total_failed}")
    print()
    
    # Comparison results
    if results['comparison']:
        comparison_failed = len(results['comparison']) - comparison_success
        print(f"COMPARISON ANALYSIS:")
        print(f"  Total: {len(results['comparison'])}")
        print(f"  ✓ Successful: {comparison_success}")
        print(f"  ✗ Failed: {comparison_failed}")
        
        if comparison_failed > 0:
            print("  Failed files:")
            for filename, success, message in results['comparison']:
                if not success:
                    print(f"    • {filename}: {message}")
        print()
    
    # Convergence results
    if results['convergence']:
        convergence_failed = len(results['convergence']) - convergence_success
        print(f"GRID CONVERGENCE ANALYSIS:")
        print(f"  Total: {len(results['convergence'])}")
        print(f"  ✓ Successful: {convergence_success}")
        print(f"  ✗ Failed: {convergence_failed}")
        
        if convergence_failed > 0:
            print("  Failed files:")
            for filename, success, message in results['convergence']:
                if not success:
                    print(f"    • {filename}: {message}")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive analysis for all experiments in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Find all .txt files recursively in the specified directory
2. Categorize them into comparison and grid convergence types
3. Run appropriate analysis for each file type in parallel
4. Save results in 'plots' directories next to each txt file

File categorization:
- Files in directories with 'refinement', 'convergence', 'grid', or 'mesh' → Grid convergence
- All other files → Comparison plots

Directory structure will be preserved with results saved locally to each txt file.
        """
    )
    
    parser.add_argument(
        "--directory", 
        default="experiments",
        help="Directory to analyze (default: experiments)"
    )
    parser.add_argument(
        "--max-workers", 
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: number of CPU cores)"
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Run only comparison analysis (skip grid convergence)"
    )
    parser.add_argument(
        "--convergence-only",
        action="store_true",
        help="Run only grid convergence analysis (skip comparisons)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only find and categorize files, don't run analysis"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.comparison_only and args.convergence_only:
        print("Error: Cannot specify both --comparison-only and --convergence-only")
        sys.exit(1)
    
    print(f"Comprehensive Analysis Runner")
    print(f"Target directory: {os.path.abspath(args.directory)}")
    print(f"Max workers: {args.max_workers or os.cpu_count()}")
    
    try:
        # Initialize runner
        runner = AnalysisRunner(args.directory, args.max_workers)
        
        # Find all txt files
        txt_files = runner.find_all_txt_files()
        
        if not txt_files:
            print(f"No .txt files found in {args.directory}")
            sys.exit(0)
        
        # Categorize files
        categories = runner.categorize_txt_files(txt_files)
        
        print(f"\nFound {len(txt_files)} .txt files:")
        print(f"  Comparison files: {len(categories['comparison'])}")
        print(f"  Grid convergence files: {len(categories['convergence'])}")
        
        # Print file details
        if categories['comparison']:
            print(f"\nComparison files:")
            for f in categories['comparison']:
                rel_path = f.relative_to(runner.base_directory)
                print(f"  • {rel_path}")
        
        if categories['convergence']:
            print(f"\nGrid convergence files:")
            for f in categories['convergence']:
                rel_path = f.relative_to(runner.base_directory)
                print(f"  • {rel_path}")
        
        if args.dry_run:
            print("\n*** DRY RUN MODE - No analysis will be performed ***")
            return 0
        
        # Determine which analysis to run
        comparison_files = [] if args.convergence_only else categories['comparison']
        convergence_files = [] if args.comparison_only else categories['convergence']
        
        if not comparison_files and not convergence_files:
            print("No files to analyze based on the specified options.")
            return 0
        
        print(f"\nRunning analysis...")
        print("=" * 80)
        
        # Run analysis
        start_time = time.time()
        results = runner.run_analysis(comparison_files, convergence_files)
        end_time = time.time()
        
        # Print summary
        print_analysis_summary(results, end_time - start_time)
        
        # Return appropriate exit code
        total_failed = (len(results['comparison']) + len(results['convergence']) - 
                       sum(1 for _, success, _ in results['comparison'] if success) -
                       sum(1 for _, success, _ in results['convergence'] if success))
        return 1 if total_failed > 0 else 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 