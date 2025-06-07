#!/usr/bin/env python3
"""
Generate LaTeX appendix from AppendixPlots directory structure.

This script scans the AppendixPlots directory and generates a complete LaTeX appendix
with proper sectioning, figure environments, and meaningful labels.

Usage:
    python generate_latex_appendix.py [--output appendix.tex] [--plots-dir AppendixPlots]
"""

import os
import argparse
from pathlib import Path
import re
from collections import defaultdict, OrderedDict


def parse_filename(filename):
    """
    Parse a PDF filename to extract experiment details.
    
    Expected format: experiment_ReXXX_meshtype_resolution_scheme_simid.pdf
    Example: lidDrivenCavity_Re100_uniform_coarse_QUICK_c2f9a4fc.pdf
    
    Returns dict with parsed components or None if parsing fails.
    """
    # Remove .pdf extension
    base_name = filename.replace('.pdf', '')
    
    # Try to match the expected pattern
    # experiment_ReXXX_meshtype_resolution_scheme_simid
    pattern = r'^(.+)_Re(\d+)_(.+)_(.+)_(.+)_([a-fA-F0-9]+)$'
    match = re.match(pattern, base_name)
    
    if match:
        experiment, re_num, mesh_type, resolution, scheme, sim_id = match.groups()
        return {
            'experiment': experiment,
            'reynolds': int(re_num),
            'mesh_type': mesh_type,
            'resolution': resolution,
            'scheme': scheme,
            'sim_id': sim_id,
            'original_filename': filename
        }
    
    return None


def abbreviate_experiment(experiment):
    """Convert experiment name to abbreviation."""
    abbrevs = {
        'lidDrivenCavity': 'LDC',
        'channelFlow': 'CF',
        'backwardFacingStep': 'BFS',
        'cylinderFlow': 'CYL'
    }
    return abbrevs.get(experiment, experiment.upper()[:3])


def abbreviate_mesh_type(mesh_type):
    """Convert mesh type to abbreviation."""
    abbrevs = {
        'uniform': 'unif',
        'unstructured': 'unstruct',
        'structured': 'struct',
        'adaptive': 'adapt'
    }
    return abbrevs.get(mesh_type, mesh_type[:4])


def abbreviate_resolution(resolution):
    """Convert resolution to abbreviation."""
    abbrevs = {
        'coarse': 'coarse',
        'medium': 'med',
        'fine': 'fine',
        'extra_fine': 'xfine',
        'ultra_fine': 'ufine'
    }
    return abbrevs.get(resolution, resolution[:4])


def abbreviate_scheme(scheme):
    """Convert scheme to abbreviation."""
    abbrevs = {
        'QUICK': 'quick',
        'Upwind': 'upwind',
        'TVD': 'tvd',
        'Central': 'central',
        'PowerLaw': 'plaw'
    }
    return abbrevs.get(scheme, scheme[:4])


def generate_label(parsed_info):
    """Generate a meaningful LaTeX label from parsed filename info."""
    exp_abbrev = abbreviate_experiment(parsed_info['experiment'])
    mesh_abbrev = abbreviate_mesh_type(parsed_info['mesh_type'])
    res_abbrev = abbreviate_resolution(parsed_info['resolution'])
    scheme_abbrev = abbreviate_scheme(parsed_info['scheme'])
    
    # Create label with clear separators but no prefixes
    label = f"sim.{exp_abbrev}.re{parsed_info['reynolds']}.{mesh_abbrev}.{res_abbrev}.{scheme_abbrev}"
    return label.lower()


def generate_caption(parsed_info):
    """Generate a descriptive caption from parsed filename info."""
    exp_name = parsed_info['experiment'].replace('_', ' ').title()
    
    caption = (f"{exp_name} simulation at Re={parsed_info['reynolds']} using "
               f"{parsed_info['mesh_type']} {parsed_info['resolution']} mesh with "
               f"{parsed_info['scheme']} discretization scheme "
               f"(Simulation ID: {parsed_info['sim_id'][:8]})")
    
    return caption


def scan_appendix_plots(plots_dir):
    """
    Scan the AppendixPlots directory and organize files by experiment and Reynolds number.
    
    Returns nested dict: experiment -> reynolds -> resolution -> list of file info
    """
    plots_path = Path(plots_dir)
    if not plots_path.exists():
        raise FileNotFoundError(f"AppendixPlots directory not found: {plots_dir}")
    
    organized_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Walk through directory structure
    for experiment_dir in plots_path.iterdir():
        if not experiment_dir.is_dir():
            continue
            
        experiment_name = experiment_dir.name
        
        for re_dir in experiment_dir.iterdir():
            if not re_dir.is_dir() or not re_dir.name.startswith('Re_'):
                continue
                
            # Extract Reynolds number
            re_match = re.match(r'Re_(\d+)', re_dir.name)
            if not re_match:
                continue
            reynolds = int(re_match.group(1))
            
            for resolution_dir in re_dir.iterdir():
                if not resolution_dir.is_dir():
                    continue
                    
                resolution = resolution_dir.name
                
                # Process PDF files in this resolution directory
                for pdf_file in resolution_dir.glob('*.pdf'):
                    parsed = parse_filename(pdf_file.name)
                    if parsed:
                        # Store relative path from AppendixPlots
                        rel_path = pdf_file.relative_to(plots_path)
                        file_info = {
                            'parsed': parsed,
                            'relative_path': str(rel_path),
                            'absolute_path': str(pdf_file),
                            'label': generate_label(parsed),
                            'caption': generate_caption(parsed)
                        }
                        organized_files[experiment_name][reynolds][resolution].append(file_info)
    
    return organized_files


def generate_latex_appendix(organized_files, output_file, plots_dir):
    """Generate the complete LaTeX appendix file as a subfile."""
    
    # Sort experiments and Reynolds numbers for consistent ordering
    sorted_experiments = sorted(organized_files.keys())
    
    latex_content = []
    
    # Subfile header (for use with subfiles package)
    latex_content.append("% LaTeX Appendix - CFD Simulation Results")
    latex_content.append("% Generated automatically from AppendixPlots directory")
    latex_content.append("% Use with subfiles package: \\subfile{path/to/this/file}")
    latex_content.append("")
    
    # Appendix section
    latex_content.append("\\appendix")
    latex_content.append("\\part{Appendices}")
    latex_content.append("")
    
    # Main appendix chapter (keep this numbered)
    latex_content.append("\\chapter{CFD Simulation Results}")
    latex_content.append("\\label{chap:cfd_results}")
    latex_content.append("")
    
    # Set TOC depth to show detailed navigation down to schemes
    latex_content.append("% Increase TOC depth to show detailed navigation")
    latex_content.append("\\setcounter{tocdepth}{4}")
    latex_content.append("")
    
    # Brief introduction
    latex_content.append("This appendix presents the complete set of CFD simulation results organized by experiment type, Reynolds number, mesh resolution, and discretization scheme.")
    latex_content.append("")
    
    # Generate the actual appendix content
    for experiment in sorted_experiments:
        exp_title = experiment.replace('_', ' ').title()
        # Use starred section (unnumbered) but add to TOC
        latex_content.append(f"\\section*{{{exp_title} Results}}")
        latex_content.append(f"\\addcontentsline{{toc}}{{section}}{{{exp_title} Results}}")
        latex_content.append(f"\\label{{sec:{experiment.lower()}_results}}")
        latex_content.append("")
        
        reynolds_numbers = sorted(organized_files[experiment].keys())
        for re_num in reynolds_numbers:
            # Use starred subsection (unnumbered) but add to TOC
            latex_content.append(f"\\subsection*{{Reynolds Number {re_num}}}")
            latex_content.append(f"\\addcontentsline{{toc}}{{subsection}}{{Reynolds Number {re_num}}}")
            latex_content.append(f"\\label{{subsec:{experiment.lower()}_re{re_num}}}")
            latex_content.append("")
            
            resolutions = sorted(organized_files[experiment][re_num].keys(), 
                               key=lambda x: ['coarse', 'medium', 'fine', 'extra_fine', 'ultra_fine'].index(x) 
                               if x in ['coarse', 'medium', 'fine', 'extra_fine', 'ultra_fine'] else 999)
            
            for resolution in resolutions:
                # Use starred subsubsection (unnumbered) but add to TOC
                latex_content.append(f"\\subsubsection*{{{resolution.title()} Mesh}}")
                latex_content.append(f"\\addcontentsline{{toc}}{{subsubsection}}{{{resolution.title()} Mesh}}")
                latex_content.append(f"\\label{{subsubsec:{experiment.lower()}_re{re_num}_{resolution}}}")
                latex_content.append("")
                
                # Group files by mesh type and scheme
                files = organized_files[experiment][re_num][resolution]
                grouped_files = defaultdict(lambda: defaultdict(list))
                
                for file_info in files:
                    mesh_type = file_info['parsed']['mesh_type']
                    scheme = file_info['parsed']['scheme']
                    grouped_files[mesh_type][scheme].append(file_info)
                
                # Sort mesh types and schemes
                sorted_mesh_types = sorted(grouped_files.keys())
                
                for mesh_type in sorted_mesh_types:
                    sorted_schemes = sorted(grouped_files[mesh_type].keys())
                    
                    for scheme in sorted_schemes:
                        # Use starred paragraph (unnumbered) but add to TOC
                        paragraph_title = f"{mesh_type.title()} Mesh with {scheme} Scheme"
                        latex_content.append(f"\\paragraph*{{{paragraph_title}}}")
                        latex_content.append(f"\\addcontentsline{{toc}}{{paragraph}}{{{paragraph_title}}}")
                        latex_content.append(f"\\label{{para:{experiment.lower()}_re{re_num}_{resolution}_{mesh_type}_{scheme}}}")
                        latex_content.append("")
                        
                        scheme_files = grouped_files[mesh_type][scheme]
                        for file_info in scheme_files:
                            # Create figure environment
                            latex_content.append("\\begin{figure}[H]")
                            latex_content.append("    \\centering")
                            
                            # Use the actual plots directory name, not hardcoded "AppendixPlots"
                            fig_path = f"{plots_dir}/{file_info['relative_path']}"
                            latex_content.append(f"    \\includegraphics[width=0.85\\linewidth]{{{fig_path}}}")
                            
                            latex_content.append(f"    \\caption{{{file_info['caption']}}}")
                            latex_content.append(f"    \\label{{{file_info['label']}}}")
                            latex_content.append("\\end{figure}")
                            latex_content.append("")
                
                # Add some space between resolutions
                if resolution != resolutions[-1]:
                    latex_content.append("\\clearpage")
                    latex_content.append("")
        
        # Add page break between experiments
        if experiment != sorted_experiments[-1]:
            latex_content.append("\\clearpage")
            latex_content.append("")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX appendix subfile generated: {output_file}")
    print("Usage: Place this file and the AppendixPlots folder in your Overleaf project,")
    print(f"then add \\subfile{{{output_file.replace('.tex', '')}}} to your main document.")
    
    # Print summary statistics
    total_figures = 0
    for experiment in organized_files:
        for re_num in organized_files[experiment]:
            for resolution in organized_files[experiment][re_num]:
                total_figures += len(organized_files[experiment][re_num][resolution])
    
    print(f"Total figures included: {total_figures}")
    print(f"Experiments covered: {len(sorted_experiments)}")
    
    return total_figures


def main():
    """Main function to handle command line arguments and generate appendix."""
    parser = argparse.ArgumentParser(description='Generate LaTeX appendix subfile from AppendixPlots directory')
    parser.add_argument('--output', '-o', default='Appendix_CFD_Results.tex',
                       help='Output LaTeX subfile (default: Appendix_CFD_Results.tex)')
    parser.add_argument('--plots-dir', '-p', default='AppendixPlots',
                       help='Path to AppendixPlots directory (default: AppendixPlots)')
    
    args = parser.parse_args()
    
    try:
        # Scan the AppendixPlots directory
        print(f"Scanning {args.plots_dir} directory...")
        organized_files = scan_appendix_plots(args.plots_dir)
        
        if not organized_files:
            print("No PDF files found in the specified directory structure.")
            return 1
        
        # Generate LaTeX appendix
        print(f"Generating LaTeX appendix subfile...")
        generate_latex_appendix(organized_files, args.output, args.plots_dir)
        
        print("Done!")
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    exit(main()) 