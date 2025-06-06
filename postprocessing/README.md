# Postprocessing Scripts

This directory contains the essential postprocessing scripts for NaviFlow CFD simulations.

## Directory Structure

```
postprocessing/
├── master_postprocess.py              # Master script for all postprocessing tasks
├── run_grid_convergence_analysis.py   # Grid convergence analysis for all txt files
├── postprocess.py                     # Core postprocessing (plots, verification)
├── compare_lid_driven_cavity.py       # Ghia benchmark comparison for LDC
├── generate_appendix.py               # Generate thesis figures for AppendixPlots
├── generate_latex_appendix.py         # Generate LaTeX code from AppendixPlots
├── lid_cavity_comparisons/            # Output directory for Ghia comparisons
├── reynolds_convergence_studies/       # Convergence analysis results
└── README.md                          # This file
```

## Master Script Usage

The `master_postprocess.py` script provides a unified interface for all postprocessing tasks:

### Basic Usage

```bash
# Process everything in experiments directory
python postprocessing/master_postprocess.py

# Process specific experiment directory
python postprocessing/master_postprocess.py --experiment-dir experiments/Collocated/lidDrivenCavity

# Postprocess only (no appendix generation)
python postprocessing/master_postprocess.py --postprocess-only

# Appendix only (thesis figures + LaTeX, no standard postprocessing)
python postprocessing/master_postprocess.py --appendix-only

# Include grid convergence analysis
python postprocessing/master_postprocess.py --include-convergence
```

### What the Master Script Does

1. **Standard Postprocessing** (unless `--appendix-only`):
   - Runs `postprocess.py --all` for each experiment
   - Creates plots in `results/plots/` directory
   - Generates verification plots (Ghia comparison for LDC)
   - Creates convergence analysis plots

2. **Ghia Comparison** (unless `--appendix-only`):
   - Automatically runs comparison for all lid-driven cavity experiments
   - Saves results to `postprocessing/lid_cavity_comparisons/`

3. **Grid Convergence Analysis** (if `--include-convergence`):
   - Runs convergence analysis for all txt files in `lid_cavity_comparisons/`
   - Saves plots in the same directory as each txt file

4. **Appendix Generation** (unless `--postprocess-only`):
   - Creates thesis figures in `AppendixPlots/` directory
   - Generates LaTeX code for the appendix

## Individual Scripts

### Core Postprocessing

```bash
# Run for single experiment
python postprocessing/postprocess.py --config experiments/path/to/config.yaml --all
```

**Note**: During standard postprocessing, no "thesis figures" are created in the results directories. Individual plots are saved to `results/plots/` only.

### Ghia Benchmark Comparison

```bash
# Create config list file
echo "experiments/lidDrivenCavity/Re_100/config.yaml" > configs.txt
echo "experiments/lidDrivenCavity/Re_400/config.yaml" >> configs.txt

# Run comparison
python postprocessing/compare_lid_driven_cavity.py --config-list configs.txt --output-dir comparison_plots
```

### Grid Convergence Analysis

```bash
# Run convergence analysis for all txt files in lid_cavity_comparisons
python postprocessing/run_grid_convergence_analysis.py --max-workers 6

# Run for specific directory only
python postprocessing/run_grid_convergence_analysis.py --base-dir postprocessing/lid_cavity_comparisons/gridRefinement
```

**Note**: This script automatically finds all `.txt` files in the lid_cavity_comparisons directory and runs `compare_lid_driven_cavity.py` for each one individually, saving the results in the same directory as the respective txt file.

### Appendix Generation

```bash
# Generate thesis figure for single experiment
python postprocessing/generate_appendix.py --config experiments/path/to/config.yaml

# Generate LaTeX code from AppendixPlots
python postprocessing/generate_latex_appendix.py --output appendix.tex --plots-dir AppendixPlots
```

## Output Locations

- **Standard plots**: `experiments/{experiment}/results/plots/`
- **Thesis figures**: `AppendixPlots/{experiment}/Re_{reynolds}/{resolution}/`
- **Ghia comparisons**: `postprocessing/lid_cavity_comparisons/`
- **LaTeX code**: `AppendixPlots/appendix.tex`

## Key Features

- **Mesh-agnostic**: Works with structured, unstructured, and adaptive meshes
- **Experiment-aware**: Automatically detects experiment type for appropriate verification
- **Modular**: Can run individual components or everything together
- **Error handling**: Continues processing even if individual experiments fail
- **Professional plots**: Uses science-style matplotlib formatting throughout

## Requirements

All required files must be present in `results/` directory:
- `U_final.npy`
- `p_final.npy`
- `residuals.npz`
- `cell_centers.npz`
- `metadata.yaml`

## Verification and Validation

- **Lid-driven cavity**: Automatic comparison with Ghia benchmark data
- **Channel flow**: Poiseuille flow verification (if available)
- **Convergence analysis**: Residual history and field plots
- **Force coefficients**: Automatic plotting if available (e.g., cylinder flow)
