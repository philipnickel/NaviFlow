# Postprocessing Utilities

This directory contains comprehensive postprocessing tools for CFD experiments, working with both staggered and collocated mesh data formats.

## 📊 Available Utilities

### 1. `generate_appendix.py` - Comprehensive Single-Page Summary
Creates a professional single-page Appendix PDF combining all key results: metadata, validation plots, field visualizations, and residual analysis.

**Usage:**
```bash
# For staggered experiments:
python postprocessing/generate_appendix.py --config experiments/Staggered/02\ GMG\ BiCGSTAB/pseudo_config.yaml
python postprocessing/generate_appendix.py --config experiments/Staggered/05\ geo_multigrid/pseudo_config.yaml
python postprocessing/generate_appendix.py --config experiments/Staggered/07\ AMG_CG/pseudo_config.yaml

# For collocated experiments:
python postprocessing/generate_appendix.py --config experiments/Collocated/lidDrivenCavity/debugging/config.yaml

# With custom output path:
python postprocessing/generate_appendix.py --config path/to/config.yaml --output my_appendix.pdf
```

**Generated Content:**
- Title with experiment name and simulation ID
- Metadata summary (parameters, solvers, timing, residuals)
- Validation plot (Ghia comparison for lid-driven cavity)
- Flow field visualizations (velocity magnitude, u/v components, pressure)
- Streamlines visualization
- Residual history and final residual fields

### 2. `generate_appendix_all.py` - Batch Appendix Generation
Automatically generates thesis figures for all valid experiments in the experiments directory.

**Usage:**
```bash
# Process all experiments with default settings:
python postprocessing/generate_appendix_all.py

# Use custom output directory and more workers:
python postprocessing/generate_appendix_all.py --output-dir thesis_figures --max-workers 8

# Preview what would be processed (dry run):
python postprocessing/generate_appendix_all.py --dry-run
```

**Features:**
- Automatically discovers all valid experiments (with config files and results)
- Parallel processing with configurable worker count
- Comprehensive error handling and progress reporting
- Generates professional single-page thesis figures for each experiment

### 3. `postprocess.py` - Standard Individual Plots
Generates standard individual postprocessing plots including field visualizations, residual analysis, and validation plots.

**Usage:**
```bash
# Generate all standard plots:
python postprocessing/postprocess.py --config path/to/config.yaml --all

# Works with both formats:
python postprocessing/postprocess.py --config experiments/Collocated/lidDrivenCavity/debugging/config.yaml --all
python postprocessing/postprocess.py --config experiments/Staggered/02\ GMG\ BiCGSTAB/pseudo_config.yaml --all
```

**Generated Plots:**
- Individual field plots (u-velocity, v-velocity, velocity magnitude, pressure)
- Individual residual plots (u-residual, v-residual, continuity residual)
- Combined flow fields visualization
- Residual history plot
- Experiment-specific validation (Ghia for lid-driven cavity, Poiseuille for channel flow)
- Force coefficients (if available for cylinder flow)
- Streamlines visualization

### 4. `compare_lid_driven_cavity.py` - Multi-Experiment Comparison
Compares multiple lid-driven cavity experiments against each other and Ghia's benchmark data, with support for both config list files and direct experiment paths.

#### **Primary Usage (Config List Method):**
Create a text file containing config file paths and compare multiple experiments:

```bash
# 1. Create a config list file (configs.txt):
echo "experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml" > configs.txt
echo "experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml" >> configs.txt
echo "experiments/lidDrivenCavity/ForReport/uniform/fine/Re_100/config.yaml" >> configs.txt
echo "experiments/Staggered/02 GMG BiCGSTAB/pseudo_config.yaml" >> configs.txt

# 2. Run comparison:
python postprocessing/compare_lid_driven_cavity.py --config-list configs.txt

# 3. With custom output directory:
python postprocessing/compare_lid_driven_cavity.py --config-list configs.txt --output-dir my_comparisons/
```

#### **Config List File Format:**
```
# Comments are supported (lines starting with #)
# Compare different mesh resolutions for Re=100 (Collocated experiments)
experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml
experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml
experiments/lidDrivenCavity/ForReport/uniform/fine/Re_100/config.yaml

# Compare with Staggered experiments (using pseudo_config.yaml)
experiments/Staggered/02 GMG BiCGSTAB/pseudo_config.yaml

# Compare different mesh types for Re=400
experiments/lidDrivenCavity/ForReport/uniform/medium/Re_400/config.yaml
experiments/lidDrivenCavity/ForReport/unstructured/medium/Re_400/config.yaml

# Empty lines are ignored
```

#### **Legacy Usage (Direct Experiment Paths):**
For backward compatibility, you can still specify experiment directories directly:

```bash
# Compare different mesh resolutions:
python postprocessing/compare_lid_driven_cavity.py \
    --experiments experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100 \
    experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100 \
    experiments/lidDrivenCavity/ForReport/uniform/fine/Re_100

# Compare different numerical methods:
python postprocessing/compare_lid_driven_cavity.py \
    --experiments experiments/Staggered/02\ GMG\ BiCGSTAB \
    experiments/Collocated/lidDrivenCavity/debugging
```

#### **Features:**
- **Automatic grouping**: Groups experiments by Reynolds number automatically
- **Centerline extraction**: Extracts u-velocity along vertical centerline (x=0.5) and v-velocity along horizontal centerline (y=0.5)
- **Benchmark comparison**: Compares against Ghia's benchmark data for Re=100, 400, 1000, 3200, 5000
- **Professional plots**: Creates publication-quality comparison plots with legends and simulation IDs
- **Mesh support**: Handles both structured uniform and unstructured meshes seamlessly
- **Error handling**: Graceful handling of missing files with informative warnings
- **Output organization**: Saves plots to `postprocessing/lid_cavity_comparisons/` by default

#### **Generated Output:**
- `lid_driven_cavity_comparison_Re_100.pdf`
- `lid_driven_cavity_comparison_Re_400.pdf`
- `lid_driven_cavity_comparison_Re_1000.pdf`
- etc. (one file per Reynolds number found)

Each plot shows:
- Left panel: u-velocity along vertical centerline (y vs u at x=0.5)
- Right panel: v-velocity along horizontal centerline (x vs v at y=0.5)
- Ghia et al. benchmark data as hollow circles
- Numerical results as lines with markers
- Legend showing mesh details and discretization schemes
- Simulation IDs for traceability

### 5. `grid_convergence_study.py` - Grid Convergence Analysis
Performs grid convergence analysis by calculating combined L2 errors between numerical solutions and Ghia's benchmark data for velocity components, then plots the errors as a function of grid size to determine the order of accuracy.

#### **Usage:**
```bash
# 1. Create a config list with different mesh resolutions for the same scheme:
echo "experiments/Collocated/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml" > mesh_study.txt
echo "experiments/Collocated/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml" >> mesh_study.txt
echo "experiments/Collocated/lidDrivenCavity/ForReport/uniform/fine/Re_100/config.yaml" >> mesh_study.txt

# 2. Run grid convergence study:
python postprocessing/grid_convergence_study.py --config-list mesh_study.txt

# 3. With custom output directory:
python postprocessing/grid_convergence_study.py --config-list mesh_study.txt --output-dir convergence_results/
```

#### **Config List Requirements:**
- All experiments must be at the same Reynolds number
- Experiments should represent different mesh resolutions of the same numerical scheme
- Ghia benchmark data must be available for that Reynolds number (Re=100, 400, 1000, 3200, 5000)
- Each experiment needs `U_final.npy`, `p_final.npy`, and `metadata.yaml` files

#### **Analysis Process:**
The script performs the following steps:
1. **Load experiment data** from each config in the list
2. **Extract centerline velocity profiles** (u at x=0.5, v at y=0.5)
3. **Interpolate** numerical solution to Ghia's reference points
4. **Calculate individual L2 errors** for both u and v velocity components
5. **Combine errors** using root-mean-square for overall velocity error
6. **Estimate grid size** using h = 1/√N (where N is number of cells)
7. **Fit power law** to determine order of accuracy: error = A × h^p
8. **Generate professional plot** with reference slopes and error quantification

#### **Generated Output:**
- **PDF Plot**: `{config_filename}_grid_convergence_Re_{Re}.pdf`
  - Single plot showing velocity convergence with L2 error vs grid size
  - Reference slopes for 1st order (O(h¹)) and 2nd order (O(h²)) accuracy
  - Observed order of accuracy displayed as text annotation
  - Cell count and scheme information in legend
  - Simulation IDs displayed below plot with markers
  
- **CSV Data**: `{config_filename}_grid_convergence_Re_{Re}.csv`
  - Complete numerical results with grid sizes, combined velocity errors, and theoretical predictions
  - Useful for further analysis or custom plotting

#### **Professional Plotting Features:**
- **Mathematical notation**: Proper LaTeX formatting (e.g., $h$, $\mathcal{O}(h^2)$)
- **Reference slopes**: Theoretical 1st and 2nd order convergence lines
- **Observed order**: Text annotation showing actual convergence rate
- **Grid lines**: Both major and minor grid lines for easy reading
- **Meaningful legends**: Show cell count, scheme, and mesh type for each experiment
- **Simulation IDs**: Displayed below plot for traceability
- **Clean single plot**: Combined velocity error instead of separate u/v components

#### **Example Results:**
```
Found 3 experiments for grid convergence study:
  - 3969 cells, Upwind, uniform
  - 16129 cells, Upwind, uniform
  - 65025 cells, Upwind, uniform

Performing convergence study for Re = 100

Calculating L2 errors:
    3969 cells: h=0.0159, L2_velocity=5.31e-03
   16129 cells: h=0.0079, L2_velocity=3.57e-03
   65025 cells: h=0.0039, L2_velocity=1.89e-03

Order of accuracy:
  velocity magnitude: 1.85
```

### 6. `postprocess_all.py` - Batch Processing
Automatically finds and processes all experiments in parallel using multiple CPU cores.

**Usage:**
```bash
# Process all experiments in the experiments/ directory:
python postprocessing/postprocess_all.py
```

**Features:**
- Automatically discovers all `config.yaml` files in the experiments directory
- Processes experiments in parallel using multiple CPU cores
- Provides progress tracking and error reporting
- Caps parallel processes to avoid system overload

## 🔧 Data Format Requirements

All utilities expect the following data files in the `results/` directory:
- `U_final.npy`: Final velocity field (N×2 array)
- `p_final.npy`: Final pressure field (flattened)
- `cell_centers.npz`: Cell center coordinates (x, y arrays)
- `residuals.npz`: Residual history (u, v, cont arrays)
- `u_residual.npy`, `v_residual.npy`, `continuity_field.npy`: Final residual fields
- `metadata.yaml`: Simulation metadata

## 🎯 Quick Start Examples

```bash
# Generate appendix for a single experiment:
python postprocessing/generate_appendix.py --config experiments/Staggered/02\ GMG\ BiCGSTAB/pseudo_config.yaml

# Generate appendices for all experiments:
python postprocessing/generate_appendix_all.py

# Generate all standard plots:
python postprocessing/postprocess.py --config experiments/Staggered/02\ GMG\ BiCGSTAB/pseudo_config.yaml --all

# Compare lid-driven cavity experiments (config list method):
echo "experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml" > lid_configs.txt
echo "experiments/lidDrivenCavity/ForReport/uniform/fine/Re_100/config.yaml" >> lid_configs.txt
python postprocessing/compare_lid_driven_cavity.py --config-list lid_configs.txt

# Grid convergence study:
echo "experiments/Collocated/lidDrivenCavity/ForReport/uniform/coarse/Re_100/config.yaml" > mesh_study.txt
echo "experiments/Collocated/lidDrivenCavity/ForReport/uniform/medium/Re_100/config.yaml" >> mesh_study.txt
python postprocessing/grid_convergence_study.py --config-list mesh_study.txt

# Compare experiments (legacy method):
python postprocessing/compare_lid_driven_cavity.py --experiments exp1/ exp2/ exp3/

# Process all experiments:
python postprocessing/postprocess_all.py
```

## 📁 Output Structure

Each utility creates organized output:
- **Individual plots**: `results/plots/` directory within each experiment
- **Appendices**: `results/thesis_figure_*.pdf` (or custom output directory)
- **Comparisons**: `postprocessing/lid_cavity_comparisons/` (or custom output directory)
- **Convergence studies**: `postprocessing/convergence_studies/` (or custom output directory)
- **Thesis figures**: Custom output directory or individual experiment results directories

### Typical Output Structure:
```
experiments/
├── lidDrivenCavity/ForReport/uniform/medium/Re_100/
│   └── results/
│       ├── plots/              # Individual plots from postprocess.py
│       └── thesis_figure_*.pdf # Appendix from generate_appendix.py
├── ...
└── 

postprocessing/
├── lid_cavity_comparisons/     # Comparison plots from compare_lid_driven_cavity.py
│   ├── lid_driven_cavity_comparison_Re_100.pdf
│   ├── lid_driven_cavity_comparison_Re_400.pdf
│   └── ...
├── convergence_studies/        # Grid convergence analysis from grid_convergence_study.py
│   ├── mesh_study_grid_convergence_Re_100.pdf
│   ├── mesh_study_grid_convergence_Re_100.csv
│   └── ...
└── README.md
```
