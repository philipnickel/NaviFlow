# Postprocessing Utilities

Comprehensive CFD postprocessing tools for analysis, visualization, and validation.

## 🚀 Quick Reference

| Script | Purpose | Example |
|--------|---------|---------|
| `postprocess.py` | Individual plots | `python postprocessing/postprocess.py --config path/to/config.yaml --all` |
| `generate_appendix.py` | Single-page summary | `python postprocessing/generate_appendix.py --config path/to/config.yaml` |
| `generate_appendix_all.py` | Batch appendix generation | `python postprocessing/generate_appendix_all.py --experiments-dir experiments` |
| `postprocess_all.py` | Comprehensive batch processing | `python postprocessing/postprocess_all.py --experiments-dir experiments` |
| `compare_lid_driven_cavity.py` | Multi-experiment comparison | `python postprocessing/compare_lid_driven_cavity.py --config-list configs.txt` |
| `grid_convergence_study.py` | Grid convergence analysis | `python postprocessing/grid_convergence_study.py --config-list mesh_study.txt` |
| `run_all_analysis.py` | Unified comparison & convergence | `python postprocessing/run_all_analysis.py --directory experiments` |
| `run_all_lid_cavity_comparisons.py` | Parallel lid cavity comparisons | `python postprocessing/run_all_lid_cavity_comparisons.py` |
| `run_all_grid_convergence_studies.py` | Parallel grid convergence studies | `python postprocessing/run_all_grid_convergence_studies.py` |

## 📋 Command Details

### `postprocess.py` - Individual Experiment Plots
```bash
python postprocessing/postprocess.py --config experiments/example/config.yaml --all
```
**Arguments:**
- `--config PATH`: Path to config.yaml file (required)
- `--all`: Generate all standard plots (recommended)

**Output:** Individual field plots, residuals, validation plots in `results/plots/`

---

### `generate_appendix.py` - Single-Page Summary
```bash
python postprocessing/generate_appendix.py --config experiments/example/config.yaml
```
**Arguments:**
- `--config PATH`: Path to config.yaml file (required)
- `--output FILE`: Custom output PDF path (optional)

**Output:** Professional single-page PDF with metadata, validation, and flow fields

---

### `generate_appendix_all.py` - Batch Appendix Generation
```bash
python postprocessing/generate_appendix_all.py --experiments-dir experiments --max-workers 4
```
**Arguments:**
- `--experiments-dir DIR`: Directory containing experiments (default: experiments)
- `--output-dir DIR`: Output directory for all figures (optional)
- `--max-workers N`: Parallel workers (default: 4)
- `--dry-run`: Preview without generating (optional)

**Output:** Thesis figures for all valid experiments

---

### `postprocess_all.py` - Comprehensive Batch Processing
```bash
python postprocessing/postprocess_all.py --experiments-dir experiments --max-workers 4
```
**Arguments:**
- `--experiments-dir DIR`: Directory containing experiments (default: experiments)
- `--output-dir DIR`: Output directory for thesis figures (optional)
- `--max-workers N`: Parallel workers (default: CPU count, capped at 8)
- `--postprocess-only`: Run only individual postprocessing (optional)
- `--appendix-only`: Run only thesis figure generation (optional)
- `--dry-run`: Preview without running (optional)

**Output:** Both individual plots and thesis figures for all experiments

---

### `compare_lid_driven_cavity.py` - Multi-Experiment Comparison
```bash
# Create config list file
echo "experiments/Re_100/coarse/config.yaml" > configs.txt
echo "experiments/Re_100/medium/config.yaml" >> configs.txt

python postprocessing/compare_lid_driven_cavity.py --config-list configs.txt
```
**Arguments:**
- `--config-list FILE`: Text file with config paths (recommended)
- `--experiments DIR1 DIR2...`: Direct experiment paths (legacy)
- `--output-dir DIR`: Output directory (default: postprocessing/lid_cavity_comparisons)

**Output:** Comparison plots grouped by Reynolds number with Ghia benchmark

---

### `grid_convergence_study.py` - Grid Convergence Analysis
```bash
# Create mesh study file
echo "experiments/coarse/config.yaml" > mesh_study.txt
echo "experiments/medium/config.yaml" >> mesh_study.txt
echo "experiments/fine/config.yaml" >> mesh_study.txt

python postprocessing/grid_convergence_study.py --config-list mesh_study.txt
```
**Arguments:**
- `--config-list FILE`: Text file with config paths (required)
- `--output-dir DIR`: Output directory (default: postprocessing/convergence_studies)

**Output:** Convergence plots and CSV data showing order of accuracy

---

### `run_all_analysis.py` - Unified Comparison & Convergence
```bash
python postprocessing/run_all_analysis.py --directory experiments --max-workers 4
```
**Arguments:**
- `--directory DIR`: Directory to analyze (default: experiments)
- `--max-workers N`: Parallel workers (default: CPU count)
- `--comparison-only`: Run only comparisons (optional)
- `--convergence-only`: Run only grid convergence (optional)
- `--dry-run`: Preview without running (optional)

**Output:** Automatically categorizes .txt files and runs appropriate analysis

---

### `run_all_lid_cavity_comparisons.py` - Parallel Lid Cavity Comparisons
```bash
python postprocessing/run_all_lid_cavity_comparisons.py --max-workers 4
```
**Arguments:**
- `--max-workers N`: Parallel workers (default: CPU count)

**Output:** Processes all .txt files in lid_cavity_comparisons directory

---

### `run_all_grid_convergence_studies.py` - Parallel Grid Convergence Studies
```bash
python postprocessing/run_all_grid_convergence_studies.py --max-workers 4
```
**Arguments:**
- `--max-workers N`: Parallel workers (default: CPU count)

**Output:** Processes all .txt files in gridRefinement directory

## 📁 Input Requirements

**Data Files (in experiment's `results/` directory):**
- `U_final.npy`, `p_final.npy`: Final velocity and pressure fields
- `cell_centers.npz`: Cell center coordinates
- `residuals.npz`: Residual history
- `metadata.yaml`: Simulation metadata

**Config Files:**
- `config.yaml`: Standard collocated experiments
- `pseudo_config.yaml`: Staggered experiments (legacy)

## 🎯 Common Workflows

**Single Experiment Analysis:**
```bash
python postprocessing/postprocess.py --config experiment/config.yaml --all
python postprocessing/generate_appendix.py --config experiment/config.yaml
```

**Batch Processing All Experiments:**
```bash
python postprocessing/postprocess_all.py
```

**Lid-Driven Cavity Study:**
```bash
# Create comparison file
echo "exp1/config.yaml" > study.txt
echo "exp2/config.yaml" >> study.txt

# Compare experiments
python postprocessing/compare_lid_driven_cavity.py --config-list study.txt

# Grid convergence analysis
python postprocessing/grid_convergence_study.py --config-list study.txt
```

**Comprehensive Analysis:**
```bash
python postprocessing/run_all_analysis.py --directory experiments
```

## 📊 Output Structure

```
experiments/
├── experiment1/
│   └── results/
│       ├── plots/              # Individual plots
│       └── thesis_figure_*.pdf # Appendix
└── experiment2/...

postprocessing/
├── lid_cavity_comparisons/     # Comparison plots
├── convergence_studies/        # Grid convergence plots
└── README.md
```
