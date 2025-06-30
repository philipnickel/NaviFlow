# LaTeX Appendix Generator for Thesis Projects

This script automatically generates a LaTeX appendix subfile from the `AppendixPlots` directory structure, designed for seamless integration with thesis projects using the `subfiles` package.

## Overview

The `generate_latex_appendix.py` script scans your `AppendixPlots` directory and creates a standalone LaTeX subfile with:

- Hierarchical organization by experiment type, Reynolds number, and mesh resolution
- Detailed table of contents navigation down to individual schemes
- Proper figure environments with meaningful labels and captions
- Full compatibility with the `subfiles` package for Overleaf
- Consistent formatting following LaTeX best practices

## Quick Start for Thesis Projects

### 1. Generate the Appendix

```bash
# Run from the main directory (where AppendixPlots exists)
python postprocessing/generate_latex_appendix.py
```

This generates `Appendix_CFD_Results.tex` in the current directory.

### 2. Add to Your Thesis

Upload both files to your Overleaf project:
- `Appendix_CFD_Results.tex`
- `AppendixPlots/` folder (with all your PDFs)

Then add this line to your main thesis document where you want the appendix:

```latex
% --- Appendix --- 
\subfile{Appendix_CFD_Results}
```

### 3. Required Packages

Make sure your thesis includes these packages in the preamble:

```latex
\usepackage{subfiles}  % For including subfiles
\usepackage{graphicx}  % For including figures
\usepackage{float}     % For [H] figure placement
```

## Usage Options

### Basic Usage

```bash
python postprocessing/generate_latex_appendix.py
```

### Custom Options

```bash
# Specify custom output filename
python postprocessing/generate_latex_appendix.py --output My_Appendix.tex

# Specify custom plots directory
python postprocessing/generate_latex_appendix.py --plots-dir /path/to/plots

# Both custom options
python postprocessing/generate_latex_appendix.py --output My_Appendix.tex --plots-dir CustomPlots
```

## Generated Structure

The script creates a LaTeX appendix with this hierarchy:

```
Appendices
└── CFD Simulation Results
    ├── Unknown Results
    │   └── Reynolds Number 20
    │       └── Coarse Mesh
    │           └── Unstructured Mesh with TVD Scheme
    ├── Channelflow Results  
    │   └── Reynolds Number 10
    │       ├── Coarse Mesh
    │       │   └── Unstructured Mesh with Upwind Scheme
    │       └── Medium Mesh
    │           └── Unstructured Mesh with TVD Scheme
    └── Liddrivencavity Results
        ├── Reynolds Number 100
        │   ├── Coarse Mesh
        │   │   ├── Uniform Mesh with QUICK Scheme
        │   │   ├── Uniform Mesh with TVD Scheme
        │   │   ├── Uniform Mesh with Upwind Scheme
        │   │   ├── Unstructured Mesh with QUICK Scheme
        │   │   ├── Unstructured Mesh with TVD Scheme
        │   │   └── Unstructured Mesh with Upwind Scheme
        │   ├── Medium Mesh (6 schemes)
        │   └── Fine Mesh (5 schemes)
        ├── Reynolds Number 400 (7 schemes)
        ├── Reynolds Number 1000 (8 schemes)
        ├── Reynolds Number 3200 (8 schemes)
        └── Reynolds Number 5000 (4 schemes)
```

## Features

### 🎯 **Detailed Navigation**
- Table of contents shows all levels down to individual mesh/scheme combinations
- Clickable hyperlinks for easy navigation in PDF viewers
- Proper sectioning that integrates with your thesis TOC

### 🏷️ **Smart Labeling**
- Meaningful figure labels like `sim:ldc_100_uni_c_q_c2f9a4fc`
- Abbreviations: LDC (Lid Driven Cavity), Uni (Uniform), c (coarse), Q (QUICK)
- Includes simulation ID for precise referencing

### 📝 **Descriptive Captions**
- Auto-generated captions: "Lid Driven Cavity simulation at Re=100 using uniform coarse mesh with QUICK discretization scheme (Simulation ID: c2f9a4fc)"
- Consistent formatting across all figures

### 🔗 **Thesis Integration**
- Compatible with `subfiles` package
- No document structure conflicts
- Seamless integration with existing thesis chapters

## Directory Structure Requirements

Your `AppendixPlots` directory should follow this structure:

```
AppendixPlots/
├── experimentName/
│   ├── Re_XXX/
│   │   ├── resolution/
│   │   │   └── experimentName_ReXXX_meshtype_resolution_scheme_simid.pdf
│   │   └── ...
│   └── ...
└── ...
```

### Example Valid Filenames
- `lidDrivenCavity_Re100_uniform_coarse_QUICK_c2f9a4fc.pdf`
- `channelFlow_Re10_unstructured_medium_TVD_da7eeea1.pdf`
- `backwardFacingStep_Re5000_structured_fine_Upwind_8db9dbd1.pdf`

## Troubleshooting

### Common Issues

1. **Missing figures**: Ensure PDF filenames follow the exact naming convention
2. **Path issues**: Make sure `AppendixPlots` folder is in the same directory as the script
3. **Compilation errors**: Verify all required LaTeX packages are included in your thesis preamble

### File Requirements

- **Required packages**: `subfiles`, `graphicx`, `float`, `hyperref`
- **File format**: PDFs only (other formats will be ignored)
- **Naming convention**: Must match the pattern exactly for proper parsing

## Integration Example

Here's how to integrate the appendix into your thesis:

```latex
\documentclass{article}

\usepackage{subfiles}
\usepackage{graphicx}
\usepackage{float}
\usepackage{hyperref}

\begin{document}

% Your thesis content here...

% --- Appendix --- 
\subfile{Appendix_CFD_Results}

% --- References ---
\bibliographystyle{plain}
\bibliography{Settings/References}

\end{document}
```

The generated appendix will automatically:
- Add itself to your thesis table of contents
- Create proper appendix numbering (Appendix A, A.1, A.1.1, etc.)
- Provide detailed navigation at all levels
- Include all figures with consistent formatting 