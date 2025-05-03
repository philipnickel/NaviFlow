# Mesh Generation & Visualization

This directory contains tools for generating and visualizing meshes for CFD simulations using the NaviFlow-Collocated framework.

## Contents

- `generate_meshes.py` - Script to generate meshes for CFD experiments
- `visualize_meshes.py` - Script to visualize meshes with ParaView
- `output/` - Directory for basic mesh examples
- `experiments/` - Directory containing organized experiment-specific meshes

## Mesh Generation

Generate meshes for specific experiment types:

```bash
# Generate meshes for all experiments
python -m meshing.generate_meshes

# Generate meshes for a specific experiment
python -m meshing.generate_meshes lidDrivenCavity
```

Available experiment types:
- `lidDrivenCavity` - Standard lid-driven cavity benchmark
- `channelFlow` - Rectangular channel flow
- `flowAroundCylinder` - Flow around a circular obstacle
- `backwardFacingStep` - Flow over a backward-facing step

## Mesh Visualization

Visualize meshes using ParaView's Python API:

```bash
# Visualize all experiment meshes
pvpython meshing/visualize_meshes.py

# Visualize meshes for a specific experiment
pvpython meshing/visualize_meshes.py meshing/experiments/lidDrivenCavity

# Visualize a specific mesh file
pvpython meshing/visualize_meshes.py meshing/experiments/lidDrivenCavity/structuredUniform/lidDrivenCavity_uniform.msh
```

## Directory Structure

Generated mesh files are organized as follows:
```
meshing/
    experiments/
        [experiment_type]/
            structuredUniform/
                [experiment_type]_uniform.msh
                [experiment_type]_uniform.png (if visualized)
            structuredRefined/
                [experiment_type]_refined.msh
                [experiment_type]_refined.png (if visualized)
            unstructured/
                [experiment_type]_unstructured.msh
                [experiment_type]_unstructured.png (if visualized)
```

## Requirements

- Python 3.x
- NumPy
- gmsh (Python bindings)
- ParaView (for visualization)
  - When using ParaView visualization, run the scripts with `pvpython` or `pvbatch`
  - These executables are included with the ParaView installation

## Supported Mesh Types

1. **Structured Uniform** - Regular Cartesian grid with uniform spacing
2. **Structured Refined** - Cartesian grid with refinement near specified boundary
3. **Unstructured** - Mesh with varying element sizes, optionally with obstacles

## Using ParaView with Python

There are several ways to run ParaView Python scripts:

1. **pvpython**: Interactive stand-alone or client/server execution
2. **pvbatch**: Non-interactive, distributed batch processing 
3. **From ParaView GUI**: Tools → Python Shell

These methods automatically set up the PYTHONPATH to include ParaView's libraries.

If you want to use a regular Python interpreter, you need to manually add ParaView's libraries and Python modules to your PYTHONPATH. 