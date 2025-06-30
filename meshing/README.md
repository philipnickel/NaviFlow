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

## Mesh Format

For each mesh, a single file format is used:

- `.msh` - Native GMSH format (version 4.1) with physical groups for boundaries
  - Used for both simulation and visualization
  - Contains all boundary information needed for the solver
  - Can be directly read by ParaView using the meshio plugin

This simplified approach avoids format conversion issues and enables direct visualization of the simulation meshes.

## Mesh Visualization

### Setting up the ParaView meshio Plugin

To visualize the `.msh` files in ParaView, you need to install the meshio plugin:

1. Install the meshio Python package in ParaView's Python environment:
   ```bash
   pip install meshio
   ```
   
2. Download the ParaView meshio plugin from:
   - https://github.com/nschloe/meshio/blob/main/tools/paraview-meshio-plugin.py

3. In ParaView, go to Tools → Manage Plugins → Load New and select the downloaded plugin file
4. Check the "Auto Load" option to load it automatically in future sessions

### Visualizing Meshes

Once the plugin is installed, you can visualize meshes using:

```bash
# Visualize all experiment meshes
pvpython meshing/visualize_meshes.py

# Visualize meshes for a specific experiment
pvpython meshing/visualize_meshes.py meshing/experiments/lidDrivenCavity

# Visualize a specific mesh file
pvpython meshing/visualize_meshes.py meshing/experiments/lidDrivenCavity/structuredRefined/lidDrivenCavity_refined.msh
```

Alternatively, you can open the `.msh` files directly in the ParaView GUI.

## Directory Structure

Generated mesh files are organized as follows:
```
meshing/
    experiments/
        [experiment_type]/
            structuredUniform/
                [experiment_type]_uniform.msh        # GMSH file (for both simulation and visualization)
                [experiment_type]_uniform.pdf        # Visualization output (if generated)
            structuredRefined/
                # Similar structure as above
            unstructured/
                # Similar structure as above
```

## Requirements

- Python 3.x
- NumPy
- gmsh (Python bindings)
- ParaView (for visualization)
  - With the meshio plugin installed
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