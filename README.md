# NaviFlow Collocated Solver

A modular, HPC-ready SIMPLE CFD solver.

## Structure

- `naviflow_collocated/`: Core solver implementation
- `tests/`: Unit and integration tests
- `experiments/`: Simulation setups and results
- `shared_configs/`: Reusable configuration snippets
- `docs/`: Project documentation

## Usage

### Preprocessing
1. Configure meshes for experiments:
```bash
python generate_meshes.py [args]
```

### Running Simulations
Run the script using bash:
```bash
bash run_experiment.sh --experiment lidDrivenCavity --reynolds 100 --mesh-type uniform --mesh-size coarse
```

Debug mode:
```bash
bash run_experiment.sh --experiment lidDrivenCavity --debug
```

### Available Options

#### Required Arguments
- `--experiment`: Name of the experiment to run
  - Options: `lidDrivenCavity`, `channelFlow`, `cylinderFlow`

#### Normal Mode Arguments (required when not in debug mode)
- `--reynolds`: Reynolds number for the simulation
  - Options: `100`, `400`, `1000`, `3200`, `5000`
- `--mesh-type`: Type of mesh to use
  - Options: `uniform`, `unstructured`
- `--mesh-size`: Size of the mesh
  - Options: `coarse`, `medium`, `fine`

#### Optional Arguments
- `--debug`: Run in debug mode using config from experiments/<experiment>/debugging/config.yaml
  - When this flag is set, mesh and Reynolds parameters are not required
- `--postprocess`: Automatically run postprocessing after simulation completes
  - Generates all plots and analysis in the appropriate directory
  - Can be used with both normal and debug modes

### Examples

```bash
# Run simulation with postprocessing
bash run_experiment.sh --experiment lidDrivenCavity --reynolds 100 --mesh-type uniform --mesh-size coarse --postprocess

# Run debug mode with postprocessing
bash run_experiment.sh --experiment channelFlow --debug --postprocess
```

### Channel Flow Examples
```bash
# Run channel flow with unstructured medium mesh at Re=10
bash run_experiment.sh --experiment channelFlow --reynolds 10 --mesh-type unstructured --mesh-size medium --postprocess
```

### Cylinder Flow Examples
```bash
# Run cylinder flow with unstructured medium mesh at Re=5
bash run_experiment.sh --experiment cylinderFlow --reynolds 5 --mesh-type unstructured --mesh-size medium --postprocess

# Run cylinder flow with unstructured medium mesh at Re=20
bash run_experiment.sh --experiment cylinderFlow --reynolds 20 --mesh-type unstructured --mesh-size medium --postprocess
```

# profiling
```bash
python -m cProfile -o lidDrivenCavity.prof main.py --config experiments/lidDrivenCavity/ForReport/unstructured/fine/Re_100/config.yaml
```

### Postprocessing
You can also run postprocessing separately:
```bash
# Process all results for an experiment
python postprocess.py --experiment lidDrivenCavity --all

# Process specific experiment results
python naviflow_collocated/utils/postprocess/postprocess.py --experiment lidDrivenCavity/staggered --all
```

### Comparing Multiple Lid-Driven Cavity Experiments
To compare multiple lid-driven cavity experiments against Ghia's benchmark data, use the `compare_lid_driven_cavity.py` script. This script generates comparison plots showing velocity profiles along the vertical and horizontal centerlines for each Reynolds number.

Example usage:
```bash
python naviflow_collocated/utils/postprocess/compare_lid_driven_cavity.py \
    --experiments experiments/lidDrivenCavity/ForReport/uniform/coarse/Re_100 \
    experiments/lidDrivenCavity/ForReport/uniform/medium/Re_100 \
    experiments/lidDrivenCavity/ForReport/unstructured/coarse/Re_100 \
    --output-dir experiments/lidDrivenCavity/ForReport/comparisons
```

This command will:
1. Load data from each experiment directory.
2. Group experiments by Reynolds number.
3. For each Reynolds number, create comparison plots showing:
   - u-velocity along the vertical centerline (x=0.5)
   - v-velocity along the horizontal centerline (y=0.5)
4. Compare with Ghia's benchmark data.
5. Include a legend showing mesh resolution, scheme, and mesh type.
6. Save plots as PDFs in the specified output directory.

These commands can be run either locally or from SSH CLI.
