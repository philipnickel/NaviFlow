# NaviFlow Collocated Solver

A modular, HPC-ready SIMPLE CFD solver.

## Structure

- `naviflow_collocated/`: Core solver implementation
- `tests/`: Unit and integration tests
- `experiments/`: Simulation setups and results
- `shared_configs/`: Reusable configuration snippets
- `docs/`: Project documentation

## Usage


## preprocessing
1. configure meshes for experiments
Run generate_meshes.py (with CLI args possibly)

## Run the solver 
Either use the vs code tasks configured
or
main.py --experiment lid_driven_cavity (optional to overwrite some stuff with CLI args)

## postprocessing 
python postprocess.py --experiment lid_driven_cavity --all

python naviflow_collocated/utils/postprocess/postprocess.py --experiment lidDrivenCavity/staggered --all

either locally or from SSH CLI
