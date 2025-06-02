# Usage Guide: Running the SIMPLE Solver Locally or on DTU HPC

This guide explains how to configure, test, and run the SIMPLE CFD solver locally or on the DTU HPC cluster. Configuration is centralized in a YAML config file.

---

## 🗂️ Configuration Overview

To organize CFD simulation configurations efficiently, especially for multiple experiments, use a modular YAML file structure with a single entrypoint `simulation.yaml` per experiment. This file references paths to domain, mesh, algorithm, solver, and HPC configs, promoting reuse and clarity.

Example file structure for an experiment `exp001_simple_cg` inside the `experiments/` folder:
```
experiments/
  ├── exp001_simple/
  │   ├── simulation.yaml       # ← SIMPLE solver configuration
  │   └── results/
  └── exp002_piso/
      ├── simulation.yaml       # ← PISO solver configuration  
      └── results/
shared_configs/
  ├── domain/
  │   ├── boundaries_lid_driven_cavity.yaml
  │   └── boundaries_cyllinder_flow.yaml
  ├── hpc/ 
  │   ├── base_gpu.yaml
  │   ├── base_cpu.yaml
  ├── physics/
  │   ├── incompressible_water.yaml
  │   └── air_25C.yaml
meshes/
  └── lid_driven/
      ├── structured/
      │   ├── uniform/
      │   │   ├── lid_031.msh
      │   │   ├── lid_063.msh
      │   │   └── ...
      │   └── boundary_refined/
      │       ├── lid_031_refined.msh
      │       └── ...
      └── unstructured/
          ├── uniform/
          │   ├── lid_031_unstr.msh
          │   └── ...
          └── boundary_refined/
              ├── lid_031_unstr_refined.msh
              └── ...
```

### Examples:

`simulation.yaml`:
```yaml
name: "exp001_simple_cg"
note: "Baseline SIMPLE test with PETSc BiCGSTAB"
date: 2025-05-01
tags: ["lid_driven_cavity", "structured", "SIMPLE"]

domain:
  mesh: ../../meshes/lid_driven/structured/uniform/lid_127.msh
  boundary_conditions: ../../shared_configs/domain/boundaries_lid_driven_cavity.yaml

physics: ../../shared_configs/physics/incompressible_water.yaml

algorithm:
  type: SIMPLE
  transient: false
  relaxation_factors:
    velocity: 0.7
    pressure: 0.3
  max_iterations: 3000
  convergence_criteria:
    velocity_residual: 1e-6
    pressure_residual: 1e-5
  discretization: powerlaw

solvers:
  momentum:
    type: petsc
    method: bicgstab
    tolerance: 1e-8
    max_iterations: 1000
  pressure:
    type: custom
    method: geometric_multigrid
    tolerance: 1e-6
    max_iterations: 100

hpc: ../../shared_configs/hpc/base_gpu.yaml

postprocessing:
  residual_plot: true
  save_every: 100
  save_fields: true
  validation:
    reference: ghia
```

---

### Description:
- `simulation.yaml` is the single entrypoint configuration file for the experiment. It references other shared configuration files via relative paths.
- Shared reusable configuration files live in the `shared_configs/` directory and are referenced from the experiment's `simulation.yaml`.
- Mesh files are now organized separately in the `meshes/` directory structured by problem type, mesh structure, and resolution.
- This structure promotes reuse and modularity by separating experiment-specific metadata from common domain, solver, and HPC configurations.

---

## 🖥️ 1. Running Locally 
No job scripts or HPC required. Just run:
```bash
python main.py -config experiments/simple_cg/simulation.yaml
```
The simulation is launched using the single `simulation.yaml` file, which includes references to domain, mesh, algorithm, solver parameters, and HPC resources.


---

## 🧠 2. Running on DTU HPC 

### 🔹 Step 1: Estimate Walltime
Run a dry run job (3 iterations) on the HPC:
```bash
python launch_job.py -estimateWall -config experiments/simple_cg/simulation.yaml
```
This:
- Generates and submits a temporary job: `submit_wallprobe.sh`
- Runs 3 SIMPLE iterations
- Logs timing and memory usage
- Prints a walltime estimate like:
  ```
  ⏱ Time per iteration: 1.3 sec
  🧠 Peak memory: 5.2 GB
  ✅ Suggested walltime (3000 iters): 01:15:00
  📊 Estimated Peak Memory Usage for Full Simulation: 5.5 GB (includes buffer for residual history)
  ```

### 🔹 Step 2: Submit the full job with estimated walltime and estimated memory requirement
Execute the following: 
```bash
python launch_job.py -wall_time 1:15 -memory 5.5 -config experiments/simple_cg/simulation.yaml
```

The simulation is launched using the single `simulation.yaml` file, which includes references to domain, mesh, algorithm, solver parameters, and HPC resources.

This:
- Reads your single `simulation.yaml` config file
- Generates `submit.sh` with full settings
- Submits it via `bsub`

---

### 📈 Monitoring Your Job on HPC
```bash
bstat -u                     # View job status
bstat -C                     # Check CPU efficiency
bstat -M                     # Check memory usage
```

---
