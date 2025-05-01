# SIMPLE Solver Tech Stack & Optimization Strategy

This document outlines the recommended tech stack and high-performance computing (HPC) strategy for implementing a fast, scalable, and maintainable SIMPLE CFD solver in Python.

---

## Goals
- Build the **fastest possible CFD solver** in Python while maintaining correctness
- Support execution from **MacBook (CPU)** to **HPC clusters (MPI + GPU)**
- Be **modular**, **maintainable**, and **configurable via YAML**
- Support both **matrix-based** and **matrix-free** solving

---

## Technology Stack

### Language / Libraries
| Layer            | Tool               | Purpose                               |
|------------------|--------------------|----------------------------------------|
| Core logic       | Python             | High-level structure, flexibility      |
| Discretization   | Numba              | Native-speed coefficient assembly      |
| Linear solvers   | PETSc + petsc4py   | Scalable CPU/GPU + MPI solvers        |
| Config           | PyYAML             | All settings in `simulation.yaml`         |
| Parallelism      | MPI (via PETSc)    | Domain decomposition & scalability    |
| GPU backend      | CUDA via PETSc     | GPU acceleration of linear systems    |

---


## Performance Optimization Guidelines

### Numba (CPU)
- Use `@njit(parallel=True)` when possible 
- Avoid in-loop allocation (`np.zeros`) — preallocate and reuse
- Keep PETSc Vecs out of Numba functions — convert to NumPy outside

### Matrix Assembly
- Use triplet (`row`, `col`, `val`) format for COO → CSR
- Assemble once when possible; reassemble only if coefficients change

### Matrix-Free Operators
- Implement `apply_operator(x)` in Numba for MatShell
- Avoid unnecessary recomputation of geometry/interpolation weights

### PETSc Solvers
- Matrix-based or matrix-free depending on memory and mesh
- Use:
  - `ksp_type:  bicgstab | gmres`
  - `pc_type: gamg `
  - GPU: `mat_type: aijcusparse`, `vec_type: cuda`
- Let PETSc handle MPI partitioning

### Residual Checking
- Avoid `.getArray()` on GPU PETSc Vecs mid-iteration
- Use physical (unrelaxed) residuals

---
## Code Structure
```text
project_root/
├── experiments/
│   ├── exp001_simple/
│   │   ├── simulation.yaml       # SIMPLE solver configuration
│   │   └── results/
│   └── exp002_piso/
│       ├── simulation.yaml       # PISO solver configuration
│       └── results/
├── shared_configs/
│   ├── domain/
│   │   ├── boundaries_lid_driven_cavity.yaml
│   │   └── boundaries_cyllinder_flow.yaml
│   ├── hpc/
│   │   ├── base_gpu.yaml
│   │   └── base_cpu.yaml
│   ├── physics/
│   │   ├── incompressible_water.yaml
│   │   └── air_25C.yaml
├── meshes/
│   └── lid_driven/
│       ├── structured/
│       │   ├── uniform/
│       │   │   ├── lid_031.msh
│       │   │   ├── lid_063.msh
│       │   │   └── ...
│       │   └── boundary_refined/
│       │       ├── lid_031_refined.msh
│       │       └── ...
│       └── unstructured/
│           ├── uniform/
│           │   ├── lid_031_unstr.msh
│           │   └── ...
│           └── boundary_refined/
│               ├── lid_031_unstr_refined.msh
│               └── ...
├── naviflow_collocated/          # Core solver implementation (modular, OOP)
      -- mesh 
          -- base.py 
          -- structured.py
          -- unstructured.py
│   ├── core/
│   │   ├── simple_loop.py
│   │   ├── fields.py
│   │   ├── solver_interface.py
│   │   └── rhie_chow.py
│   ├── discretization/
│   │   ├── base_discretization.py
│   │   ├── convection/
│   │   │   ├── base_convection.py
│   │   │   ├── upwind.py
│   │   │   └── power_law.py
│   │   ├── diffusion/
│   │   │   ├── base_diffusion.py
│   │   │   └── central_diff.py
│   │   ├── gradient/
│   │   │   ├── base_gradient.py
│   │   │   └── gauss.py
│   │   └── interpolation/
│   │       ├── base_interpolation.py
│   │       └── linear.py
│   ├── linear_solvers/
│   │   ├── base_solver.py
│   │   ├── petsc_solver.py
│   │   ├── scipy_solver.py
│   │   └── custom_solver.py
│   ├── assembly/
│   │   ├── base_assembler.py
│   │   ├── momentum_eq.py
│   │   ├── pressure_eq.py
│   │   └── matrix_utils.py
│   └── utils/
│       └── logger.py
├── utils/
│   ├── config_loader.py
│   ├── preprocessing.py         # Mesh generation and setup
│   ├── postprocessing.py
│   └── validation.py
├── tests/                       # Unit & integration tests (Moukalled-aligned)
│   ├── conftest.py                  # Mesh-agnostic test setup
│   ├── test_mesh.py
│   ├── test_fields.py
│   ├── test_discretization.py
│   ├── test_matrix_assembly.py
│   ├── test_rhie_chow.py
│   ├── test_pressure_correction.py
│   ├── test_velocity_correction.py
│   ├── test_residuals.py
│   ├── test_under_relaxation.py
│   ├── test_utils.py
│   └── test_synthetic_cases.py
├── main.py                      # Entry point to launch experiments
└── launch_job.py                # HPC job launcher
```

---
## Summary
- Use Numba for fast, flexible discretization
- Use PETSc (via petsc4py) for high-performance scalable solves
- Allow clean switching between modes
- All config is runtime-switchable via YAML
- Design to scale from local debugging to full GPU-MPI deployment
