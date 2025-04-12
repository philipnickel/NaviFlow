# Geometric Multigrid Preconditioned Conjugate Gradient

This directory contains examples of using the Geometric Multigrid method as a preconditioner for the Conjugate Gradient solver.

## Features

- `GeoMultigridPrecondCGSolver`: This pressure solver combines the robustness of the Conjugate Gradient method with the efficiency of the Multigrid method as a preconditioner.
  
- The solver leverages:
  - Multigrid's ability to quickly reduce low-frequency error components
  - CG's ability to achieve very accurate solutions 

## Theory

Multigrid methods are excellent at handling low-frequency error components but may converge slowly for certain problems. Conversely, the Conjugate Gradient method is very efficient for symmetric positive definite systems but can struggle with ill-conditioned matrices.

By using Multigrid as a preconditioner for CG, we get:
1. Faster convergence than standard CG
2. More robust convergence than standalone Multigrid in some cases
3. Better handling of complex boundary conditions and highly anisotropic grids

## Example Script

- **geomg_cg.py**: Demonstrates using the Geometric Multigrid Preconditioned CG solver for a lid-driven cavity problem.

## Running the Example

```bash
python geomg_cg.py
```

Results will be saved in the `results/` directory. 