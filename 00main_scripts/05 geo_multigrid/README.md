# NaviFlow Multigrid Solver Smoothers

This directory contains multigrid solvers with different smoothing techniques for CFD pressure correction equations.

## Implemented Smoothers

### 1. Gauss-Seidel Smoother
- Standard point-wise Gauss-Seidel iteration
- Fast convergence when used as a smoother in multigrid
- Efficient for moderately-sized problems
- Defined in `naviflow_oo/solver/pressure_solver/gauss_seidel.py`

### 2. Line Gauss-Seidel Smoother
- Solves entire lines (rows or columns) at once using tridiagonal solvers
- Better at handling anisotropic problems with strong coupling in one direction
- More computationally intensive per iteration than point-wise methods
- Configured to alternate between x and y sweeps for optimal convergence
- Defined in `naviflow_oo/solver/pressure_solver/line_gauss_seidel.py`

## Performance Comparison

From our tests on the standard lid-driven cavity problem (Re=100):

| Smoother             | Time (s) | Iterations | Residual  | Max Divergence |
|----------------------|----------|------------|-----------|----------------|
| Standard Gauss-Seidel| 4.98     | 116        | 9.96e-04  | 1.82e-04       |
| Line Gauss-Seidel    | 276.0    | 149        | 9.97e-04  | 7.93e-05       |

### Observations:
- Standard Gauss-Seidel is significantly faster for our implementation, likely due to simpler per-iteration cost
- Line Gauss-Seidel achieves slightly better mass conservation (lower divergence)
- The computational overhead of Line Gauss-Seidel outweighs its convergence benefits in this test case

## Usage

Each smoother can be used either:
1. Directly as a standalone solver
2. As a smoother within the multigrid framework

### Example: Using Line Gauss-Seidel in Multigrid

```python
# Create Line Gauss-Seidel smoother
line_gs_smoother = LineGaussSeidelSolver(
    omega=1.1,            # Slight over-relaxation
    direction='alternating',  # Alternate between x and y sweeps
    max_iterations=5,     # Iterations per smoothing step
    tolerance=1e-5        # Tolerance for smoother
)

# Create multigrid solver with this smoother
multigrid_solver = MultiGridSolver(
    smoother=line_gs_smoother,
    pre_smoothing=3,      # Pre-smoothing steps
    post_smoothing=3,     # Post-smoothing steps
    cycle_type='v'        # V-cycle type
)
```

## When to Use Each Smoother

- **Standard Gauss-Seidel**: Best for general use cases and isotropic problems
- **Line Gauss-Seidel**: May be beneficial for highly anisotropic problems or stretched grids where coupling along lines is strong

## Demo Scripts

- `line_gs_test.py`: Tests Line Gauss-Seidel as a standalone solver
- `line_gs_vcycle.py`: Uses Line Gauss-Seidel in multigrid V-cycle
- `compare_smoothers.py`: Compares performance of different smoothers 