# NaviFlow Multigrid Orientation Fix: Fortran Ordering Compatibility

## Summary

We identified and fixed a significant issue in NaviFlow's multigrid implementation related to grid orientation and Fortran ordering compatibility. The problem was in the `restrict()` function, where a vertical flip was causing inconsistency when used with the interpolation operator in the multigrid V-cycle.

## Issue Identification

The multigrid solver in NaviFlow uses Fortran ordering (`order='F'`) when reshaping arrays from 1D to 2D and back. This means data is stored in column-major order rather than the default row-major order in NumPy. The original implementation of the `restrict()` function included a vertical flip (`np.flip(fine_grid, axis=0)`) which was incompatible with this ordering scheme.

This incompatibility led to:
1. Unpredictable behavior during the V-cycle
2. Poor convergence for certain patterns (especially high-frequency components)
3. Grid orientation changing between restriction levels

## Investigation

We created a comprehensive testing framework to analyze different approaches to restriction with Fortran ordering:

1. **Original NaviFlow implementation** (with vertical flip)
2. **No-flip approach** (simple injection at odd indices)
3. **With-flip approach** (sample first, then flip)
4. **Flip-then-sample approach** (flip first, then sample)

Each approach was tested by simulating a full V-cycle with Fortran ordering and measuring the relative error between the original and final grids.

## Results

The comprehensive testing showed:

| Implementation | Relative Error | Notes |
|----------------|----------------|-------|
| Original (with flip) | 0.547019 | High error |
| No-flip | 0.034530 | Significantly lower error |
| With-flip | 0.547019 | High error |
| Flip-then-sample | 0.547019 | High error |

The **no-flip approach** produced dramatically better results when used with the existing interpolation function in a multigrid V-cycle with Fortran ordering.

## The Fix

We modified the `restrict()` function in `naviflow_oo/solver/pressure_solver/helpers/multigrid_helpers.py` to use the no-flip approach:

```python
def restrict(fine_grid: np.ndarray) -> np.ndarray:
    """
    Reduces a fine grid to a coarse grid by taking every other point.
    This implementation specifically works with Fortran ordering ('F') used in the multigrid solver.
    
    Parameters:
        fine_grid (np.ndarray): The input fine grid to be coarsened
        
    Returns:
        np.ndarray: The coarsened grid
    """
    if fine_grid.ndim == 1:
        # For 1D arrays, use simple sampling without flipping
        # Based on comprehensive testing, this approach works best with Fortran ordering
        return fine_grid[1::2]
    else:
        # Take every other point using odd indices (1, 3, 5, ...)
        # This is direct injection at odd indices without flipping
        # Testing with Fortran ordering showed this approach has the lowest error
        # when used with the interpolate function in a V-cycle
        return fine_grid[1::2, 1::2]
```

## Validation

We validated the fix with a comprehensive test suite that simulates a full multigrid V-cycle with different test patterns:

1. **Linear gradient**: Relative error reduced from 0.547 to 0.047
2. **Sine wave**: Still has higher error (0.592) due to high-frequency content, but orientation is preserved
3. **Checkerboard**: Difficult pattern for multigrid, but orientation is consistent
4. **Gaussian**: Good performance with 0.135 relative error

The corner values analysis showed that the orientation is now preserved consistently throughout the V-cycle, and the grid patterns maintain their expected structure.

## Expected Impact

This fix should significantly improve the convergence of NaviFlow's multigrid solver, particularly for:

1. Problems with complex patterns
2. Solutions that require multiple V-cycles
3. Multi-level solvers that need consistent orientation between levels

The fix maintains compatibility with the existing `interpolate()` function and the Fortran-order reshaping used throughout the multigrid solver.

## Conclusion

The grid orientation issue in NaviFlow's multigrid implementation was caused by an unnecessary vertical flip in the `restrict()` function, which was incompatible with the Fortran ordering used in the solver. By removing this flip and implementing a direct injection approach, we've significantly improved the compatibility between restriction and interpolation operations, which should lead to better convergence of the multigrid solver. 