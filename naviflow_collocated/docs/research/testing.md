
# Critical Validation Tests (Pre-SIMPLE)

## 1. Geometric Integrity Checks

### Cell volumes positive and non-zero

### Face areas positive and non-zero

### Face normals magnitude matches face areas

## 2. Connectivity Validation

### Owner/neighbor cell indices valid

### Boundary faces have neighbor_cells == -1

## 3. Mesh Quality Metrics

### Skewness (Wolf Dynamics recommendation)
### Non-orthogonality (Moukalled Ch 8)
### Aspect ratio check

## 4. Boundary Condition Validation

### At least one moving wall BC for lid-driven cavity
### BC face indices within valid range



## Mandatory Mesh Quality Thresholds

| Metric               | Maximum Allowable | Failure Action                |
|----------------------|-------------------|--------------------------------|
| Skewness Angle       | 85°               | Re-mesh with better orthogonality |
| Aspect Ratio         | 5:1               | Apply mesh smoothing           |
| Cell Volume Ratio    | 100:1             | Gradual mesh transition zones  |
| Non-ortho Correction | 0.3 (|T_f|/|d_CF|) | Enable non-orthogonal correctors |

## Implementation Notes

1. **Numba Compatibility**:
   - Use `numba.typed.List` for dynamic data
   - Prefer `float64` over `float32` for stability
   - Avoid object types and Python dicts

2. **Performance Optimization**:


3. **Boundary Handling**:
- Encode BC types as integers:
  - 0: Dirichlet
  - 1: Neumann
  - 2: Moving Wall
  - 3: Symmetry

4. **Data Layout**:
- Use C-order arrays for memory efficiency
- Precompute often-used terms (d_CF, face interpolation factors)
- Store cell-to-face connectivity as CSR format

This structure ensures mesh-agnostic SIMPLE implementation while maintaining Numba compatibility and numerical stability per Moukalled's FVM methodology.










