# Mesh-Agnostic CFD Solver Implementation

This directory contains the implementation of a mesh-agnostic SIMPLE algorithm for solving the incompressible Navier-Stokes equations on arbitrary mesh topologies.

## Key Features

1. **Mesh-Agnostic Design**
   - Works with structured uniform, structured non-uniform, and unstructured meshes
   - Abstract mesh interface for accessing topology, geometry, and connectivity
   - Face-based flux calculations and coefficient assembly

2. **Collocated Grid Arrangement**
   - All variables (u, v, p) stored at cell centers
   - Rhie-Chow interpolation to prevent pressure-velocity decoupling
   - Consistent face velocity interpolation for mass flux calculation

3. **Numerical Methods**
   - SIMPLE algorithm for pressure-velocity coupling
   - Power-Law scheme for convection-diffusion discretization
   - Direct solver and AMG solver options for linear systems

4. **Performance Monitoring**
   - Comprehensive profiler for tracking execution time
   - Convergence monitoring and solver statistics
   - HDF5 output for post-processing and analysis

## Usage

The main example is in `mesh_agnostic_cavity.py`, which demonstrates the lid-driven cavity flow using different mesh types.

```python
# Choose mesh type: 'structured_uniform', 'structured_nonuniform', or 'unstructured'
MESH_TYPE = 'structured_uniform'
RESOLUTION = 21  # Number of points in each direction
REYNOLDS = 100   # Reynolds number
```

## Implementation Details

### Mesh Interface

The mesh interface provides a consistent way to access:
- Cell and face count
- Owner and neighbor relationships
- Face areas and normals
- Cell volumes and centers
- Face centers

### Rhie-Chow Interpolation

For collocated grids, we use Rhie-Chow interpolation to calculate face velocities:

1. Interpolate cell-centered velocities to faces: `u_face_avg = 0.5 * (u[owner] + u[neighbor])`
2. Calculate pressure gradients at cells
3. Interpolate the product of pressure gradients and momentum coefficients
4. Apply correction: `u_face = u_face_avg - d_u_face * (dp_dx_face - 0.5 * (dp_dx_owner + dp_dx_neighbor))`

### SIMPLE Algorithm

1. Solve momentum equations for intermediate velocities u* and v*
2. Calculate mass fluxes using Rhie-Chow interpolation
3. Solve pressure correction equation
4. Update pressure with relaxation
5. Correct velocities using pressure gradient
6. Check for convergence

## Results

The solver has been tested with:
- Structured uniform meshes
- Structured non-uniform meshes
- Unstructured meshes

Performance depends on mesh size, flow conditions, and solver parameters. Typical convergence for lid-driven cavity at Re=100 takes 10-30 iterations. 