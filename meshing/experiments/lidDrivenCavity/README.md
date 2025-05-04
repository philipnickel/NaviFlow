# lidDrivenCavity Experiment

Classic lid-driven cavity problem with moving top wall

## Mesh Types

- **structuredUniform**: Uniform mesh for lid-driven cavity
- **structuredRefined**: Mesh with refinement near the top (lid)

## File Formats

- **.msh**: Native GMSH format with all boundary information
  - Used for both simulation and visualization
  - For visualization in ParaView, use the meshio plugin
