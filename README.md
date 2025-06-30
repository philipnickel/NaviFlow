# NaviFlow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NaviFlow is a finite volume based solver written in Python for the 2D incompressible Navier-Stokes equations. The solver features both collocated and staggered grid implementations of the SIMPLE algorithm. 
Based on the finite volume methodology of Versteeg & Malalasekera, Moukalled et al., and Ferziger & Perić, it was developed as part of my bachelor's thesis in applied mathematics at the Technical University of Denmark together with Aske Funch Schrøder Nielsen, under the supervision of Associate Professor Allan P. Engsig-Karup (DTU Compute) and Professor Gustaaf Jacobs (San Diego State University).

## Why Make This Public?
While there are countless publicly available codes for the lid-driven cavity problem using structured grids and simple numerical methods (likely because often educational books present the methods this way), there are few implementations that handle unstructured meshes while remaining accessible and not overly complex. The code might have some rough edges or mistakes (thesis deadline...), but it produces good results and might be helpful for others implementing similar methods. 
If you have questions about specifics, feel free to reach out at philipnickel@outlook.dk.


## Table of Contents
- [Features](#features)
- [Highlighted Results and Features](#highlighted-results-and-features)
  - [Solver Performance](#solver-performance)
  - [Code Verification (MMS)](#code-verification-mms)
  - [Mesh Generation](#mesh-generation)
  - [Lid-Driven Cavity Flow](#lid-driven-cavity-flow)
  - [Flow past a cylinder (Re = 20)](#flow-past-a-cylinder-re--20)
  - [Unsteady flow past a cylinder (Re = 100)](#unsteady-flow-past-a-cylinder-re--100)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Technical Details and Implementation](#technical-details-and-implementation)
- [License](#license)

## Features

- Solvers for both collocated and staggered grid arrangements
- Convection scheme implementations:
  - TVD with MUSCL, OSPRE, and H-CUBIC limiters
  - QUICK scheme
  - Power Law scheme
  - First-order Upwind
- Unstructured and structured mesh support with non-orthogonal and skewness corrections
- Steady and unsteady flow simulations with implicit Euler and Crank-Nicolson time integration
- Gradient computation using least-squares method
- Pressure-velocity coupling via SIMPLE algorithm
- Linear solvers:
  - PETSc and SciPy integration for various linear solvers and preconditioners.
  - Custom geometric multigrid implementation (V-cycle, W-cycle and Full Multigrid) as standalone solvers or preconditioners.


## Highlighted Results and Features

### Solver Performance

The BiCGSTAB solver from SciPy achieves optimal scaling when preconditioned with the custom Gauss-Seidel smoothed multigrid V-cycle implementation. The Full Multigrid (FMG) implementation demonstrates optimal complexity, with wall time scaling linearly with problem size. Wall times were profiled using CProfile, measuring the pressure correction step of the SIMPLE algorithm over 5 iterations across different mesh resolutions.

<p align="center">
  <img src="assets/wall_times_plot.png" width="70%"/>
  <br>
  <em>Wall time scaling comparison for pressure equation solvers</em>
</p>

### Code Verification (MMS)

Method of Manufactured Solutions verifies the spatial discretization of convection-diffusion terms:

$$ 
\rho(\vec{u} \cdot \nabla)\phi - \mu \nabla^2\phi = f
$$

Using a manufactured velocity field:

$$ 
\vec{u} = \begin{pmatrix}
\cos(\pi y)\sin(\pi x) \\
-\cos(\pi x)\sin(\pi y)
\end{pmatrix}
$$

Source term $f$ is derived analytically from the manufactured solution. Results show expected order of accuracy for both structured and unstructured meshes ($\mu=0.1$, $\rho=1.0$).

<p align="center">
  <img src="assets/verification/mms_structured.png" width="48%"/>
  <img src="assets/verification/mms_unstructured.png" width="48%"/>
  <br>
  <em>Momentum convergence on structured (left) and unstructured (right) meshes</em>
</p>

### Lid-Driven Cavity Flow
This section demonstrates the solver's capability to handle flows across a wide range of Reynolds numbers (Re=100 to Re=5000) and its performance with different numerical schemes and mesh types.

#### Flow Pattern Evolution with Reynolds Number

Results obtained using first-order Upwind scheme on a uniform mesh (16,129 CVs).

<p align="center">
  <img src="assets/lid_driven_cavity/streamlines_legacy_Re100.png" width="30%"/>
  <img src="assets/lid_driven_cavity/streamlines_legacy_Re400.png" width="30%"/>
  <img src="assets/lid_driven_cavity/streamlines_legacy_Re1000.png" width="30%"/>
  <br>
  <img src="assets/lid_driven_cavity/streamlines_legacy_Re3200.png" width="30%"/>
  <img src="assets/lid_driven_cavity/streamlines_legacy_Re5000.png" width="30%"/>
  <br>
  <em>Streamlines: Re=100, 400, 1000, 3200, and 5000</em>
</p>

<p align="center">
  <img src="assets/lid_driven_cavity/ghia_comparison_Re100.png" width="30%"/>
  <img src="assets/lid_driven_cavity/ghia_comparison_Re400.png" width="30%"/>
  <img src="assets/lid_driven_cavity/ghia_comparison_Re1000.png" width="30%"/>
  <br>
  <img src="assets/lid_driven_cavity/ghia_comparison_Re3200.png" width="30%"/>
  <img src="assets/lid_driven_cavity/ghia_comparison_Re5000.png" width="30%"/>
  <br>
  <em>Validation against Ghia et al. (1982) benchmark data</em>
</p>

#### Numerical Scheme Comparison

Different convection schemes (Upwind, Power Law, QUICK and TVD) were tested to evaluate their accuracy and stability. The results below show velocity profiles compared against Ghia's benchmark data.

<p align="center">
  <img src="assets/lid_driven_cavity/schemes_Re400.png" width="48%"/>
  <img src="assets/lid_driven_cavity/schemes_Re1000.png" width="48%"/>
  <br>
  <em>Comparison of discretization schemes at Re=400 (left) and Re=1000 (right).</em>
</p>

<img src="assets/lid_driven_cavity/residuals_Re1000.png" width="35%" align="right" style="margin-left: 20px; margin-bottom: 10px;"/>

The residual history (right) demonstrates stable convergence behavior for the momentum and continuity equations using the TVD scheme at Re=1000.

<div style="clear: both;"></div>

#### Mesh Type Comparison

The solver supports both structured (uniform) and unstructured meshes. Below is a comparison at Re=5000, demonstrating improved results by locally refining the mesh at the top corners in the unstructured case while keeping the number of control volumes in the same range.

<p align="center">
  <img src="assets/lid_driven_cavity/ghia_Re5000_structured.png" width="48%"/>
  <img src="assets/lid_driven_cavity/ghia_Re5000_unstructured.png" width="48%"/>
  <br>
  <em>Re=5000: Comparison between structured (left) and unstructured (right) meshes.</em>
</p>

### Flow past a cylinder (Re = 20)

<p align="center">
  <img src="assets/cylinder_flow/Re20_velocity_magnitude.png" width="90%"/>
  <br>
  <em>Velocity magnitude for steady flow over a cylinder at Re=20.</em>
</p>


<div style="clear: both;"></div>

### Unsteady flow past a cylinder (Re = 100)

The solver's behavior in a transient simulation of flow over a cylinder at Re=100 is explored under two different mesh conditions.

**Case 1: Dispersive Behavior**

<p align="center">
  <img src="assets/cylinder_flow/transient_Re100_dispersive.gif" width="90%"/>
  <br>
  <em>Pseudo-steady state solution at Re=100 on a standard mesh.</em>
</p>

<img src="assets/cylinder_flow/transient_Re100_dispersive_residuals.png" width="35%" align="right" style="margin-left: 20px; margin-bottom: 10px;"/>

On a standard mesh, the solver's numerical dispersion seems to dampen the physical instabilities leading to convergence to an unphysical pseudo-steady state.

<div style="clear: both;"></div>

<div align="center">
<table style="width:85%; font-size:90%; margin: auto; text-align:left;">
  <tr style="border-bottom: 1px solid #eee;">
    <td style="padding: 4px; width: 30%;"><b>Discretization schemes</b></td>
    <td style="padding: 4px;">TVD (MUSCL Limiter), Crank-Nicolson</td>
  </tr>
  <tr style="border-bottom: 1px solid #eee;">
    <td style="padding: 4px;"><b>Mesh</b></td>
    <td style="padding: 4px;">16,242 Control Volumes</td>
  </tr>
  <tr>
    <td style="padding: 4px;"><b>Params</b></td>
    <td style="padding: 4px;">Re = 100, dt = 0.0025s</td>
  </tr>
</table>
</div>

**Case 2: Physical Instability**

<p align="center">
  <img src="assets/cylinder_flow/transient_Re100_refined_mesh.gif" width="90%"/>
  <br>
  <em>Vortex shedding with mesh with the bottom left corner refined.</em>
</p>

<img src="assets/cylinder_flow/transient_Re100_refined_mesh_residuals.png" width="35%" align="right" style="margin-left: 20px; margin-bottom: 10px;"/>

While the setup already introduces asymmetry through the cylinder placement, additional mesh refinement in the bottom left corner helps maintain the physical instability.

<div style="clear: both;"></div>

<div align="center">
<table style="width:85%; font-size:90%; margin: auto; text-align:left;">
  <tr style="border-bottom: 1px solid #eee;">
    <td style="padding: 4px; width: 30%;"><b>Discretization schemes</b></td>
    <td style="padding: 4px;">TVD (MUSCL Limiter), Crank-Nicolson</td>
  </tr>
  <tr style="border-bottom: 1px solid #eee;">
    <td style="padding: 4px;"><b>Mesh</b></td>
    <td style="padding: 4px;">11,280 Control Volumes</td>
  </tr>
  <tr>
    <td style="padding: 4px;"><b>Params</b></td>
    <td style="padding: 4px;">Re = 100, dt = 0.0025s</td>
  </tr>
</table>
</div>

## Dependencies

This project has a number of dependencies, which can be viewed in the `requirements.txt` file.


## Usage

The main entry point for running a simulation is `main.py`. It requires a configuration file that specifies the problem setup, mesh, physical properties, and solver parameters.

To run a simulation, use the following command:

```bash
python main.py --config path/to/your/config.yaml
```

Example configuration files can be found in the `experiments/` directory. For instance, to run a lid-driven cavity flow simulation:

```bash
python main.py --config experiments/Collocated/lidDrivenCavity/TVD/medium/config.yaml
```

## Technical Details and Implementation

### Numerical Methods
- **Discretization Schemes**:
  - Convection: TVD schemes with various limiters (MUSCL, OSPRE, H-CUBIC), QUICK, Power Law, and Upwind schemes (some implemented via deferred correction approach).
  - Diffusion: Central differencing with non-orthogonal and skewness corrections
  - Pressure-Velocity Coupling: SIMPLE algorithm 
  - Time Integration: Euler implicit and Crank-Nicolson schemes

- **Gradient Formulations**:
  - Least-squares 

- **Mesh Support**:
  - Unstructured and structured grids
  - Non-orthogonal skewed meshes
  - Local refinement capabilities 


- **Linear Solvers**:
  - PETSc and SciPy integration for various linear solvers and preconditioners.
  - Custom geometric multigrid implementation (V-cycle, W-cycle and Full Multigrid) for pressure equation
  - Nullspace removal for pressure correction via PETSc or 'classic' pressure node pinning.

- **Boundary Conditions**:
  - No-slip and prescribed velocity (Dirichlet)
  - Inlet and outlet boundary conditions.

### Key References
1. Versteeg, H.K. and Malalasekera, W. (2007) "An Introduction to Computational Fluid Dynamics: The Finite Volume Method"
2. Moukalled, F., Mangani, L. and Darwish, M. (2016) "The Finite Volume Method in Computational Fluid Dynamics"
3. Ferziger, J.H. and Perić, M. (2002) "Computational Methods for Fluid Dynamics"
4. Ghia, U., Ghia, K.N. and Shin, C.T. (1982) "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method", Journal of Computational Physics, 48(3), 387-411
5. Schäfer, M., Turek, S., Durst, F., Krause, E. and Rannacher, R. (1996) "Benchmark Computations of Laminar Flow Around a Cylinder", Notes on Numerical Fluid Mechanics, 48, 547-566

### Keywords and Technical Tags
`computational-fluid-dynamics` `cfd` `navier-stokes` `finite-volume` `python` `petsc` `numba` `deferred-correction` `tvd-schemes` `quick-scheme` `power-law` `rhie-chow` `least-squares-gradients` `green-gauss` `non-orthogonal-correction` `skewness-correction` `unstructured-mesh` `structured-mesh` `gmsh` `multigrid` `high-performance-computing` `lid-driven-cavity` `cylinder-flow` `scientific-computing` `incompressible-flow`

## Project Structure

-   `naviflow_collocated/`: Source code for the collocated grid solver.
-   `naviflow_staggered/`: Source code for the staggered grid solver.
-   `experiments/`: Contains configuration files and results for various simulation cases.
-   `meshing/`: Scripts and tools for mesh generation.
-   `postprocessing/`: Scripts for analyzing and visualizing simulation results.
-   `tests/`: Unit and integration tests.

## License

This project is licensed under the MIT License.

