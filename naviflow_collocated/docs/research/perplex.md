<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# SIMPLE Algorithm for Lid-Driven Cavity Flow: Implementation Research

Before diving into the main content, let me summarize the key findings of this research: The SIMPLE algorithm implemented on a collocated grid requires careful treatment of the pressure-velocity coupling through Rhie-Chow interpolation to avoid checkerboard oscillations. For mesh-agnostic implementation, face-based data structures and operator-based abstractions are essential. The pressure correction equation forms the core of the algorithm, while extensions to PISO can be achieved through iterative corrector steps.

## Fundamentals of the SIMPLE Algorithm

### Mathematical Formulation

- **Governing Equations**: Incompressible Navier-Stokes equations in integral form:

```
∮(ρv)·dS = 0
∮(ρv⊗v)·dS = -∮pI·dS + ∮μ∇v·dS
```

- **Discrete Momentum Equation**: Using operator notation from Moukalled[^1_6]:

```
a_Cv_C + ∑(a_Fv_F) = -V_C(∇p)_C + b̂_C
```

Or in compact form:

```
v_C + H_C[v-v_C] = -D^v_C(∇p)_C + B^v_C
```

- **Pressure-Velocity Coupling**: Core of SIMPLE algorithm[^1_6]:

1. Solve momentum with guessed pressure p* to get v*
2. Define corrections v' and p' where v = v* + v' and p = p* + p'
3. Derive v'_C = -H_C[v'] - D^v_C(∇p')_C
4. Simplify by neglecting H_C[v'] term
5. Substitute into continuity to get pressure correction equation
- **Pressure Correction Equation**[^1_6]:

```
∑(ρ_fD_f(p'_F-p'_C)) = -∑(ṁ*_f)
```


### Implementation Strategy

- **Solution Sequence**:

1. Initialize fields
2. Begin iteration loop:
        - Compute face velocities and mass fluxes
        - Solve momentum equations
        - Formulate and solve pressure correction equation
        - Correct pressure and velocity fields
        - Check convergence
- **Data Structures**:

```python
class Field:
    # Base class for scalar/vector fields
    
class ScalarField(Field):
    # Handles pressure, p'
    
class VectorField(Field):
    # Handles velocity
    
class FaceField(Field):
    # Handles face fluxes
```

- **Operator-Based Approach**:

```python
class MomentumOperator:
    def assemble_and_solve(self, velocity, pressure):
        # Returns updated velocity field
        
class PressureCorrectionOperator:
    def assemble_and_solve(self, velocity, mass_flux):
        # Returns pressure correction field
```


### Enhancement Opportunities

- **SIMPLEC Extension**[^1_6]: Approximate H_C[v'] as v'_C·H_C[^1_1] to improve convergence
- **PISO Extension**: Add multiple pressure correction steps
- **Matrix-Free Implementation**: Use PETSc's Python matrix type for custom operators[^1_7]
- **Dynamic Relaxation**: Adjust under-relaxation factors based on residual history


## Collocated Grid Implementation

### Mathematical Considerations

- **Pressure-Velocity Decoupling**: Collocated grids suffer from checkerboard oscillations
- **Rhie-Chow Interpolation**[^1_3]: Add third-derivative pressure term to prevent oscillations:

```
ṁ_f = ṁ_f^central + D_f[(p_C-p_F) - ½((∇p)_C+(∇p)_F)·d_CF]
```

- **Pressure Gradient at Faces**[^1_6]: Requires special treatment to ensure consistency
- **Momentum Interpolation**: Ensures conservation of mass and momentum


### Implementation Aspects

- **Face Flux Computation**:

```python
def compute_face_flux(self, velocity, pressure):
    # Basic interpolation
    flux_central = interpolate_to_face(velocity) · face_area
    
    # Pressure correction for Rhie-Chow
    p_diff = pressure_C - pressure_F
    p_grad_interp = 0.5*(pressure_gradient_C + pressure_gradient_F) · face_vector
    pressure_term = diffusion_coefficient * (p_diff - p_grad_interp)
    
    return flux_central + pressure_term
```

- **Flexible Grid Interface**:

```python
class Mesh:
    def get_cell_centers(self):
        # Returns cell center coordinates
    
    def get_face_centers(self):
        # Returns face center coordinates
    
    def get_face_areas(self):
        # Returns face area vectors
    
    def get_cell_volumes(self):
        # Returns cell volumes
    
    def get_cell_to_face_connectivity(self):
        # Returns mapping from cells to faces
    
    def get_face_to_cell_connectivity(self):
        # Returns mapping from faces to neighboring cells
```


### Enhancement Opportunities

- **Deferred Correction Approach**: For higher-order schemes on non-orthogonal meshes
- **Consistent Face Interpolation**: Ensure consistency between face fluxes and cell values
- **Optimized Data Structures**: Memory layout for cache efficiency and vectorization


## Discretization of Key Terms

### Diffusion Term

#### Mathematical Considerations

- **General Form**[^1_6]: (μ∇v)_f·S_f
- **Orthogonal Component**: μ_f (v_F - v_C)/|d_CF| |S_f| cos(θ)
- **Non-Orthogonal Correction**: Required when mesh faces are not orthogonal to cell centers
- **Face Viscosity**: Harmonic mean preferred for discontinuous properties


#### Implementation Aspects

```python
def compute_diffusion_coefficient(self, face_index, viscosity_field):
    # Get neighboring cells
    cell_C, cell_F = self.mesh.get_face_neighbors(face_index)
    
    # Get geometric quantities
    face_normal = self.mesh.get_face_normal(face_index)
    face_area = self.mesh.get_face_area(face_index)
    distance_CF = self.mesh.get_cell_distance(cell_C, cell_F)
    
    # Compute harmonic mean of viscosity
    visc_C = viscosity_field[cell_C]
    visc_F = viscosity_field[cell_F]
    visc_f = 2 * visc_C * visc_F / (visc_C + visc_F)
    
    # Compute diffusion coefficient
    orthogonality = dot(face_normal, self.mesh.get_cell_vector(cell_C, cell_F))
    orthogonality /= (distance_CF * face_area)
    
    diffusion_coeff = visc_f * face_area * orthogonality / distance_CF
    
    # Add non-orthogonal correction if needed
    if orthogonality &lt; 0.95:  # threshold for orthogonality
        # Implement correction
        pass
    
    return diffusion_coeff
```


#### Enhancement Opportunities

- **Over-relaxed Approach**: For highly non-orthogonal meshes
- **Adaptive Correction**: Adjust non-orthogonal correction based on mesh quality
- **Tensor Diffusivity**: Support for anisotropic diffusion


### Convection Term

#### Mathematical Considerations

- **General Form**[^1_6]: ∮(ρv⊗v)·dS ≈ ∑_f (ρv)_f · S_f v_f
- **Face Value Schemes**:
    - 1st-order upwind: v_f = v_upwind
    - 2nd-order central: v_f = λv_C + (1-λ)v_F where λ is interpolation factor
    - QUICK: Quadratic interpolation using additional upstream point
- **Stability**: Higher-order schemes may require flux limiting


#### Implementation Aspects

```python
class ConvectionScheme:
    def compute_face_value(self, field, face_index, mass_flux):
        pass

class UpwindScheme(ConvectionScheme):
    def compute_face_value(self, field, face_index, mass_flux):
        cell_C, cell_F = self.mesh.get_face_neighbors(face_index)
        if mass_flux &gt;= 0:
            return field[cell_C]
        else:
            return field[cell_F]

class CentralScheme(ConvectionScheme):
    def compute_face_value(self, field, face_index, mass_flux):
        cell_C, cell_F = self.mesh.get_face_neighbors(face_index)
        lambda_f = self.compute_interpolation_factor(face_index)
        return lambda_f * field[cell_C] + (1 - lambda_f) * field[cell_F]
```


#### Enhancement Opportunities

- **TVD Schemes**: Implement high-resolution schemes with flux limiters
- **WENO Schemes**: For better accuracy in regions with sharp gradients
- **Hybrid Schemes**: Blend low and high-order schemes based on local flow conditions


### Gradient Computation

#### Mathematical Considerations

- **Green-Gauss Method**[^1_5]: ∇ϕ_C = (1/V_C) ∑_f ϕ_f S_f
- **Least-Squares Method**: Minimizes error in gradient approximation
- **Face Values**: Require accurate interpolation from cell centers
- **Boundary Contributions**: Must incorporate boundary conditions properly


#### Implementation Aspects

```python
class GradientCalculator:
    def calculate(self, field):
        pass

class GreenGaussGradient(GradientCalculator):
    def calculate(self, field):
        gradient = VectorField(self.mesh)
        for cell in self.mesh.cells:
            cell_volume = self.mesh.get_cell_volume(cell)
            grad_sum = Vector(0, 0, 0)
            
            for face in self.mesh.get_cell_faces(cell):
                face_value = self.interpolate_to_face(field, face)
                face_normal = self.mesh.get_face_normal(face)
                face_area = self.mesh.get_face_area(face)
                
                grad_sum += face_value * face_normal * face_area
            
            gradient[cell] = grad_sum / cell_volume
        
        return gradient
```


#### Enhancement Opportunities

- **Weighted Least-Squares**: Better accuracy on distorted meshes
- **Approximate Map Gradient**: More efficient on large parallel meshes
- **Limiters**: Prevent overshoots in gradient calculation near discontinuities


### Boundary Conditions

#### Mathematical Considerations

- **Wall Boundary**[^1_6]: No-slip condition (v_wall = 0 or specified velocity)
- **Inlet Boundary**[^1_6]: Specified velocity or pressure
- **Outlet Boundary**[^1_6]: Specified pressure or zero gradient
- **Pressure at Walls**[^1_6]: Requires special treatment since no explicit condition
- **Lid-Driven Cavity BCs**: Moving top wall, stationary other walls


#### Implementation Aspects

```python
class BoundaryCondition:
    def apply(self, field):
        pass

class DirichletBC(BoundaryCondition):
    def __init__(self, boundary_patch, value):
        self.boundary_patch = boundary_patch
        self.value = value
    
    def apply(self, field):
        for face in self.boundary_patch.faces:
            cell = self.mesh.get_face_cell(face)
            # Set boundary value or modify cell value based on boundary value
            field.set_boundary_value(face, self.value)

class LidDrivenCavityBCs:
    def __init__(self, mesh, lid_velocity):
        self.mesh = mesh
        self.lid_velocity = lid_velocity
        
    def setup(self):
        # Create boundary conditions for each patch
        self.velocity_bcs = [
            DirichletBC(self.mesh.get_patch("top"), VectorValue(self.lid_velocity, 0, 0)),
            DirichletBC(self.mesh.get_patch("bottom"), VectorValue(0, 0, 0)),
            DirichletBC(self.mesh.get_patch("left"), VectorValue(0, 0, 0)),
            DirichletBC(self.mesh.get_patch("right"), VectorValue(0, 0, 0))
        ]
        # Pressure BCs require special treatment in SIMPLE
```


#### Enhancement Opportunities

- **Ghost Cell Approach**: Implement ghost cells for complex boundary conditions
- **Higher-Order Wall Treatment**: Better approximation of near-wall gradients
- **Adaptive Wall Functions**: For high Reynolds number flows


## SIMPLE Algorithm Implementation

### Core Algorithm Structure

```python
class SIMPLE:
    def __init__(self, mesh, properties, settings):
        self.mesh = mesh
        self.properties = properties
        self.settings = settings
        
        # Create fields
        self.velocity = VectorField(mesh)
        self.pressure = ScalarField(mesh)
        self.pressure_correction = ScalarField(mesh)
        self.mass_flux = FaceField(mesh)
        
        # Create operators
        self.momentum_operator = MomentumOperator(mesh, properties, settings)
        self.pressure_correction_operator = PressureCorrectionOperator(mesh, properties, settings)
        
        # Create utilities
        self.gradient_calculator = GreenGaussGradient(mesh)
        self.face_interpolator = FaceInterpolator(mesh)
        
    def solve(self):
        # Initialize fields
        self.initialize()
        
        # Iteration loop
        for iteration in range(self.settings.max_iterations):
            # 1. Calculate face velocities and mass fluxes using Rhie-Chow
            self.calculate_mass_fluxes()
            
            # 2. Solve momentum equations
            self.momentum_operator.assemble_and_solve(self.velocity, self.pressure)
            
            # 3. Calculate and solve pressure correction equation
            self.calculate_pressure_correction()
            
            # 4. Correct pressure and velocity
            self.correct_fields()
            
            # 5. Check convergence
            if self.check_convergence():
                break
                
    def calculate_mass_fluxes(self):
        pressure_gradient = self.gradient_calculator.calculate(self.pressure)
        
        for face in self.mesh.internal_faces:
            cell_C, cell_F = self.mesh.get_face_neighbors(face)
            
            # Basic interpolation of velocity to face
            velocity_f = self.face_interpolator.interpolate(self.velocity, face)
            
            # Pressure term for Rhie-Chow interpolation
            pressure_diff = self.pressure[cell_F] - self.pressure[cell_C]
            d_CF = self.mesh.get_cell_vector(cell_C, cell_F)
            pressure_grad_interp = 0.5 * (pressure_gradient[cell_C] + pressure_gradient[cell_F])
            pressure_term = self.diffusion_coefficient(face) * (pressure_diff - dot(pressure_grad_interp, d_CF))
            
            # Compute mass flux
            face_velocity = velocity_f + pressure_term
            self.mass_flux[face] = dot(face_velocity, self.mesh.get_face_normal(face)) * self.properties.density
```


### Pressure Correction Equation

```python
class PressureCorrectionOperator:
    def assemble_and_solve(self, velocity, mass_flux):
        # 1. Create coefficient matrix and RHS vector
        A = self.create_matrix()
        b = self.create_vector()
        
        # 2. Assemble coefficients for internal faces
        for face in self.mesh.internal_faces:
            cell_C, cell_F = self.mesh.get_face_neighbors(face)
            
            # Diffusion coefficient
            D_f = self.calculate_diffusion_coefficient(face)
            
            # Add contribution to matrix
            A.add_value(cell_C, cell_C, D_f)
            A.add_value(cell_F, cell_F, D_f)
            A.add_value(cell_C, cell_F, -D_f)
            A.add_value(cell_F, cell_C, -D_f)
            
            # Add negative mass flux to RHS
            b.add_value(cell_C, -mass_flux[face])
            b.add_value(cell_F, mass_flux[face])
        
        # 3. Apply boundary conditions
        self.apply_boundary_conditions(A, b)
        
        # 4. Solve linear system
        pressure_correction = self.linear_solver.solve(A, b)
        
        return pressure_correction
```


### Field Correction

```python
def correct_fields(self):
    # 1. Correct pressure with under-relaxation
    alpha_p = self.settings.pressure_relaxation
    for cell in self.mesh.cells:
        self.pressure[cell] += alpha_p * self.pressure_correction[cell]
    
    # 2. Correct velocity with momentum equation
    alpha_v = self.settings.velocity_relaxation
    pressure_correction_gradient = self.gradient_calculator.calculate(self.pressure_correction)
    
    for cell in self.mesh.cells:
        velocity_correction = -self.momentum_operator.get_inverse_diagonal(cell) * pressure_correction_gradient[cell]
        self.velocity[cell] += alpha_v * velocity_correction
    
    # 3. Correct face mass fluxes
    for face in self.mesh.internal_faces:
        cell_C, cell_F = self.mesh.get_face_neighbors(face)
        D_f = self.calculate_diffusion_coefficient(face)
        mass_flux_correction = -D_f * (self.pressure_correction[cell_F] - self.pressure_correction[cell_C])
        self.mass_flux[face] += mass_flux_correction
```


## Testing and Validation

### Unit Tests

```python
def test_linear_gradient():
    # Create mesh and linear field
    mesh = create_uniform_mesh(10, 10)
    field = ScalarField(mesh)
    
    # Set linear variation
    for cell in mesh.cells:
        x, y, z = mesh.get_cell_center(cell)
        field[cell] = 2*x + 3*y
    
    # Calculate gradient
    gradient_calculator = GreenGaussGradient(mesh)
    gradient = gradient_calculator.calculate(field)
    
    # Check gradient is constant and correct
    for cell in mesh.cells:
        assert abs(gradient[cell].x - 2.0) &lt; 1e-10
        assert abs(gradient[cell].y - 3.0) &lt; 1e-10
        assert abs(gradient[cell].z - 0.0) &lt; 1e-10

def test_flux_continuity():
    # Create mesh and velocity field
    mesh = create_uniform_mesh(10, 10)
    velocity = VectorField(mesh)
    
    # Set uniform velocity
    for cell in mesh.cells:
        velocity[cell] = VectorValue(1.0, 0.0, 0.0)
    
    # Calculate face fluxes
    face_interpolator = FaceInterpolator(mesh)
    mass_flux = FaceField(mesh)
    
    for face in mesh.internal_faces:
        velocity_f = face_interpolator.interpolate(velocity, face)
        face_normal = mesh.get_face_normal(face)
        face_area = mesh.get_face_area(face)
        mass_flux[face] = 1.0 * dot(velocity_f, face_normal) * face_area
    
    # Check flux conservation for each cell
    for cell in mesh.cells:
        net_flux = 0.0
        for face in mesh.get_cell_faces(cell):
            if face in mesh.internal_faces:
                cell_C, cell_F = mesh.get_face_neighbors(face)
                sign = 1.0 if cell == cell_C else -1.0
                net_flux += sign * mass_flux[face]
            # Handle boundary faces
        
        assert abs(net_flux) &lt; 1e-10
```


### Full Solver Validation

```python
def validate_lid_driven_cavity(reynolds_number):
    # 1. Setup case
    mesh = create_uniform_mesh(100, 100)
    properties = FluidProperties(density=1.0, viscosity=1.0/reynolds_number)
    settings = SolverSettings(max_iterations=1000, tolerance=1e-6)
    
    # 2. Create and run solver
    solver = SIMPLE(mesh, properties, settings)
    ldc_bcs = LidDrivenCavityBCs(mesh, lid_velocity=1.0)
    ldc_bcs.setup()
    
    solver.solve()
    
    # 3. Extract centerline velocities
    centerline_u = []
    centerline_v = []
    
    for cell in mesh.cells:
        x, y, z = mesh.get_cell_center(cell)
        if abs(x - 0.5) &lt; 1e-10:
            centerline_v.append((y, solver.velocity[cell].y))
        if abs(y - 0.5) &lt; 1e-10:
            centerline_u.append((x, solver.velocity[cell].x))
    
    # 4. Compare with Ghia et al. data
    ghia_data = load_ghia_data(reynolds_number)
    compare_with_reference(centerline_u, ghia_data['u'])
    compare_with_reference(centerline_v, ghia_data['v'])
```


## Extensibility for Future Development

### PISO Algorithm Extension

```python
class PISO(SIMPLE):
    def __init__(self, mesh, properties, settings):
        super().__init__(mesh, properties, settings)
        self.n_correctors = settings.n_correctors
    
    def solve(self):
        # Initialize fields
        self.initialize()
        
        # Iteration loop for transient solution
        for time_step in range(self.settings.n_time_steps):
            # Predictor step (same as SIMPLE)
            self.calculate_mass_fluxes()
            self.momentum_operator.assemble_and_solve(self.velocity, self.pressure)
            
            # Multiple corrector steps
            for corrector in range(self.n_correctors):
                # Calculate and solve pressure correction
                self.calculate_pressure_correction()
                
                # Correct pressure
                for cell in self.mesh.cells:
                    self.pressure[cell] += self.pressure_correction[cell]  # No under-relaxation in PISO
                
                # Correct velocity and fluxes
                self.correct_fields(with_relaxation=False)
            
            # Advance time
            self.time += self.settings.time_step
```


### Matrix-Free Implementation

```python
class MatrixFreeMomentumOperator:
    def __init__(self, mesh, properties, settings):
        self.mesh = mesh
        self.properties = properties
        self.settings = settings
    
    def mult(self, A, x, y):
        """Implement matrix-vector product for momentum operator using PETSc Python matrix type."""
        # Convert PETSc Vec to numpy array
        x_array = x.getArray()
        y_array = y.getArray()
        
        # Reshape to vector field format
        velocity = self.reshape_to_vector_field(x_array)
        result = VectorField(self.mesh)
        
        # Apply operator
        for cell in self.mesh.cells:
            # Diagonal term
            result[cell] = self.get_diagonal(cell) * velocity[cell]
            
            # Off-diagonal terms (neighbors)
            for neighbor in self.mesh.get_cell_neighbors(cell):
                face = self.mesh.get_face_between(cell, neighbor)
                coeff = self.calculate_coefficient(cell, neighbor, face)
                result[cell] += coeff * velocity[neighbor]
        
        # Copy result back to PETSc Vec
        result_array = self.reshape_to_array(result)
        y.setArray(result_array)
```


### Mesh Agnostic Design

```python
class MeshAgnosticOperator:
    def __init__(self, mesh):
        self.mesh = mesh
        
    def compute_cell_gradient(self, field, cell):
        """Compute gradient at cell center using face values."""
        gradient = Vector(0, 0, 0)
        cell_volume = self.mesh.get_cell_volume(cell)
        
        for face in self.mesh.get_cell_faces(cell):
            face_value = self.get_face_value(field, face)
            face_normal = self.mesh.get_face_normal(face)
            face_area = self.mesh.get_face_area(face)
            
            gradient += face_value * face_normal * face_area
        
        return gradient / cell_volume
    
    def get_face_value(self, field, face):
        """Get field value at face, handling both internal and boundary faces."""
        if face in self.mesh.internal_faces:
            cell_C, cell_F = self.mesh.get_face_neighbors(face)
            lambda_f = self.compute_interpolation_factor(face)
            return lambda_f * field[cell_C] + (1 - lambda_f) * field[cell_F]
        else:
            # Handle boundary face
            cell = self.mesh.get_face_cell(face)
            return self.apply_boundary_condition(field, face, cell)
```


## Configuration through YAML

```yaml
solver:
  type: "SIMPLE"
  max_iterations: 5000
  convergence_tolerance: 1e-6
  relaxation:
    pressure: 0.3
    velocity: 0.7

discretization:
  convection_scheme: "upwind"  # Options: upwind, central, quick
  gradient_method: "green_gauss"  # Options: green_gauss, least_squares
  non_orthogonal_correction: true

physical_properties:
  density: 1.0
  viscosity: 0.01  # Re = 100 for cavity with length 1 and lid velocity 1

boundary_conditions:
  top:
    type: "velocity"
    value: [1.0, 0.0, 0.0]
  bottom:
    type: "no_slip"
  left:
    type: "no_slip"
  right:
    type: "no_slip"

output:
  write_interval: 100
  vtk_output: true
  convergence_history: true
```


## Conclusion and Next Steps

The presented design follows Moukalled's guidelines for a mesh-agnostic implementation of the SIMPLE algorithm for lid-driven cavity flow. The modular, object-oriented approach with clear separation of concerns enables easy extension to PISO and other algorithms. Key aspects include:

1. Rhie-Chow interpolation for pressure-velocity coupling on collocated grids
2. Face-based data structures for mesh-agnostic operations
3. Operator abstractions for clean algorithm implementation
4. PETSc integration for scalable linear system solving
5. Comprehensive testing strategy from unit tests to full validation

Next development steps should focus on:

- Implementing higher-order discretization schemes
- Enhancing boundary condition handling
- Adding support for non-orthogonal meshes
- Extending to transient cases with PISO algorithm
- Optimizing parallel performance using domain decomposition

This design provides a solid foundation for a modular, extensible CFD solver capable of handling the lid-driven cavity benchmark and extending to more complex flow problems.

<div style="text-align: center">⁂</div>

[^1_1]: FVM-Mouka.pdf

[^1_2]: https://www.youtube.com/watch?v=pJbhmB3xrAs

[^1_3]: https://scicomp.stackexchange.com/questions/26487/rhie-chow-interpolation-on-pde-level

[^1_4]: https://clima.github.io/OceananigansDocumentation/v0.40.0/verification/lid_driven_cavity/

[^1_5]: https://tuprints.ulb.tu-darmstadt.de/7043/1/tomislav-maric-phd-thesis.pdf

[^1_6]: https://gidropraktikum.narod.ru/Moukalled-et-al-FVM-OpenFOAM-Matlab.pdf

[^1_7]: https://petsc.org/release/petsc4py/petsc_python_types.html

[^1_8]: https://www.femto.eu/femto_story/lid-driven-cavity-flow/

[^1_9]: https://hammer.purdue.edu/articles/thesis/CODE_AND_MESH_AGNOSTIC_NON-LINEAR_MULTISCALE_ANALYSIS_AND_MACHINE_LEARNING_MODELS_FOR_DESIGN_AND_ANALYSIS_OF_HETEROGENEOUSLY_INTEGRATED_ELECTRONIC_PACKAGES/28049828

[^1_10]: http://www.msaidi.ir/upload/Ghia1982.pdf

[^1_11]: https://research.manchester.ac.uk/files/38921872/fld4240.pdf

[^1_12]: https://www.youtube.com/watch?v=VRKSPnvJyiU

[^1_13]: https://scholar.google.com/citations?user=0sBOmHYAAAAJ

[^1_14]: https://arc.aiaa.org/doi/10.2514/1.J058853

[^1_15]: https://www.cfd-online.com/Forums/main/181670-implementing-simple-unstructured-collocated-grid.html

[^1_16]: https://www.academia.edu/63579501/A_finite_volume_algorithm_for_all_speed_flows

[^1_17]: https://github.com/deepmorzaria/Lid-Driven-Cavity-Collocated-Grid

[^1_18]: https://www.academia.edu/20236865/THE_FINITE_VOLUME_METHOD_IN_CFD_by_F_Moukalled_L_Mangani_M_Darwish

[^1_19]: https://nht.xjtu.edu.cn/paper/en/2005204new.pdf

[^1_20]: https://www.aub.edu.lb/msfea/research/Documents/CFD-P15.pdf

[^1_21]: https://petsc.org/release/manual/getting_started/

[^1_22]: https://www.reddit.com/r/CFD/comments/17q1t18/staggered_vs_collocated_grids/

[^1_23]: https://citeseerx.ist.psu.edu/document?repid=rep1\&type=pdf\&doi=f1ac4e75f9688f3514ef6dda0eef4cb23c0fcb12

[^1_24]: https://www.cfd-online.com/Wiki/Lid-driven_cavity_problem

[^1_25]: https://fenix.tecnico.ulisboa.pt/downloadFile/2815144904097976/Thesis.pdf

[^1_26]: https://scicomp.stackexchange.com/questions/2137/efficiency-of-using-petsc4py-vs-c-c-fortran

[^1_27]: https://www.atlantis-press.com/article/23165.pdf

[^1_28]: https://www.youtube.com/watch?v=yqZ59Xn_aF8

[^1_29]: https://www.fifty2.eu/innovation/lid-driven-cavity-2d-in-preonlab

[^1_30]: https://mooseframework.inl.gov/workshop/

[^1_31]: https://petsc.org

[^1_32]: https://www.sciencedirect.com/science/article/pii/S2090447922001654

[^1_33]: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2022.985440/full

[^1_34]: https://scispace.com/authors/marwan-darwish-5956capdr6

[^1_35]: https://www.mcs.anl.gov/petsc/documentation/tutorials/INL05/tutorial.pdf

[^1_36]: https://vbn.aau.dk/files/16650398/SIMPLE_Pressure_Correction_Algorithm

[^1_37]: https://gist.github.com/ivan-pi/3e9326d18a366ffe6a8e5bfda6353219

[^1_38]: https://dl.acm.org/doi/10.1145/3474124.3474152

[^1_39]: https://www.sciencedirect.com/science/article/pii/S0307904X23003797

[^1_40]: https://orbit.dtu.dk/files/267167240/Smit2021_Article_TopologyOptimizationUsingPETSc_1_.pdf

[^1_41]: https://orbit.dtu.dk/files/3190785/2008_138.pdf

[^1_42]: https://gist.github.com/ivan-pi/caa6c6737d36a9140fbcf2ea59c78b3c

[^1_43]: https://www.umr-cnrm.fr/accord/IMG/pdf/pmap_fvm_accord_asm_2024.pdf

[^1_44]: https://arc.aiaa.org/doi/10.2514/1.J055581

[^1_45]: https://courses.grainger.illinois.edu/cs554/fa2011/notes/petscpde.pdf

[^1_46]: https://numba.pydata.org/numba-doc/dev/user/5minguide.html

[^1_47]: https://www.sciencedirect.com/science/article/abs/pii/S0098135498002701

[^1_48]: https://www.aub.edu.lb/msfea/research/Documents/CFD-P38.pdf

[^1_49]: http://websites.umich.edu/~mdolaboratory/pdf/He2020b.pdf

[^1_50]: https://github.com/deepmorzaria/Lid-Driven-Cavity-Collocated-Grid

[^1_51]: https://www.grc.nasa.gov/www/wind/valid/cavity/cavity.html

[^1_52]: https://scicomp.stackexchange.com/questions/26487/rhie-chow-interpolation-on-pde-level

[^1_53]: https://docs.mfem.org/html/examples_2petsc_2ex1p_8cpp_source.html

[^1_54]: https://numba.pydata.org

[^1_55]: https://www.sciencedirect.com/science/article/abs/pii/S004579301000349X

[^1_56]: https://scholar.uwindsor.ca/cgi/viewcontent.cgi?article=5517\&context=etd

[^1_57]: https://arxiv.org/html/2408.12171v1

[^1_58]: https://meetingorganizer.copernicus.org/EGU25/session/53470

[^1_59]: https://arxiv.org/abs/2210.05495

[^1_60]: https://scicomp.stackexchange.com/questions/25927/simple-methods-for-solving-2d-steady-incompressible-flow

[^1_61]: https://gidropraktikum.narod.ru/Moukalled-et-al-FVM-OpenFOAM-Matlab.pdf

