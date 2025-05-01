# SIMPLE Solver: Incremental Implementation Checklist

## Phase 0: Mesh Geometry
- [ ] Validate cell volumes, face areas > 0
- [ ] Ensure consistent face normals and topology
- [ ] Boundary patch classification (e.g., lid)

## Phase 1: Fields
- [ ] Allocate cell-centered fields: `u`, `v`, `p`
- [ ] Initialize internal and boundary values
- [ ] Assert all values are finite (no NaNs, Infs)

## Phase 2: Diffusion Operator
- [ ] Central differencing diffusion stencil
- [ ] Check matrix symmetry and diagonal dominance
- [ ] Solve Laplace problem for benchmark (manufactured sol.)

## Phase 3: Momentum Assembly (no pressure yet)
- [ ] Upwind convection (Power Law)
- [ ] Compute `aP`, `aN`, `Su` from face loop
- [ ] Momentum residuals decrease on stand-alone solve

## Phase 4: Pressure Correction Equation
- [ ] Assemble p'-equation from continuity + Rhie–Chow
- [ ] Correct face fluxes using stored `d_u`, `d_v`
- [ ] p' solution improves mass balance

## Phase 5: Velocity Correction
- [ ] Compute ∇p' and apply velocity correction
- [ ] Apply velocity BCs again after correction
- [ ] Mass conservation restored

## Phase 6: SIMPLE Loop
- [ ] Apply under-relaxation correctly (not in residuals)
- [ ] Residuals: Res_u, Res_v, Res_mass decrease
- [ ] Lid-driven cavity converges to known vortex pattern

## Validation
- [ ] u_center = 0.1565 ± 1e-3 (Ghia et al.)
- [ ] All residuals < 1e-6 within 1000 iterations


## Testing: 


```text

tests/
├── conftest.py                    # Mesh loader fixture for parameterized testing
├── test_mesh.py                   # Mesh geometry and connectivity
│   ├── test_face_geometry()
│   └── test_patch_classification()
├── test_fields.py                 # Field initialization and BCs
│   ├── test_cell_centered_storage()
│   └── test_boundary_conditions_initial()
├── test_discretization.py         # Power law, central differencing, gradient
│   ├── test_central_diffusion_symmetry()
│   ├── test_power_law_behavior()
│   ├── test_gradient_gauss_accuracy()
│   └── test_interpolation_linear_vs_mid()
├── test_matrix_assembly.py        # Momentum matrix assembly
│   ├── test_momentum_flux_assembly()
│   ├── test_diagonal_dominance()
│   └── test_assembled_stencils_face_based()
├── test_rhie_chow.py              # Rhie–Chow interpolation
│   ├── test_rhie_chow_flux_structure()
│   └── test_checkerboard_suppression()
├── test_pressure_correction.py    # p' equation and flux correction
│   ├── test_p_prime_matrix()
│   └── test_p_correction_application()
├── test_velocity_correction.py    # Correction and BC reapplication
│   ├── test_corrected_velocity()
│   └── test_bc_applied_after_correction()
├── test_residuals.py              # Physical convergence
│   ├── test_physical_residuals_unrelaxed()
│   ├── test_mass_imbalance_residual()
│   └── test_convergence_threshold_met()
├── test_under_relaxation.py       # Under-relaxation logic
│   ├── test_post_solve_relaxation()
│   └── test_relaxation_bounds_valid()
├── test_utils.py                  # Config, preprocessing helpers
│   ├── test_config_loader()
│   └── test_mesh_preprocessing()
└── test_synthetic_cases.py        # Integration and benchmark tests
    ├── test_minimal_cavity_stability()
    └── test_ghia_center_velocity_match()

```