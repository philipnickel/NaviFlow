# SIMPLE Solver Math Considerations (Moukalled-based)

This document outlines the mathematical considerations and best practices for implementing the SIMPLE algorithm on a collocated mesh, following the treatment in Moukalled et al.'s book.

---

## Collocated Grid Structure
- Store all variables (`u`, `v`, `p`) at **cell centers**.
- Use **Rhie–Chow interpolation** to prevent checkerboarding in pressure field.

---

## Momentum Equations
- Discretize using the **Power Law** scheme (stable up to Re ~10,000).
- Diffusion: central differencing with linear interpolation at face centers.
- Face-based assembly:
  - Use geometric data: face area vectors `Sf`, distance vectors `d`, interpolation weights `fx`
  - Loop over internal faces, compute contributions to aP, aN, and source term

### Notes:
- Compute face flux `F_f` using upwind-interpolated velocities
- Apply **under-relaxation** only after solving, not during residual checks
- Maintain diagonal dominance (Practice B for BCs)

---

## Pressure Correction Equation
- Derived from mass conservation + Rhie-Chow corrected face fluxes
- Solve for pressure correction `p'`:
  ```
  A_p' * p' = b_p'
  ```
- Flux calculation:
  ```
  u_f = interpolated(u) - d_f * grad(p')
  ```

### Notes:
- Use Rhie-Chow interpolation at all internal faces
- `d_u`, `d_v` are stored and reused (inverse of momentum diagonal coefficients)
- Pressure correction is applied as:
  ```
  p := p_old + alpha_p * p'
  ```

---

## Velocity Correction
- Correct intermediate velocities `u*`, `v*` using pressure correction:
  ```
  u := u* - d_u * grad(p')
  v := v* - d_v * grad(p')
  ```

### Notes:
- Face-based gradient computation
- Apply boundary conditions again if velocity field is modified near walls

---

## Boundary Conditions
- Use **Practice B** from Moukalled:
  - Modify matrix coefficients and RHS instead of overriding solution
  - Maintain matrix consistency and diagonal dominance

---

## Residuals and Convergence
- Compute **physical residuals** from **unrelaxed** momentum equations
- Monitor normalized residuals:
  ```
  Res_u = ||A*u - b|| / ||b||
  ```
- Typical convergence threshold: `1e-6` or better

### Notes:
- Do not check convergence on relaxed solution fields
- Track u-residual, v-residual, and mass imbalance separately

---

## Under-Relaxation
- Apply under-relaxation only to updated fields:
  ```
  phi := phi_old + alpha * (phi_new - phi_old)
  ```
- Typical values:
  - `alpha_u`, `alpha_v`: 0.3–0.8
  - `alpha_p`: 0.1–0.5

---

## Summary
- Follow Moukalled’s face-based FVM strategy with Rhie–Chow interpolation
- Apply Practice B for BCs
- Assemble coefficients and fluxes at faces using mesh geometry
- Separate solve and relaxation steps
- Check convergence based on physical (unrelaxed) residuals

