
To verify the **convection** and **diffusion discretization functions** in isolation *before matrix assembly*, use targeted MMS tests that validate individual term implementations. Below is a step-by-step verification strategy:

---

## **1. Convection Term Verification**

### **Test Case Setup**

- **Manufactured Solution**:
Choose a scalar field (e.g., temperature) with non-uniform velocity:

$$
\phi = \sin(\pi x)\cos(\pi y), \quad \mathbf{u} = [2y(1-x^2), -2x(1-y^2)]
$$
- **Governing Equation**:
Convection-only:

$$
\nabla \cdot (\rho \mathbf{u} \phi) = S_{\text{conv}}
$$

Derive \$ S_{conv} \$ symbolically.


### **Verification Steps**

1. **Compute Exact Convective Flux**
Use symbolic math to calculate face fluxes (e.g., with SymPy):

```python
from sympy import symbols, diff, sin, cos
x, y = symbols('x y')
phi = sin(pi*x)*cos(pi*y)
u = 2*y*(1 - x**2)
v = -2*x*(1 - y**2)

# Convective flux F = (u*phi)_x + (v*phi)_y
F_exact = diff(u*phi, x) + diff(v*phi, y)
```

2. **Test Convection Scheme Implementation**
For each scheme (upwind, central, QUICK):

```python
def test_upwind_convection():
    phi_face = upwind_interpolation(phi_cell, u_face)  # Your function
    F_numerical = sum(u_face * phi_face * face_area)
    assert abs(F_numerical - F_exact) &lt; 1e-4
```

3. **Boundary-Specific Checks**
    - Inlet/outlet: Verify flux conservation \$ \sum F_{in} = \sum F_{out} \$.
    - Walls: Confirm zero advective flux where \$ \mathbf{u} \cdot \mathbf{n} = 0 \$.

---

## **2. Diffusion Term Verification**

### **Test Case Setup**

- **Manufactured Solution**:
Steep scalar gradient to stress diffusion:

$$
\phi = e^{-10(x^2 + y^2)}, \quad \Gamma = 1.0 \quad (\text{diffusion coefficient})
$$
- **Governing Equation**:
Diffusion-only:

$$
\nabla \cdot (\Gamma \nabla \phi) = S_{\text{diff}}
$$


### **Verification Steps**

1. **Compute Exact Diffusive Flux**
Derive \$ \nabla \phi \$ and \$ S_{diff} \$ analytically:

```python
phi = sp.exp(-10*(x**2 + y**2))
grad_phi_x = diff(phi, x)
grad_phi_y = diff(phi, y)
S_diff = diff(Gamma*grad_phi_x, x) + diff(Gamma*grad_phi_y, y)
```

2. **Test Diffusion Scheme Implementation**
For orthogonal/non-orthogonal meshes:

```python
def test_gauss_diffusion():
    grad_phi = gauss_gradient(phi_cell, mesh)  # Your function
    flux = Gamma * grad_phi.dot(face_normal)
    assert np.allclose(flux, grad_phi_exact.dot(face_normal), rtol=1e-3)
```

3. **Non-Orthogonal Correction Check**
On skewed meshes, verify the over-relaxed approach from [Wolf Dynamics](https://www.wolfdynamics.com/wiki/fvm_crash_intro.pdf):

$$
\nabla \phi_{\text{face}} = \underbrace{\frac{\phi_N - \phi_P}{|\mathbf{d}|}}_{\text{orthogonal}} + \underbrace{k (\nabla \phi - (\nabla \phi \cdot \mathbf{e})\mathbf{e})}_{\text{non-orthogonal}}
$$

Test with blending factor \$ k = 0.33 \$ and \$ k = 1.0 \$.

---

## **3. Isolated Term Validation Workflow**

### **Pre-Matrix Checks**

| **Function** | **Test** | **Pass Criteria** |
| :-- | :-- | :-- |
| `computeConvectionFlux` | Compare against exact flux for 5x5 mesh | L2 error < 1e-4, order ≈ scheme accuracy |
| `computeDiffusionFlux` | Check gradient symmetry on skewed hex mesh | Max residual < 1e-6 |
| `applyBoundaryConditions` | Enforce Dirichlet/Neumann on irregular cells | Boundary flux error < 1e-9 |

### **Automation via pytest**

```python
# Example test for central differencing diffusion
def test_diffusion_central():
    mesh = generate_quad_mesh(16, 16, skewed=True)
    phi = manufactured_solution(mesh)
    flux_numerical = compute_diffusion_flux(phi, mesh, scheme='central')
    flux_exact = precomputed_flux(mesh)  # From SymPy
    assert np.linalg.norm(flux_numerical - flux_exact) &lt; 1e-5
```


---

## **4. Special Cases to Cover**

### **Convection Tests**

- **High Peclet Number**: \$ Pe = 1000 \$ to check stability of upwind schemes.
- **Variable Density**: \$ \rho = 1 + \sin(x) \$ to validate \$ \rho \mathbf{u} \$ handling.


### **Diffusion Tests**

- **Anisotropic Diffusion**: \$ \Gamma = $$
\begin{bmatrix} 1 &amp; 0.5 \\ 0.5 &amp; 1 \end{bmatrix}
$$ \$.
- **Non-Linear Coefficients**: \$ \Gamma = \phi^2 \$ with automatic differentiation.

---

## **Debugging Failed Tests**

| **Symptom** | **Likely Culprit** | **Debug Action** |
| :-- | :-- | :-- |
| First-order error decay | Missing higher-order reconstruction (MUSCL) | Check gradient limiter implementation. |
| Asymmetric flux errors | Incorrect face normal direction | Visualize face normals in ParaView. |
| Boundary spikes | Ghost cell values not mirrored | Print boundary cell/face indices. |


---

By isolating convection/diffusion functions and comparing against manufactured solutions *before* matrix assembly, you can localize errors to specific numerical kernels. This aligns with verification practices in [SU2](https://su2code.github.io/docs_v7/Convective-Schemes/) and OpenFOAM’s `gaussLaplacianSchemes`.

<div style="text-align: center">⁂</div>

[^11_1]: https://orbit.dtu.dk/files/262739936/FinalReport_JesperRolandKj_rgaardQwist.pdf

[^11_2]: https://su2code.github.io/docs_v7/Convective-Schemes/

[^11_3]: https://research.tue.nl/files/1436139/458484.pdf

[^11_4]: https://www.aoe.vt.edu/content/dam/aoe_vt_edu/people/faculty/cjroy/Publications-Articles/IJNMF.final.2004.pdf

[^11_5]: https://www.wolfdynamics.com/wiki/fvm_crash_intro.pdf

[^11_6]: https://ntrs.nasa.gov/api/citations/20040084089/downloads/20040084089.pdf

[^11_7]: https://www.osti.gov/servlets/purl/805879

[^11_8]: https://www.eccomas2016.org/proceedings/pdf/5839.pdf

[^11_9]: https://www.esaim-m2an.org/articles/m2an/pdf/2009/06/m2an0777.pdf

[^11_10]: https://mfix.netl.doe.gov/doc/vvuq-manual/main/html/mms/mms-ex-02.html

[^11_11]: https://www.osti.gov/servlets/purl/793406

[^11_12]: https://www.tfd.chalmers.se/~hani/kurser/OS_CFD_2021/IlyaMorev/Report_Morev.pdf

[^11_13]: https://www.wias-berlin.de/people/john/BETREUUNG/master_khrais.pdf

[^11_14]: https://arxiv.org/pdf/1611.05795.pdf

[^11_15]: https://doras.dcu.ie/20241/2/Numerical_modelling_of_convection-reaction-diffusion_problems_using_electrical_analogues.pdf

[^11_16]: https://dragonfly.tam.cornell.edu/teaching/mae5230-cfd-intro-notes.pdf

[^11_17]: https://wwwold.mathematik.tu-dortmund.de/papers/KuzminMoellerTurek2003.pdf

[^11_18]: https://scicomp.stackexchange.com/questions/36903/diffusivity-matrix-assembly-in-nonlinear-finite-element-analysis

[^11_19]: https://cimec.org.ar/ojs/index.php/mc/article/download/3854/3775

[^11_20]: https://www.sciencedirect.com/topics/engineering/diffusion-matrix

