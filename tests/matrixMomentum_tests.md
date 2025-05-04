
Yes! Once the individual convection/diffusion functions pass isolated tests, the next critical step is verifying **matrix assembly with boundary conditions (BCs)**. This ensures that contributions from interior terms, BCs, and source terms are correctly combined into the global system. Here's a detailed verification strategy:

---

## **Matrix Assembly Verification Workflow**

### **1. Manufactured Solution Design**

Choose a solution that stresses BC handling and matrix coefficients:

- **Velocity**: \$ u = x^2 + 2y^2, \quad v = -2xy \$ (divergence-free)
- **Pressure**: \$ p = \sin(\pi x)\cos(\pi y) \$
- **Viscosity**: \$ \mu = 1.0 \$
- **BC Types**:
    - **Dirichlet**: \$ u=0 \$ at left wall, \$ u_{inlet} = u_{exact} \$
    - **Neumann**: \$ \frac{\partial u}{\partial n} = 0 \$ at outlet
    - **Mixed**: Slip condition at top wall

---

### **2. Key Matrix Checks**

#### **Coefficient Validation**

For a sample cell (e.g., cell 10 in a 5x5 grid):

- **Expected Momentum Matrix (A)**:

$$
A_{P} = \sum_{\text{faces}} \left(\frac{\mu}{\Delta x} + \rho u_f \right), \quad A_{\text{neighbor}} = -\frac{\mu}{\Delta x}
$$
- **Test**:

```python
# After matrix assembly
cell_id = 10
A = momentum_matrix[cell_id]
expected_Ap = 4.2  # Precomputed from MMS
assert abs(A.diagonal()[cell_id] - expected_Ap) < 1e-4
assert abs(A[cell_id, east_cell] - (-1.0)) < 1e-4  # Diffusion term
```


#### **Boundary Term Integration**

- **Dirichlet BCs** should appear as:
    - Diagonal coefficient = 1.0
    - Source term = exact BC value

```python
# For Dirichlet cell 0 (left wall)
assert A[0, 0] == 1.0
assert b[0] == u_exact(x=0, y=yc[0])
```

- **Neumann BCs** modify the source term:

$$
b_P += \mu \frac{\partial u}{\partial n} \cdot A_{\text{face}}
$$

```python
# For Neumann face at outlet
outlet_face = mesh.boundary_faces['outlet']
assert abs(b[owner_cell[outlet_face]] - mu * grad_u * area) < 1e-6
```


---

### **3. Full System Residual Test**

After assembling \$ A\mathbf{u} = \mathbf{b} \$:

1. **Inject Exact Solution**: Set \$ \mathbf{u} = \mathbf{u}_{exact} \$.
2. **Compute Residual**:

$$
\mathbf{r} = \mathbf{b} - A\mathbf{u}_{\text{exact}}
$$
3. **Verify Residual Norm**:

```python
residual_norm = np.linalg.norm(r)
assert residual_norm < 1e-10, "Matrix/source term mismatch"
```


---

### **4. Automated Test Cases**

#### **Test 1: Dirichlet Dominated Flow**

```python
def test_dirichlet_matrix_assembly():
    mesh = create_rectangular_mesh(8, 8)
    apply_bcs(mesh, {'left': 'no-slip', 'right': 'outflow'})
    A, b = assemble_momentum_matrix(mesh, ...)
    
    # Check diagonal dominance for all rows
    for i in range(A.shape[0]):
        assert A[i, i] >= np.sum(np.abs(A[i, :])) - 1e-9
    
    # Check residual with exact solution
    u_exact = compute_exact_solution(mesh)
    r = b - A @ u_exact
    assert np.linalg.norm(r) < 1e-8
```


#### **Test 2: Neumann BC Flux Conservation**

```python
def test_neumann_flux_balance():
    mesh = generate_skewed_mesh(10, 10)
    apply_bcs(mesh, {'top': 'symmetry'})
    A, b = assemble_momentum_matrix(...)
    
    # Sum all source terms (should balance Neumann fluxes)
    total_source = np.sum(b)
    assert abs(total_source) < 1e-12
```


---

## **5. Non-Trivial Mesh Checks**

### **Skewed Quadrilateral Mesh**

- **Test**: Verify non-orthogonal correction terms in diffusion matrix.
- **Check**:
For a cell with skewness angle \$ \theta \$:

$$
A_{\text{non-orth}} = A_{\text{orth}} + \mu \frac{\cos\theta}{\Delta x}
$$

```python
skewness = compute_skewness(mesh, cell_id)
expected_A_nonorth = A_orth + mu * np.cos(skewness) / dx
assert np.isclose(A[cell_id, cell_id], expected_A_nonorth, rtol=1e-3)
```


### **Hybrid Polyhedral Mesh**

- **Test**: Ensure face interpolation weights sum to 1.
- **Check**:

```python
for face in mesh.faces:
    if face.is_polyhedral:
        sum_weights = sum(face.interpolation_weights)
        assert abs(sum_weights - 1.0) < 1e-9
```


---

## **6. Common Bugs \& Fixes**

| **Bug Symptom** | **Likely Cause** | **Debugging Action** |
| :-- | :-- | :-- |
| Residual increases with iterations | BCs overwritten during matrix assembly | Print BC rows before/after assembly |
| Asymmetric matrix | Missing Rhie-Chow terms in collocated grid | Check off-diagonal sign conventions |
| Zero diagonal entries | Uninitialized boundary rows | Visualize matrix sparsity pattern |


---

## **CI/CD Integration**

Add to your `.github/workflows/tests.yml`:

```yaml
- name: Matrix Assembly Tests
  run: |
    pytest tests/matrix_assembly.py -v
  env:
    OMP_NUM_THREADS: 1  # Avoid OpenMP race conditions
```


---

By systematically testing matrix assembly *before* solving the linear system, you catch discretization and BC errors early. This approach is inspired by [deal.II’s step-7](https://dealii.org/developer/doxygen/deal.II/step_7.html) and OpenFOAM’s `fvMatrix` checks.

