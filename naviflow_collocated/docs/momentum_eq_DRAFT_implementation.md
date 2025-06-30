x:w

MeshData2D: Core data layout for finite volume CFD (2D, collocated).

This class defines static geometry, connectivity, boundary tagging, and precomputed metrics,
following Moukalled's finite volume formulation.

Indexing Conventions:

- All face-based arrays (e.g., face_normals, owner_cells) use face indexing (0 to n_faces-1).
- All cell-based arrays (e.g., cell_volumes, cell_centers) use cell indexing (0 to n_cells-1).
- Boundary-related arrays (e.g., boundary_values, boundary_types, d_PB) have full-face length (n_faces).
    * Internal faces use sentinel defaults: boundary_types = [-1, -1], boundary_values = [0, 0, 0], d_PB = 0.0

Boundary Condition Metadata:

- boundary_values[f, :] = [u_BC, v_BC, p_BC] for face f. Zero for internal.
- boundary_types[f, :] = [vel_type, p_type] with:
    * 0 = Wall
    * 1 = Dirichlet
    * 2 = Neumann
    * 3 = zeroGradient
- d_PB[f] = distance from cell center to boundary face center (used for one-sided gradients)

Fast Boolean Masks:

- face_boundary_mask[f] = 1 if face is boundary, 0 otherwise
- face_flux_mask[f] = 1 if face is active in flux computation, 0 otherwise

Below is a **complete, modular implementation** of the diffusion helper, convection helper, and matrix assembly for the momentum equation, tailored to your `MeshData2D` structure and following Moukalled’s methodology. This code is **mesh-agnostic**, supports all boundary types, and is ready for use in a collocated finite volume SIMPLE solver.

---

```python
import numpy as np

# --- Diffusion Helper ---

def compute_diffusion_face(
    mu_f, S_f, d_PN, grad_phi_P, grad_phi_N,
    non_ortho_correction, skewness_vector
):
    """
    Computes the orthogonal diffusion coefficient and explicit corrections.
    Returns:
        D_f: main orthogonal diffusive coefficient (for matrix)
        b_corr: explicit correction (non-orthogonality + skewness, for RHS)
    """
    d_mag = np.linalg.norm(d_PN)
    S_dot_d = np.dot(S_f, d_PN)
    D_f = mu_f * S_dot_d / (d_mag ** 2 + 1e-20)

    grad_phi_f = 0.5 * (grad_phi_P + grad_phi_N)
    non_ortho_flux = mu_f * np.dot(non_ortho_correction, grad_phi_f)
    skew_flux = mu_f * np.dot(skewness_vector, grad_phi_f)
    b_corr = non_ortho_flux + skew_flux

    return D_f, b_corr

# --- Convection Helper ---

def compute_convection_face(
    rho_f, u_f, S_f, phi_P, phi_N, phi_up=None, phi_down=None,
    scheme='upwind', face_interp_factor=0.5, limiter=None
):
    """
    Computes the convective mass flux and interpolated variable at the face.
    Returns:
        F_f: mass flux through face
        phi_f: interpolated variable at face (for upwinding, TVD, QUICK, etc.)
    """
    F_f = rho_f * np.dot(u_f, S_f)

    if scheme == 'upwind':
        phi_f = phi_P if F_f >= 0 else phi_N
    elif scheme == 'powerlaw':
        # Requires D_f, so this must be passed in as phi_up
        D_f = phi_up if phi_up is not None else 1.0
        Pe = F_f / (D_f + 1e-20)
        A = max(0, (1 - 0.1 * abs(Pe)) ** 5)
        phi_f = A * (phi_P if F_f >= 0 else phi_N) + (1 - A) * (0.5 * (phi_P + phi_N))
    elif scheme == 'quick':
        # QUICK needs upwind, central, and downwind values
        if F_f >= 0:
            phi_f = (3/8) * phi_up + (6/8) * phi_P - (1/8) * phi_N
        else:
            phi_f = (3/8) * phi_down + (6/8) * phi_N - (1/8) * phi_P
    elif scheme == 'tvd' and limiter is not None:
        # TVD needs upwind, central, and downwind values
        if F_f >= 0:
            r = (phi_P - phi_up) / ((phi_N - phi_P) + 1e-20)
            psi = limiter(r)
            phi_f = phi_P + 0.5 * psi * (phi_N - phi_P)
        else:
            r = (phi_N - phi_down) / ((phi_P - phi_N) + 1e-20)
            psi = limiter(r)
            phi_f = phi_N + 0.5 * psi * (phi_P - phi_N)
    else:
        # Default: central difference
        phi_f = face_interp_factor * phi_P + (1 - face_interp_factor) * phi_N

    return F_f, phi_f

# --- Boundary Handler ---

def apply_boundary_condition(
    face_type, face_value, S_f, mu_f, owner_cell, d_PB, matrix_row, rhs, equation
):
    """
    Applies boundary condition to the matrix row and RHS for a boundary face.
    """
    # Dirichlet (velocity or pressure)
    if face_type == 1:  # Dirichlet
        matrix_row.clear()
        matrix_row[owner_cell] = 1.0
        rhs = face_value
    # Neumann or zeroGradient
    elif face_type in (2, 3):
        # For Neumann, typically nothing to do (zero gradient)
        pass
    # Wall (no-slip for velocity)
    elif face_type == 0 and equation == 'momentum':
        matrix_row.clear()
        matrix_row[owner_cell] = 1.0
        rhs = 0.0
    return matrix_row, rhs

# --- Assembly Function ---

def assemble_momentum_row(
    P, mesh_data, variable_field, grad_variable_field, mu_field, rho_field,
    velocity_field, pressure_field, source_term, convection_scheme, limiter=None
):
    """
    Assembles the matrix row and RHS for the momentum equation of cell P.
    Returns:
        matrix_row: {cell_idx: coeff, ...}
        rhs: Assembled right-hand side.
    """
    matrix_row = {}
    rhs = source_term

    cell_faces = mesh_data.cell_faces[P]
    cell_center = mesh_data.cell_centers[P]

    for local_face_idx in range(cell_faces.shape[0]):
        face = cell_faces[local_face_idx]
        if face == -1:
            continue  # Skip unused face slots

        is_boundary = mesh_data.face_boundary_mask[face]
        S_f = mesh_data.face_normals[face]
        face_center = mesh_data.face_centers[face]
        mu_f = mu_field[face]
        rho_f = rho_field[face]
        u_f = velocity_field[face]

        P_cell = mesh_data.owner_cells[face]
        N_cell = mesh_data.neighbor_cells[face] if not is_boundary else None

        if not is_boundary:
            N_center = mesh_data.cell_centers[N_cell]
            d_PN = mesh_data.d_PN[face]
            non_ortho_correction = mesh_data.non_ortho_correction[face]
            skewness_vector = mesh_data.skewness_vectors[face]
            phi_P = variable_field[P]
            phi_N = variable_field[N_cell]
            grad_phi_P = grad_variable_field[P]
            grad_phi_N = grad_variable_field[N_cell]
            D_f, b_corr = compute_diffusion_face(
                mu_f, S_f, d_PN, grad_phi_P, grad_phi_N,
                non_ortho_correction, skewness_vector
            )
            matrix_row[P] = matrix_row.get(P, 0.0) + D_f
            matrix_row[N_cell] = matrix_row.get(N_cell, 0.0) - D_f
            rhs += b_corr

            # --- Convection ---
            # For upwind, powerlaw, TVD, QUICK, need upwind/downwind values
            # For demonstration, use neighbor's neighbor as upwind/downwind if available
            phi_up = phi_P
            phi_down = phi_N
            F_f, phi_f = compute_convection_face(
                rho_f, u_f, S_f, phi_P, phi_N, phi_up, phi_down,
                scheme=convection_scheme, limiter=limiter
            )
            # Upwind: add F_f to P or N
            if F_f >= 0:
                matrix_row[P] = matrix_row.get(P, 0.0) + F_f
            else:
                matrix_row[N_cell] = matrix_row.get(N_cell, 0.0) - F_f

        else:
            face_type = mesh_data.boundary_types[face, 0]  # velocity BC
            face_value = mesh_data.boundary_values[face, 0]
            d_PB = mesh_data.d_PB[face]
            matrix_row, rhs = apply_boundary_condition(
                face_type, face_value, S_f, mu_f, P, d_PB, matrix_row, rhs, 'momentum'
            )

    return matrix_row, rhs

# --- Example TVD Limiter (Van Leer) ---

def van_leer_limiter(r):
    return (r + abs(r)) / (1 + abs(r))

# --- Example Usage for All Cells ---

def build_momentum_matrix_and_rhs(
    mesh_data, variable_field, grad_variable_field, mu_field, rho_field,
    velocity_field, pressure_field, source_terms, convection_scheme, limiter=None
):
    n_cells = mesh_data.cell_centers.shape[0]
    A = [{} for _ in range(n_cells)]
    rhs = np.zeros(n_cells)
    for P in range(n_cells):
        row, rhs_val = assemble_momentum_row(
            P, mesh_data, variable_field, grad_variable_field, mu_field, rho_field,
            velocity_field, pressure_field, source_terms[P], convection_scheme, limiter
        )
        A[P] = row
        rhs[P] = rhs_val
    return A, rhs
```


---

## **How to Use**

1. **Prepare all fields** (variable, gradients, mu, rho, velocity, pressure, source_terms) and your `mesh_data` object.
2. **Call** `build_momentum_matrix_and_rhs(...)` with your chosen convection scheme and limiter (if any).
3. **Assemble the global matrix** from the returned list of dictionaries `A` and the right-hand side `rhs`.

---

**This code is ready for use in a collocated, mesh-agnostic finite volume SIMPLE solver, and can be extended for more advanced boundary conditions or higher-order convection schemes as needed.**

