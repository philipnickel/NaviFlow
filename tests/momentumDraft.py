import os
import numpy as np
import sympy as sp

# --- Diffusion Helper ---

def compute_diffusion_face( 
        f, grad_u, mesh, mu
):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]
    mu_f = mu if isinstance(mu, float) else mu[f]

    # Geometric quantities
    delta = mesh.delta_PN[f] + 1e-14
    e_f = mesh.unit_dPN[f]
    S_f = mesh.face_normals[f]
    t_f = mesh.non_ortho_correction[f]  # tangential vector
    d_f = mesh.skewness_vectors[f]      # face centroid shift

    # Orthogonal diffusive conductance
    D_f = mu_f * np.dot(np.ascontiguousarray(S_f), np.ascontiguousarray(e_f)) / delta
    a_PP = D_f
    a_PN = -D_f

    # Gradient interpolation (central)
    grad_u_P = grad_u[P]
    grad_u_N = grad_u[N] if N >= 0 else grad_u_P
    grad_u_f = 0.5 * (grad_u_P + grad_u_N)

    # Explicit non-orthogonal + skewness correction
    b_corr = -mu_f * np.dot(np.ascontiguousarray(grad_u_f), np.ascontiguousarray(t_f + d_f)) 

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
        matrix_row = {}
        matrix_row[owner_cell] = 1.0
        rhs = face_value
    # Neumann or zeroGradient
    elif face_type in (2, 3):
        # For Neumann, typically nothing to do (zero gradient)
        pass
    # Wall (no-slip for velocity)
    elif face_type == 0 and equation == 'momentum':
        matrix_row = {}
        matrix_row[owner_cell] = 1.0
        rhs = 0.0
    return matrix_row, rhs

def assemble_momentum_matrix_face_based(mesh, phi, grad_phi, mu, rho, u_field, convection_scheme='upwind', limiter=None):
    n_cells = mesh.cell_volumes.shape[0]
    row, col, data = [], [], []
    b = np.zeros(n_cells)

    # Internal faces
    for f in mesh.internal_faces:
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]

        D_f, b_d = compute_diffusion_face(f, grad_phi, mesh, mu)
        phi_P = phi[P]
        phi_N = phi[N]
        u_f = u_field[f]
        S_f = mesh.face_normals[f]
        rho_f = rho[f]

        F_f, phi_f = compute_convection_face(rho_f, u_f, S_f, phi_P, phi_N, phi_P, phi_N, scheme=convection_scheme, limiter=limiter)
        a_c_P = F_f if F_f >= 0 else 0.0
        a_c_N = -F_f if F_f < 0 else 0.0

        a_P = D_f + a_c_P
        a_N = -D_f + a_c_N
        b_total = b_d

        # Matrix entries (antisymmetric)
        row.extend([P, P, N, N])
        col.extend([P, N, P, N])
        data.extend([a_P, a_N, -a_N, -a_P])

        # RHS
        b[P] -= b_total
        b[N] -= b_total

    # Boundary faces
    for f in mesh.boundary_faces:
        P = mesh.owner_cells[f]
        face_type = mesh.boundary_types[f, 0]
        face_value = mesh.boundary_values[f, 0]
        S_f = mesh.face_normals[f]
        mu_f = mu[f]
        rho_f = rho[f]
        u_f = u_field[f]
        d_PB = mesh.d_PB[f]

        if face_type == 1:  # Dirichlet
            row.append(P)
            col.append(P)
            data.append(1.0)
            b[P] = face_value
        elif face_type in (2, 3):  # Neumann
            continue
        elif face_type == 0:  # Wall
            row.append(P)
            col.append(P)
            data.append(1.0)
            b[P] = 0.0

    return np.array(row), np.array(col), np.array(data), b

# --- Example TVD Limiter (Van Leer) ---

def van_leer_limiter(r):
    return (r + abs(r)) / (1 + abs(r))

# --- MMS Test ---

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from naviflow_collocated.mesh.mesh_loader import load_mesh

def plot_field(mesh, field, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    sc = ax.scatter(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1],
                    c=field, cmap="viridis", s=30, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.75)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")

def run_mms_test(mesh_file, bc_file, u_exact_fn, grad_fn, rhs_fn, mu, rho, u_field_fn, tag):
    mesh = load_mesh(mesh_file, bc_file)

    phi_exact = u_exact_fn(mesh.cell_centers)
    grad_phi = grad_fn(mesh.cell_centers)
    # u_field should be per-face, so evaluate at face centers
    u_field = u_field_fn(mesh.face_centers)

    # Convert mu and rho to arrays of per-face values if needed
    n_faces = mesh.face_centers.shape[0]
    if np.isscalar(mu):
        mu = np.full(n_faces, mu)
    if np.isscalar(rho):
        rho = np.full(n_faces, rho)

    convection_scheme = 'upwind'

    row, col, data, b = assemble_momentum_matrix_face_based(
        mesh, phi_exact, grad_phi, mu, rho, u_field, convection_scheme
    )

    rhs = rhs_fn(mesh.cell_centers)
    b_rhs = rhs * mesh.cell_volumes

    A = coo_matrix((data, (row, col)), shape=(mesh.cell_centers.shape[0],) * 2).tocsr()
    phi_numeric = spsolve(A, b_rhs)

    err = np.abs(phi_numeric - phi_exact)
    max_err, l2_err = np.max(err), np.sqrt(np.mean(err ** 2))
    print(f"[{tag}] Max error: {max_err:.2e}, L2 error: {l2_err:.2e}")
    # Print scales of numerical and exact values
    print(f"[{tag}] Numerical min: {np.min(phi_numeric):.2e}, max: {np.max(phi_numeric):.2e}")
    print(f"[{tag}] Exact min: {np.min(phi_exact):.2e}, max: {np.max(phi_exact):.2e}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plot_field(mesh, phi_numeric, ax=axs[0], title="Numerical")
    plot_field(mesh, phi_exact, ax=axs[1], title="Exact")
    plot_field(mesh, err, ax=axs[2], title="Error")
    plt.tight_layout()
    outdir = Path("tests/test_output")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"mms_{tag}.png", dpi=300)
    plt.close()

# === MMS Functions ===
def u_zero(xy): return np.zeros((xy.shape[0], 2))

def u_sin(xy): return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])
def grad_sin(xy):
    gx = np.pi * np.cos(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])
    gy = np.pi * np.sin(np.pi * xy[:, 0]) * np.cos(np.pi * xy[:, 1])
    return np.stack([gx, gy], axis=1)
def rhs_sin_diffusion(xy): return -2 * np.pi**2 * u_sin(xy)

def u_uniform_x(xy): return np.tile([1.0, 0.0], (xy.shape[0], 1))

def u_cos(xy): return np.cos(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])
def grad_cos(xy):
    gx = -np.pi * np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])
    gy =  np.pi * np.cos(np.pi * xy[:, 0]) * np.cos(np.pi * xy[:, 1])
    return np.stack([gx, gy], axis=1)
def rhs_cos_convection(xy): return -np.pi * np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])

def rhs_combined(xy):
    return rhs_sin_diffusion(xy) + np.pi * np.cos(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])

def test_flux_balance_center_cell(mesh_file, u_fn, grad_fn, mu):
    from naviflow_collocated.mesh.mesh_loader import load_mesh

    mesh = load_mesh(mesh_file, None)
    phi = u_fn(mesh.cell_centers)
    grad_phi = grad_fn(mesh.cell_centers)

    center_cell = mesh.cell_centers.shape[0] // 2
    volume = mesh.cell_volumes[center_cell]

    flux_sum = 0.0
    for face in mesh.cell_faces[center_cell]:
        if face == -1:
            continue
        owner = mesh.owner_cells[face]
        neighbor = mesh.neighbor_cells[face] if face < mesh.neighbor_cells.shape[0] else -1

        if neighbor == -1 or (owner != center_cell and neighbor != center_cell):
            continue

        P = owner if owner == center_cell else neighbor
        N = neighbor if owner == center_cell else owner

        d_PN = mesh.d_PN[face]
        S_f = mesh.face_normals[face]
        t_f = mesh.non_ortho_correction[face]
        d_f = mesh.skewness_vectors[face]
        mu_f = mu[face] if not np.isscalar(mu) else mu

        grad_P = grad_phi[P]
        grad_N = grad_phi[N] if N >= 0 else grad_P
        grad_f = 0.5 * (grad_P + grad_N)

        D_f = mu_f * np.dot(S_f, d_PN) / (np.dot(d_PN, d_PN) + 1e-20)
        phi_P = phi[P]
        phi_N = phi[N] if N >= 0 else phi_P
        flux_diff = D_f * (phi_N - phi_P) - mu_f * np.dot(grad_f, t_f + d_f)

        flux_sum += flux_diff if P == center_cell else -flux_diff

    return flux_sum, center_cell, volume

# === Run Tests ===
if __name__ == "__main__":
    mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh"

    run_mms_test(mesh_file, "shared_configs/domain/sanityCheckDiffusion.yaml",
                 u_sin, grad_sin, rhs_sin_diffusion,
                 mu=1.0, rho=0.0, u_field_fn=u_zero, tag="diffusion")
    """

    run_mms_test(mesh_file, "shared_configs/domain/sanityCheckConvection.yaml",
                 u_cos, grad_cos, rhs_cos_convection,
                 mu=0.0, rho=1.0, u_field_fn=u_uniform_x, tag="convection")

    run_mms_test(mesh_file, "shared_configs/domain/sanityCheckCombined.yaml",
                 u_sin, grad_sin, rhs_combined,
                 mu=1.0, rho=1.0, u_field_fn=u_uniform_x, tag="combined")
    """

    mesh = load_mesh(mesh_file, None)

    flux, center, vol = test_flux_balance_center_cell(mesh_file, u_sin, grad_sin, mu=1.0)
    source_val = rhs_sin_diffusion(mesh.cell_centers)[center] * vol
    print(f"[Flux Test] Diffusive flux sum at cell {center}: {flux:.6e}")
    print(f"[Flux Test] Expected source * volume: {source_val:.6e}")
    print(f"[Flux Test] Absolute error: {abs(flux - source_val):.2e}")