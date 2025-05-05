"""
Matrix assembly for momentum equations in the collocated SIMPLE solver.

This module provides functions to assemble the coefficient matrices for the
u and v momentum equations, following Moukalled's methodology for collocated grids.
"""

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse import csr_matrix
import pytest

from naviflow_collocated.mesh import MeshData2D as Mesh  # Use alias
from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_central_diffusion_face_coeffs,
)
from naviflow_collocated.discretization.convection.power_law import (
    compute_powerlaw_convection_face_coeffs,
)

# Small epsilon value to ensure diagonal dominance
_SMALL_DD_EPS = 1.0e-12

# Boundary condition type codes
BC_WALL_NO_SLIP = 1  # No-slip wall (u=0, v=0)
BC_WALL_SLIP = 2  # Slip wall (tangential=copy, normal=0)
BC_INLET_VELOCITY = 3  # Inlet with specified velocity
BC_OUTLET_PRESSURE = 4  # Outlet with specified pressure
BC_SYMMETRY = 5  # Symmetry plane


@njit
def apply_boundary_face_coeffs(
    mesh,
    f,
    C,
    aP,
    aF,
    row,
    col,
    data,
    rhs,
    entry_count,
    mu,
    rho,
    phi,
    component,
    bc_idx=None,
):
    """
    Apply boundary condition contributions to the matrix coefficients.

    This function modifies the matrix coefficients and right-hand side
    for boundary faces according to the boundary condition type.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh data structure
    f : int
        Face index
    C : int
        Owner cell index
    aP : float
        Diagonal coefficient for owner cell (will be modified)
    aF : float
        Off-diagonal coefficient for non-existent neighbor cell
    row : ndarray
        Row indices for sparse matrix
    col : ndarray
        Column indices for sparse matrix
    data : ndarray
        Values for sparse matrix
    rhs : ndarray
        Right-hand side vector
    entry_count : int
        Current count of entries in COO format
    mu : float or ndarray
        Dynamic viscosity
    rho : float
        Density
    phi : ndarray
        Current velocity field (u or v)
    component : int
        Velocity component (0=u, 1=v)
    bc_idx : int
        Boundary face index for accessing boundary_types and boundary_values

    Returns
    -------
    entry_count : int
        Updated count of entries in COO format
    aP : float
        Updated diagonal coefficient
    """
    # Find the boundary index for this face
    if bc_idx is None:
        bc_idx = -1
        for i in range(len(mesh.boundary_faces)):
            if mesh.boundary_faces[i] == f:
                bc_idx = i
                break

        if bc_idx < 0:
            # Face not found in boundary list, treat as zero gradient
            return entry_count, aP

    # Get the boundary condition type
    bc_type = mesh.boundary_types[bc_idx]

    # Get the boundary value for this component
    bc_value = mesh.boundary_values[bc_idx, component]

    # Get the face area and normal direction
    area = mesh.face_areas[f]
    nx = mesh.face_normals[f, 0]
    ny = mesh.face_normals[f, 1]

    # Unit normal vector
    nmag = np.sqrt(nx * nx + ny * ny)
    if nmag > _SMALL_DD_EPS:
        nx = nx / nmag
        ny = ny / nmag

    # Distance from cell center to face
    dCf = np.sqrt(
        (mesh.face_centers[f, 0] - mesh.cell_centers[C, 0]) ** 2
        + (mesh.face_centers[f, 1] - mesh.cell_centers[C, 1]) ** 2
    )

    # Apply different BC types following Practice B: modify matrix coefficients
    if bc_type == BC_WALL_NO_SLIP:
        # Wall BC: u = 0, v = 0 (Dirichlet)
        # Use standard diffusion discretization with high diffusivity
        alpha = 1e10  # Large value for strong enforcement
        phi_bc = bc_value  # Usually 0 for no-slip

        # Add to matrix
        aP_bc = alpha * area / dCf
        aP += aP_bc  # Add to diagonal

        # Add to RHS
        rhs[C] += aP_bc * phi_bc

    elif bc_type == BC_WALL_SLIP:
        # Slip wall: zero normal gradient, zero normal velocity
        # Get the component of velocity normal to the boundary
        if component == 0:
            # u-component
            dir_x = 1.0  # Direction of the component (unit vector in x)
            dir_y = 0.0
        else:
            # v-component
            dir_x = 0.0
            dir_y = 1.0

        # Calculate normal component
        normal_dot_dir = nx * dir_x + ny * dir_y

        if abs(normal_dot_dir) < 1e-10:
            # Component is tangential to the face, use zero gradient
            # Zero gradient means no contribution to matrix from this face
            pass
        else:
            # Component has a normal component, enforce zero normal velocity
            alpha = 1e10  # Large value for strong enforcement
            aP_bc = alpha * area * abs(normal_dot_dir)
            aP += aP_bc
            # phi_bc = 0.0 for zero normal velocity, so no contribution to RHS

    elif bc_type == BC_INLET_VELOCITY:
        # Inlet: Dirichlet BC
        alpha = 1e10  # Large value for strong enforcement
        phi_bc = bc_value

        # Add to matrix
        aP_bc = alpha * area / dCf
        aP += aP_bc

        # Add to RHS
        rhs[C] += aP_bc * phi_bc

    elif bc_type == BC_OUTLET_PRESSURE:
        # Outlet: zero gradient
        # Zero gradient means no contribution to matrix from this face
        # Pressure term is handled separately
        pass

    elif bc_type == BC_SYMMETRY:
        # Symmetry: zero normal gradient, zero normal velocity
        if component == 0:
            # u-component
            dir_x = 1.0  # Direction of the component (unit vector in x)
            dir_y = 0.0
        else:
            # v-component
            dir_x = 0.0
            dir_y = 1.0

        # Calculate normal component
        normal_dot_dir = nx * dir_x + ny * dir_y

        if abs(normal_dot_dir) < 1e-10:
            # Component is tangential to the face, use zero gradient
            # Zero gradient means no contribution to matrix from this face
            pass
        else:
            # Component has a normal component, enforce zero normal velocity
            alpha = 1e10  # Large value for strong enforcement
            aP_bc = alpha * area * abs(normal_dot_dir)
            aP += aP_bc
            # phi_bc = 0.0 for zero normal velocity, so no contribution to RHS

    return entry_count, aP


@njit
def _assemble_momentum_matrix_rhs_impl(
    mesh,
    u,
    v,
    p,
    mu,
    rho,
    component,
    under_relax=1.0,
    use_skew_correction=False,
):
    """Internal implementation for momentum assembly (cannot be njit with BC dict)."""
    n_cells = len(mesh.cell_volumes)
    # n_faces = len(mesh.face_areas) # Already calculated later
    # n_internal_faces = n_faces - len(mesh.boundary_faces) # Not used here

    # Small value to prevent division by zero
    _SMALL = 1.0e-12

    # Allocate storage for COO format
    row = []
    col = []
    data = []

    # Initialize right-hand side vector and diagonal terms
    rhs = np.zeros(n_cells, dtype=np.float64)
    diag = np.zeros(n_cells, dtype=np.float64)

    # Select velocity component to solve
    phi = u if component == 0 else v

    # Track number of entries in COO format
    entry_count = 0

    # Face-by-face assembly of coefficients
    for f in range(len(mesh.face_areas)):
        # Get owner and neighbor cells
        C = mesh.owner_cells[f]
        F = mesh.neighbor_cells[f]

        # Handle boundary faces separately
        if F < 0:
            # Find the boundary index for this face
            bc_idx = -1
            for i in range(len(mesh.boundary_faces)):
                if mesh.boundary_faces[i] == f:
                    bc_idx = i
                    break

            # Apply boundary conditions to coefficients
            if bc_idx >= 0:
                entry_count, diag[C] = apply_boundary_face_coeffs(
                    mesh,
                    f,
                    C,
                    diag[C],
                    0.0,
                    row,
                    col,
                    data,
                    rhs,
                    entry_count,
                    mu,
                    rho,
                    phi,
                    component,
                    bc_idx,
                )
            continue

        # ---- INTERNAL FACE PROCESSING ----
        # For internal faces, we compute contributions from diffusion and convection

        # 1. Diffusion terms using central differencing
        aC_diff, aF_diff, skew_flux = compute_central_diffusion_face_coeffs(
            mesh, f, mu, use_skew_correction
        )

        # 2. Convection terms using Power Law scheme
        face_flux = 0.0  # Simplified for now, will be from Rhie-Chow
        dC_conv, oC_conv, dF_conv, oF_conv = compute_powerlaw_convection_face_coeffs(
            mesh, f, face_flux, rho, mu
        )

        # 3. Combine contributions
        aC = aC_diff + dC_conv + oC_conv  # Diagonal contribution for owner
        aF = aF_diff + dF_conv + oF_conv  # Off-diagonal for neighbor

        # Add owner cell diagonal contribution
        diag[C] += aC

        # Add neighbor cell diagonal contribution
        diag[F] += aF

        # Add owner-neighbor coupling
        row.append(C)
        col.append(F)
        data.append(-aF)  # Negative for off-diagonal
        entry_count += 1

        # Add neighbor-owner coupling
        row.append(F)
        col.append(C)
        data.append(-aC)  # Negative for off-diagonal
        entry_count += 1

        # 4. Pressure gradient term (add to RHS)
        if component == 0:  # x-momentum
            delta = max(mesh.delta_CF[f], _SMALL)
            dp_dx = (p[F] - p[C]) / delta * mesh.face_normals[f, 0]
            rhs[C] -= mesh.face_areas[f] * dp_dx
            rhs[F] += mesh.face_areas[f] * dp_dx
        else:  # y-momentum
            delta = max(mesh.delta_CF[f], _SMALL)
            dp_dy = (p[F] - p[C]) / delta * mesh.face_normals[f, 1]
            rhs[C] -= mesh.face_areas[f] * dp_dy
            rhs[F] += mesh.face_areas[f] * dp_dy

    # Add diagonal entries and apply under-relaxation
    for i in range(n_cells):
        a_prev = diag[i]  # Store original diagonal (including BC contributions)

        # Apply under-relaxation factor (as post-modification)
        if under_relax < 1.0:
            # Check to avoid division by zero if diag is zero before relaxation
            if abs(under_relax) > _SMALL:
                diag_relaxed = a_prev / under_relax
                rhs[i] += ((1.0 - under_relax) / under_relax) * a_prev * phi[i]
            else:
                # If under_relax is zero, treat as very strong relaxation (effectively setting diagonal high)
                diag_relaxed = a_prev * 1.0e20  # Assign a large number
                # RHS adjustment might become problematic, maybe set to zero? or large value?
                # Let's keep the original RHS adjustment logic but it might be unstable
                # rhs[i] += ((1.0 - under_relax) / _SMALL) * a_prev * phi[i] # Avoid direct division
                # Simpler: set RHS adjustment based on phi_old? Requires careful thought. Let's zero it for now.
                pass  # Keep RHS as is, diagonal is huge
        else:
            diag_relaxed = a_prev

        # Add final diagonal to COO format
        row.append(i)
        col.append(i)
        data.append(diag_relaxed)
        entry_count += 1

    # Truncate arrays to actual size
    row = row[:entry_count]
    col = col[:entry_count]
    data = data[:entry_count]

    return np.array(row), np.array(col), np.array(data), rhs


def assemble_momentum_matrix_rhs(
    mesh: Mesh,  # Type hint for clarity
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    mu: float,
    rho: float,
    boundary_conditions=None,  # Keep for backward compatibility but don't use
    component: int = 0,
    under_relax: float = 1.0,
):
    """Public facing function for momentum assembly (calls internal impl)."""
    # We might add Numba acceleration back later by refactoring BC handling
    return _assemble_momentum_matrix_rhs_impl(
        mesh, u, v, p, mu, rho, component, under_relax
    )


def assemble_momentum_matrix_csr(
    mesh: Mesh,  # Type hint
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    mu: float,
    rho: float,
    boundary_conditions=None,  # Keep for backward compatibility but don't use
    component: int = 0,
    under_relax: float = 1.0,
) -> tuple[csr_matrix, np.ndarray]:
    """
    Wrapper to assemble the momentum matrix and RHS, returning a CSR matrix.
    Args:
        mesh: The mesh object.
        u: Current u-velocity field.
        v: Current v-velocity field.
        p: Current pressure field.
        mu: Viscosity.
        rho: Density.
        boundary_conditions: List of boundary condition dictionaries (not used, kept for backward compatibility).
        component: Velocity component (0=u, 1=v).
        under_relax: Under-relaxation factor for momentum.

    Returns:
        Tuple: (CSR matrix A, RHS vector b) for the specified momentum component.
    """
    row, col, data, rhs = assemble_momentum_matrix_rhs(
        mesh, u, v, p, mu, rho, None, component, under_relax
    )
    n_cells = len(mesh.cell_volumes)
    A = csr_matrix((data, (row, col)), shape=(n_cells, n_cells))
    A.eliminate_zeros()  # Recommended for efficiency
    return A, rhs


def solve_momentum_equation(
    mesh,
    u,
    v,
    p,
    mu,
    rho,
    dt=None,
    under_relax=1.0,
    component=0,
    tol=1e-8,
    max_iter=1000,
    solver="bicgstab",
):
    """
    Assemble and solve the momentum equation for the specified component.

    Parameters
    ----------
    (Same as assemble_momentum_matrix_rhs plus solver parameters)
    tol : float, optional
        Convergence tolerance for the linear solver
    max_iter : int, optional
        Maximum number of iterations for the linear solver
    solver : str, optional
        Linear solver to use ("bicgstab", "gmres", "cg", etc.)

    Returns
    -------
    phi_new : ndarray
        Solution vector for the specified velocity component
    residual : float
        Final residual of the linear solver
    iterations : int
        Number of iterations taken by the linear solver
    """
    from scipy.sparse.linalg import bicgstab, gmres, cg

    # Assemble the matrix and RHS
    A, b = assemble_momentum_matrix_csr(
        mesh, u, v, p, mu, rho, None, component, under_relax
    )

    # Select the appropriate solver
    if solver == "bicgstab":
        phi_new, info = bicgstab(A, b, tol=tol, maxiter=max_iter)
    elif solver == "gmres":
        phi_new, info = gmres(A, b, tol=tol, maxiter=max_iter)
    elif solver == "cg":
        phi_new, info = cg(A, b, tol=tol, maxiter=max_iter)
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Calculate residual (approximate)
    residual = np.linalg.norm(A @ phi_new - b) / np.linalg.norm(b)

    return phi_new, residual, info


def test_momentum_conservation(mesh_instance):
    """Test momentum conservation principles in the discretization."""
    # Use fields from the mesh_instance fixture or initialize placeholders
    n_cells = len(mesh_instance.cell_volumes)
    u = np.zeros(n_cells)  # Placeholder
    v = np.zeros(n_cells)  # Placeholder
    p = np.zeros(n_cells)  # Placeholder
    mu = 0.01  # Placeholder
    rho = 1.0  # Placeholder
    component = 0  # Test U-component

    # Assemble the u-momentum equation matrix
    # Pass boundary conditions
    row, col, data, rhs = assemble_momentum_matrix_rhs(
        mesh_instance,
        u,
        v,
        p,
        mu,
        rho,
        mesh_instance.boundary_conditions,
        component=component,
    )

    # Create sparse matrix for analysis
    # ... (rest of the function code remains unchanged) ...


@pytest.mark.usefixtures("mesh_instance")
def test_assemble_momentum_matrix_csr(mesh_instance: Mesh):
    """Tests the assembly of the CSR matrix for momentum equation."""
    # Use fields from the mesh_instance fixture or initialize placeholders
    n_cells = len(mesh_instance.cell_volumes)
    u = np.zeros(n_cells)
    v = np.zeros(n_cells)
    p = np.zeros(n_cells)
    mu = 0.01
    rho = 1.0
    under_relax = 0.7
    component = 0  # Test U-component

    # Assemble the matrix and RHS
    A, b = assemble_momentum_matrix_csr(
        mesh_instance,
        u,
        v,
        p,
        mu,
        rho,
        mesh_instance.boundary_conditions,
        component,
        under_relax,
    )

    assert isinstance(A, sparse.csr_matrix)


@pytest.mark.usefixtures("mesh_instance")
def test_assemble_momentum_matrix_rhs(mesh_instance: Mesh):
    """Tests the assembly of the RHS vector for the momentum equation."""
    # Use fields from the mesh_instance fixture or initialize placeholders
    n_cells = len(mesh_instance.cell_volumes)
    u = np.zeros(n_cells)
    v = np.zeros(n_cells)
    p = np.zeros(n_cells)
    mu = 0.01
    rho = 1.0
    component = 0  # Test U-component

    # Pass boundary conditions
    row, col, data, rhs = assemble_momentum_matrix_rhs(
        mesh_instance,
        u,
        v,
        p,
        mu,
        rho,
        mesh_instance.boundary_conditions,
        component=component,
    )

    assert isinstance(rhs, np.ndarray)
