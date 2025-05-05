import numpy as np
from scipy.sparse import csr_matrix

from naviflow_collocated.mesh import MeshData2D as Mesh  # Use alias


def assemble_pressure_correction_matrix_rhs(
    mesh: Mesh,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rho: float,
    Ap_u: np.ndarray,
    Ap_v: np.ndarray,
    boundary_conditions=None,  # Keep for backward compatibility, not used
    under_relax_p: float = 1.0,  # Typically not relaxed during assembly
):
    """
    Assembles the pressure correction matrix (A) and RHS (b) using Rhie-Chow interpolation.

    A p' = b

    Parameters:
    - mesh: The mesh data.
    - u: The velocity component in the x-direction.
    - v: The velocity component in the y-direction.
    - p: The pressure field.
    - rho: The density of the fluid.
    - Ap_u: The coefficient matrix for the x-direction velocity equation.
    - Ap_v: The coefficient matrix for the y-direction velocity equation.
    - boundary_conditions: The list of boundary conditions (not used, kept for backward compatibility).
    - under_relax_p: The under-relaxation factor for pressure (default is 1.0).

    Returns:
    - row: The row indices of the sparse matrix.
    - col: The column indices of the sparse matrix.
    - data: The data values of the sparse matrix.
    - rhs: The RHS vector.
    """
    n_cells = len(mesh.cell_volumes)
    n_faces = len(mesh.face_areas)

    # Initialize arrays for COO sparse matrix format
    row = []
    col = []
    data = []

    # Initialize RHS vector
    rhs = np.zeros(n_cells)

    # Initialize diagonal coefficients for each cell
    diag = np.zeros(n_cells)

    # Small value to prevent division by zero
    _SMALL = 1.0e-12

    # Loop over all internal faces to assemble the matrix
    for f in range(n_faces):
        # Get owner and neighbor cells
        owner = mesh.owner_cells[f]
        neighbor = mesh.neighbor_cells[f]

        # Skip boundary faces for now (negative neighbor index)
        if neighbor < 0:
            continue

        # Get face area vector and magnitude
        face_normal = mesh.face_normals[f]
        face_area = mesh.face_areas[f]

        # Calculate d_CF (distance between cell centers)
        d_CF = mesh.delta_CF[f]

        # Calculate face coefficients based on Rhie-Chow interpolation
        # A_p at the face is interpolated from cell values
        Ap_u_owner = Ap_u[owner]
        Ap_v_owner = Ap_v[owner]
        Ap_u_neighbor = Ap_u[neighbor]
        Ap_v_neighbor = Ap_v[neighbor]

        # Interpolate 1/A_p to the face - linear weight fx from mesh
        fx = mesh.face_interp_factors[f]

        # Use harmonic mean for better stability near discontinuities
        Ap_u_face = 1.0 / (
            fx / (Ap_u_owner + _SMALL) + (1.0 - fx) / (Ap_u_neighbor + _SMALL)
        )
        Ap_v_face = 1.0 / (
            fx / (Ap_v_owner + _SMALL) + (1.0 - fx) / (Ap_v_neighbor + _SMALL)
        )

        # Calculate the coefficient between owner and neighbor
        # This is d_(face)/A_p * rho * area^2 / d_CF for each component
        # Simplified to area/d_CF * (area_x^2/Ap_u + area_y^2/Ap_v)

        # Normal components
        norm_x = face_normal[0] / (face_area + _SMALL)
        norm_y = face_normal[1] / (face_area + _SMALL)

        # Calculate coefficient (Moukalled Eq 11.52)
        coeff = (
            face_area
            * ((norm_x * norm_x * Ap_u_face) + (norm_y * norm_y * Ap_v_face))
            / (d_CF + _SMALL)
        )

        # Add to the matrix in COO format
        # Owner diagonal
        diag[owner] += coeff

        # Neighbor diagonal
        diag[neighbor] += coeff

        # Off-diagonal terms (negative because moving to RHS)
        row.append(owner)
        col.append(neighbor)
        data.append(-coeff)  # Off-diagonal is negative

        row.append(neighbor)
        col.append(owner)
        data.append(-coeff)  # Off-diagonal is negative

        # Calculate mass flux across the face for the RHS
        # mass flux = rho * A * (u*nx + v*ny)
        u_face = fx * u[owner] + (1.0 - fx) * u[neighbor]
        v_face = fx * v[owner] + (1.0 - fx) * v[neighbor]

        mass_flux = rho * face_area * (u_face * norm_x + v_face * norm_y)

        # Add contribution to RHS (mass imbalance)
        rhs[owner] -= mass_flux  # Outflow from owner is negative contribution
        rhs[neighbor] += mass_flux  # Inflow to neighbor is positive contribution

    # Handle boundary faces
    for i, bf in enumerate(mesh.boundary_faces):
        f = bf  # Face index
        owner = mesh.owner_cells[f]

        # Get face normal and area
        face_normal = mesh.face_normals[f]
        face_area = mesh.face_areas[f]
        norm_x = face_normal[0] / (face_area + _SMALL)
        norm_y = face_normal[1] / (face_area + _SMALL)

        # Get boundary condition type
        bc_type = mesh.boundary_types[i]

        # For pressure correction, most boundaries are treated as zero gradient
        # This means no contribution to the matrix (just to the RHS for mass flux)

        # Calculate mass flux based on boundary type
        if bc_type == 1:  # No-slip wall
            # Zero velocity at wall, no mass flux
            mass_flux = 0.0
        elif bc_type == 3:  # Inlet
            # Fixed velocity at inlet
            u_bc = mesh.boundary_values[i, 0]
            v_bc = mesh.boundary_values[i, 1]
            mass_flux = rho * face_area * (u_bc * norm_x + v_bc * norm_y)
        else:  # Other boundaries default to zero gradient
            # Use cell center value
            u_face = u[owner]
            v_face = v[owner]
            mass_flux = rho * face_area * (u_face * norm_x + v_face * norm_y)

        # Add to RHS
        rhs[owner] -= mass_flux

    # Add diagonal terms to COO arrays
    for i in range(n_cells):
        row.append(i)
        col.append(i)
        data.append(diag[i])

    return np.array(row), np.array(col), np.array(data), rhs


def assemble_pressure_correction_matrix_csr(
    mesh: Mesh,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rho: float,
    Ap_u: np.ndarray,
    Ap_v: np.ndarray,
    boundary_conditions=None,  # Keep for backward compatibility, not used
    under_relax_p: float = 1.0,
) -> tuple[csr_matrix, np.ndarray]:
    row, col, data, rhs = assemble_pressure_correction_matrix_rhs(
        mesh, u, v, p, rho, Ap_u, Ap_v, None, under_relax_p
    )
    n_cells = len(mesh.cell_volumes)
    A = csr_matrix((data, (row, col)), shape=(n_cells, n_cells))
    A.eliminate_zeros()  # Recommended for efficiency
    return A, rhs
