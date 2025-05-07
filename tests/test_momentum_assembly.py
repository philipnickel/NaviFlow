from naviflow_collocated.discretization.diffusion.central_diff import (
    compute_diffusive_flux,
)
from naviflow_collocated.discretization.convection.upwind import (
    compute_convective_flux_upwind,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path  # Use pathlib for path handling

from naviflow_collocated.assembly.momentumMatrix import assemble_momentum_matrix


# --- Discrete RHS computation helper ---
def compute_discrete_rhs(u, grad_u, mesh, rho, mu, include_convection=True):
    n_cells = len(mesh.cell_volumes)
    rhs = np.zeros(n_cells)

    for f in mesh.internal_faces:
        # Diffusion
        P, N, a_PP_d, a_PN_d, b_corr_d = compute_diffusive_flux(f, u, grad_u, mesh, mu)
        rhs[P] += a_PP_d * u[P] + a_PN_d * u[N] + b_corr_d

        if include_convection:
            uf = np.array([1.0, 0.0])
            P_c, N_c, a_PP_c, a_PN_c, b_corr_c = compute_convective_flux_upwind(
                f, u, mesh, uf, rho
            )
            rhs[P_c] += a_PP_c * u[P_c] + a_PN_c * u[N_c] + b_corr_c

    for f in mesh.boundary_faces:
        P = mesh.owner_cells[f]
        bc_val = mesh.boundary_values[f, 0]
        d_PB_val = mesh.d_PB[f]
        S_f = mesh.face_normals[f] * mesh.face_areas[f]

        # Diffusion
        diff_coeff = mu * np.linalg.norm(S_f) / (d_PB_val + 1e-14)
        rhs[P] += diff_coeff * (bc_val - u[P])

        if include_convection:
            uf = np.array([1.0, 0.0])
            m_dot_f = rho * np.dot(uf, S_f)
            if m_dot_f < 0:
                rhs[P] += -m_dot_f * (bc_val - u[P])

    return rhs


# --- Helper Plotting Function ---
def plot_mms_results(
    mesh,
    mesh_label,
    test_name,
    u_exact,
    Ax,
    expected_rhs,
    filename_suffix,
):
    """Generates and saves plots for MMS test results."""

    # --- Define output directory ---
    # Corrected path to be inside the tests directory
    output_dir = Path("tests/test_output/momentum_equations")
    output_dir.mkdir(
        parents=True, exist_ok=True
    )  # Create directory if it doesn't exist
    # --- End Define output directory ---
    x_coords = mesh.cell_centers[:, 0]
    y_coords = mesh.cell_centers[:, 1]

    # residual field
    fields_to_plot = {
        "Ax (Assembled)": Ax,
        "Expected RHS": expected_rhs,
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))  # Changed to 1x2 layout
    fig.suptitle(f"MMS Results: {test_name} ({mesh_label})", fontsize=16)

    all_axes = axes.flatten()
    plot_titles = list(fields_to_plot.keys())

    for i in range(len(all_axes)):
        ax = all_axes[i]
        if i < len(plot_titles):
            title = plot_titles[i]
            field = fields_to_plot[title]

            # Check if there are enough points to create a triangulation
            if np.sum(~np.isnan(field)) >= 3:
                triang = tri.Triangulation(x_coords, y_coords)
                if triang.triangles.shape[0] > 0:
                    contour = ax.tricontourf(triang, field, cmap="viridis", levels=15)
                    fig.colorbar(contour, ax=ax)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No triangles in triangulation",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )
            else:  # Handle cases where all data is masked or too few points
                ax.text(
                    0.5,
                    0.5,
                    "Not enough data to plot",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal", "box")
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    # Construct filename with the new directory
    filename = output_dir / f"mms_results_{mesh_label}_{filename_suffix}.png"
    plt.savefig(filename)
    print(f"Saved MMS plot: {filename}")
    plt.close(fig)  # Close the figure to free memory


# --- End Helper Plotting Function ---


def test_assemble_momentum_matrix_structure(mesh_instance, mesh_label):
    mesh = mesh_instance
    num_cells = len(mesh.cell_volumes)
    # Dummy velocity field: constant
    u = np.ones(num_cells)
    # Dummy gradient: sinusoidal
    grad_u = np.zeros((num_cells, 2))
    for i, x in enumerate(mesh.cell_centers):
        grad_u[i, 0] = np.sin(np.pi * x[0])
        grad_u[i, 1] = np.cos(np.pi * x[1])
    rho = 1.0
    mu = 0.01
    a_P, a_N_list, b_P = assemble_momentum_matrix(u, grad_u, mesh, rho, mu)
    # Assert main diagonal length
    assert len(a_P) == num_cells
    # Assert off-diagonal structure: list of (P, N, value)
    assert isinstance(a_N_list, list)
    for entry in a_N_list:
        assert isinstance(entry, tuple)
        assert len(entry) == 3
        P, N, value = entry
        assert isinstance(P, (int, np.integer))
        assert isinstance(N, (int, np.integer))
    # Assert source term length
    assert len(b_P) == num_cells
    # Assert main diagonal not all zeros
    assert np.any(np.abs(a_P) > 0)


def test_assemble_momentum_matrix_mms(mesh_instance, mesh_label):
    mesh = mesh_instance
    num_cells = len(mesh.cell_volumes)
    rho = 1.0
    mu = 0.01

    # Manufactured solution: u(x, y) = sin(pi x) * sin(pi y)
    u_exact = np.zeros(num_cells)
    grad_u_exact = np.zeros((num_cells, 2))

    for i, x in enumerate(mesh.cell_centers):
        u_exact[i] = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        grad_u_exact[i, 0] = np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
        grad_u_exact[i, 1] = np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])

    # Assemble matrix and source
    a_P, a_N_list, b_P = assemble_momentum_matrix(u_exact, grad_u_exact, mesh, rho, mu)

    # Apply matrix to exact solution
    Ax = np.copy(b_P)
    for P, N, val in a_N_list:
        Ax[P] += val * u_exact[N]
    Ax += a_P * u_exact

    # Discrete RHS using FV operators for direct comparison
    expected_rhs_integrated = compute_discrete_rhs(u_exact, grad_u_exact, mesh, rho, mu)

    # Plot results before assertion
    plot_mms_results(
        mesh,
        mesh_label,
        "ConvectionDiffusion",
        u_exact,
        Ax,
        expected_rhs_integrated,
        "conv_diff_mms_volscaled",
    )
