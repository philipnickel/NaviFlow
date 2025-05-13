import numpy as np
import os
from utils.plot_style import plt
from naviflow_collocated.discretization.gradient.leastSquares import (
    compute_cell_gradients,
)


def test_gradient_mms(mesh_instance, mesh_label):
    """
    MMS test for least-squares gradient on u(x, y) = sin(pi x) * sin(pi y)
    """
    n_cells = mesh_instance.cell_volumes.shape[0]
    # only run test on large meshes
    if n_cells < 1000:
        return

    x = mesh_instance.cell_centers[:, 0]
    y = mesh_instance.cell_centers[:, 1]

    u = np.sin(np.pi * x) * np.sin(np.pi * y)
    grad_u_exact = np.zeros((len(u), 2))
    grad_u_exact[:, 0] = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    grad_u_exact[:, 1] = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)

    grad_u_computed = compute_cell_gradients(mesh_instance, u)

    err = grad_u_computed - grad_u_exact
    rel_err = np.linalg.norm(err) / (np.linalg.norm(grad_u_exact) + 1e-14)

    os.makedirs("tests/test_output", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].quiver(
        x,
        y,
        grad_u_exact[:, 0],
        grad_u_exact[:, 1],
        color="blue",
        alpha=0.6,
        label="Exact",
    )
    axs[0].quiver(
        x,
        y,
        grad_u_computed[:, 0],
        grad_u_computed[:, 1],
        color="red",
        alpha=0.6,
        label="Computed",
    )
    axs[0].set_title("Gradient Field")
    axs[0].set_aspect("equal")
    axs[0].legend()

    err_mag = np.linalg.norm(err, axis=1)
    sc = axs[1].scatter(x, y, c=err_mag, cmap="viridis", s=8)
    axs[1].set_title("Gradient Error Magnitude")
    axs[1].set_aspect("equal")
    plt.colorbar(sc, ax=axs[1])

    plt.tight_layout()
    os.makedirs("tests/test_output/gradients", exist_ok=True)
    plt.savefig(f"tests/test_output/gradients/gradient_mms_{mesh_label}.png", dpi=300)
    plt.close()

    assert rel_err < 3e-1, (
        f"Gradient MMS failed for {mesh_label}: rel_error = {rel_err:.2e}"
    )
