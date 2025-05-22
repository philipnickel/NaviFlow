import numpy as np
import os
from utils.plot_style import plt
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.discretization.gradient.leastSquares import (
    compute_cell_gradients,
)
#from naviflow_collocated.discretization.gradient.gauss import (
#    compute_cell_gradients,
#k)


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

    u = np.zeros(len(x))#np.sin(4*np.pi * (x + y)) + np.cos(4 * np.pi * x * y) 
    grad_u_exact = np.zeros((len(u), 2))
    #grad_u_exact[:, 0] = 4 * np.pi * (-y * np.sin(np.pi * x * y) + np.cos (np.pi * (4*x + 4*y)))
    #grad_u_exact[:, 1] = 4 * np.pi * (-x * np.sin(np.pi * x * y) + np.cos (np.pi * (4*x + 4*y)))

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
    axs[1].set_title(f"Gradient Error Magnitude")
    axs[1].set_aspect("equal")
    plt.colorbar(sc, ax=axs[1])

    plt.tight_layout()
    os.makedirs("tests/test_output/gradients", exist_ok=True)
    plt.savefig(f"tests/test_output/gradients/gradient_mms_{mesh_label}.png", dpi=300)
    plt.close()

    #assert rel_err < 3e-1, (
    #    f"Gradient MMS failed for {mesh_label}: rel_error = {rel_err:.2e}"
    #)


mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh" 
bc_file = "shared_configs/domain/boundaries_lid_driven_cavity.yaml" 
mesh = load_mesh(mesh_file, bc_file)

def test_gradient_mms(mesh):
    """
    MMS test for least-squares gradient on u(x, y) = sin(pi x) * sin(pi y)
    """
    n_cells = mesh.cell_volumes.shape[0]
    # only run test on large meshes

    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]

    u = np.zeros(len(x))#np.sin(4*np.pi * (x + y)) + np.cos(4 * np.pi * x * y) 
    grad_u_exact = np.zeros((len(u), 2))
    #grad_u_exact[:, 0] = 4 * np.pi * (-y * np.sin(np.pi * x * y) + np.cos (np.pi * (4*x + 4*y)))
    #grad_u_exact[:, 1] = 4 * np.pi * (-x * np.sin(np.pi * x * y) + np.cos (np.pi * (4*x + 4*y)))

    grad_u_computed = compute_cell_gradients(mesh, u)

    print(f"Max grad_u_computed: {np.max(grad_u_computed)}")
    print(f"Min grad_u_computed: {np.min(grad_u_computed)}")
    print(f"grad_u_computed: {grad_u_computed}")
    # check for any None
    if np.any(grad_u_computed == None):
        print("None found in grad_u_computed")
    else:
        print("No None found in grad_u_computed")

    err = grad_u_computed - grad_u_exact
    print(f"L2 of error: {np.linalg.norm(err)}")
    
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
    axs[1].set_title(f"Gradient Error Magnitude")
    axs[1].set_aspect("equal")
    plt.colorbar(sc, ax=axs[1])

    plt.tight_layout()
    os.makedirs("tests/test_output/gradients", exist_ok=True)
    plt.savefig(f"tests/test_output/gradients/gradient_mms.png", dpi=300)
    plt.close()

    #assert rel_err < 3e-1, (
    #    f"Gradient MMS failed for {mesh_label}: rel_error = {rel_err:.2e}"
    #)

if __name__ == "__main__":
    test_gradient_mms(mesh)
