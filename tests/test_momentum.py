import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.assembly.momentumMatrix import assemble_diffusion_convection_matrix
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients


# New test: Lid-driven cavity
def test_momentum_solver_lid_driven(mesh_file, bc_file, mu=0.01, rho=1.0):
    """
    Tests the momentum matrix assembly for lid-driven cavity (top wall drives flow).
    Tests both u and v components.
    """
    print("[Test] Lid-driven cavity momentum solve")

    mesh = load_mesh(mesh_file, bc_file)
    n_cells = mesh.cell_centers.shape[0]

    for comp_idx, comp_name in enumerate(['u', 'v']):
        p_field = np.zeros(n_cells)  # No pressure gradient
        grad_p = np.zeros((n_cells, 2))

        phi_dummy = np.zeros(n_cells)
        grad_dummy = np.zeros((n_cells, 2))
        u_field_dummy = np.zeros((n_cells, 2))  # unused

        row, col, data, b = assemble_diffusion_convection_matrix(
            mesh, phi_dummy, grad_dummy, rho=rho, mu=mu,
            u_field=u_field_dummy, component_idx=comp_idx
        )

        A = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        rhs = b  # no body force

        phi_numeric = spsolve(A, rhs)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1],
                        c=phi_numeric, cmap='viridis', s=35, edgecolor='k')
        plt.colorbar(sc, ax=ax, label=f'{comp_name} velocity')
        ax.set_title(f'Lid-Driven: Velocity Field ({comp_name}-component)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f"tests/test_output/test_momentum_lid_driven_{comp_name}.png", dpi=300)
        plt.close()

        # Checks
        assert not np.any(np.isnan(phi_numeric)), f"NaNs in {comp_name}-velocity!"
        if comp_name == 'u':
            assert np.max(phi_numeric) > 0.1, "No flow induced by lid in u-component!"
        print(f"[OK] Lid-driven flow computed for {comp_name}. Max {comp_name} = {np.max(phi_numeric):.3e}")


# Example usage
if __name__ == "__main__":
    mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/medium/lidDrivenCavity_uniform_medium.msh"
    bc_file = "shared_configs/domain/sanityCheckMomentum.yaml"  # BCs: u = 0 on all walls
    test_momentum_solver_lid_driven(mesh_file, bc_file)
    mesh_file = "meshing/experiments/lidDrivenCavity/unstructured/medium/lidDrivenCavity_unstructured_medium.msh"
    bc_file = "shared_configs/domain/sanityCheckMomentum.yaml"  # BCs: u = 0 on all walls
    test_momentum_solver_lid_driven(mesh_file, bc_file)
