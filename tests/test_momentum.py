import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.assembly.momentumMatrix import assemble_diffusion_convection_matrix
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients


# New test: Lid-driven cavity
def test_momentum_solver_lid_driven(mesh_file, u_field, bc_file, mu=0.01, rho=1.0):
    """
    Tests the momentum matrix assembly for lid-driven cavity (top wall drives flow).
    Tests both u and v components.
    """
    print("[Test] Lid-driven cavity momentum solve")

    mesh = load_mesh(mesh_file, bc_file)
    n_cells = mesh.cell_centers.shape[0]
    phi_full = np.zeros((n_cells, 2))

    for comp_idx, comp_name in enumerate(['u', 'v']):

           # Extract the specific component as a scalar field before passing to gradient calculation
        scalar_field = u_field[:, comp_idx]
        grad_component = compute_cell_gradients(mesh, scalar_field)

        row, col, data, b = assemble_diffusion_convection_matrix(
            mesh=mesh, 
            grad_phi=grad_component, 
            rho=rho, 
            mu=mu,
            u_field=u_field, 
            component_idx=comp_idx
        )

        A = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        rhs = b   # no body force

        phi_numeric = spsolve(A, rhs)
        phi_full[:, comp_idx] = phi_numeric

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
        print(f"[OK] Lid-driven flow computed for {comp_name}. max = {np.max(phi_numeric):.3e}")
    return phi_full

def plot_upwind_directions(mesh_file, bc_file, u_field_fn, title="Upwind Direction Visualization"):
    mesh = load_mesh(mesh_file, bc_file)
    u_field = u_field_fn(mesh)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect('equal')

    green_count = 0
    red_count = 0

    for f in mesh.internal_faces:
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]
        S_f = mesh.vector_S_f[f]
        alpha = mesh.face_interp_factors[f]
        u_f = (1 - alpha) * u_field[P] + alpha * u_field[N]
        flux = np.dot(u_f, S_f)

        x, y = mesh.face_centers[f]

        dx, dy = u_f / (np.linalg.norm(u_f) + 1e-12) * 0.02

        color = 'green' if flux >= 0 else 'red'
        if flux >= 0:
            green_count += 1
        else:
            red_count += 1
        ax.arrow(x - dx / 2, y - dy / 2, dx, dy, head_width=0.01, color=color, length_includes_head=True)

    print(f"Green arrows: {green_count}, Red arrows: {red_count}")

    ax.scatter(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], c='black', s=5, label='Cell Centers')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("tests/test_output/upwind_face_directions.png", dpi=300)
    plt.close()

def lid_top_wall_flow(mesh):
    u_field = np.zeros((mesh.cell_centers.shape[0], 2))
    u_field[:, 0] = 1.0
    return u_field



# Example usage
if __name__ == "__main__":
    Re = 100
    mu = 1.0 / Re
    rho = 1
    mesh = load_mesh("meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh", "shared_configs/domain/sanityCheckMomentum.yaml")

    n_cells = mesh.cell_centers.shape[0]
    u_field_dummy = np.zeros((n_cells, 2))  # unused
    u_field_dummy[:, 0] = 1.0


    mesh_file = "meshing/experiments/lidDrivenCavity/unstructured/medium/lidDrivenCavity_unstructured_medium.msh"
    bc_file = "shared_configs/domain/sanityCheckMomentum.yaml"  # BCs: u = 1 on top wall 0 on all other walls
    #test_momentum_solver_lid_driven(mesh_file, bc_file, mu, rho)
    mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh"
    bc_file = "shared_configs/domain/sanityCheckMomentum.yaml"  # BCs: u = 1 on top wall 0 on all other walls
    phi = u_field_dummy
    for i in range(1):
        phi = test_momentum_solver_lid_driven(mesh_file, phi, bc_file, mu, rho)


"""
    plot_upwind_directions(
        #"meshing/experiments/lidDrivenCavity/unstructured/coarse/lidDrivenCavity_unstructured_coarse.msh",

        "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh",
        "shared_configs/domain/sanityCheckMomentum.yaml",
        lid_top_wall_flow
    )
"""