import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.assembly.momentumMatrix import assemble_diffusion_convection_matrix
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients


# New test: Lid-driven cavity
def test_momentum_solver_lid_driven(mesh, u_field, mu=0.01, rho=1.0):
    """
    Tests the momentum matrix assembly for lid-driven cavity (top wall drives flow).
    Tests both u and v components.
    """
    print("[Test] Lid-driven cavity momentum solve")

    n_cells = mesh.cell_centers.shape[0]
    phi_full = np.zeros((n_cells, 2))

    for comp_idx, comp_name in enumerate(['u', 'v']):

           # Extract the specific component as a scalar field before passing to gradient calculation
        phi = u_field[:, comp_idx]
        grad_phi = compute_cell_gradients(mesh, u_field[:, comp_idx])

        row, col, data, b_correction = assemble_diffusion_convection_matrix(
            mesh, phi, grad_phi, u_field,
            rho, mu, comp_idx
        ) 

        A = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        rhs = b_correction   # no body force

        phi_numeric = spsolve(A, rhs)
        phi_full[:, comp_idx] = phi_numeric

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Check if mesh is structured by attempting to form a square grid
        n = int(np.sqrt(len(phi_numeric)))
        if n*n == len(phi_numeric):  # Structured mesh
            # Reshape data into 2D grid for matshow
            phi_2d = phi_numeric.reshape(n, n)
            phi_2d = np.rot90(phi_2d)  # Rotate 90 degrees counterclockwise
            im = ax.matshow(phi_2d, cmap='viridis', aspect='equal')
            ax.set_xlabel('y')
            ax.set_ylabel('x')
        else:  # Unstructured mesh
            # Create tricontour plot colored by velocity values
            triang = plt.matplotlib.tri.Triangulation(mesh.cell_centers[:, 0],
                                                    mesh.cell_centers[:, 1])
            im = ax.tricontourf(triang, phi_numeric,
                              levels=20,
                              cmap='viridis')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            
        plt.colorbar(im, ax=ax, label=f'{comp_name} velocity')
        ax.set_title(f'Lid-Driven: Velocity Field ({comp_name}-component)')
        plt.tight_layout()
        plt.savefig(f"tests/test_output/test_momentum_lid_driven_{comp_name}.png", dpi=300)
        plt.close()

        # Checks
        assert not np.any(np.isnan(phi_numeric)), f"NaNs in {comp_name}-velocity!"
        if comp_name == 'u':
            assert np.max(phi_numeric) > 0.1, "No flow induced by lid in u-component!"
        print(f"[OK] Lid-driven flow computed for {comp_name}. max = {np.max(phi_numeric):.3e}")
    return phi_full
"""
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
"""

def lid_top_wall_flow(mesh):
    u_field = np.zeros((mesh.cell_centers.shape[0], 2))
    u_field[:, 0] = 1.0
    return u_field



if __name__ == "__main__":
    Re = 1000
    mu = 1.0 / Re
    rho = 1
    mesh = load_mesh("meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh", "shared_configs/domain/boundaries_lid_driven_cavity.yaml")
    #mesh = load_mesh("meshing/experiments/lidDrivenCavity/unstructured/fine/lidDrivenCavity_unstructured_fine.msh", "shared_configs/domain/boundaries_lid_driven_cavity.yaml")

    n_cells = mesh.cell_centers.shape[0]
    u_field_dummy = np.zeros((n_cells, 2))  
    u_field_dummy[:, 0] = 1.0
    u_field_dummy[:, 1] = 0.0


    #test_momentum_solver_lid_driven(mesh_file, bc_file, mu, rho)
    #mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh"
    phi = u_field_dummy
    _ = test_momentum_solver_lid_driven(mesh, phi, mu, rho)


"""
    plot_upwind_directions(
        #"meshing/experiments/lidDrivenCavity/unstructured/coarse/lidDrivenCavity_unstructured_coarse.msh",

        "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh",
        "shared_configs/domain/sanityCheckMomentum.yaml",
        lid_top_wall_flow
    )
"""