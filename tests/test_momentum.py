import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from petsc4py import PETSc
import matplotlib.pyplot as plt
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.assembly.momentumMatrix import assemble_diffusion_convection_matrix
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients
import os


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
            mesh, grad_phi, u_field,
            rho, mu, comp_idx, phi, beta=1.0
        )

        A = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        rhs = b_correction   # no body force

        # PETSc solve without MPI: local sequential solve using shared memory
        A_petsc = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        A_petsc.assemble()
        b_petsc = PETSc.Vec().createWithArray(rhs)
        x_petsc = PETSc.Vec().createSeq(n_cells)
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setType("bcgs")
        ksp.getPC().setType("hypre")
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)
        ksp.setFromOptions()
        ksp.solve(b_petsc, x_petsc)
        phi_numeric = x_petsc.getArray()
        phi_full[:, comp_idx] = phi_numeric


        # Checks
        assert not np.any(np.isnan(phi_numeric)), f"NaNs in {comp_name}-velocity!"
        print(f"[OK] Lid-driven flow computed for {comp_name}. max = {np.max(phi_numeric):.3e}")
    return phi_full



if __name__ == "__main__":
    Re = 100
    mu = 1.0 / Re
    print("mu = ", mu)
    rho = 1.0
    #mesh = load_mesh("meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh", "shared_configs/domain/boundaries_lid_driven_cavity.yaml")
    mesh = load_mesh("meshing/experiments/lidDrivenCavity/unstructured/fine/lidDrivenCavity_unstructured_fine.msh", "shared_configs/domain/boundaries_lid_driven_cavity.yaml")

    n_cells = mesh.cell_centers.shape[0]
    u_field_dummy = np.zeros((n_cells, 2))  


    #test_momentum_solver_lid_driven(mesh_file, bc_file, mu, rho)
    #mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh"
    from numba import get_num_threads
    print(get_num_threads())
    import os
    os.environ["NUMBA_DEBUG_ARRAY_OPT_STATS"] = "1"
    phi_it = u_field_dummy
    for i in range(2):
        phi_it = test_momentum_solver_lid_driven(mesh, phi_it, mu, rho)
        continue

    velocity_magnitude = np.sqrt(phi_it[:, 0]**2 + phi_it[:, 1]**2)


    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Check if mesh is structured by attempting to form a square grid
    n = int(np.sqrt(len(phi_it)))
    if n*n == len(phi_it):  # Structured mesh
        # Reshape data into 2D grid for matshow
        phi_2d = phi_it.reshape(n, n)
        phi_2d = np.rot90(phi_2d)  # Rotate 90 degrees counterclockwise
        im = ax.matshow(phi_2d, cmap='viridis', aspect='equal')
        ax.set_xlabel('y')
        ax.set_ylabel('x')
    else:  # Unstructured mesh
        # Create tricontour plot colored by velocity values
        triang = plt.matplotlib.tri.Triangulation(mesh.cell_centers[:, 0],
                                                mesh.cell_centers[:, 1])
        im = ax.tricontourf(triang, velocity_magnitude,
                            levels=20,
                            cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        
    plt.colorbar(im, ax=ax, label=f'velocity')
    ax.set_title(f'Lid-Driven: Velocity Field')
    plt.tight_layout()
    plt.savefig(f"tests/test_output/test_momentum_lid_driven_u.png", dpi=300)
    plt.close()
