import numpy as np
import matplotlib.pyplot as plt
from naviflow_oo.preprocessing.mesh.structured_mesh import StructuredMesh

def verify_structured_mesh():
    mesh = StructuredMesh(
        n_cells_x=5,
        n_cells_y=5,
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1,
        is_uniform=True
    )

    # --- Basic shape consistency ---
    assert mesh.n_cells == (mesh.nx - 1) * (mesh.ny - 1), "Incorrect cell count"
    assert mesh.n_faces == len(mesh.get_owner_neighbor()[0]), "Face count mismatch"
    assert mesh.n_nodes == mesh.nx * mesh.ny, "Incorrect node count"

    # --- Owner/neighbor checks ---
    owners, neighbors = mesh.get_owner_neighbor()
    for i, (own, nei) in enumerate(zip(owners, neighbors)):
        assert not (own == -1 and nei == -1), f"Face {i} has no owner or neighbor"
        if own != -1 and nei != -1:
            assert own != nei, f"Face {i} has identical owner and neighbor"

    # --- Boundary indexing ---
    for side in ['left', 'right', 'top', 'bottom']:
        idx = mesh.get_boundary_cell_indices(side)
        assert idx is not None and len(idx) > 0, f"Boundary '{side}' missing cell indices"

    # --- Face normals: existence, unit length, finite values ---
    normals = mesh.get_face_normals()
    assert normals.shape == (mesh.n_faces, 2), "Normals shape mismatch"
    norms = np.linalg.norm(normals, axis=1)
    for i, n in enumerate(norms):
        assert np.all(np.isfinite(normals[i])), f"Face {i} normal has NaN or Inf"
        if n > 1e-10:
            assert np.isclose(n, 1.0, atol=1e-6), f"Face {i} normal not unit length: {n}"
        else:
            raise AssertionError(f"Face {i} normal is zero (likely missing or wrong)")

    # --- Face areas ---
    areas = mesh.get_face_areas()
    assert areas.shape == (mesh.n_faces,), "Face area shape mismatch"
    assert np.all(np.isfinite(areas)), "Face areas contain NaN or Inf"
    assert np.all(areas >= 0), "Negative face area found"

    # --- Cell volumes ---
    volumes = mesh.get_cell_volumes()
    assert volumes.shape == (mesh.n_cells,), "Cell volume shape mismatch"
    assert np.all(np.isfinite(volumes)), "Cell volumes contain NaN or Inf"
    assert np.all(volumes > 0), "Zero or negative cell volume found"

    # --- Cell centers in bounds ---
    cell_centers = mesh.get_cell_centers()
    assert np.all(np.isfinite(cell_centers)), "Cell centers contain NaN or Inf"
    assert np.all((cell_centers[:, 0] >= mesh.xmin) & (cell_centers[:, 0] <= mesh.xmax)), "Cell center X out of bounds"
    assert np.all((cell_centers[:, 1] >= mesh.ymin) & (cell_centers[:, 1] <= mesh.ymax)), "Cell center Y out of bounds"

    # --- Face centers in bounds ---
    face_centers = mesh.get_face_centers()
    assert np.all(np.isfinite(face_centers)), "Face centers contain NaN or Inf"
    assert np.all((face_centers[:, 0] >= mesh.xmin) & (face_centers[:, 0] <= mesh.xmax)), "Face center X out of bounds"
    assert np.all((face_centers[:, 1] >= mesh.ymin) & (face_centers[:, 1] <= mesh.ymax)), "Face center Y out of bounds"

    # --- Interpolation weights must be consistent ---
    for face_idx in range(mesh.n_faces):
        g_C, g_F = mesh.get_face_interpolation_factors(face_idx)
        assert np.isfinite(g_C) and np.isfinite(g_F), f"NaN/Inf in interpolation at face {face_idx}"
        assert np.isclose(g_C + g_F, 1.0, atol=1e-6), f"g_C + g_F != 1 at face {face_idx}"

    # --- Node positions: finite and in bounds ---
    node_coords = mesh.get_node_positions()
    assert np.all(np.isfinite(node_coords)), "Node coordinates contain NaN or Inf"
    assert np.all((node_coords[:, 0] >= mesh.xmin) & (node_coords[:, 0] <= mesh.xmax)), "Node X out of bounds"
    assert np.all((node_coords[:, 1] >= mesh.ymin) & (node_coords[:, 1] <= mesh.ymax)), "Node Y out of bounds"

    # --- Face distances: positive, finite ---
    face_dists = mesh.get_face_distances()
    assert np.all(np.isfinite(face_dists)), "Face distances contain NaN or Inf"
    assert np.all(face_dists > 0), "Zero or negative face-to-cell distance found"

    # --- Optional: Cell overlap / duplicate detection (degeneracy) ---
    unique_cell_centers = np.unique(cell_centers.round(decimals=10), axis=0)
    assert len(unique_cell_centers) == mesh.n_cells, "Duplicate cell centers detected"

    # --- Optional: Check consistency between cell count and returned arrays ---
    assert mesh.get_cell_centers().shape[0] == mesh.n_cells
    assert mesh.get_cell_volumes().shape[0] == mesh.n_cells

    # --- Visualization (optional) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh.plot(ax=ax, title="Mesh Verification")
    ax.quiver(face_centers[:, 0], face_centers[:, 1], normals[:, 0], normals[:, 1],
              color='r', label='Normals')
    for i, (x, y) in enumerate(face_centers):
        ax.text(x, y, str(i), color='blue', fontsize=8)
    for i, (x, y) in enumerate(cell_centers):
        ax.text(x, y, f'C{i}', color='green', fontsize=8)
    ax.legend()
    plt.show()

    print("✅ Mesh verification complete — all tests passed.")

if __name__ == '__main__':
    verify_structured_mesh()
