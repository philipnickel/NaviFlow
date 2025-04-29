import numpy as np
import matplotlib.pyplot as plt
from naviflow_oo.preprocessing.mesh.unstructured import UnstructuredUniform

def verify_unstructured_mesh():
    MESH_SIZE_CENTER_UNIFORM = 0.5
    XMIN = 0
    XMAX = 1
    YMIN = 0
    YMAX = 1
    mesh= UnstructuredUniform(
        mesh_size=MESH_SIZE_CENTER_UNIFORM,
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX
    )

    # --- Sanity checks ---
    assert mesh.n_cells > 0, "Mesh has no cells"
    assert mesh.n_faces > 0, "Mesh has no faces"
    assert mesh.n_nodes > 0, "Mesh has no nodes"

    # --- Owner/Neighbor relations ---
    owners, neighbors = mesh.get_owner_neighbor()
    for i, (own, nei) in enumerate(zip(owners, neighbors)):
        assert not (own == -1 and nei == -1), f"Face {i} has no owner or neighbor"
        if own != -1 and nei != -1:
            assert own != nei, f"Face {i} has identical owner and neighbor"

    # --- Normals ---
    normals = mesh.get_face_normals()
    assert normals.shape == (mesh.n_faces, 2), "Normals shape mismatch"
    norms = np.linalg.norm(normals, axis=1)
    for i, n in enumerate(norms):
        assert np.all(np.isfinite(normals[i])), f"Face {i} normal has NaN/Inf"
        if n > 1e-10:
            assert np.isclose(n, 1.0, atol=1e-6), f"Face {i} normal not unit length: {n}"
        else:
            # Allow zero norm for degenerate faces if necessary, but flag it.
            # Or raise AssertionError if zero-length faces are strictly forbidden.
            print(f"Warning: Face {i} normal has near-zero length: {n}")
            # raise AssertionError(f"Face {i} normal is zero (likely missing or wrong)")

    # --- Areas & Volumes ---
    areas = mesh.get_face_areas()
    volumes = mesh.get_cell_volumes()
    assert np.all(np.isfinite(areas)) and np.all(areas >= 0), "Invalid face areas"
    assert np.all(np.isfinite(volumes)) and np.all(volumes > 0), "Invalid cell volumes"

    # --- Centers ---
    cell_centers = mesh.get_cell_centers()
    face_centers = mesh.get_face_centers()
    node_coords = mesh.get_node_positions()
    for arr, name in [(cell_centers, "cell centers"), (face_centers, "face centers"), (node_coords, "node coords")]:
        assert np.all(np.isfinite(arr)), f"{name} contains NaN/Inf"

    # --- Check Normal Direction ---
    # Ensure normals point outward from owner cell (or handle convention)
    tol = 1e-9 # Tolerance for dot product check
    for i in range(mesh.n_faces):
        owner = owners[i]
        if owner != -1:
            vec_owner_to_face = face_centers[i] - cell_centers[owner]
            # Check if vector is near zero (face center coincides with cell center)
            if np.linalg.norm(vec_owner_to_face) > tol:
                 assert np.dot(normals[i], vec_owner_to_face) >= -tol, (
                     f"Face {i} normal {normals[i]} not pointing outward from owner {owner}. "
                     f"Vector O->F: {vec_owner_to_face}, Dot: {np.dot(normals[i], vec_owner_to_face)}"
                 )

    # --- Interpolation weights ---
    for face_idx in range(mesh.n_faces):
        g_C, g_F = mesh.get_face_interpolation_factors(face_idx)
        assert np.isfinite(g_C) and np.isfinite(g_F), f"Face {face_idx} interpolation has NaN/Inf"
        assert np.isclose(g_C + g_F, 1.0, atol=1e-6), f"Face {face_idx} interpolation weights do not sum to 1"

    # --- Face distances ---
    face_dists = mesh.get_face_distances()
    assert np.all(np.isfinite(face_dists))
    assert np.all(face_dists > 0)

    # --- Duplicate cells ---
    unique_cell_centers = np.unique(cell_centers.round(decimals=10), axis=0)
    assert len(unique_cell_centers) == mesh.n_cells, "Duplicate cell centers detected"

    # --- Hanging Faces Check ---
    # Ensure each face connects at least two nodes
    for i, face_nodes in enumerate(mesh._faces): # Accessing private member for debugging
        assert len(face_nodes) >= 2, f"Face {i} has insufficient node connectivity (Nodes: {face_nodes})"

    # --- Orphan Faces Check ---
    # Ensure all faces are referenced by at least one cell
    referenced_faces = set()
    for cell_faces in mesh._cells: # Accessing private member for debugging
        referenced_faces.update(cell_faces)
    assert len(referenced_faces) == mesh.n_faces, (
        f"Mismatch between number of faces ({mesh.n_faces}) and referenced faces ({len(referenced_faces)}). "
        f"Orphan faces might exist."
    )

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh.plot(ax=ax, title="Unstructured Mesh Verification")
    ax.quiver(face_centers[:, 0], face_centers[:, 1], normals[:, 0], normals[:, 1], color='r', label='Normals')
    for i, (x, y) in enumerate(face_centers):
        ax.text(x, y, str(i), color='blue', fontsize=8)
    for i, (x, y) in enumerate(cell_centers):
        ax.text(x, y, f'C{i}', color='green', fontsize=8)
    ax.legend()
    plt.show()

    print("✅ Unstructured mesh verification complete — all tests passed.")

if __name__ == '__main__':
    verify_unstructured_mesh()