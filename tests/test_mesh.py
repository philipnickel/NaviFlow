import numpy as np
import os
import matplotlib.pyplot as plt
from naviflow_collocated.mesh.mesh_data import MeshData2D


def test_basic_mesh_integrity(mesh_instance):
    mesh = mesh_instance
    assert isinstance(mesh, MeshData2D)

    n_cells = mesh.cell_volumes.shape[0]
    n_faces = mesh.face_areas.shape[0]

    # Core shape checks
    assert mesh.cell_centers.shape == (n_cells, 2)
    assert mesh.face_centers.shape == (n_faces, 2)
    assert mesh.face_normals.shape == (n_faces, 2)
    assert mesh.owner_cells.shape[0] == n_faces
    assert mesh.neighbor_cells.shape[0] == n_faces

    # Physical quantities
    assert np.all(mesh.cell_volumes > 0), "All cell volumes should be > 0"
    assert np.all(mesh.face_areas > 0), "All face areas should be > 0"

    # Connectivity validity
    assert np.all(mesh.owner_cells >= 0)
    assert np.all(mesh.neighbor_cells[mesh.boundary_faces] == -1)
    assert np.all((mesh.face_interp_factors >= 0) & (mesh.face_interp_factors <= 1))

    # Structured-specific
    if mesh.is_structured:
        assert mesh.is_orthogonal, "Structured meshes must be flagged orthogonal"


def test_face_cell_symmetry(mesh_instance):
    mesh = mesh_instance
    n_faces = mesh.face_areas.shape[0]
    counts = np.zeros(n_faces, dtype=np.int32)
    for face_list in mesh.cell_faces:
        for f in face_list:
            if f >= 0:
                counts[f] += 1
    assert np.all((counts == 1) | (counts == 2)), (
        "Each face must belong to 1 or 2 cells"
    )


def test_unique_face_vertices(mesh_instance):
    mesh = mesh_instance
    face_keys = [tuple(sorted(face.tolist())) for face in mesh.face_vertices]
    assert len(set(face_keys)) == len(face_keys), "Duplicate face vertices detected"


def test_face_normal_orientation(mesh_instance):
    mesh = mesh_instance
    internal = mesh.neighbor_cells >= 0
    P = mesh.owner_cells[internal]
    N = mesh.neighbor_cells[internal]
    vec = mesh.cell_centers[N] - mesh.cell_centers[P]
    dot = np.einsum("ij,ij->i", vec, mesh.face_normals[internal])
    assert np.all(dot > 0), "Some face normals do not point from owner to neighbor"


def test_delta_PN_matches_geometry(mesh_instance):
    mesh = mesh_instance
    mask = mesh.neighbor_cells >= 0
    P = mesh.owner_cells[mask]
    N = mesh.neighbor_cells[mask]
    expected = np.linalg.norm(mesh.cell_centers[N] - mesh.cell_centers[P], axis=1)
    assert np.allclose(mesh.delta_PN[mask], expected, rtol=1e-3), "Mismatch in delta_PN"


def test_non_ortho_projection_decomposition(mesh_instance):
    mesh = mesh_instance
    internal = mesh.neighbor_cells >= 0
    vec_pn = mesh.d_PN[internal]
    n_hat = mesh.face_normals[internal] / (
        np.linalg.norm(mesh.face_normals[internal], axis=1)[:, None] + 1e-12
    )
    proj_len = np.einsum("ij,ij->i", vec_pn, n_hat)
    t_f = mesh.non_ortho_correction[internal]
    reconstructed = proj_len[:, None] * n_hat + t_f
    assert np.allclose(vec_pn, reconstructed, rtol=1e-6), (
        "Projection decomposition failed"
    )


def test_structured_mesh_alignment(mesh_instance):
    mesh = mesh_instance
    if mesh.is_structured:
        aligned = np.einsum("ij,ij->i", mesh.face_normals, mesh.e_f) / (
            mesh.face_areas + 1e-12
        )
        assert np.allclose(aligned, 1.0, atol=1e-2), (
            "Structured mesh has misaligned normals"
        )


def test_boundary_patch_consistency(mesh_instance):
    mesh = mesh_instance
    assert mesh.boundary_faces.shape[0] == mesh.boundary_patches.shape[0], (
        "Mismatch in boundary face/patch count"
    )
    assert mesh.boundary_types.shape[0] == mesh.boundary_faces.shape[0], (
        "Mismatch in boundary face/type count"
    )
    assert mesh.boundary_values.shape[0] == mesh.boundary_faces.shape[0], (
        "Mismatch in boundary face/value shape"
    )


def test_graph_connectivity_and_plot(mesh_instance):
    mesh = mesh_instance
    os.makedirs("tests/test_output", exist_ok=True)
    mesh_type = "structured" if mesh.is_structured else "unstructured"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax1, ax2 = axes

    face_c = mesh.face_centers
    face_n = mesh.face_normals
    face_v = mesh.face_vertices
    cell_c = mesh.cell_centers
    boundary_mask = np.zeros(len(face_c), dtype=bool)
    boundary_mask[mesh.boundary_faces] = True

    # === Subplot 1: Mesh geometry with real face edges and face normals ===
    ax1.set_title("Mesh Geometry and Face Normals")
    ax1.set_aspect("equal")

    # Plot actual faces as lines between vertices
    for i, verts in enumerate(face_v):
        v0, v1 = mesh.vertices[verts[0]], mesh.vertices[verts[1]]
        color = "red" if boundary_mask[i] else "gray"
        ax1.plot([v0[0], v1[0]], [v0[1], v1[1]], color=color, lw=1.2)

    # Cell centers
    ax1.scatter(cell_c[:, 0], cell_c[:, 1], s=10, color="blue", label="Cell Centers")

    # Face centers
    ax1.scatter(
        face_c[~boundary_mask, 0],
        face_c[~boundary_mask, 1],
        s=10,
        color="gray",
        label="Internal Faces",
    )
    ax1.scatter(
        face_c[boundary_mask, 0],
        face_c[boundary_mask, 1],
        s=20,
        color="red",
        label="Boundary Faces",
    )

    # Scaled face normals for visibility
    arrow_len = 0.1 * np.mean(mesh.face_areas)
    norms = np.linalg.norm(face_n, axis=1)[:, None] + 1e-12
    face_n_vis = (face_n / norms) * arrow_len
    ax1.quiver(
        face_c[:, 0],
        face_c[:, 1],
        face_n_vis[:, 0],
        face_n_vis[:, 1],
        color="black",
        scale=0.5,
        width=0.002,
        alpha=0.5,
        label="Face Normals",
    )

    # Label faces with face ID and owner/neighbor info
    ax1.legend(fontsize="x-small")

    # === Subplot 2: Face–Cell bipartite graph ===
    ax2.set_title("Face–Cell Connectivity")
    ax2.set_aspect("equal")

    # Draw edges from cell to face
    for c, faces in enumerate(mesh.cell_faces):
        for f in faces:
            if f >= 0:
                x0, y0 = cell_c[c]
                x1, y1 = face_c[f]
                ax2.plot([x0, x1], [y0, y1], color="gray", linewidth=0.6)

    # Plot nodes
    ax2.scatter(cell_c[:, 0], cell_c[:, 1], s=15, color="blue", label="Cells")
    ax2.scatter(face_c[:, 0], face_c[:, 1], s=15, color="orange", label="Faces")

    ax2.legend(fontsize="x-small")

    # Final save
    plt.suptitle(f"Mesh Connectivity Check: {mesh_type.capitalize()}")
    plt.tight_layout()
    plt.savefig(f"tests/test_output/mesh_graph_detailed_{mesh_type}.pdf", dpi=300)
    plt.close()
