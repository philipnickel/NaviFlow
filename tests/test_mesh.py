import numpy as np
import os
import matplotlib.pyplot as plt
from naviflow_collocated.mesh.mesh_data import MeshData2D
import warnings
import uuid


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
    for f in mesh.boundary_faces:
        assert mesh.boundary_patches[f] >= 0, (
            f"Missing or invalid boundary patch for face {f}"
        )
        assert mesh.boundary_types[f] >= 0, (
            f"Missing or invalid boundary type for face {f}"
        )
        assert not np.isnan(mesh.boundary_values[f, 0]), (
            f"Missing boundary value for face {f}"
        )


def test_graph_connectivity_and_plot(mesh_instance):
    mesh = mesh_instance
    os.makedirs("tests/test_output", exist_ok=True)
    mesh_type = "structured" if mesh.is_structured else "unstructured"

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Mesh Geometry and Face Normals")
    ax.set_aspect("equal")

    face_c = mesh.face_centers
    face_n = mesh.face_normals
    face_v = mesh.face_vertices
    cell_c = mesh.cell_centers
    boundary_mask = np.zeros(len(face_c), dtype=bool)
    boundary_mask[mesh.boundary_faces] = True

    # Plot actual faces as lines between vertex coordinates
    for i, verts in enumerate(face_v):
        v0, v1 = mesh.vertices[verts[0]], mesh.vertices[verts[1]]
        color = "red" if boundary_mask[i] else "gray"
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], color=color, lw=0.5)

    # Cell centers
    ax.scatter(cell_c[:, 0], cell_c[:, 1], s=1, color="blue", label="Cell Centers")

    # Face centers
    ax.scatter(
        face_c[~boundary_mask, 0],
        face_c[~boundary_mask, 1],
        s=1,
        color="gray",
        label="Internal Faces",
    )
    ax.scatter(
        face_c[boundary_mask, 0],
        face_c[boundary_mask, 1],
        s=1,
        color="red",
        label="Boundary Faces",
    )

    # Face normals (rescaled for visibility)
    arrow_len = 0.3 * np.mean(mesh.face_areas)
    norms = np.linalg.norm(face_n, axis=1)[:, None] + 1e-12
    face_n_vis = (face_n / norms) * arrow_len
    ax.quiver(
        face_c[:, 0],
        face_c[:, 1],
        face_n_vis[:, 0],
        face_n_vis[:, 1],
        color="black",
        scale=1,
        width=0.002,
        alpha=0.5,
        label="Face Normals",
    )
    """
    # Label faces with face ID and owner/neighbor info
    for f in range(len(face_c)):
        x, y = face_c[f]
        owner = mesh.owner_cells[f]
        neighbor = mesh.neighbor_cells[f]
        label = f"F{f}\nP{owner}" + (f"\nN{neighbor}" if neighbor >= 0 else "")
        ax.text(x, y, label, fontsize=6, color="green", ha="center", va="center")
    """
    ax.legend(fontsize="x-small")

    plt.suptitle(f"Mesh Visualization: {mesh_type.capitalize()}")
    plt.tight_layout()
    plt.savefig(f"tests/test_output/mesh_geometry_{mesh_type}.pdf", dpi=300)
    plt.close()


def warn_if_geom_issues_and_visualize(
    mesh_instance,
    label=None,
    skew_angle_threshold_deg=30,
    ortho_angle_threshold_deg=80,
    relative_skew_threshold=0.05,
):
    mesh = mesh_instance
    internal = mesh.neighbor_cells >= 0
    P = mesh.owner_cells[internal]
    N = mesh.neighbor_cells[internal]

    cell_P = mesh.cell_centers[P]
    cell_N = mesh.cell_centers[N]
    face_c = mesh.face_centers[internal]
    d_PN = mesh.d_PN[internal]
    e_f = mesh.e_f[internal]

    # === Skewness ===
    center_midpoint = 0.5 * (cell_P + cell_N)
    skew_vectors = face_c - center_midpoint
    skew_magnitude = np.linalg.norm(skew_vectors, axis=1)
    center_line = cell_N - cell_P
    center_length = np.linalg.norm(center_line, axis=1)

    skew_angle_cos = np.einsum("ij,ij->i", center_line, skew_vectors) / (
        center_length * skew_magnitude + 1e-12
    )
    skew_angles_deg = np.degrees(np.arccos(np.clip(skew_angle_cos, -1.0, 1.0)))

    skew_angle_threshold = np.cos(np.radians(skew_angle_threshold_deg))
    length_mask = skew_magnitude > (relative_skew_threshold * center_length)
    skew_mask = (skew_angle_cos < skew_angle_threshold) & length_mask

    # === Non-Orthogonality ===
    cos_theta = np.einsum("ij,ij->i", e_f, d_PN) / (
        np.linalg.norm(e_f, axis=1) * np.linalg.norm(d_PN, axis=1) + 1e-12
    )
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    ortho_mask = angle_deg > ortho_angle_threshold_deg

    if np.any(skew_mask) or np.any(ortho_mask):
        os.makedirs("tests/test_output", exist_ok=True)
        mesh_id = label or f"mesh_{uuid.uuid4().hex[:8]}"
        plot_path = f"tests/test_output/mesh_geom_issues_{mesh_id}.pdf"

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"Mesh Geometry Issues: {mesh_id}")
        ax.set_aspect("equal")

        # === Draw cell polygons ===
        for c, cell_faces in enumerate(mesh.cell_faces):
            polygon = []
            for f in cell_faces:
                if f >= 0:
                    verts = mesh.face_vertices[f]
                    for v in verts:
                        polygon.append(tuple(mesh.vertices[v]))
            if polygon:
                polygon = np.array(polygon)
                ax.plot(polygon[:, 0], polygon[:, 1], color="lightgray", lw=0.6)

        # === Face centers ===
        ax.scatter(
            mesh.face_centers[:, 0],
            mesh.face_centers[:, 1],
            s=3,
            color="gray",
            alpha=0.3,
        )

        # === Skewed Faces ===
        if np.any(skew_mask):
            ax.scatter(
                face_c[skew_mask, 0],
                face_c[skew_mask, 1],
                s=2,
                color="orange",
                label="Skewed Face Centers",
            )
            for idx, i in enumerate(np.where(skew_mask)[0][:50]):
                ax.plot(
                    [cell_P[i][0], cell_N[i][0]],
                    [cell_P[i][1], cell_N[i][1]],
                    "k--",
                    lw=0.4,
                    alpha=0.4,
                )
                ax.plot(
                    [center_midpoint[i][0], face_c[i][0]],
                    [center_midpoint[i][1], face_c[i][1]],
                    "orange",
                    lw=0.3,
                    label="Skew Vector" if idx == 0 else "",
                )
                ax.scatter(
                    center_midpoint[i][0],
                    center_midpoint[i][1],
                    s=2,
                    color="green",
                    marker="x",
                    label="Ideal Center" if idx == 0 else "",
                )
                ax.text(
                    face_c[i][0],
                    face_c[i][1],
                    f"{skew_angles_deg[i]:.1f}°",
                    fontsize=6,
                    color="darkorange",
                    ha="left",
                    va="bottom",
                )

        # === Non-Orthogonal Faces ===
        if np.any(ortho_mask):
            ax.scatter(
                face_c[ortho_mask, 0],
                face_c[ortho_mask, 1],
                s=2,
                color="red",
                label="Non-Orthogonal",
            )
            for idx, i in enumerate(np.where(ortho_mask)[0][:50]):
                origin = face_c[i]
                scale = 0.5 * np.linalg.norm(d_PN[i])
                ef_vec = e_f[i] / (np.linalg.norm(e_f[i]) + 1e-12) * scale
                dpn_vec = d_PN[i] / (np.linalg.norm(d_PN[i]) + 1e-12) * scale
                ax.arrow(
                    origin[0],
                    origin[1],
                    ef_vec[0],
                    ef_vec[1],
                    head_width=0.01,
                    color="black",
                    alpha=0.6,
                    label="e_f" if idx == 0 else "",
                )
                ax.arrow(
                    origin[0],
                    origin[1],
                    dpn_vec[0],
                    dpn_vec[1],
                    head_width=0.01,
                    color="blue",
                    alpha=0.5,
                    label="d_PN" if idx == 0 else "",
                )
                ax.text(
                    face_c[i][0],
                    face_c[i][1],
                    f"{angle_deg[i]:.1f}°",
                    fontsize=6,
                    color="red",
                    ha="right",
                    va="top",
                )

        ax.legend(fontsize="small")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        warnings.warn(
            f"{np.sum(skew_mask)} skewed and {np.sum(ortho_mask)} non-orthogonal faces "
            f"in {mesh_id}. Saved: {plot_path}"
        )


def test_mesh_geometry_diagnostics(mesh_instance, mesh_label):
    warn_if_geom_issues_and_visualize(mesh_instance, label=mesh_label)
