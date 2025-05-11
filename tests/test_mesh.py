import numpy as np
import os
from matplotlib.patches import Polygon
from naviflow_collocated.mesh.mesh_data import MeshData2D
from utils.plot_style import plt
from matplotlib import rcParams
import pytest
from naviflow_collocated.mesh.mesh_loader import load_mesh, BC_WALL, BC_DIRICHLET, BC_NEUMANN, BC_ZEROGRADIENT

# --- Matplotlib colors for diagnostic plotting ---
colors = rcParams["axes.prop_cycle"].by_key()["color"]


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
    vec = mesh.d_PN[internal]  # Use precomputed owner→neighbor vectors from meshdata
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


def test_boundary_patch_consistency(mesh_instance):
    mesh = mesh_instance
    for f in mesh.boundary_faces:
        assert mesh.boundary_patches[f] >= 0, (
            f"Missing or invalid boundary patch for face {f}"
        )
        assert mesh.boundary_types[f, 0] >= 0, (
            f"Missing or invalid velocity boundary type for face {f}"
        )
        assert mesh.boundary_types[f, 1] >= 0, (
            f"Missing or invalid pressure boundary type for face {f}"
        )
        assert not np.isnan(mesh.boundary_values[f, 0]), (
            f"Missing boundary velocity u value for face {f}"
        )
        assert not np.isnan(mesh.boundary_values[f, 1]), (
            f"Missing boundary velocity v value for face {f}"
        )
        assert not np.isnan(mesh.boundary_values[f, 2]), (
            f"Missing boundary pressure value for face {f}"
        )


# --- Unified mesh visual diagnostics ---
def test_mesh_visual_diagnostics(mesh_instance, mesh_label):
    """
    Visual diagnostics for mesh integrity.
    only plots for small meshes
    """
    mesh = mesh_instance
    if mesh.cell_volumes.shape[0] > 200:
        return

    mesh = mesh_instance
    os.makedirs("tests/test_output", exist_ok=True)
    path = f"tests/test_output/mesh_diagnostics_{mesh_label}.pdf"

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")
    # Draw cell polygons
    for c, face_ids in enumerate(mesh.cell_faces):
        verts_idx = []
        for f in face_ids:
            if f >= 0:
                verts_idx.extend(mesh.face_vertices[f].tolist())
        if not verts_idx:
            continue
        verts_idx = list(dict.fromkeys(verts_idx))
        verts = mesh.vertices[verts_idx]
        center = mesh.cell_centers[c]
        angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
        poly_coords = verts[np.argsort(angles)]
        ax.add_patch(Polygon(poly_coords, facecolor="none", edgecolor="gray", lw=1))

    # Cell centres
    ax.scatter(
        mesh.cell_centers[:, 0],
        mesh.cell_centers[:, 1],
        s=6,
        color="blue",
        zorder=3,
        label="Cell Centres",
    )

    # Annotate with cell info: ID, volume, type (internal/boundary) using LaTeX-style formatting
    for cid, (x, y) in enumerate(mesh.cell_centers):
        # Determine cell type: boundary if any face is boundary
        face_ids = mesh.cell_faces[cid]
        is_boundary = any((f >= 0 and f in mesh.boundary_faces) for f in face_ids)
        cell_type = "boundary" if is_boundary else "internal"
        vol = mesh.cell_volumes[cid]
        ax.annotate(
            f"$\\text{{Cell}}\\ {cid}$\n$\\text{{Vol}} = {vol:.4g}$\n$\\text{{Type}}: \\text{{{cell_type}}}$",
            xy=(x, y),
            xytext=(x + 0.01, y + 0.01),
            textcoords="data",
            fontsize=7,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", lw=0.2
            ),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.2),
        )

    # --- Face Normals Visualization ---
    face_normals = mesh.face_normals
    face_normal_magnitude = np.linalg.norm(face_normals, axis=1)
    unit_dPN_mags = np.linalg.norm(mesh.unit_dPN, axis=1)
    scale_factor = 0.05
    # Plot area-weighted face normals (S_f)
    ax.quiver(
        mesh.face_centers[:, 0],
        mesh.face_centers[:, 1],
        face_normals[:, 0] * scale_factor,
        face_normals[:, 1] * scale_factor,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=colors[0],
        width=0.001,
        alpha=0.8,
        label="Face Normals (S_f)",
    )

    # Plot unit normals (unit_dPN) -- not scaled by magnitude, just unit vectors
    ax.quiver(
        mesh.face_centers[:, 0],
        mesh.face_centers[:, 1],
        mesh.unit_dPN[:, 0] * scale_factor * 0.5,
        mesh.unit_dPN[:, 1] * scale_factor * 0.5,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=colors[1],
        width=0.001,
        alpha=0.8,
        label="Unit Normal ($\\hat{n}_f$)",
    )

    # Removed separate annotation of face normal magnitudes to avoid duplication

    # d_PN and non-ortho vectors (no scaling)
    internal = mesh.neighbor_cells >= 0
    t_f = mesh.non_ortho_correction[internal]

    ax.quiver(
        mesh.face_centers[internal, 0],
        mesh.face_centers[internal, 1],
        t_f[:, 0],
        t_f[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=colors[3],
        width=0.0015,
        alpha=0.8,
        label="Non-Ortho Correction ($\\vec{k}_f$)",
    )

    # Optional skewness overlay (no scaling)
    skew = mesh.skewness_vectors[internal]
    ax.quiver(
        mesh.face_centers[internal, 0],
        mesh.face_centers[internal, 1],
        -skew[:, 0],
        -skew[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=colors[4],
        width=0.0012,
        alpha=0.8,
        label="Skew Vector ($\\vec{d}_f$)",
    )

    # Plot combined correction vectors: t_f + d_f (no scaling)
    correction_sum = t_f + skew
    ax.quiver(
        mesh.face_centers[internal, 0],
        mesh.face_centers[internal, 1],
        correction_sum[:, 0],
        correction_sum[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=colors[5],
        width=0.001,
        alpha=0.8,
        label="Correction Sum ($\\vec{k}_f + \\vec{d}_f$)",
    )

    # Add per-face labels: face ID, owner, neighbor, area, boundary type/internal using annotation with arrow and yellow box (LaTeX-style)
    for i, (fc, owner, neighbor, area, mag, e_mag) in enumerate(
        zip(
            mesh.face_centers,
            mesh.owner_cells,
            mesh.neighbor_cells,
            mesh.face_areas,
            face_normal_magnitude,
            unit_dPN_mags,
        )
    ):
        if i in mesh.boundary_faces:
            patch = mesh.boundary_patches[i]
            # Try to get boundary type name if available (optional)
            # patch mapping: top=3, bottom=1, left=4, right=2, etc.
            patch_name = {3: "Top", 1: "Bottom", 4: "Left", 2: "Right"}.get(
                patch, "unknown"
            )
            btype_str = f"$\\text{{Boundary:}}\\ {patch_name}$"
        else:
            btype_str = "$\\text{Internal}$"
        ax.annotate(
            f"$\\text{{Face}}\\ {i}$\n"
            f"$\\text{{O:}}\\ {owner}\\ \\text{{N:}}\\ {neighbor}$\n"
            f"$\\text{{Area}} = {area:.3g}$\n"
            f"$|\\vec{{S}}_f| = {mag:.3f}$\n"
            f"$|\\hat{{n}}_f| = {e_mag:.3f}$\n"
            f"{btype_str}",
            xy=(fc[0], fc[1]),
            xytext=(fc[0] + 0.01, fc[1] + 0.01),
            textcoords="data",
            fontsize=7,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor="white", edgecolor="black", lw=0.2
            ),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.2),
        )
    ax.legend(
        fontsize="small",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=0.9,
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    assert os.path.exists(path), f"Failed to create {path}"


# --- Comprehensive mesh data completeness check ---
def test_mesh_data_completeness(mesh_instance):
    """
    Comprehensive sanity checks on the MeshData2D object to ensure that
    array shapes, value ranges, and face/cell masks are mutually consistent.
    This catches silent geometry or connectivity errors early.
    """
    mesh = mesh_instance

    # --- Basic size references ---
    n_cells = mesh.cell_volumes.shape[0]
    n_faces = mesh.face_areas.shape[0]

    # --- Shape checks ---
    assert mesh.cell_centers.shape == (n_cells, 2), "cell_centers shape mismatch"
    assert mesh.face_centers.shape == (n_faces, 2), "face_centers shape mismatch"
    assert mesh.face_normals.shape == (n_faces, 2), "face_normals shape mismatch"
    assert mesh.face_vertices.shape[0] == n_faces, "face_vertices count mismatch"
    assert mesh.owner_cells.shape[0] == n_faces, "owner_cells count mismatch"
    assert mesh.neighbor_cells.shape[0] == n_faces, "neighbor_cells count mismatch"
    assert mesh.boundary_types.shape == (n_faces, 2), "boundary_types shape mismatch"
    assert mesh.boundary_values.shape == (n_faces, 3), "boundary_values shape mismatch"

    # --- Finite‑value checks ---
    import numpy as np

    numeric_fields = [
        "cell_volumes",
        "cell_centers",
        "face_areas",
        "face_normals",
        "face_centers",
        "vertices",
        "d_PN",
        "unit_dPN",
        "delta_PN",
        "delta_Pf",
        "delta_fN",
        "non_ortho_correction",
        "face_interp_factors",
        "d_PB",
        "rc_interp_weights",
    ]
    for field in numeric_fields:
        arr = getattr(mesh, field)
        assert np.all(np.isfinite(arr)), f"{field} contains NaN or Inf"

    # --- Connectivity consistency ---
    all_faces = set(range(n_faces))
    internal = set(mesh.internal_faces.tolist())
    boundary = set(mesh.boundary_faces.tolist())
    assert internal.isdisjoint(boundary), "internal_faces and boundary_faces overlap"
    assert internal | boundary == all_faces, (
        "Some faces are neither internal nor boundary"
    )

    # Boundary faces: neighbor == -1 and boundary_type >= 0
    assert np.all(mesh.neighbor_cells[mesh.boundary_faces] == -1), (
        "Boundary neighbor check failed"
    )
    assert np.all(mesh.boundary_types[mesh.boundary_faces, 0] >= 0), (
        "Velocity boundary types missing on boundary faces"
    )
    assert np.all(mesh.boundary_types[mesh.boundary_faces, 1] >= 0), (
        "Pressure boundary types missing on boundary faces"
    )

    # Internal faces: neighbor >= 0 and boundary_type == -1
    assert np.all(mesh.neighbor_cells[mesh.internal_faces] >= 0), (
        "Internal neighbor check failed"
    )
    assert np.all(mesh.boundary_types[mesh.internal_faces, 0] == -1), (
        "Internal faces should have velocity boundary_type -1"
    )
    assert np.all(mesh.boundary_types[mesh.internal_faces, 1] == -1), (
        "Internal faces should have pressure boundary_type -1"
    )

    # Interpolation factors should be within [0, 1]
    assert np.all(
        (mesh.face_interp_factors >= 0.0) & (mesh.face_interp_factors <= 1.0)
    ), "face_interp_factors out of range [0, 1]"

    # Physical quantities should be positive
    assert np.all(mesh.cell_volumes > 0), "Non‑positive cell volume detected"
    assert np.all(mesh.face_areas > 0), "Non‑positive face area detected"

    # --- Full-face indexing checks for boundary fields ---
    # boundary_types: -1 for internal, >=0 for boundary (already checked above)
    # boundary_values: e.g. zero vector for internal faces
    # d_PB: 0 for internal faces
    # These checks assume that boundary_values is at least 1D (n_faces, ...) and d_PB is (n_faces,) or (n_faces, ...)
    # For boundary_values, zero vector for internal faces
    if mesh.boundary_values.ndim == 2:
        zero_vec = np.zeros(mesh.boundary_values.shape[1])
        # Check internal faces get zero vector
        assert np.allclose(
            mesh.boundary_values[mesh.internal_faces], zero_vec, atol=1e-12
        ), "boundary_values for internal faces should be zero vector"
    else:
        # fallback: just check internal faces are zero
        assert np.allclose(
            mesh.boundary_values[mesh.internal_faces], 0.0, atol=1e-12
        ), "boundary_values for internal faces should be zero"
    # d_PB: 0 for internal faces
    assert np.allclose(mesh.d_PB[mesh.internal_faces], 0.0, atol=1e-12), (
        "d_PB should be 0 for internal faces"
    )
