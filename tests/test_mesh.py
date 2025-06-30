import numpy as np
import os
from matplotlib.patches import Polygon
from naviflow_collocated.mesh.mesh_data import MeshData2D
from utils.plot_style import plt
from matplotlib import rcParams
import pytest

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
    assert mesh.vector_S_f.shape == (n_faces, 2)
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


def test_vector_S_f_orientation(mesh_instance):
    mesh = mesh_instance
    internal_mask = mesh.neighbor_cells >= 0
    vec_d_CE_internal = mesh.vector_d_CE[internal_mask]
    vector_S_f_internal = mesh.vector_S_f[internal_mask]
    dot_product = np.einsum("ij,ij->i", vec_d_CE_internal, vector_S_f_internal)
    assert np.all(dot_product > -1e-9), "Some internal face S_f vectors do not align with owner to neighbor direction (d_CE)"
    boundary_mask = ~internal_mask
    owner_centers_bf = mesh.cell_centers[mesh.owner_cells[boundary_mask]]
    vec_owner_to_face_center_bf = mesh.face_centers[boundary_mask] - owner_centers_bf
    vector_S_f_bf = mesh.vector_S_f[boundary_mask]
    dot_product_bf = np.einsum("ij,ij->i", vec_owner_to_face_center_bf, vector_S_f_bf)
    assert np.all(dot_product_bf > -1e-9), "Some boundary face S_f vectors do not point outwards from owner cell"


def test_vector_d_CE_magnitude_matches_geometry(mesh_instance):
    mesh = mesh_instance
    mask = mesh.neighbor_cells >= 0
    P_coords = mesh.cell_centers[mesh.owner_cells[mask]]
    N_coords = mesh.cell_centers[mesh.neighbor_cells[mask]]
    expected_d_CE_mag = np.linalg.norm(N_coords - P_coords, axis=1)
    actual_d_CE_mag = np.linalg.norm(mesh.vector_d_CE[mask], axis=1)
    assert np.allclose(actual_d_CE_mag, expected_d_CE_mag, rtol=1e-6), "Mismatch in vector_d_CE magnitude"


def test_over_relaxed_decomposition(mesh_instance):
    mesh = mesh_instance
    reconstructed_S_f = mesh.vector_E_f + mesh.vector_T_f
    assert np.allclose(mesh.vector_S_f, reconstructed_S_f, atol=1e-9), \
        "Over-relaxed decomposition S_f = E_f + T_f failed."
    
    internal_mask = mesh.neighbor_cells >= 0
    dot_Tf_Sf = np.einsum("ij,ij->i", mesh.vector_T_f[internal_mask], mesh.vector_S_f[internal_mask])
    assert np.allclose(dot_Tf_Sf, 0, atol=1e-9), "vector_T_f is not orthogonal to vector_S_f for some internal faces."


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
    if mesh.cell_volumes.shape[0] > 400:
        return

    # --- Uniform scaling for vector field visualization ---
    vector_scale = 0.2

    os.makedirs("tests/test_output/mesh_test", exist_ok=True)
    # Create subfolders for different Tf magnitudes
    for rank in ['largest', 'second_largest', 'third_largest']:
        os.makedirs(f"tests/test_output/mesh_test/{rank}_Tf", exist_ok=True)

    path = f"tests/test_output/mesh_test/mesh_diagnostics_{mesh_label}.pdf"

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")
    ax.axis('off')
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
        ax.add_patch(Polygon(poly_coords, facecolor="none", edgecolor="gray", lw=1.0))

    # Cell centres
    ax.scatter(
        mesh.cell_centers[:, 0],
        mesh.cell_centers[:, 1],
        s=6,
        color=colors[0],
        zorder=3,
        label="Cell Centres",
    )

    # --- Face Normals Visualization ---
    internal_mask = mesh.neighbor_cells >= 0
    boundary_mask = ~internal_mask

    # --- Plotting for INTERNAL FACES --- 
    if np.any(internal_mask):
        # Find the top 3 internal faces with largest T_f magnitudes
        internal_faces = np.where(internal_mask)[0]
        T_f_magnitude = np.linalg.norm(mesh.vector_T_f[internal_mask], axis=1)
        top_3_indices = np.argsort(T_f_magnitude)[-3:][::-1]  # Get indices of top 3 largest
        top_3_faces = internal_faces[top_3_indices]
        top_3_magnitudes = T_f_magnitude[top_3_indices]

        for rank, (face_idx, magnitude) in enumerate(zip(top_3_faces, top_3_magnitudes)):
            rank_name = ['largest', 'second_largest', 'third_largest'][rank]
            fig_int, ax_int = plt.subplots(figsize=(11, 11))
            ax_int.set_aspect("equal")
            ax_int.axis('off')
        
            # Create mask for just this face
            single_internal_mask = np.zeros_like(internal_mask)
            single_internal_mask[face_idx] = True

            # Draw all cells with gray outlines
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
                if c == mesh.owner_cells[face_idx]:
                    ax_int.add_patch(Polygon(poly_coords, facecolor='#A7C7E7', edgecolor='#24527A', lw=0.6, alpha=0.7, label='Owner Cell'))
                elif c == mesh.neighbor_cells[face_idx]:
                    ax_int.add_patch(Polygon(poly_coords, facecolor='#B7E7A7', edgecolor='#2A7A24', lw=0.6, alpha=0.7, label='Neighbor Cell'))
                else:
                    ax_int.add_patch(Polygon(poly_coords, facecolor='none', edgecolor='#B0B0B0', lw=0.6))

            # Highlight the face in question (thinner, dark orange)
            face_verts = mesh.vertices[mesh.face_vertices[face_idx]]
            ax_int.add_patch(Polygon(face_verts, facecolor='none', edgecolor='#D2691E', lw=0.6))

            # Cell centres (only for owner and neighbor)
            owner_centroid = mesh.cell_centers[mesh.owner_cells[face_idx]]
            neighbor_centroid = mesh.cell_centers[mesh.neighbor_cells[face_idx]]
            ax_int.scatter(
                [owner_centroid[0], neighbor_centroid[0]],
                [owner_centroid[1], neighbor_centroid[1]],
                s=20,  # Increased from 16
                color=colors[0],
                zorder=3,
                label="Cell Centroids",
            )

            # Add legend
            ax_int.legend(loc='upper right', frameon=True, framealpha=0.9)

            fc_internal = mesh.face_centers[single_internal_mask]
            cc_owner_internal = mesh.cell_centers[mesh.owner_cells[single_internal_mask]]

            # 1. vector_S_f (Internal, scaled, at face centers)
            ax_int.quiver(fc_internal[:, 0], fc_internal[:, 1],
                      mesh.vector_S_f[single_internal_mask, 0] * vector_scale,
                      mesh.vector_S_f[single_internal_mask, 1] * vector_scale,
                      angles="xy", scale_units="xy", scale=1, color=colors[0],
                      width=0.001, alpha=1.0)

            # 2. vector_d_CE (Internal, from Owner Cell Center)
            ax_int.quiver(cc_owner_internal[:, 0], cc_owner_internal[:, 1],
                      mesh.vector_d_CE[single_internal_mask, 0],
                      mesh.vector_d_CE[single_internal_mask, 1],
                      angles="xy", scale_units="xy", scale=1, color=colors[0],
                      width=0.001, alpha=0.2)
            
            # Add annotations for owner and neighbor cell centroids
            for i in range(cc_owner_internal.shape[0]):
                # Owner cell centroid 'C'
                ax_int.annotate('C', xy=cc_owner_internal[i],
                        xytext=(cc_owner_internal[i][0] - 0.45, cc_owner_internal[i][1] - 0.45),
                        fontsize=16, color=colors[0])  # Increased from 14
                
                # Neighbor cell centroid 'E'
                neighbor_center = mesh.cell_centers[mesh.neighbor_cells[single_internal_mask][i]]
                ax_int.annotate('E', xy=neighbor_center,
                        xytext=(neighbor_center[0] - 0.45, neighbor_center[1] - 0.45),
                        fontsize=16, color=colors[0])  # Increased from 14
                
                # d_CE annotation
                v = mesh.vector_d_CE[single_internal_mask][i]
                mid = cc_owner_internal[i] + 0.25 * v
                label_pos = mid + mesh.vector_T_f[single_internal_mask][i] / np.linalg.norm(mesh.vector_T_f[single_internal_mask][i]) * 0.3
                ax_int.annotate(r"$\vec{d}_{CE}$", xy=cc_owner_internal[i], xytext=(label_pos[0], label_pos[1]),
                            fontsize=12, color=colors[0])

            # 3. unit_vector_n
            ax_int.quiver(fc_internal[:, 0], fc_internal[:, 1],
                      mesh.unit_vector_n[single_internal_mask, 0] * vector_scale*2,
                      mesh.unit_vector_n[single_internal_mask, 1] * vector_scale*2,
                      angles="xy", scale_units="xy", scale=1, color=colors[0],
                      width=0.002, alpha=1.0)

            # Annotate unit_vector_n
            for i in range(fc_internal.shape[0]):
                v = mesh.unit_vector_n[single_internal_mask][i] * vector_scale*2
                mid = fc_internal[i] + 0.6 * v
                label_pos = mid + mesh.vector_T_f[single_internal_mask][i] / np.linalg.norm(mesh.vector_T_f[single_internal_mask][i]) * 0.3
                ax_int.annotate(r"$\vec{n}$", xy=fc_internal[i], xytext=(label_pos[0], label_pos[1]),
                            fontsize=12, color=colors[0])

            # Only plot Tf, Ef, and df'f if Tf magnitude is not too small (for unstructured grids)
            Tf_magnitude = np.linalg.norm(mesh.vector_T_f[single_internal_mask][0])
            if Tf_magnitude > 1e-10:  # Only plot for unstructured grids
                # 4. vector_E_f
                ax_int.quiver(fc_internal[:, 0], fc_internal[:, 1],
                          mesh.vector_E_f[single_internal_mask, 0] * vector_scale,
                          mesh.vector_E_f[single_internal_mask, 1] * vector_scale,
                          angles="xy", scale_units="xy", scale=1, color=colors[0],
                          width=0.001, alpha=1.0)

                # 5. vector_T_f
                tf_origin = fc_internal + mesh.vector_E_f[single_internal_mask] * vector_scale
                ax_int.quiver(tf_origin[:, 0], tf_origin[:, 1],
                          mesh.vector_T_f[single_internal_mask, 0] * vector_scale,
                          mesh.vector_T_f[single_internal_mask, 1] * vector_scale,
                          angles="xy", scale_units="xy", scale=1, color=colors[0],
                          width=0.001, alpha=1.0)

                # Annotate S_f, E_f, T_f
                for i in range(fc_internal.shape[0]):
                    S_vec = mesh.vector_S_f[single_internal_mask][i] * vector_scale
                    E_vec = mesh.vector_E_f[single_internal_mask][i] * vector_scale
                    T_vec = mesh.vector_T_f[single_internal_mask][i] * vector_scale

                    O_S = fc_internal[i]
                    O_E = fc_internal[i]
                    O_T = tf_origin[i]

                    mid_S = O_S + 0.5 * S_vec
                    mid_E = O_E + 0.5 * E_vec
                    mid_T = O_T + 0.5 * T_vec

                    triangle_center = O_S + 0.5 * (mid_T - O_S)

                    label_pos_S = mid_S + T_vec / np.linalg.norm(T_vec) * 0.3
                    ax_int.annotate(r"$\vec{S}_f$", xy=O_S, xytext=label_pos_S,
                                fontsize=11, color=colors[0])

                    label_pos_E = mid_E - T_vec / np.linalg.norm(T_vec) * 0.3
                    ax_int.annotate(r"$\vec{E}_f$", xy=O_E, xytext=label_pos_E,
                                fontsize=11, color=colors[0])

                    dir_T = mid_T - triangle_center
                    label_pos_T = triangle_center + 1.4 * dir_T
                    ax_int.annotate(r"$\vec{T}_f$", xy=O_T, xytext=label_pos_T,
                                fontsize=11, color=colors[0])

                # Add df'f vector plotting at the end
                # vector_skewness (Internal)
                ax_int.quiver(fc_internal[:, 0]- mesh.vector_skewness[single_internal_mask, 0] , fc_internal[:, 1]- mesh.vector_skewness[single_internal_mask, 1] ,
                          mesh.vector_skewness[single_internal_mask, 0],
                          mesh.vector_skewness[single_internal_mask, 1],
                          angles="xy", scale_units="xy", scale=1, color=colors[0],
                          width=0.002, alpha=1.0)

                # Annotate vector_skewness
                for i in range(fc_internal.shape[0]):
                    v = mesh.vector_skewness[single_internal_mask][i]
                    mid = fc_internal[i]- mesh.vector_skewness[single_internal_mask][i] + 0.5 * v
                    label_pos = mid - mesh.unit_vector_n[single_internal_mask][i] * vector_scale * 1.0
                    ax_int.annotate(r"$\vec{d}_{f'f}$", xy=fc_internal[i], xytext=(label_pos[0], label_pos[1]),
                                fontsize=10, color=colors[0])
            else:
                # For structured grids, only annotate S_f
                for i in range(fc_internal.shape[0]):
                    S_vec = mesh.vector_S_f[single_internal_mask][i] * vector_scale
                    mid_S = fc_internal[i] + 0.5 * S_vec
                    label_pos_S = mid_S + np.array([0.3, 0.3])  # Increased offset for structured grid
                    ax_int.annotate(r"$\vec{S}_f$", xy=fc_internal[i], xytext=label_pos_S,
                                fontsize=11, color=colors[0])

            plt.tight_layout()
            int_path = f"tests/test_output/mesh_test/{rank_name}_Tf/mesh_interior_face_{mesh_label}.pdf"
            plt.savefig(int_path, dpi=300)
            plt.close()
            assert os.path.exists(int_path), f"Failed to create {int_path}"

    # --- Plot for boundary face with largest Tf ---
    if np.any(boundary_mask):
        # Find the top 3 boundary faces with largest T_f magnitudes
        boundary_faces = np.where(boundary_mask)[0]
        T_f_magnitude = np.linalg.norm(mesh.vector_T_f[boundary_mask], axis=1)
        top_3_indices = np.argsort(T_f_magnitude)[-3:][::-1]  # Get indices of top 3 largest
        top_3_faces = boundary_faces[top_3_indices]
        top_3_magnitudes = T_f_magnitude[top_3_indices]

        for rank, (face_idx, magnitude) in enumerate(zip(top_3_faces, top_3_magnitudes)):
            rank_name = ['largest', 'second_largest', 'third_largest'][rank]
            fig_bnd, ax_bnd = plt.subplots(figsize=(11, 11))
            ax_bnd.set_aspect("equal")
            ax_bnd.axis('off')
        
            # Create mask for just this face
            single_boundary_mask = np.zeros_like(boundary_mask)
            single_boundary_mask[face_idx] = True

            # Draw all cells with gray outlines
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
                if c == mesh.owner_cells[face_idx]:
                    ax_bnd.add_patch(Polygon(poly_coords, facecolor='#A7C7E7', edgecolor='#24527A', lw=0.6, alpha=0.7, label='Owner Cell'))
                else:
                    ax_bnd.add_patch(Polygon(poly_coords, facecolor='none', edgecolor='#B0B0B0', lw=0.6))

            # Highlight the face in question (thinner, dark orange)
            face_verts = mesh.vertices[mesh.face_vertices[face_idx]]
            ax_bnd.add_patch(Polygon(face_verts, facecolor='none', edgecolor='#D2691E', lw=0.6))

            # Cell centres (only for owner cell)
            owner_centroid = mesh.cell_centers[mesh.owner_cells[face_idx]]
            ax_bnd.scatter(
                [owner_centroid[0]],
                [owner_centroid[1]],
                s=20,  # Increased from 16
                color=colors[0],
                zorder=3,
                label="Cell Centroids",
            )

            # Add legend
            ax_bnd.legend(loc='upper right', frameon=True, framealpha=0.9)

            fc_boundary = mesh.face_centers[single_boundary_mask]
            cc_owner_boundary = mesh.cell_centers[mesh.owner_cells[single_boundary_mask]]

            # Plot all vectors for this face
            # 1. vector_S_f
            ax_bnd.quiver(fc_boundary[:, 0], fc_boundary[:, 1],
                      mesh.vector_S_f[single_boundary_mask, 0] * vector_scale,
                      mesh.vector_S_f[single_boundary_mask, 1] * vector_scale,
                      angles="xy", scale_units="xy", scale=1, color=colors[0],
                      width=0.002, alpha=1.0)

            # Annotate boundary vector_S_f
            for i in range(fc_boundary.shape[0]):
                S_vec = mesh.vector_S_f[single_boundary_mask][i] * vector_scale
                T_vec =  mesh.vector_T_f[single_boundary_mask][i] * vector_scale
                O_S = fc_boundary[i]
                O_T = O_S + mesh.vector_E_f[single_boundary_mask][i] * vector_scale
                mid_S = O_S + 0.5 * S_vec
                mid_T = O_T + 0.5 * T_vec
                triangle_center = O_S + 0.5 * (mid_T - O_S)
                label_pos_S = mid_S + T_vec / (np.linalg.norm(T_vec) + 1e-12) * 0.6  # Increased from 0.3
                ax_bnd.annotate(r"$\vec{S}_f$", xy=O_S, xytext=label_pos_S,
                            fontsize=11, color=colors[0])

            # 2. Vector from Owner Cell Center to Boundary Face Center (P->f)
            vec_Pf_boundary = fc_boundary - cc_owner_boundary
            ax_bnd.quiver(cc_owner_boundary[:, 0], cc_owner_boundary[:, 1],
                      vec_Pf_boundary[:, 0],
                      vec_Pf_boundary[:, 1],
                      angles="xy", scale_units="xy", scale=1, color=colors[0],
                      width=0.002, alpha=0.2)

            # Annotate boundary d_Pf
            for i in range(cc_owner_boundary.shape[0]):
                Pf_orth = np.array([-vec_Pf_boundary[i][1], vec_Pf_boundary[i][0]])
                Pf_mid = cc_owner_boundary[i] + vec_Pf_boundary[i] * 0.5
                Pf_tip = Pf_mid + Pf_orth * 0.4  # Increased from 0.3
                ax_bnd.annotate(r"$\vec{d}_{Pf}$", xy=cc_owner_boundary[i],
                        xytext=(Pf_tip[0] + 0.05, Pf_tip[1] + 0.05),  # Increased from 0.03
                        fontsize=11, color=colors[0])

            # Only plot Tf, Ef, and df'f if Tf magnitude is not too small (for unstructured grids)
            Tf_magnitude = np.linalg.norm(mesh.vector_T_f[single_boundary_mask][0])
            if Tf_magnitude > 1e-10:  # Only plot for unstructured grids
                # 3. vector_E_f (Boundary, scaled, at face centers)
                ax_bnd.quiver(fc_boundary[:, 0], fc_boundary[:, 1],
                          mesh.vector_E_f[single_boundary_mask, 0] * vector_scale,
                          mesh.vector_E_f[single_boundary_mask, 1] * vector_scale,
                          angles="xy", scale_units="xy", scale=1, color=colors[0],
                          width=0.002, alpha=1.0)

                # Annotate vector_E_f
                for i in range(fc_boundary.shape[0]):
                    E_vec = mesh.vector_E_f[single_boundary_mask][i] * vector_scale
                    mid = fc_boundary[i] + 0.5 * E_vec
                    T_vec = mesh.vector_T_f[single_boundary_mask][i]
                    T_hat = T_vec / (np.linalg.norm(T_vec) + 1e-12)
                    label_pos = mid - T_hat/(np.linalg.norm(T_hat) + 1e-12) * 0.6  # Increased from 0.3
                    ax_bnd.annotate(r"$\vec{E}_f$", xy=fc_boundary[i], xytext=(label_pos[0], label_pos[1]),
                                fontsize=11, color=colors[0])

                # 4. vector_T_f (Boundary, scaled, at tf_origin_boundary)
                tf_origin_boundary = fc_boundary + mesh.vector_E_f[single_boundary_mask] * vector_scale
                ax_bnd.quiver(tf_origin_boundary[:, 0], tf_origin_boundary[:, 1],
                          mesh.vector_T_f[single_boundary_mask, 0] * vector_scale,
                          mesh.vector_T_f[single_boundary_mask, 1] * vector_scale,
                          angles="xy", scale_units="xy", scale=1, color=colors[0],
                          width=0.002, alpha=1.0)

                # Annotate vector_T_f
                for i in range(fc_boundary.shape[0]):
                    T_vec = mesh.vector_T_f[single_boundary_mask][i] * vector_scale
                    O_T = tf_origin_boundary[i]
                    mid_T = O_T + 0.5 * T_vec
                    dir_T = T_vec / (np.linalg.norm(T_vec) + 1e-12)
                    label_pos = mid_T + dir_T * 0.7
                    ax_bnd.annotate(r"$\vec{T}_f$", xy=O_T, xytext=(label_pos[0], label_pos[1]),
                                fontsize=11, color=colors[0])
            else:
                # For structured grids, only annotate S_f
                for i in range(fc_boundary.shape[0]):
                    S_vec = mesh.vector_S_f[single_boundary_mask][i] * vector_scale
                    mid_S = fc_boundary[i] + 0.5 * S_vec
                    # Place Sf annotation in a different direction than n
                    n_vec = mesh.unit_vector_n[single_boundary_mask][i] * vector_scale * 2
                    label_pos_S = mid_S + np.array([0.4, 0.4])  # Increased offset and moved diagonally
                    ax_bnd.annotate(r"$\vec{S}_f$", xy=fc_boundary[i], xytext=label_pos_S,
                                fontsize=11, color=colors[0])

            plt.tight_layout()
            bnd_path = f"tests/test_output/mesh_test/{rank_name}_Tf/mesh_boundary_face_{mesh_label}.pdf"
            plt.savefig(bnd_path, dpi=300)
            plt.close()
            assert os.path.exists(bnd_path), f"Failed to create {bnd_path}"


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
    assert mesh.vector_S_f.shape == (n_faces, 2), "vector_S_f shape mismatch"
    assert mesh.face_vertices.shape[0] == n_faces, "face_vertices count mismatch"
    assert mesh.owner_cells.shape[0] == n_faces, "owner_cells count mismatch"
    assert mesh.neighbor_cells.shape[0] == n_faces, "neighbor_cells count mismatch"
    assert mesh.boundary_types.shape == (n_faces, 2), "boundary_types shape mismatch"
    assert mesh.boundary_values.shape == (n_faces, 3), "boundary_values shape mismatch"

    # --- Finite‑value checks ---
    numeric_fields = [
        "cell_volumes", "cell_centers", "face_areas", "vector_S_f", "face_centers", 
        "vertices", "vector_d_CE", "unit_vector_n", "unit_vector_e", 
        "vector_E_f", "vector_T_f", "vector_skewness",
        "face_interp_factors", "d_Cb", "rc_interp_weights"
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
    # d_Cb: 0 for internal faces (or for faces not in boundary_faces)
    # These checks assume that boundary_values is at least 1D (n_faces, ...) and d_Cb is (n_faces,) or (n_faces, ...)
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
    # d_Cb: 0 for internal faces (or for faces not in boundary_faces)
    non_boundary_mask = np.ones(n_faces, dtype=bool)
    if len(mesh.boundary_faces) > 0:
      non_boundary_mask[mesh.boundary_faces] = False
    assert np.allclose(mesh.d_Cb[non_boundary_mask], 0.0, atol=1e-12), \
        "d_Cb should be 0 for internal faces"


def test_internal_face_vector_sanity(mesh_instance):
    """
    Performs detailed sanity checks on geometric vectors for internal faces.
    """
    mesh = mesh_instance
    if mesh.internal_faces.shape[0] == 0:
        pytest.skip("No internal faces in this mesh to test vector sanity.")

    internal_mask = np.zeros(mesh.face_areas.shape[0], dtype=bool)
    internal_mask[mesh.internal_faces] = True

    # 1. vector_d_CE (Owner-to-Neighbor Vector)
    vec_d_CE_internal = mesh.vector_d_CE[internal_mask]
    mag_d_CE_internal = np.linalg.norm(vec_d_CE_internal, axis=1)
    assert np.all(mag_d_CE_internal > 1e-12), \
        "Internal faces: vector_d_CE magnitude should be positive."
    
    # Re-verify d_CE direction and magnitude (subset of test_vector_d_CE_magnitude_matches_geometry)
    P_coords = mesh.cell_centers[mesh.owner_cells[internal_mask]]
    N_coords = mesh.cell_centers[mesh.neighbor_cells[internal_mask]]
    expected_d_CE_vec = N_coords - P_coords
    assert np.allclose(vec_d_CE_internal, expected_d_CE_vec, atol=1e-9), \
        "Internal faces: vector_d_CE does not match owner-to-neighbor cell center vector."

    # 2. unit_vector_e (Unit Vector along d_CE)
    unit_e_internal = mesh.unit_vector_e[internal_mask]
    mag_unit_e_internal = np.linalg.norm(unit_e_internal, axis=1)
    assert np.allclose(mag_unit_e_internal, 1.0, atol=1e-9), \
        "Internal faces: unit_vector_e magnitude is not 1.0."
    # Check alignment with d_CE (normalized dot product should be ~1)
    # (d_CE / |d_CE|) . unit_e  (where unit_e is already supposed to be d_CE / |d_CE|)
    normalized_d_CE = vec_d_CE_internal / (mag_d_CE_internal[:, np.newaxis] + 1e-12)
    dot_prod_e_dCE = np.einsum("ij,ij->i", unit_e_internal, normalized_d_CE)
    assert np.allclose(dot_prod_e_dCE, 1.0, atol=1e-9), \
        "Internal faces: unit_vector_e is not aligned with vector_d_CE."

    # 3. Over-Relaxed Decomposition (vector_E_f, vector_T_f)
    vec_S_f_internal = mesh.vector_S_f[internal_mask]
    vec_E_f_internal = mesh.vector_E_f[internal_mask]
    vec_T_f_internal = mesh.vector_T_f[internal_mask]

   

    # T_f should be orthogonal to S_f
    dot_Tf_Sf = np.einsum("ij,ij->i", vec_T_f_internal, vec_S_f_internal)
    assert np.allclose(dot_Tf_Sf, 0.0, atol=1e-9), \
        "Internal faces: vector_T_f is not orthogonal to vector_S_f."
    
    # S_f = E_f + T_f (already in test_over_relaxed_decomposition, but good for completeness here too)
    assert np.allclose(vec_S_f_internal, vec_E_f_internal + vec_T_f_internal, atol=1e-9), \
        "Internal faces: S_f != E_f + T_f."

    # 4. vector_skewness
    vec_skew_internal = mesh.vector_skewness[internal_mask]
    unit_n_internal = mesh.unit_vector_n[internal_mask]
    # Skewness vector should be orthogonal to face normal unit_vector_n
    dot_skew_n = np.einsum("ij,ij->i", vec_skew_internal, unit_n_internal)
    assert np.allclose(dot_skew_n, 0.0, atol=1e-9), \
        "Internal faces: vector_skewness is not orthogonal to unit_vector_n."

    # 5. face_interp_factors (alpha_f)
    interp_factors_internal = mesh.face_interp_factors[internal_mask]
    assert np.all((interp_factors_internal >= 0.0) & (interp_factors_internal <= 1.0)), \
        "Internal faces: face_interp_factors are out of range [0, 1]."
