import numpy as np
import meshio
from collections import defaultdict
from naviflow_collocated.mesh.mesh_data import MeshData2D
import yaml


def parse_physical_names(msh_filename):
    with open(msh_filename, "r") as f:
        lines = f.readlines()

    phys_names = {}
    inside = False
    for line in lines:
        if line.strip() == "$PhysicalNames":
            inside = True
            continue
        if line.strip() == "$EndPhysicalNames":
            break
        if inside:
            parts = line.strip().split()
            if len(parts) >= 3:
                dim, tag, *name_parts = parts
                name = " ".join(name_parts).strip('"')
                phys_names[int(tag)] = name
    return phys_names


def load_mesh(filename, bc_config_file=None):
    """
    Load a 2D mesh (structured or unstructured) from a Gmsh .msh file
    and return a MeshData2D object with boundary tagging.
    """
    physical_names = parse_physical_names(filename)
    boundary_conditions = {}
    if bc_config_file is not None:
        with open(bc_config_file, "r") as f:
            boundary_conditions = yaml.safe_load(f)

    mesh = meshio.read(filename)
    points = mesh.points[:, :2]

    if "triangle" in mesh.cells_dict:
        return _load_unstructured_mesh(
            mesh, points, physical_names, boundary_conditions
        )
    elif "quad" in mesh.cells_dict:
        return _load_structured_mesh(mesh, points, physical_names, boundary_conditions)
    else:
        raise ValueError("Unsupported mesh type: must contain triangle or quad cells")


def _load_unstructured_mesh(mesh, points, physical_names, boundary_conditions):
    triangles = np.array(mesh.cells_dict["triangle"], dtype=np.int64)
    boundary_lines = np.array(mesh.cells_dict.get("line", []), dtype=np.int64)
    boundary_tags = np.array(
        mesh.cell_data_dict["gmsh:physical"].get("line", []), dtype=np.int64
    )
    return _build_meshdata2d(
        points,
        triangles,
        boundary_lines,
        boundary_tags,
        is_structured=False,
        physical_id_to_name=physical_names,
        boundary_conditions=boundary_conditions,
    )


def _load_structured_mesh(mesh, points, physical_names, boundary_conditions):
    quads = np.array(mesh.cells_dict["quad"], dtype=np.int64)
    boundary_lines = np.array(mesh.cells_dict.get("line", []), dtype=np.int64)
    boundary_tags = np.array(
        mesh.cell_data_dict["gmsh:physical"].get("line", []), dtype=np.int64
    )
    return _build_meshdata2d(
        points,
        quads,
        boundary_lines,
        boundary_tags,
        is_structured=True,
        physical_id_to_name=physical_names,
        boundary_conditions=boundary_conditions,
    )


def _build_meshdata2d(
    points,
    cells,
    boundary_lines,
    boundary_tags,
    is_structured,
    physical_id_to_name,
    boundary_conditions,
):
    def sorted_edge(a, b):
        return tuple(sorted((a, b)))

    n_cells = len(cells)
    cell_centers = np.mean(points[cells], axis=1)
    cell_volumes = _calculate_cell_volumes(points, cells)

    face_map = defaultdict(list)
    for cid, cell in enumerate(cells):
        for i in range(len(cell)):
            edge = sorted_edge(cell[i], cell[(i + 1) % len(cell)])
            face_map[edge].append(cid)

    face_centers, face_normals, face_areas = [], [], []
    face_vertices, owner_cells, neighbor_cells = [], [], []

    for edge, cids in face_map.items():
        v0, v1 = points[edge[0]], points[edge[1]]
        center = 0.5 * (v0 + v1)
        normal = np.array([v1[1] - v0[1], -(v1[0] - v0[0])])
        length = np.linalg.norm(normal)
        if length > 0:
            normal /= length

        owner = cids[0]
        neighbor = cids[1] if len(cids) == 2 else -1

        ref_vec = (cell_centers[neighbor] if neighbor >= 0 else center) - cell_centers[
            owner
        ]
        if np.dot(ref_vec, normal) < 0:
            normal *= -1

        face_vertices.append(edge)
        owner_cells.append(owner)
        neighbor_cells.append(neighbor)
        face_centers.append(center)
        face_normals.append(normal * length)
        face_areas.append(length)

    face_centers = np.array(face_centers)
    face_normals = np.array(face_normals)
    face_areas = np.array(face_areas)
    face_vertices = np.array(face_vertices)
    owner_cells = np.array(owner_cells)
    neighbor_cells = np.array(neighbor_cells)

    n_faces = len(face_areas)
    d_PN = np.zeros((n_faces, 2))
    e_f = np.zeros((n_faces, 2))
    delta_PN = np.zeros(n_faces)
    delta_Pf = np.zeros(n_faces)
    delta_fN = np.zeros(n_faces)
    non_ortho_correction = np.zeros((n_faces, 2))
    face_interp_factors = np.zeros(n_faces)

    for f in range(n_faces):
        P = owner_cells[f]
        N = neighbor_cells[f]
        x_f = face_centers[f]
        x_P = cell_centers[P]
        vec_pf = x_f - x_P
        delta_Pf[f] = np.linalg.norm(vec_pf)

        if N >= 0:
            x_N = cell_centers[N]
            vec_pn = x_N - x_P
            d_PN[f] = vec_pn
            delta_PN[f] = np.linalg.norm(vec_pn)
            e_f[f] = vec_pn / (delta_PN[f] + 1e-12)
            delta_fN[f] = np.linalg.norm(x_N - x_f)
            n_hat = face_normals[f] / (np.linalg.norm(face_normals[f]) + 1e-12)
            proj_len = np.dot(vec_pn, n_hat)
            t_f = vec_pn - proj_len * n_hat
            non_ortho_correction[f] = t_f
            face_interp_factors[f] = delta_Pf[f] / (delta_PN[f] + 1e-12)
        else:
            e_f[f] = vec_pf / (np.linalg.norm(vec_pf) + 1e-12)
            face_interp_factors[f] = 1.0

    # Construct mapping from edge → face index
    edge_to_face = {tuple(sorted(edge)): i for i, edge in enumerate(face_vertices)}

    # Boundary tagging
    # Initialize boundary-related arrays to be n_faces long with default values
    boundary_faces_list = []  # Temporary list to collect global IDs of boundary faces
    boundary_patches = np.full(n_faces, -1, dtype=np.int64)  # Default patch ID -1
    boundary_types = np.full(
        n_faces, -1, dtype=np.int64
    )  # Default type -1 (internal/unspecified)
    # Initialize boundary_values with 0.0 instead of NaN as a safer default for testing
    boundary_values = np.full((n_faces, 2), 0.0, dtype=np.float64)  # Default 0.0

    internal_faces = np.where(neighbor_cells >= 0)[0]

    for i, line in enumerate(boundary_lines):  # i is the index for boundary_tags
        edge = sorted_edge(*line)
        if edge in edge_to_face:
            face_id = edge_to_face[edge]  # This is the global face index
            patch_tag_from_mesh_file = boundary_tags[i]  # gmsh physical tag
            patch_name = physical_id_to_name.get(patch_tag_from_mesh_file, "unknown")
            # Use patch_name to look up boundary conditions, default to empty dict if not found
            bc = boundary_conditions.get(patch_name, {})

            boundary_faces_list.append(face_id)
            boundary_patches[face_id] = patch_tag_from_mesh_file  # Use global face_id

            bc_type_str = bc.get("type", "unknown").lower()

            # Assign types and values based on config or defaults
            if bc_type_str == "wall":
                boundary_types[face_id] = 1  # Use global face_id
                boundary_values[face_id] = bc.get(
                    "value", [0.0, 0.0]
                )  # Default wall value [0,0]
            elif bc_type_str == "dirichlet":
                boundary_types[face_id] = 2  # Use global face_id
                # Ensure value is a list/array of size 2, default to [0,0] if missing/invalid
                val = bc.get("value", [0.0, 0.0])
                if not isinstance(val, (list, np.ndarray)) or len(val) != 2:
                    print(
                        f"Warning: Invalid Dirichlet value for patch '{patch_name}', using [0,0]. Value: {val}"
                    )
                    val = [0.0, 0.0]
                boundary_values[face_id] = val
            elif bc_type_str == "zerogradient":  # Match case-insensitivity
                boundary_types[face_id] = 3  # Use global face_id
                # Keep NaN for zeroGradient as it signals specific handling might be needed downstream
                boundary_values[face_id] = [np.nan, np.nan]
            else:  # Default or unknown type from YAML/config
                # Assign a default type (e.g., 0) but keep the default value ([0,0] set during init)
                boundary_types[face_id] = 0
                # No need to explicitly set boundary_values[face_id] here, it keeps the default [0,0]

    boundary_faces = np.array(
        sorted(list(set(boundary_faces_list))), dtype=np.int64
    )  # Ensure unique sorted global IDs

    # Compute distances from owner cell centers to boundary face centers
    # This array is used by the MeshData2D constructor, but not in mesh_data_spec.
    # For now, keep its original logic tied to the boundary_faces array.
    boundary_dists = np.array(
        [
            np.linalg.norm(face_centers[f] - cell_centers[owner_cells[f]])
            for f in boundary_faces
        ]
    )

    # Initialize d_PB with 0.0, then calculate for boundary faces (aligns with MeshData2D spec comment)
    d_PB = np.zeros(n_faces, dtype=np.float64)
    for f in boundary_faces:  # Iterate through global boundary face indices
        P = owner_cells[f]
        x_P = cell_centers[P]
        x_f = face_centers[f]
        # Calculate norm, ensure it's not NaN (shouldn't be if centers are valid)
        dist = np.linalg.norm(x_f - x_P)
        d_PB[f] = (
            dist if not np.isnan(dist) else 0.0
        )  # Assign distance, default to 0 if NaN somehow occurs

    # Construct cell_faces as a padded ndarray (NumPy-compatible for Numba)
    cell_face_lists = [[] for _ in range(n_cells)]
    for fid, (owner, neighbor) in enumerate(zip(owner_cells, neighbor_cells)):
        cell_face_lists[owner].append(fid)
        if neighbor >= 0:
            cell_face_lists[neighbor].append(fid)

    max_faces = max(len(faces) for faces in cell_face_lists)
    cell_faces = -np.ones((n_cells, max_faces), dtype=np.int64)
    for i, faces in enumerate(cell_face_lists):
        cell_faces[i, : len(faces)] = faces

    return MeshData2D(
        cell_volumes=cell_volumes,
        cell_centers=cell_centers,
        face_areas=face_areas,
        face_normals=face_normals,
        face_centers=face_centers,
        owner_cells=owner_cells,
        neighbor_cells=neighbor_cells,
        face_vertices=face_vertices,
        vertices=points,
        d_PN=d_PN,
        e_f=e_f,
        delta_PN=delta_PN,
        delta_Pf=delta_Pf,
        delta_fN=delta_fN,
        non_ortho_correction=non_ortho_correction,
        face_interp_factors=face_interp_factors,
        internal_faces=internal_faces,
        boundary_faces=boundary_faces,
        boundary_patches=boundary_patches,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        boundary_dists=boundary_dists,
        d_PB=d_PB,
        cell_faces=cell_faces,
        is_structured=is_structured,
        is_orthogonal=is_structured,
        is_conforming=True,
    )


def _calculate_cell_volumes(points, cells):
    if cells.shape[1] == 3:  # triangles
        a = points[cells[:, 0]]
        b = points[cells[:, 1]]
        c = points[cells[:, 2]]
        return 0.5 * np.abs(
            a[:, 0] * (b[:, 1] - c[:, 1])
            + b[:, 0] * (c[:, 1] - a[:, 1])
            + c[:, 0] * (a[:, 1] - b[:, 1])
        )
    elif cells.shape[1] == 4:  # quads
        a = points[cells[:, 0]]
        b = points[cells[:, 1]]
        c = points[cells[:, 2]]
        d = points[cells[:, 3]]
        tri1 = 0.5 * np.abs(
            a[:, 0] * (b[:, 1] - d[:, 1])
            + b[:, 0] * (d[:, 1] - a[:, 1])
            + d[:, 0] * (a[:, 1] - b[:, 1])
        )
        tri2 = 0.5 * np.abs(
            b[:, 0] * (c[:, 1] - d[:, 1])
            + c[:, 0] * (d[:, 1] - b[:, 1])
            + d[:, 0] * (b[:, 1] - c[:, 1])
        )
        return tri1 + tri2
    else:
        raise ValueError("Unsupported cell shape for volume calculation")
