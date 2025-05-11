import numpy as np
import meshio
from numba.typed import List
from collections import defaultdict
from naviflow_collocated.mesh.mesh_data import MeshData2D
import yaml

# Define boundary condition type mapping constants
# These should be used consistently across the codebase
BC_WALL = 0
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_ZEROGRADIENT = 3
BC_CONVECTIVE = 4
BC_SYMMETRY = 5

# Map from string names to type constants
BC_TYPE_MAP = {
    "wall": BC_WALL,
    "dirichlet": BC_DIRICHLET,
    "neumann": BC_NEUMANN,
    "zerogradient": BC_ZEROGRADIENT,
    "convective": BC_CONVECTIVE,
    "symmetry": BC_SYMMETRY,
}

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
            boundary_config = yaml.safe_load(f)
        boundary_conditions = boundary_config.get("boundaries", {})

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
        physical_id_to_name=physical_names,
        boundary_conditions=boundary_conditions,
    )


def _build_meshdata2d(
    points,
    cells,
    boundary_lines,
    boundary_tags,
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

        edge_vec = v1 - v0
        edge_len = np.linalg.norm(edge_vec)
        if edge_len > 0:
            n_hat = np.array([edge_vec[1], -edge_vec[0]]) / edge_len
        else:
            n_hat = np.zeros(2)

        owner = cids[0]
        neighbor = cids[1] if len(cids) == 2 else -1

        ref_vec = (cell_centers[neighbor] if neighbor >= 0 else center) - cell_centers[
            owner
        ]
        if np.dot(n_hat, ref_vec) < 0:
            n_hat *= -1

        face_vertices.append(edge)
        owner_cells.append(owner)
        neighbor_cells.append(neighbor)
        face_centers.append(center)
        face_normals.append(n_hat * edge_len)
        face_areas.append(edge_len)

    face_centers = np.array(face_centers)
    face_normals = np.array(face_normals)
    face_normal_mags = np.linalg.norm(face_normals, axis=1)
    assert np.allclose(face_normal_mags, face_areas, rtol=1e-10), (
        "Face normal magnitudes do not match face areas."
    )
    face_areas = np.array(face_areas)
    face_vertices = np.array(face_vertices)
    owner_cells = np.array(owner_cells)
    neighbor_cells = np.array(neighbor_cells)

    n_faces = len(face_areas)
    d_PN = np.zeros((n_faces, 2))
    unit_dPN = np.zeros((n_faces, 2))
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
            unit_dPN[f] = vec_pn / (delta_PN[f] + 1e-12)
            delta_fN[f] = np.linalg.norm(x_N - x_f)
            n_hat = face_normals[f] / (np.linalg.norm(face_normals[f]) + 1e-12)
            proj_len = np.dot(vec_pn, n_hat)
            t_f = vec_pn - proj_len * n_hat
            non_ortho_correction[f] = t_f
        else:
            unit_dPN[f] = vec_pf / (np.linalg.norm(vec_pf) + 1e-12)

    internal_faces = np.where(neighbor_cells >= 0)[0]
    unit_dPN_mags = np.linalg.norm(unit_dPN[internal_faces], axis=1)
    assert np.allclose(unit_dPN_mags, 1.0, rtol=1e-10), (
        "unit_dPN vectors on internal faces are not unit vectors."
    )

    # Construct mapping from edge → face index
    edge_to_face = {tuple(sorted(edge)): i for i, edge in enumerate(face_vertices)}

    # Map physical tag -> boundary condition config
    tag_to_bc = {
        tag: boundary_conditions.get(name, {})
        for tag, name in physical_id_to_name.items()
    }

    # Boundary tagging
    # Initialize boundary-related arrays to be n_faces long with default values
    boundary_faces_list = []  # Temporary list to collect global IDs of boundary faces
    boundary_patches = np.full(n_faces, -1, dtype=np.int64)  # Default patch ID -1
    boundary_types = np.full(
        (n_faces, 2), -1, dtype=np.int64
    )  # Default type -1 [vel_type, p_type]
    # Initialize boundary_values with 0.0 as a safer default
    boundary_values = np.full((n_faces, 3), 0.0, dtype=np.float64)  # [u, v, p]

    for i, line in enumerate(boundary_lines):
        edge = sorted_edge(*line)
        if edge not in edge_to_face:
            continue

        face_id = edge_to_face[edge]
        patch_tag = boundary_tags[i]
        patch_name = physical_id_to_name.get(patch_tag, "unknown")
        bc = tag_to_bc.get(patch_tag, {})

        x_f = face_centers[face_id]  # <-- Add this line for evaluating function BCs

        # --- Velocity BC ---
        vel_bc = bc.get("velocity", {})
        vel_bc_type_str = vel_bc.get("bc", "zerogradient").lower()
        vel_bc_type = BC_TYPE_MAP.get(vel_bc_type_str, BC_ZEROGRADIENT)
        vel_value_raw = vel_bc.get("value", [0.0, 0.0])

        # Process velocity boundary condition value
        if isinstance(vel_value_raw, str):
            try:
                vel_value = eval(vel_value_raw, {"np": np, "x": x_f})
            except Exception as e:
                raise ValueError(f"Failed to evaluate velocity BC expression: '{vel_value_raw}' at x = {x_f}: {e}")
        elif isinstance(vel_value_raw, list):
            # Process each element in the list
            vel_value = []
            for item in vel_value_raw:
                if isinstance(item, str):
                    try:
                        vel_value.append(eval(item, {"np": np, "x": x_f}))
                    except Exception as e:
                        raise ValueError(f"Failed to evaluate velocity BC expression: '{item}' at x = {x_f}: {e}")
                else:
                    vel_value.append(item)
        else:
            vel_value = vel_value_raw

        if callable(vel_value):
            vel_value = vel_value(x_f)

        # --- Pressure BC ---
        p_bc = bc.get("pressure", {})
        p_bc_type_str = p_bc.get("bc", "zerogradient").lower()
        p_bc_type = BC_TYPE_MAP.get(p_bc_type_str, BC_ZEROGRADIENT)
        p_value_raw = p_bc.get("value", 0.0)
        if isinstance(p_value_raw, str):
            try:
                p_value = eval(p_value_raw, {"np": np, "x": x_f})
            except Exception as e:
                raise ValueError(f"Failed to evaluate pressure BC expression: '{p_value_raw}' at x = {x_f}: {e}")
        else:
            p_value = p_value_raw

        if callable(p_value):
            p_value = p_value(x_f)

        boundary_faces_list.append(face_id)
        boundary_patches[face_id] = patch_tag
        boundary_types[face_id] = [vel_bc_type, p_bc_type]

        if isinstance(vel_value, list) and len(vel_value) >= 2:
            boundary_values[face_id, 0] = vel_value[0]
            boundary_values[face_id, 1] = vel_value[1]
        elif isinstance(vel_value, (int, float)):
            boundary_values[face_id, 0] = vel_value
            boundary_values[face_id, 1] = 0.0
        else:
            boundary_values[face_id, 0] = 0.0
            boundary_values[face_id, 1] = 0.0

        boundary_values[face_id, 2] = p_value


    boundary_faces = np.array(
        sorted(list(set(boundary_faces_list))), dtype=np.int64
    )  # Ensure unique sorted global IDs

    # --- Construct binary masks ---
    face_boundary_mask = np.zeros(n_faces, dtype=np.int64)
    face_flux_mask = np.ones(n_faces, dtype=np.int64)  # assume all contribute for now

    face_boundary_mask[boundary_faces] = 1

    # Optional: refine face_flux_mask if needed later, e.g.:
    # for f in boundary_faces:
    #     vel_type, p_type = boundary_types[f]
    #     if vel_type == BC_SYMMETRY and p_type == BC_SYMMETRY:
    #         face_flux_mask[f] = 0  # skip flux computation on symmetry faces


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

    skewness_vectors = np.zeros_like(face_centers)
    for f in range(n_faces):
        P = owner_cells[f]
        x_P = cell_centers[P]
        x_f = face_centers[f]
        if neighbor_cells[f] >= 0:
            N = neighbor_cells[f]
            x_N = cell_centers[N]
            vec_pn = d_PN[f]                    # C→F line
            x_P_to_f = face_centers[f] - x_P

            # Face unit normal (already outward‑oriented)
            n_hat = face_normals[f] / (np.linalg.norm(face_normals[f]) + 1e-12)

            # Parameter t where the P–N line intersects the infinite face line:
            denom = np.dot(n_hat, vec_pn) + 1e-12
            alpha_f = np.dot(n_hat, x_P_to_f) / denom
            alpha_f = np.clip(alpha_f, 0.0, 1.0)     # keep within the segment

            face_interp_factors[f] = alpha_f
            x_interp = x_P + alpha_f * vec_pn        # lies on the face plane
            skew_vec = face_centers[f] - x_interp

            # Ensure skew_vec is tangential to the face (n̂·skew≈0)
            skewness_vectors[f] = skew_vec - np.dot(skew_vec, n_hat) * n_hat
        else:
            skewness_vectors[f] = np.zeros(2)
            face_interp_factors[f] = 1.0

    # Verify skew vectors are tangential (orthogonal to face normal)
    tangential_check = np.abs(np.sum(skewness_vectors * face_normals, axis=1))
    assert np.all(tangential_check < 1e-10), "Skew vectors not tangential to faces"

    vec_Pf = np.zeros((n_faces, 2))
    vec_fN = np.zeros((n_faces, 2))

    for f in range(n_faces):
        P = owner_cells[f]
        x_P = cell_centers[P]
        x_f = face_centers[f]
        vec_Pf[f] = x_f - x_P

        N = neighbor_cells[f]
        if N >= 0:
            x_N = cell_centers[N]
            vec_fN[f] = x_N - x_f
        else:
            vec_fN[f] = np.zeros(
                2
            )  # Or leave unassigned if truly unused for boundaries

    assert np.allclose(np.linalg.norm(vec_Pf, axis=1), delta_Pf, rtol=1e-12)
    internal_faces = np.where(neighbor_cells >= 0)[0]
    assert np.allclose(
        np.linalg.norm(vec_fN[internal_faces], axis=1),
        delta_fN[internal_faces],
        rtol=1e-12,
    )

    # --- Green–Gauss reconstruction weights ---------------------------------
    epsilon = 1e-12
    rc_interp_weights = np.zeros(n_faces)
    mask = (
        (delta_PN > epsilon)
        & (face_interp_factors > 0.0)
        & (face_interp_factors < 1.0)
    )
    rc_interp_weights[mask] = 1.0 / (
        face_interp_factors[mask]
        * (1.0 - face_interp_factors[mask])
        * delta_PN[mask]
    )


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
        unit_dPN=unit_dPN,
        delta_PN=delta_PN,
        delta_Pf=delta_Pf,
        delta_fN=delta_fN,
        vec_Pf=vec_Pf,
        vec_fN=vec_fN,
        non_ortho_correction=non_ortho_correction,
        face_interp_factors=face_interp_factors,
        internal_faces=internal_faces,
        boundary_faces=boundary_faces,
        boundary_patches=boundary_patches,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        d_PB=d_PB,
        cell_faces=cell_faces,
        rc_interp_weights=rc_interp_weights,
        skewness_vectors=skewness_vectors,
        face_boundary_mask=face_boundary_mask,
        face_flux_mask=face_flux_mask,
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
