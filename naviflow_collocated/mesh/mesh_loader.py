import numpy as np
import meshio
from collections import defaultdict
from naviflow_collocated.mesh.mesh_data import MeshData2D


def load_mesh(filename):
    """
    Load a 2D mesh (structured or unstructured) from a Gmsh .msh file
    and return a MeshData2D object with boundary tagging.
    """
    mesh = meshio.read(filename)
    points = mesh.points[:, :2]

    if "triangle" in mesh.cells_dict:
        return _load_unstructured_mesh(mesh, points)
    elif "quad" in mesh.cells_dict:
        return _load_structured_mesh(mesh, points)
    else:
        raise ValueError("Unsupported mesh type: must contain triangle or quad cells")


def _load_unstructured_mesh(mesh, points):
    triangles = np.array(mesh.cells_dict["triangle"], dtype=np.int64)
    boundary_lines = np.array(mesh.cells_dict.get("line", []), dtype=np.int64)
    boundary_tags = np.array(
        mesh.cell_data_dict["gmsh:physical"].get("line", []), dtype=np.int64
    )
    return _build_meshdata2d(
        points, triangles, boundary_lines, boundary_tags, is_structured=False
    )


def _load_structured_mesh(mesh, points):
    quads = np.array(mesh.cells_dict["quad"], dtype=np.int64)
    boundary_lines = np.array(mesh.cells_dict.get("line", []), dtype=np.int64)
    boundary_tags = np.array(
        mesh.cell_data_dict["gmsh:physical"].get("line", []), dtype=np.int64
    )
    return _build_meshdata2d(
        points, quads, boundary_lines, boundary_tags, is_structured=True
    )


def _build_meshdata2d(points, cells, boundary_lines, boundary_tags, is_structured):
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
    boundary_faces = []
    boundary_patches = []
    for i, line in enumerate(boundary_lines):
        edge = sorted_edge(*line)
        if edge in edge_to_face:
            boundary_faces.append(edge_to_face[edge])
            boundary_patches.append(boundary_tags[i])

    boundary_faces = np.array(boundary_faces, dtype=np.int64)
    boundary_patches = np.array(boundary_patches, dtype=np.int64)
    boundary_types = np.zeros_like(boundary_patches)
    boundary_values = np.zeros((len(boundary_faces), 2), dtype=np.float64)
    internal_faces = np.where(neighbor_cells >= 0)[0]

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
