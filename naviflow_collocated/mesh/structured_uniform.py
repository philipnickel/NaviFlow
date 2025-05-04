import numpy as np
import gmsh
from numba import njit, prange
from numba import types
from numba.types import UniTuple, ListType, Tuple
from numba.typed import Dict, List
from .mesh_data import MeshData2D

def calculate_face_normals(points, edges):
    normals = np.empty((len(edges), 2))
    for i in prange(len(edges)):
        n1, n2 = edges[i]
        dx = points[n2][0] - points[n1][0]
        dy = points[n2][1] - points[n1][1]
        normals[i] = np.array([dy, -dx])
        norm = np.sqrt(normals[i, 0] ** 2 + normals[i, 1] ** 2)
        normals[i] /= norm if norm > 1e-12 else 1.0
    return normals

def generate(L=1.0, nx=50, ny=50, lc=0.1, output_filename=None, model_name=None):
    if model_name is None:
        model_name = "structured_uniform_gmsh"

    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.clear()
    gmsh.model.add(model_name) 

    # Define points and geometry
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(L, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(L, L, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, L, 0, lc)

    c1 = gmsh.model.geo.addLine(p1, p2)
    c2 = gmsh.model.geo.addLine(p2, p3)
    c3 = gmsh.model.geo.addLine(p3, p4)
    c4 = gmsh.model.geo.addLine(p4, p1)

    cl1 = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
    s1 = gmsh.model.geo.addPlaneSurface([cl1])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [c4], tag=1, name="left_boundary")
    gmsh.model.addPhysicalGroup(1, [c1], tag=2, name="bottom_boundary")
    gmsh.model.addPhysicalGroup(1, [c2], tag=3, name="right_boundary")
    gmsh.model.addPhysicalGroup(1, [c3], tag=4, name="top_boundary")
    gmsh.model.addPhysicalGroup(2, [s1], tag=5, name="fluid_domain")

    gmsh.model.mesh.setTransfiniteCurve(c4, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(c2, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(c1, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(c3, nx + 1)
    gmsh.model.mesh.setTransfiniteSurface(s1)
    gmsh.model.mesh.setRecombine(2, s1)

    # ✅ Set robust export options
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.SaveElementTagType", 2)
    gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.option.setNumber("Mesh.Binary", 0)

    gmsh.model.mesh.generate(2)

    if output_filename:
        try:
            gmsh.write(output_filename)
            print(f"Mesh saved to {output_filename}")
        except Exception as e:
            print(f"Error saving mesh to {output_filename}: {e}")

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.array(coords).reshape(-1, 3)[:, :2]
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2, s1)
    quad_tags_gmsh, quad_nodes_gmsh = [], []
    if 3 in elem_types:
        idx = list(elem_types).index(3)
        quad_tags_gmsh = elem_tags_list[idx]
        quad_nodes_gmsh = elem_node_tags_list[idx]
    cells = np.array([node_map[int(tag)] for tag in quad_nodes_gmsh], dtype=np.int64).reshape(-1, 4)

    all_edge_nodes_list, all_edge_tags_list = [], []
    edge_tag_to_phys_tag = {}
    for dim, tag in gmsh.model.getEntities(1):
        phys_tags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if phys_tags:
            physical_tag = phys_tags[0]
            e_types, e_tags_nested, e_node_tags_nested = gmsh.model.mesh.getElements(dim, tag)
            if 1 in e_types:
                idx = list(e_types).index(1)
                e_tags = e_tags_nested[idx]
                e_node_tags = e_node_tags_nested[idx]
                all_edge_tags_list.extend(e_tags)
                all_edge_nodes_list.extend(e_node_tags)
                for edge_tag in e_tags:
                    edge_tag_to_phys_tag[int(edge_tag)] = physical_tag

    edges_np = np.array([node_map[int(tag)] for tag in all_edge_nodes_list], dtype=np.int64).reshape(-1, 2)
    all_edge_tags_np = np.array(all_edge_tags_list, dtype=np.int64)

    cell_centers = np.array([np.mean(points[cell], axis=0) for cell in cells])
    face_centers = np.array([np.mean(points[edge], axis=0) for edge in edges_np])
    face_areas = np.array([np.linalg.norm(points[edge[1]] - points[edge[0]]) for edge in edges_np])
    owner, neighbor = build_owner_neighbor(cells, edges_np)

    boundary_faces_indices = np.where(neighbor == -1)[0]
    boundary_gmsh_edge_tags = all_edge_tags_np[boundary_faces_indices]
    boundary_types = np.array([edge_tag_to_phys_tag.get(int(t), 0) for t in boundary_gmsh_edge_tags], dtype=np.int64)

    return MeshData2D(
        cell_volumes=np.full(len(cells), (L/nx)*(L/ny)),
        face_areas=face_areas,
        face_normals=calculate_face_normals(points, edges_np),
        face_centers=face_centers,
        cell_centers=cell_centers,
        owner_cells=owner,
        neighbor_cells=neighbor,
        boundary_faces=boundary_faces_indices,
        boundary_types=boundary_types,
        boundary_values=np.zeros((len(edges_np), 2)),
        boundary_patches=np.zeros(len(edges_np), dtype=np.int64),
        face_interp_factors=np.full(len(edges_np), 0.5),
        d_CF=np.zeros((len(edges_np), 2)),
        non_ortho_correction=np.zeros((len(edges_np), 2)),
        is_structured=True,
        is_orthogonal=True,
        is_conforming=True
    )

def build_owner_neighbor(cells, edges):
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.ListType(types.int64)
    edge_map = Dict.empty(key_type=key_type, value_type=value_type)

    for cell_idx, cell in enumerate(cells):
        if len(cell) == 3:
            cell_edges = [(cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])]
        elif len(cell) == 4:
            cell_edges = [(cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[3]), (cell[3], cell[0])]
        else:
            continue
        for edge in cell_edges:
            edge_int = (int(edge[0]), int(edge[1]))
            sorted_edge = sorted(edge_int)
            key = (np.int64(sorted_edge[0]), np.int64(sorted_edge[1]))
            if key not in edge_map:
                edge_map[key] = List.empty_list(types.int64)
            edge_map[key].append(cell_idx)

    owner = np.full(len(edges), -1, dtype=np.int64)
    neighbor = np.full(len(edges), -1, dtype=np.int64)

    for i, edge_nodes in enumerate(edges):
        edge_int = (int(edge_nodes[0]), int(edge_nodes[1]))
        sorted_edge_nodes = sorted(edge_int)
        key = (np.int64(sorted_edge_nodes[0]), np.int64(sorted_edge_nodes[1]))
        if key in edge_map:
            cells_connected = edge_map[key]
            owner[i] = cells_connected[0]
            if len(cells_connected) > 1:
                neighbor[i] = cells_connected[1]
    return owner, neighbor