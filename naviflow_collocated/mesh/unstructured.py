import numpy as np
import gmsh
from .mesh_data import MeshData2D
from .structured_uniform import calculate_face_normals, build_owner_neighbor 
from numba import njit, prange

def generate(L=1.0, obstacle_radius=0.1, output_filename=None):
    model_name = "unstructured_gmsh"
    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.clear()
    gmsh.model.add(model_name) 

    # Create geometry
    gmsh.model.occ.addRectangle(0, 0, 0, 2.2 * L, 0.41 * L, tag=1)
    obstacle = gmsh.model.occ.addDisk(0.2 * L, 0.2 * L, 0, obstacle_radius, obstacle_radius, tag=2)

    cut_result, _ = gmsh.model.occ.cut([(2, 1)], [(2, obstacle)], removeObject=True, removeTool=True)
    surface_tag = cut_result[0][1]
    gmsh.model.occ.synchronize()

    # Identify boundary curves
    boundary_entities = gmsh.model.getBoundary([(2, surface_tag)], combined=False, oriented=False)
    eps = 1e-6
    center_x, center_y = 0.2 * L, 0.2 * L
    curves_around_obstacle = gmsh.model.getEntitiesInBoundingBox(
        center_x - obstacle_radius - eps, center_y - obstacle_radius - eps, -eps,
        center_x + obstacle_radius + eps, center_y + obstacle_radius + eps, eps, dim=1
    )
    obstacle_curve_tags = [c[1] for c in curves_around_obstacle]
    all_curve_tags = [c[1] for c in boundary_entities]
    channel_curve_tags = list(set(all_curve_tags) - set(obstacle_curve_tags))

    # Add physical groups
    if obstacle_curve_tags:
        gmsh.model.addPhysicalGroup(1, obstacle_curve_tags, tag=1, name="cylinder")
    if channel_curve_tags:
        gmsh.model.addPhysicalGroup(1, channel_curve_tags, tag=2, name="walls")
    gmsh.model.addPhysicalGroup(2, [surface_tag], tag=3, name="fluid_domain")

    # Mesh options to ensure grouping
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.SaveElementTagType", 2)
    gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.option.setNumber("Mesh.Binary", 0)

    # Generate mesh
    gmsh.model.mesh.generate(2)

    if output_filename:
        try:
            gmsh.write(output_filename)
            print(f"Mesh saved to {output_filename}")
        except Exception as e:
            print(f"Error saving mesh to {output_filename}: {e}")

    # Mesh extraction
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.array(coords).reshape(-1, 3)[:, :2]
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2, surface_tag)
    tri_tags_gmsh, tri_nodes_gmsh = [], []
    if 2 in elem_types:
        idx = elem_types.index(2)
        tri_tags_gmsh = elem_tags_list[idx]
        tri_nodes_gmsh = elem_node_tags_list[idx]
    cells = np.array([node_map[int(tag)] for tag in tri_nodes_gmsh], dtype=np.int64).reshape(-1, 3)

    # Boundary edges and tagging
    all_edge_nodes_list, all_edge_tags_list = [], []
    edge_tag_to_phys_tag = {}
    for dim, tag in gmsh.model.getEntities(1):
        phys_tags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if phys_tags:
            physical_tag = phys_tags[0]
            e_types, e_tags_nested, e_node_tags_nested = gmsh.model.mesh.getElements(dim, tag)
            if 1 in e_types:
                idx = e_types.index(1)
                e_tags = e_tags_nested[idx]
                e_node_tags = e_node_tags_nested[idx]
                all_edge_tags_list.extend(e_tags)
                all_edge_nodes_list.extend(e_node_tags)
                for edge_tag in e_tags:
                    edge_tag_to_phys_tag[int(edge_tag)] = physical_tag

    edges_np = np.array([node_map[int(tag)] for tag in all_edge_nodes_list], dtype=np.int64).reshape(-1, 2)
    all_edge_tags_np = np.array(all_edge_tags_list, dtype=np.int64)

    cell_centers_np = np.array([np.mean(points[cell], axis=0) for cell in cells])
    face_centers_np = np.array([np.mean(points[edge], axis=0) for edge in edges_np])
    face_areas_np = np.array([np.linalg.norm(points[edge[1]] - points[edge[0]]) for edge in edges_np])
    cell_volumes_np = calculate_cell_volumes(points, cells)

    owner, neighbor = build_owner_neighbor(cells, edges_np)

    boundary_faces_indices = np.where(neighbor == -1)[0]
    boundary_gmsh_edge_tags = all_edge_tags_np[boundary_faces_indices]
    boundary_types = np.array([edge_tag_to_phys_tag.get(int(t), 0) for t in boundary_gmsh_edge_tags], dtype=np.int64)

    return MeshData2D(
        cell_volumes=cell_volumes_np,
        face_areas=face_areas_np,
        face_normals=calculate_face_normals(points, edges_np),
        face_centers=face_centers_np,
        cell_centers=cell_centers_np,
        owner_cells=owner,
        neighbor_cells=neighbor,
        boundary_faces=boundary_faces_indices,
        boundary_types=boundary_types,
        boundary_values=np.zeros((len(edges_np), 2)),
        boundary_patches=np.zeros(len(edges_np), dtype=np.int64),
        face_interp_factors=np.full(len(edges_np), 0.5),
        d_CF=np.zeros((len(edges_np), 2)),
        non_ortho_correction=np.zeros((len(edges_np), 2)),
        is_structured=False,
        is_orthogonal=False,
        is_conforming=True
    )

@njit
def calculate_cell_volumes(points, cells):
    volumes = np.empty(len(cells))
    for i in prange(len(cells)):
        if len(cells[i]) == 3:
            a, b, c = cells[i]
            vol = 0.5 * np.abs(
                (points[b][0] - points[a][0]) * (points[c][1] - points[a][1]) -
                (points[c][0] - points[a][0]) * (points[b][1] - points[a][1])
            )
            volumes[i] = vol
        else:
            volumes[i] = 0.0
    return volumes