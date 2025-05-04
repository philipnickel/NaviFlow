import numpy as np
import gmsh
from .mesh_data import MeshData2D
from .structured_uniform import calculate_face_normals, build_owner_neighbor
from numba import njit, prange

def generate(L=1.0, nx=50, ny=50, refine_edge='left', ratio=1.2, output_filename=None, model_name=None):
    if model_name is None:
        model_name = "structured_refined_gmsh"

    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.clear()
    gmsh.model.add(model_name) 

    x = stretched_coords(nx, L, ratio, refine_edge in ['left', 'right'])
    y = stretched_coords(ny, L, ratio, refine_edge in ['top', 'bottom'])

    if refine_edge == 'right': x = L - x[::-1]
    if refine_edge == 'top': y = L - y[::-1]

    pts = [gmsh.model.geo.addPoint(x[i], y[j], 0) for j in range(ny + 1) for i in range(nx + 1)]

    horz_lines = [gmsh.model.geo.addLine(pts[j * (nx + 1) + i], pts[j * (nx + 1) + i + 1]) for j in range(ny + 1) for i in range(nx)]
    vert_lines = [gmsh.model.geo.addLine(pts[j * (nx + 1) + i], pts[(j + 1) * (nx + 1) + i]) for i in range(nx + 1) for j in range(ny)]

    for j in range(ny):
        for i in range(nx):
            bottom = horz_lines[j * nx + i]
            right = vert_lines[(i + 1) * ny + j]
            top = horz_lines[(j + 1) * nx + i]
            left = vert_lines[i * ny + j]
            gmsh.model.geo.addCurveLoop([bottom, right, -top, -left])

    bottom_b = horz_lines[:nx]
    right_b = vert_lines[nx * ny:]
    top_b = [-l for l in horz_lines[ny * nx:]][::-1]
    left_b = [-l for l in vert_lines[:ny]][::-1]

    outer_loop = gmsh.model.geo.addCurveLoop(bottom_b + right_b + top_b + left_b)
    s1 = gmsh.model.geo.addPlaneSurface([outer_loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, left_b, 1)
    gmsh.model.addPhysicalGroup(1, bottom_b, 2)
    gmsh.model.addPhysicalGroup(1, right_b, 3)
    gmsh.model.addPhysicalGroup(1, top_b, 4)
    gmsh.model.addPhysicalGroup(2, [s1], 5)

    gmsh.model.mesh.setRecombine(2, s1)
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

    elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(2, s1)
    idx = elem_types.index(3) if 3 in elem_types else None
    if idx is None:
        raise ValueError("No quadrilateral elements found.")
    cells = np.array([node_map[int(n)] for n in elem_nodes[idx]]).reshape(-1, 4)

    edge_nodes, edge_tags, tag_map = [], [], {}
    for dim, tag in gmsh.model.getEntities(1):
        phys = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if not phys:
            continue
        phys_tag = phys[0]
        e_types, tags_nested, nodes_nested = gmsh.model.mesh.getElements(dim, tag)
        if 1 not in e_types:
            continue
        e_idx = e_types.index(1)
        edge_tags.extend(tags_nested[e_idx])
        edge_nodes.extend(nodes_nested[e_idx])
        for t in tags_nested[e_idx]:
            tag_map[int(t)] = phys_tag

    edges_np = np.array([node_map[int(n)] for n in edge_nodes], dtype=np.int64).reshape(-1, 2)
    edge_tags_np = np.array(edge_tags, dtype=np.int64)

    cell_centers = np.array([np.mean(points[c], axis=0) for c in cells])
    face_centers = np.array([np.mean(points[e], axis=0) for e in edges_np])
    face_areas = np.linalg.norm(points[edges_np[:, 1]] - points[edges_np[:, 0]], axis=1)
    cell_vols = calculate_cell_volumes(points, cells)

    owner, neighbor = build_owner_neighbor(cells, edges_np)
    bface_idx = np.where(neighbor == -1)[0]
    btypes = np.array([tag_map.get(int(t), 0) for t in edge_tags_np[bface_idx]])

    return MeshData2D(
        cell_volumes=cell_vols,
        face_areas=face_areas,
        face_normals=calculate_face_normals(points, edges_np),
        face_centers=face_centers,
        cell_centers=cell_centers,
        owner_cells=owner,
        neighbor_cells=neighbor,
        boundary_faces=bface_idx,
        boundary_types=btypes,
        boundary_values=np.zeros((len(edges_np), 2)),
        boundary_patches=np.zeros(len(edges_np), dtype=np.int64),
        face_interp_factors=np.full(len(edges_np), 0.5),
        d_CF=np.zeros((len(edges_np), 2)),
        non_ortho_correction=np.zeros((len(edges_np), 2)),
        is_structured=True,
        is_orthogonal=False,
        is_conforming=True
    )

@njit
def calculate_cell_volumes(points, cells):
    vols = np.empty(len(cells))
    for i in prange(len(cells)):
        a, b, c, d = points[cells[i]]
        vols[i] = 0.5 * abs(a[0]*b[1] + b[0]*c[1] + c[0]*d[1] + d[0]*a[1] -
                            (a[1]*b[0] + b[1]*c[0] + c[1]*d[0] + d[1]*a[0]))
    return vols

def stretched_coords(n, L, ratio, apply):
    if not apply or abs(ratio - 1.0) < 1e-6:
        return np.linspace(0, L, n + 1)
    h = L * (1 - ratio) / (1 - ratio**n)
    coords = np.zeros(n + 1)
    coords[0] = 0
    for i in range(1, n + 1):
        coords[i] = coords[i - 1] + h * (ratio ** (i - 1))
    coords[-1] = L
    return coords