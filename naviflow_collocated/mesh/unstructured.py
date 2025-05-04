import numpy as np
import gmsh
from .mesh_data import MeshData2D
from .structured_uniform import calculate_face_normals, build_owner_neighbor
from numba import njit, prange

def generate(L=1.0, n_cells=1000, ratio=2.5, output_filename=None, model_name=None):
    """
    Generate unstructured mesh with refinement near boundaries and corners.
    
    Parameters:
    -----------
    L : float
        Domain length (square domain with side length L)
    n_cells : int
        Target number of cells for the mesh
    ratio : float
        Refinement ratio between boundaries and center
    output_filename : str, optional
        Filename to save the mesh to
    model_name : str, optional
        Name of the Gmsh model
    """
    if model_name is None:
        model_name = "unstructured_gmsh"

    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.clear()
    gmsh.model.add(model_name)
    
    # Approximate cell size based on target cell count
    h = L / np.sqrt(n_cells / 4.2)  # Adjusted factor based on testing
    
    # Define mesh sizes
    h_min = h / ratio  # Size at boundaries
    h_max = h * 1.2    # Size at center
    
    print(f"Estimated mesh parameters for {n_cells} cells:")
    print(f"  - Base cell size: {h:.5f}")
    print(f"  - Min cell size: {h_min:.5f} (boundaries)")
    print(f"  - Max cell size: {h_max:.5f} (center)")
    
    # Create square domain
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(L, 0, 0)
    p3 = gmsh.model.geo.addPoint(L, L, 0)
    p4 = gmsh.model.geo.addPoint(0, L, 0)
    
    # Define boundary lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left
    
    # Create surface
    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    gmsh.model.geo.synchronize()
    
    # Set up physical groups for boundaries
    bottom_tag = gmsh.model.addPhysicalGroup(1, [l1], 1)
    right_tag = gmsh.model.addPhysicalGroup(1, [l2], 2)
    top_tag = gmsh.model.addPhysicalGroup(1, [l3], 3)
    left_tag = gmsh.model.addPhysicalGroup(1, [l4], 4)
    fluid_tag = gmsh.model.addPhysicalGroup(2, [surface], 5)
    
    # Name the physical groups
    gmsh.model.setPhysicalName(1, bottom_tag, "bottom_boundary")
    gmsh.model.setPhysicalName(1, right_tag, "right_boundary")
    gmsh.model.setPhysicalName(1, top_tag, "top_boundary")
    gmsh.model.setPhysicalName(1, left_tag, "left_boundary")
    gmsh.model.setPhysicalName(2, fluid_tag, "fluid_domain")
    
    # MESH REFINEMENT: Use distance field to refine near boundaries
    # 1. Create distance field from boundaries
    field_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field_distance, "EdgesList", [l1, l2, l3, l4])
    
    # 2. Create a threshold field that varies mesh size with distance from boundaries
    field_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field_threshold, "IField", field_distance)
    gmsh.model.mesh.field.setNumber(field_threshold, "LcMin", h_min)
    gmsh.model.mesh.field.setNumber(field_threshold, "LcMax", h_max)
    gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", L/3)
    
    # 3. Set this field as the background mesh size field
    gmsh.model.mesh.field.setAsBackgroundMesh(field_threshold)
    
    # Mesh settings
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    # Generate the mesh
    gmsh.model.mesh.generate(2)
    
    # Report on mesh
    try:
        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(2, -1)
        actual_cells = sum(len(tags) for tags in elem_tags)
        print(f"Actual number of cells generated: {actual_cells}")
        print(f"Cell count ratio: {actual_cells/n_cells:.2f}x target")
    except Exception as e:
        print(f"Warning: Could not count cells: {e}")
    
    # Save mesh if filename provided
    if output_filename:
        try:
            gmsh.write(output_filename)
            print(f"Mesh saved to {output_filename}")
        except Exception as e:
            print(f"Error saving mesh to {output_filename}: {e}")
    
    # Extract mesh data for NaviFlow
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.array(coords).reshape(-1, 3)[:, :2]
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(2, surface)
    idx = list(elem_types).index(2) if 2 in elem_types else None
    if idx is None:
        raise ValueError("No triangle elements found.")
    cells = np.array([node_map[int(n)] for n in elem_nodes[idx]]).reshape(-1, 3)

    edge_nodes, edge_tags, tag_map = [], [], {}
    for dim, tag in gmsh.model.getEntities(1):
        phys = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if not phys:
            continue
        phys_tag = phys[0]
        e_types, tags_nested, nodes_nested = gmsh.model.mesh.getElements(dim, tag)
        if 1 not in e_types:
            continue
        e_idx = list(e_types).index(1) if 1 in e_types else None
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
    btypes = np.array([tag_map.get(int(t), 0) for t in edge_tags_np[bface_idx]], dtype=np.int64)

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
        is_structured=False,
        is_orthogonal=False,
        is_conforming=True
    )

@njit
def calculate_cell_volumes(points, cells):
    vols = np.empty(len(cells))
    for i in prange(len(cells)):
        a, b, c = points[cells[i]]
        # Shoelace formula for triangle area
        vols[i] = 0.5 * abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
    return vols