import numpy as np
# import pygmsh # No longer needed for geometry creation
import gmsh # Import gmsh API
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
        norm = np.sqrt(normals[i,0]**2 + normals[i,1]**2)
        normals[i] /= norm if norm > 1e-12 else 1.0
    return normals


def generate(L=1.0, nx=50, ny=50, lc=0.1, output_filename=None):
    """Generate structured uniform Cartesian mesh using gmsh API"""
    model_name = "structured_uniform_gmsh"
    if gmsh.isInitialized():
        try:
            gmsh.model.setCurrent(model_name)
        except: # Model might not exist yet
             gmsh.model.add(model_name)
    else:
        gmsh.initialize()
        gmsh.model.add(model_name)

    # Define points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(L, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(L, L, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, L, 0, lc)

    # Define lines (curves)
    c1 = gmsh.model.geo.addLine(p1, p2) # bottom
    c2 = gmsh.model.geo.addLine(p2, p3) # right
    c3 = gmsh.model.geo.addLine(p3, p4) # top
    c4 = gmsh.model.geo.addLine(p4, p1) # left

    # Define curve loop and surface
    cl1 = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
    s1 = gmsh.model.geo.addPlaneSurface([cl1])

    # Synchronize before adding physical groups and mesh constraints
    gmsh.model.geo.synchronize()

    # Add physical groups (using curve/surface tags now)
    gmsh.model.addPhysicalGroup(1, [c4], tag=1, name="left_boundary")
    gmsh.model.addPhysicalGroup(1, [c1], tag=2, name="bottom_boundary")
    gmsh.model.addPhysicalGroup(1, [c2], tag=3, name="right_boundary")
    gmsh.model.addPhysicalGroup(1, [c3], tag=4, name="top_boundary")
    gmsh.model.addPhysicalGroup(2, [s1], tag=5, name="fluid_domain")

    # Set transfinite curves (using curve tags)
    gmsh.model.mesh.setTransfiniteCurve(c4, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(c2, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(c1, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(c3, nx + 1)

    # Set transfinite surface (using surface tag)
    gmsh.model.mesh.setTransfiniteSurface(s1)
    gmsh.model.mesh.setRecombine(2, s1) # Recombine surface 1

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Save mesh if filename provided
    if output_filename:
        try:
            gmsh.write(output_filename)
            print(f"Mesh saved to {output_filename}")
        except Exception as e:
            print(f"Error saving mesh to {output_filename}: {e}")

    # Extract mesh data using gmsh API
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.array(coords).reshape(-1, 3)[:, :2] # Keep only x, y

    # Map Gmsh node tags to zero-based indices for numpy arrays
    node_map = {int(tag): i for i, tag in enumerate(node_tags)} # Ensure tags are int

    # Get quads (elementType 3)
    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2, s1)
    quad_tags_gmsh = []
    quad_nodes_gmsh = []
    if 3 in elem_types:
        idx = list(elem_types).index(3) # Find index for quads
        quad_tags_gmsh = elem_tags_list[idx]
        quad_nodes_gmsh = elem_node_tags_list[idx]

    # Convert quad node tags to zero-based indices
    # Ensure tags are int before lookup
    cells = np.array([node_map[int(tag)] for tag in quad_nodes_gmsh], dtype=np.int64).reshape(-1, 4)


    # Get all boundary edges (lines, elementType 1) and their physical tags
    all_edge_nodes_list = []
    all_edge_tags_list = []
    edge_tag_to_phys_tag = {}

    # Get all curves and check their physical groups
    curves = gmsh.model.getEntities(1)
    for dim, tag in curves:
        phys_tags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if phys_tags:
            physical_tag = phys_tags[0] # Assuming one physical tag per curve
            e_types, e_tags_nested, e_node_tags_nested = gmsh.model.mesh.getElements(dim, tag)
            if 1 in e_types: # If line elements exist for this curve
                 idx = list(e_types).index(1)
                 e_tags = e_tags_nested[idx]
                 e_node_tags = e_node_tags_nested[idx]
                 all_edge_tags_list.extend(e_tags)
                 all_edge_nodes_list.extend(e_node_tags)
                 for edge_tag in e_tags:
                     edge_tag_to_phys_tag[int(edge_tag)] = physical_tag # Ensure key is int


    # Convert edge node tags to zero-based indices
    edges_np = np.array([node_map[int(tag)] for tag in all_edge_nodes_list], dtype=np.int64).reshape(-1, 2)
    all_edge_tags_np = np.array(all_edge_tags_list, dtype=np.int64) # Store Gmsh edge tags

    # Calculate geometric properties based on extracted mesh data
    cell_centers = np.array([np.mean(points[cell], axis=0) for cell in cells])
    face_centers = np.array([np.mean(points[edge], axis=0) for edge in edges_np])
    face_areas = np.array([np.linalg.norm(points[edge[1]] - points[edge[0]]) for edge in edges_np])

    # Build connectivity using numpy arrays
    owner, neighbor = build_owner_neighbor(cells, edges_np)

    # Identify boundary faces using the neighbor information
    boundary_faces_mask = (neighbor == -1)
    boundary_faces_indices = np.where(boundary_faces_mask)[0]

    # Get the gmsh tags of the boundary edges identified by owner/neighbor
    boundary_gmsh_edge_tags = all_edge_tags_np[boundary_faces_indices]

    # Get the physical types using the map
    # Handle potential KeyError if an edge tag wasn't mapped (shouldn't happen here)
    boundary_types = np.array([edge_tag_to_phys_tag.get(int(t), 0) for t in boundary_gmsh_edge_tags], dtype=np.int64)

    # gmsh.finalize() # Don't finalize if called from tester.py

    return MeshData2D(
        cell_volumes=np.full(len(cells), (L/nx)*(L/ny)),
        face_areas=face_areas, # Use recalculated based on edges_np
        face_normals=calculate_face_normals(points, edges_np), # Use numpy array
        face_centers=face_centers, # Use recalculated based on edges_np
        cell_centers=cell_centers, # Calculated based on numpy cells
        owner_cells=owner,
        neighbor_cells=neighbor,
        boundary_faces=boundary_faces_indices, # Use indices from owner/neighbor check
        boundary_types=boundary_types, # Use physical tags derived from edge map
        boundary_values=np.zeros((len(edges_np), 2)), # Placeholder, use length of numpy array
        boundary_patches=np.zeros(len(edges_np), dtype=np.int64), # Use length of numpy array
        face_interp_factors=np.full(len(edges_np), 0.5),# Use length of numpy array
        d_CF=np.zeros((len(edges_np), 2)),# Use length of numpy array
        non_ortho_correction=np.zeros((len(edges_np), 2)), # Use length of numpy array
        is_structured=True,
        is_orthogonal=True, # This mesh is orthogonal
        is_conforming=True
    )


def build_owner_neighbor(cells, edges):
    # Define Numba types using explicit Tuple syntax
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.ListType(types.int64)

    # Initialize Numba typed dictionary using the defined types
    edge_map = Dict.empty(
        key_type=key_type,
        value_type=value_type
    )

    for cell_idx, cell in enumerate(cells):
        if len(cell) == 3: # Triangle
            cell_edges = [(cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])]
        elif len(cell) == 4: # Quadrangle
            cell_edges = [(cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[3]), (cell[3], cell[0])]
        else:
            # Handle other element types or raise error if needed
            continue 
            
        for edge in cell_edges:
            # Numba compatible tuple creation from sorted list
            # Ensure edge nodes are integers for sorting and key creation
            edge_int = (int(edge[0]), int(edge[1]))
            sorted_edge = sorted(edge_int)
            key = (np.int64(sorted_edge[0]), np.int64(sorted_edge[1]))
            if key not in edge_map:
                # Initialize with Numba typed list
                edge_map[key] = List.empty_list(types.int64)
            edge_map[key].append(cell_idx)

    owner = np.full(len(edges), -1, dtype=np.int64)
    neighbor = np.full(len(edges), -1, dtype=np.int64)

    for i, edge_nodes in enumerate(edges):
        # Numba compatible tuple creation from sorted list
        # Ensure edge nodes are integers for sorting and key creation
        edge_int = (int(edge_nodes[0]), int(edge_nodes[1]))
        sorted_edge_nodes = sorted(edge_int)
        key = (np.int64(sorted_edge_nodes[0]), np.int64(sorted_edge_nodes[1]))
        if key in edge_map:
            cells_connected = edge_map[key]
            owner[i] = cells_connected[0]
            if len(cells_connected) > 1:
                neighbor[i] = cells_connected[1]
    return owner, neighbor
