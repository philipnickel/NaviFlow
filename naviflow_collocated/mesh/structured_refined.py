import numpy as np
# import pygmsh # No longer needed
import gmsh # Import gmsh API
from .mesh_data import MeshData2D
# Import necessary functions (adjusted to relative imports)
from .structured_uniform import calculate_face_normals, build_owner_neighbor 
from numba import njit, prange

def generate(L=1.0, nx=50, ny=50, refine_edge='left', ratio=1.2, output_filename=None, model_name=None):
    """Generate stretched grid with boundary refinement using gmsh API"""
    if model_name is None:
        model_name = "structured_refined_gmsh"
        
    if gmsh.isInitialized():
        try:
            gmsh.model.setCurrent(model_name)
        except: # Model might not exist yet
             gmsh.model.add(model_name)
    else:
        gmsh.initialize()
        gmsh.model.add(model_name)

    # Create stretched coordinates
    x = stretched_coords(nx, L, ratio, refine_edge in ['left', 'right'])
    y = stretched_coords(ny, L, ratio, refine_edge in ['top', 'bottom'])
    
    if refine_edge == 'right': x = L - x[::-1]
    if refine_edge == 'top': y = L - y[::-1]

    # Define points using stretched coordinates
    pts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            pts.append(gmsh.model.geo.addPoint(x[i], y[j], 0))

    # Define lines (structured grid - horizontal then vertical)
    horz_lines = []
    for j in range(ny + 1):
        for i in range(nx):
            p1_idx = j * (nx + 1) + i
            p2_idx = j * (nx + 1) + i + 1
            horz_lines.append(gmsh.model.geo.addLine(pts[p1_idx], pts[p2_idx]))

    vert_lines = []
    for i in range(nx + 1):
        for j in range(ny):
            p1_idx = j * (nx + 1) + i
            p2_idx = (j + 1) * (nx + 1) + i
            vert_lines.append(gmsh.model.geo.addLine(pts[p1_idx], pts[p2_idx]))
            
    # Define surface(s) - Assuming a single surface for the rectangle
    loops = []
    for j in range(ny):
        for i in range(nx):
            # Indices for lines forming the quad cell
            bottom_line_idx = j * nx + i
            top_line_idx = (j+1) * nx + i
            # Correct indexing for vertical lines:
            # Left vertical line for cell (i, j) corresponds to the j-th line in the i-th column group
            left_line_idx = i * ny + j 
            # Right vertical line for cell (i, j) corresponds to the j-th line in the (i+1)-th column group
            right_line_idx = (i+1) * ny + j 

            # Ensure correct line tags from horz_lines and vert_lines
            bottom_l = horz_lines[bottom_line_idx]
            right_l = vert_lines[right_line_idx] 
            top_l = horz_lines[top_line_idx]
            left_l = vert_lines[left_line_idx]
            
            cl = gmsh.model.geo.addCurveLoop([bottom_l, right_l, -top_l, -left_l])
            # We don't actually need to create individual loops/surfaces for a structured grid
            # loops.append(cl)
            
    # Define the single outer loop and surface
    bottom_boundary = horz_lines[0:nx]
    right_boundary = vert_lines[nx*ny : (nx+1)*ny] # Corrected index for right boundary
    top_boundary = [-l for l in horz_lines[ny*nx : (ny+1)*nx]][::-1] # Reverse and negate
    left_boundary = [-l for l in vert_lines[0:ny]][::-1] # Reverse and negate
    
    outer_loop = gmsh.model.geo.addCurveLoop(bottom_boundary + right_boundary + top_boundary + left_boundary)
    s1 = gmsh.model.geo.addPlaneSurface([outer_loop])
    
    # Synchronize before adding physical groups and mesh constraints
    gmsh.model.geo.synchronize()
    
    # Add physical groups for boundaries
    left_tag = gmsh.model.addPhysicalGroup(1, left_boundary, tag=1, name="left_boundary")
    bottom_tag = gmsh.model.addPhysicalGroup(1, bottom_boundary, tag=2, name="bottom_boundary")
    right_tag = gmsh.model.addPhysicalGroup(1, right_boundary, tag=3, name="right_boundary")
    top_tag = gmsh.model.addPhysicalGroup(1, top_boundary, tag=4, name="top_boundary")
    surf_tag = gmsh.model.addPhysicalGroup(2, [s1], tag=5, name="fluid_domain")
    
    # Setting transfinite automatic is not a direct gmsh api call.
    # We need to set it for curves and surfaces manually if needed.
    # For a structured grid from points, transfinite constraints might not be needed
    # if the point distribution itself defines the structure.
    # Let's mesh without explicit transfinite for now. We might need to add them
    # if the mesh is not structured as expected.
    # gmsh.model.mesh.setTransfiniteAutomatic() # Removed
    
    # Recombine the mesh
    gmsh.model.mesh.setRecombine(2, s1)

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

    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2, s1)
    quad_tags_gmsh = []
    quad_nodes_gmsh = []
    if 3 in elem_types:
        idx = list(elem_types).index(3)
        quad_tags_gmsh = elem_tags_list[idx]
        quad_nodes_gmsh = elem_node_tags_list[idx]

    cells = np.array([node_map[int(tag)] for tag in quad_nodes_gmsh], dtype=np.int64).reshape(-1, 4)

    # Get all boundary edges and their physical tags
    all_edge_nodes_list = []
    all_edge_tags_list = []
    edge_tag_to_phys_tag = {}
    curves = gmsh.model.getEntities(1)
    for dim, tag in curves:
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

    # Calculate geometric properties
    cell_centers_np = np.array([np.mean(points[cell], axis=0) for cell in cells])
    face_centers_np = np.array([np.mean(points[edge], axis=0) for edge in edges_np])
    face_areas_np = np.array([np.linalg.norm(points[edge[1]] - points[edge[0]]) for edge in edges_np])
    cell_volumes_np = calculate_cell_volumes(points, cells) # Use existing function

    # Build connectivity
    owner, neighbor = build_owner_neighbor(cells, edges_np) # Pass numpy array

    # Identify boundary faces and types
    boundary_faces_indices = np.where(neighbor == -1)[0]
    boundary_gmsh_edge_tags = all_edge_tags_np[boundary_faces_indices]
    boundary_types = np.array([edge_tag_to_phys_tag.get(int(t), 0) for t in boundary_gmsh_edge_tags], dtype=np.int64)

    # gmsh.finalize() # Don't finalize if called from tester.py

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
        boundary_values=np.zeros((len(edges_np), 2)), # Placeholder
        boundary_patches=np.zeros(len(edges_np), dtype=np.int64), # Placeholder
        face_interp_factors=np.full(len(edges_np), 0.5), # Placeholder
        d_CF=np.zeros((len(edges_np), 2)), # Placeholder
        non_ortho_correction=np.zeros((len(edges_np), 2)), # Placeholder
        is_structured=True,
        is_orthogonal=False, # Stretched grid is not orthogonal
        is_conforming=True
    )

def stretched_coords(n, L, ratio, stretch):
    if not stretch or abs(ratio - 1.0) < 1e-6 : # Added check for ratio near 1
        return np.linspace(0, L, n+1)
    
    # Handle potential division by zero if ratio is exactly 1
    if abs(ratio - 1.0) < 1e-10:
       return np.linspace(0, L, n+1)

    # Original formula can cause issues if ratio is near 1, use alternative for stability
    # Check if ratio * (xi - 1) approaches zero, which can lead to precision issues
    # Simplified approach: use geometric series sum formula
    # Let the first interval be h, then h * ratio, h * ratio^2, ..., h * ratio^(n-1)
    # Sum = h * (1 - ratio^n) / (1 - ratio) = L
    # h = L * (1 - ratio) / (1 - ratio^n)
    
    if ratio > 0:
        h = L * (1 - ratio) / (1 - ratio**(n)) # Corrected based on n intervals
        coords = np.zeros(n + 1)
        coords[1:] = h * (1 - ratio**np.arange(1, n + 1)) / (1 - ratio)
        coords[0] = 0 # Ensure start is exactly 0
        coords[n] = L # Ensure end is exactly L
        
        # Need to compute the intermediate points based on the geometric series
        coords_new = np.zeros(n+1)
        coords_new[0] = 0
        current_pos = 0
        first_interval = L * (ratio - 1) / (ratio**n - 1) # Correct first interval size
        for i in range(1, n + 1):
            current_pos += first_interval * (ratio**(i-1))
            coords_new[i] = current_pos
        coords_new[-1] = L # Ensure the last point is exactly L
        
        return coords_new

    else: # Fallback to original method or handle negative ratio if needed
         xi = np.linspace(0, 1, n+1)
         # Prevent potential overflow/underflow with large ratios
         # Using np.expm1 for potentially better precision near ratio*xi = 0
         exp_ratio_xi = np.exp(ratio * xi)
         exp_ratio = np.exp(ratio)
         # Check for potential division by zero
         if abs(exp_ratio - 1) < 1e-12:
             return np.linspace(0, L, n+1)
         return L * (exp_ratio_xi - 1) / (exp_ratio - 1)


@njit
def calculate_cell_volumes(points, cells):
    volumes = np.empty(len(cells))
    for i in prange(len(cells)):
        cell = cells[i]
        # Assuming quad cells from recombination
        if len(cell) == 4:
            a, b, c, d = points[cell]
            # Shoelace formula for polygon area (works for convex/concave quads)
            # More robust than splitting into triangles for potentially non-planar quads
            area = 0.5 * np.abs(a[0]*b[1] + b[0]*c[1] + c[0]*d[1] + d[0]*a[1] - 
                                (a[1]*b[0] + b[1]*c[0] + c[1]*d[0] + d[1]*a[0]))
            volumes[i] = area
        else: # Fallback for other cell types if necessary
             volumes[i] = 0.0 # Or handle appropriately
    return volumes
