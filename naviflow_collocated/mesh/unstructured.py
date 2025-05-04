import numpy as np
import gmsh
import math
from .mesh_data import MeshData2D
from .structured_uniform import calculate_face_normals, build_owner_neighbor
from numba import njit, prange

def generate_naca_profile(naca_digits, chord_length=1.0, num_points=100):
    """
    Generate a NACA 4-digit airfoil profile.
    
    Parameters:
    -----------
    naca_digits : str
        The 4-digit NACA profile (e.g., '0012')
    chord_length : float
        The chord length of the airfoil
    num_points : int
        Number of points to generate along the profile
        
    Returns:
    --------
    tuple of numpy arrays
        (x_coords, y_coords) for the airfoil profile
    """
    # Parse NACA digits
    if len(naca_digits) != 4:
        raise ValueError("NACA profile must be a 4-digit string")
    
    try:
        m = int(naca_digits[0]) / 100.0    # Maximum camber
        p = int(naca_digits[1]) / 10.0     # Location of maximum camber
        t = int(naca_digits[2:]) / 100.0   # Maximum thickness
    except ValueError:
        raise ValueError("NACA digits must be integers")
    
    # Generate x-coordinates with cosine spacing (clustered at leading and trailing edge)
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta)) * chord_length
    
    # Calculate thickness distribution
    yt = 5 * t * chord_length * (0.2969 * np.sqrt(x/chord_length) - 
                                0.1260 * (x/chord_length) - 
                                0.3516 * (x/chord_length)**2 + 
                                0.2843 * (x/chord_length)**3 - 
                                0.1015 * (x/chord_length)**4)
    
    # Initialize arrays for upper and lower surface
    xu = np.zeros_like(x)
    xl = np.zeros_like(x)
    yu = np.zeros_like(x)
    yl = np.zeros_like(x)
    
    # Calculate camber line and surface coordinates
    if m == 0:  # Symmetric airfoil
        xu = x
        xl = x
        yu = yt
        yl = -yt
    else:
        # Calculate camber line
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        # Split calculations based on position relative to p
        for i in range(len(x)):
            xi = x[i] / chord_length
            if xi < p:
                yc[i] = m * (xi / p**2) * (2*p - xi)
                dyc_dx[i] = (2*m / p**2) * (p - xi)
            else:
                yc[i] = m * (1 - 2*p + (2*p - xi) / (1-p)**2)
                dyc_dx[i] = (2*m / (1-p)**2) * (p - xi)
        
        # Calculate final coordinates
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
    
    # Combine upper and lower surfaces in the correct order
    # Start from trailing edge, go around leading edge, back to trailing edge
    # Reverse lower surface points to maintain CCW order
    x_coords = np.concatenate((xu[::-1][:-1], xl))
    y_coords = np.concatenate((yu[::-1][:-1], yl))
    
    return x_coords, y_coords

def generate_custom_geometry(geom_type, **params):
    """
    Generate coordinates for custom geometry types.
    
    Parameters:
    -----------
    geom_type : str
        Type of geometry ('naca', 'file', 'points')
    params : dict
        Parameters specific to the geometry type
        
    Returns:
    --------
    tuple of numpy arrays
        (x_coords, y_coords) for the geometry
    """
    if geom_type == 'naca':
        # NACA airfoil profile
        naca_digits = params.get('digits', '0012')
        chord_length = params.get('chord', 1.0)
        num_points = params.get('points', 100)
        angle_deg = params.get('angle', 0.0)  # Angle of attack in degrees
        
        # Generate base profile
        x_coords, y_coords = generate_naca_profile(naca_digits, chord_length, num_points)
        
        # Apply rotation if needed
        if angle_deg != 0:
            angle_rad = np.radians(angle_deg)
            # Rotate coordinates around origin
            x_rot = x_coords * np.cos(angle_rad) - y_coords * np.sin(angle_rad)
            y_rot = x_coords * np.sin(angle_rad) + y_coords * np.cos(angle_rad)
            x_coords, y_coords = x_rot, y_rot
            
        return x_coords, y_coords
        
    elif geom_type == 'file':
        # Load coordinates from file
        filepath = params.get('path', '')
        if not filepath:
            raise ValueError("File path must be provided for 'file' geometry type")
        
        try:
            data = np.loadtxt(filepath)
            if data.shape[1] < 2:
                raise ValueError("File must contain at least 2 columns (x, y)")
            return data[:, 0], data[:, 1]
        except Exception as e:
            raise ValueError(f"Failed to load geometry from file: {e}")
            
    elif geom_type == 'points':
        # Use provided points directly
        points = params.get('coords', [])
        if not points or len(points) < 3:
            raise ValueError("At least 3 points must be provided for 'points' geometry type")
            
        # Convert to numpy arrays if needed
        if isinstance(points, list):
            points = np.array(points)
            
        if points.shape[1] < 2:
            raise ValueError("Points must be (x,y) pairs")
            
        return points[:, 0], points[:, 1]
        
    else:
        raise ValueError(f"Unknown geometry type: {geom_type}")

def generate(Lx=1.0, Ly=1.0, n_cells=1000, ratio=2.5, obstacle=None, output_filename=None, model_name=None):
    """
    Generate unstructured mesh with refinement near boundaries and optional internal obstacles.
    
    Parameters:
    -----------
    Lx : float
        Domain length in x-direction
    Ly : float
        Domain length in y-direction
    n_cells : int
        Target number of cells for the mesh
    ratio : float
        Refinement ratio between boundaries and center
    obstacle : dict, optional
        Dictionary specifying an internal obstacle. Supported types:
        - Circle: {'type': 'circle', 'center': (x,y), 'radius': r}
        - Rectangle: {'type': 'rectangle', 'start': (x1,y1), 'end': (x2,y2)}
        - Custom: {'type': 'custom', 'geometry': 'naca|file|points', ...params}
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
    
    # Approximate cell size based on target cell count and domain area
    area = Lx * Ly
    h = np.sqrt(area / (n_cells / 4.2))  # Adjusted factor based on testing
    
    # Define mesh sizes
    h_min = h / ratio  # Size at boundaries
    h_max = h * 1.2    # Size at center
    
    print(f"Estimated mesh parameters for {n_cells} cells:")
    print(f"  - Base cell size: {h:.5f}")
    print(f"  - Min cell size: {h_min:.5f} (boundaries)")
    print(f"  - Max cell size: {h_max:.5f} (center)")
    
    # Create domain boundary
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0)
    
    # Define boundary lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left
    
    external_boundary = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    
    # Create obstacle if specified
    obstacle_boundary = None
    obstacle_lines = []
    if obstacle:
        obstacle_type = obstacle.get('type', '').lower()
        
        if obstacle_type == 'circle':
            center = obstacle.get('center', (Lx/2, Ly/2))
            radius = obstacle.get('radius', min(Lx, Ly)/8)
            
            # Add center point
            pc = gmsh.model.geo.addPoint(center[0], center[1], 0)
            
            # Add four points around the circle
            p_circ1 = gmsh.model.geo.addPoint(center[0] + radius, center[1], 0)
            p_circ2 = gmsh.model.geo.addPoint(center[0], center[1] + radius, 0)
            p_circ3 = gmsh.model.geo.addPoint(center[0] - radius, center[1], 0)
            p_circ4 = gmsh.model.geo.addPoint(center[0], center[1] - radius, 0)
            
            # Create circle arcs
            circ1 = gmsh.model.geo.addCircleArc(p_circ1, pc, p_circ2)
            circ2 = gmsh.model.geo.addCircleArc(p_circ2, pc, p_circ3)
            circ3 = gmsh.model.geo.addCircleArc(p_circ3, pc, p_circ4)
            circ4 = gmsh.model.geo.addCircleArc(p_circ4, pc, p_circ1)
            
            obstacle_boundary = gmsh.model.geo.addCurveLoop([circ1, circ2, circ3, circ4])
            obstacle_lines = [circ1, circ2, circ3, circ4]
            
        elif obstacle_type == 'rectangle':
            start = obstacle.get('start', (Lx/4, Ly/4))
            end = obstacle.get('end', (3*Lx/4, 3*Ly/4))
            
            # Add corners of rectangle
            p_rect1 = gmsh.model.geo.addPoint(start[0], start[1], 0)
            p_rect2 = gmsh.model.geo.addPoint(end[0], start[1], 0)
            p_rect3 = gmsh.model.geo.addPoint(end[0], end[1], 0)
            p_rect4 = gmsh.model.geo.addPoint(start[0], end[1], 0)
            
            # Create rectangle lines
            rect1 = gmsh.model.geo.addLine(p_rect1, p_rect2)
            rect2 = gmsh.model.geo.addLine(p_rect2, p_rect3)
            rect3 = gmsh.model.geo.addLine(p_rect3, p_rect4)
            rect4 = gmsh.model.geo.addLine(p_rect4, p_rect1)
            
            obstacle_boundary = gmsh.model.geo.addCurveLoop([rect1, rect2, rect3, rect4])
            obstacle_lines = [rect1, rect2, rect3, rect4]
            
        elif obstacle_type == 'custom':
            # Get geometry parameters
            geom_type = obstacle.get('geometry', '')
            geom_params = obstacle.get('params', {})
            position = obstacle.get('position', (Lx/2, Ly/2))
            scale = obstacle.get('scale', 1.0)
            
            # Generate the geometry coordinates
            try:
                x_coords, y_coords = generate_custom_geometry(geom_type, **geom_params)
                
                # Apply scaling and translation
                x_coords = x_coords * scale + position[0]
                y_coords = y_coords * scale + position[1]
                
                # Add points for the geometry
                points = []
                for i in range(len(x_coords)):
                    points.append(gmsh.model.geo.addPoint(x_coords[i], y_coords[i], 0))
                
                # Create lines connecting the points
                lines = []
                for i in range(len(points)-1):
                    lines.append(gmsh.model.geo.addLine(points[i], points[i+1]))
                # Close the loop
                lines.append(gmsh.model.geo.addLine(points[-1], points[0]))
                
                obstacle_boundary = gmsh.model.geo.addCurveLoop(lines)
                obstacle_lines = lines
                
                print(f"Created custom geometry with {len(points)} points and {len(lines)} lines")
                
            except Exception as e:
                print(f"Error creating custom geometry: {e}")
                obstacle_boundary = None
                obstacle_lines = []
    
    # Create surface with or without hole
    if obstacle_boundary:
        # Create surface with hole
        surface = gmsh.model.geo.addPlaneSurface([external_boundary, obstacle_boundary])
    else:
        # Create simple surface
        surface = gmsh.model.geo.addPlaneSurface([external_boundary])
    
    gmsh.model.geo.synchronize()
    
    # Physical groups for boundaries
    bottom_tag = gmsh.model.addPhysicalGroup(1, [l1], 1)
    right_tag = gmsh.model.addPhysicalGroup(1, [l2], 2)
    top_tag = gmsh.model.addPhysicalGroup(1, [l3], 3)
    left_tag = gmsh.model.addPhysicalGroup(1, [l4], 4)
    
    # Add obstacle physical group if present
    if obstacle_lines:
        obstacle_tag = gmsh.model.addPhysicalGroup(1, obstacle_lines, 5)
        gmsh.model.setPhysicalName(1, obstacle_tag, "obstacle_boundary")
    
    # Add fluid domain
    fluid_tag = gmsh.model.addPhysicalGroup(2, [surface], 10)
    
    # Name physical groups
    gmsh.model.setPhysicalName(1, bottom_tag, "bottom_boundary")
    gmsh.model.setPhysicalName(1, right_tag, "right_boundary")
    gmsh.model.setPhysicalName(1, top_tag, "top_boundary")
    gmsh.model.setPhysicalName(1, left_tag, "left_boundary")
    gmsh.model.setPhysicalName(2, fluid_tag, "fluid_domain")
    
    # MESH REFINEMENT using distance fields
    boundary_edges = [l1, l2, l3, l4]
    if obstacle_lines:
        boundary_edges.extend(obstacle_lines)
    
    # Create distance field from all boundaries
    field_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field_distance, "EdgesList", boundary_edges)
    
    # Create threshold field that varies mesh size with distance from boundaries
    field_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field_threshold, "IField", field_distance)
    gmsh.model.mesh.field.setNumber(field_threshold, "LcMin", h_min)
    gmsh.model.mesh.field.setNumber(field_threshold, "LcMax", h_max)
    gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", min(Lx, Ly)/3)
    
    # Set this field as the background mesh size field
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