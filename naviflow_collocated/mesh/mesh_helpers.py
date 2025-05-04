import numpy as np
from collections import defaultdict

def calculate_face_normals(points, edges):
    normals = np.empty((len(edges), 2))
    for i, (n1, n2) in enumerate(edges):
        dx, dy = points[n2][0] - points[n1][0], points[n2][1] - points[n1][1]
        normals[i] = [dy, -dx]
        norm = np.linalg.norm(normals[i])
        if norm > 1e-12:
            normals[i] /= norm
    return normals

def calculate_cell_centers(points, cells):
    return np.array([np.mean(points[cell], axis=0) for cell in cells])

def calculate_face_centers(points, edges):
    return np.array([np.mean(points[edge], axis=0) for edge in edges])

def calculate_face_areas(points, edges):
    return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)

def calculate_cell_volumes(points, cells):
    """Calculate cell area for arbitrary polygon cells (2D) using Green's theorem."""
    volumes = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        x = points[cell, 0]
        y = points[cell, 1]
        volumes[i] = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return volumes

def build_owner_neighbor(cells, edges):
    edge_map = defaultdict(list)
    for i, cell in enumerate(cells):
        e = [(cell[j], cell[(j + 1) % len(cell)]) for j in range(len(cell))]
        for a, b in e:
            key = tuple(sorted((a, b)))
            edge_map[key].append(i)

    owner = np.full(len(edges), -1, dtype=np.int64)
    neighbor = np.full(len(edges), -1, dtype=np.int64)
    for i, (a, b) in enumerate(edges):
        key = tuple(sorted((a, b)))
        cells_ = edge_map.get(key, [])
        if cells_:
            owner[i] = cells_[0]
            if len(cells_) > 1:
                neighbor[i] = cells_[1]
    return owner, neighbor

def calculate_distance_vectors(cell_centers, owner, neighbor):
    """
    Calculate the distance vectors d_cf from owner cell center to neighbor cell center.
    Only defined for internal faces (where neighbor != -1).
    """
    internal_faces = neighbor != -1
    owner_valid = owner[internal_faces]
    neighbor_valid = neighbor[internal_faces]

    d_cf = np.zeros((len(owner), 2)) # Initialize for all faces
    d_cf[internal_faces] = cell_centers[neighbor_valid] - cell_centers[owner_valid]
    return d_cf

def calculate_interpolation_factors(points, edges, cell_centers, owner, neighbor):
    """
    Calculate the geometric interpolation factor fx for internal faces.
    fx = |intersection - neighbor_center| / |owner_center - neighbor_center|
    where intersection is the point where the line connecting cell centers
    intersects the face.
    """
    face_centers = calculate_face_centers(points, edges)
    fx = np.full(len(owner), -1.0) # Initialize with invalid value

    internal_faces = neighbor != -1
    owner_valid = owner[internal_faces]
    neighbor_valid = neighbor[internal_faces]
    face_centers_valid = face_centers[internal_faces]
    cell_centers_owner = cell_centers[owner_valid]
    cell_centers_neighbor = cell_centers[neighbor_valid]

    # Vector from owner to neighbor
    d_cf = cell_centers_neighbor - cell_centers_owner
    norm_d_cf = np.linalg.norm(d_cf, axis=1)

    # Vector from owner to face center
    d_cf_face = face_centers_valid - cell_centers_owner

    # Project d_cf_face onto d_cf to find the intersection point relative to owner
    # We need the component of d_cf_face along the direction of d_cf
    # projection_length = dot(d_cf_face, d_cf) / dot(d_cf, d_cf) * |d_cf|
    #                 = dot(d_cf_face, d_cf) / |d_cf|
    dot_product = np.einsum('ij,ij->i', d_cf_face, d_cf) # Efficient batch dot product

    # Avoid division by zero for zero-length d_cf (shouldn't happen in valid meshes)
    valid_norm = norm_d_cf > 1e-12
    fx_internal = np.full(len(owner_valid), 0.5) # Default to 0.5 if norms are zero

    # Calculate fx = projection_length / |d_cf|
    #            = (dot(d_cf_face, d_cf) / |d_cf|) / |d_cf|
    #            = dot(d_cf_face, d_cf) / |d_cf|^2
    fx_internal[valid_norm] = dot_product[valid_norm] / (norm_d_cf[valid_norm]**2)

    # Clamp fx to [0, 1] to handle potential floating point issues or weird geometries
    fx_internal = np.clip(fx_internal, 0.0, 1.0)

    fx[internal_faces] = fx_internal

    return fx