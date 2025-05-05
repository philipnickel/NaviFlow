import numpy as np
from collections import defaultdict


def calculate_face_normals(points, edges):
    normals = np.empty((len(edges), 2))
    for i, (n1, n2) in enumerate(edges):
        dx = points[n2][0] - points[n1][0]
        dy = points[n2][1] - points[n1][1]
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
    """Green's theorem for arbitrary 2D polygonal cell area."""
    volumes = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        x = points[cell, 0]
        y = points[cell, 1]
        volumes[i] = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return volumes


def build_owner_neighbor(cells, edges):
    # Map undirected edge (frozenset of 2 vertices) to the cells that use it
    edge_to_cells = defaultdict(list)

    for cell_id, cell in enumerate(cells):
        for i in range(len(cell)):
            a, b = cell[i], cell[(i + 1) % len(cell)]
            edge = frozenset((a, b))
            edge_to_cells[edge].append(cell_id)

    owner = np.full(len(edges), -1, dtype=np.int64)
    neighbor = np.full(len(edges), -1, dtype=np.int64)

    for i, (a, b) in enumerate(edges):
        edge = frozenset((a, b))
        cells_for_edge = edge_to_cells.get(edge, [])
        if len(cells_for_edge) >= 1:
            owner[i] = cells_for_edge[0]
        if len(cells_for_edge) == 2:
            neighbor[i] = cells_for_edge[1]

    return owner, neighbor


def calculate_distance_vectors(cell_centers, owner, neighbor):
    """
    d_cf = vector from owner to neighbor cell centers for internal faces.
    """
    internal = neighbor != -1
    d_cf = np.zeros((len(owner), 2))
    d_cf[internal] = cell_centers[neighbor[internal]] - cell_centers[owner[internal]]
    return d_cf


def calculate_interpolation_factors(points, edges, cell_centers, owner, neighbor):
    """
    Compute geometric interpolation factor fx for internal faces.
    fx = projection of vector from owner to face center onto d_cf direction,
         normalized by length of d_cf (clamped to [0, 1]).
    """
    face_centers = calculate_face_centers(points, edges)
    fx = np.full(len(owner), -1.0)

    internal = neighbor != -1
    o = owner[internal]
    n = neighbor[internal]
    fc = face_centers[internal]
    co = cell_centers[o]
    cn = cell_centers[n]

    d_cf = cn - co
    norm = np.linalg.norm(d_cf, axis=1)
    d_cf_face = fc - co

    dot = np.einsum("ij,ij->i", d_cf, d_cf_face)
    fx_internal = np.full(len(o), 0.5)
    valid = norm > 1e-12
    fx_internal[valid] = dot[valid] / norm[valid] ** 2
    fx_internal = np.clip(fx_internal, 0.0, 1.0)

    fx[internal] = fx_internal
    return fx
