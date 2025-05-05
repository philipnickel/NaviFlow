"""
Mesh loader for importing .msh files into MeshData2D format.

This module provides functionality to load meshes from Gmsh .msh files and convert
them to the MeshData2D format used by the NaviFlow solver.
"""

import numpy as np
import gmsh
import warnings
from .mesh_data import MeshData2D
from .mesh_helpers import (
    calculate_face_centers,
    calculate_face_areas,
)


def calculate_face_normals_outward(points, edges, owner_cells, cell_centers):
    """
    Calculate face normals ensuring they point outward from owner cells.

    Parameters:
    -----------
    points : ndarray
        Array of vertex coordinates
    edges : ndarray
        Array of edges defined by vertex indices
    owner_cells : ndarray
        Array of owner cell indices for each face
    cell_centers : ndarray
        Array of cell center coordinates

    Returns:
    --------
    ndarray
        Array of face normal vectors
    """
    normals = np.empty((len(edges), 2))
    face_centers = np.array([np.mean(points[edge], axis=0) for edge in edges])

    for i, (n1, n2) in enumerate(edges):
        # Calculate initial normal (perpendicular to edge)
        dx, dy = points[n2][0] - points[n1][0], points[n2][1] - points[n1][1]
        raw_normal = np.array([dy, -dx])

        # Normalize
        norm = np.linalg.norm(raw_normal)
        if norm > 1e-12:
            raw_normal /= norm

        # Check direction relative to owner cell
        owner = owner_cells[i]
        if owner >= 0:  # Skip if no owner (shouldn't happen)
            # Vector from cell center to face center
            cf_vector = face_centers[i] - cell_centers[owner]

            # If normal points inward, flip it
            if np.dot(raw_normal, cf_vector) < 0:
                raw_normal = -raw_normal

        normals[i] = raw_normal

    return normals


def calculate_cell_volumes_custom(points, cells):
    """Calculate cell area for arbitrary polygon cells (2D) using Green's theorem."""
    volumes = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        x = points[cell, 0]
        y = points[cell, 1]
        volumes[i] = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return volumes


def calculate_cell_centers(points, cells):
    """Calculate cell centers for arbitrary polygon cells."""
    return np.array([np.mean(points[cell], axis=0) for cell in cells])


def assess_mesh_quality(n_cells, face_counts):
    """
    Assess mesh quality based on cell-face ownership distribution.

    Parameters:
    -----------
    n_cells : int
        Total number of cells in the mesh
    face_counts : ndarray
        Number of faces owned by each cell

    Returns:
    --------
    dict
        Dictionary with quality metrics
    """
    orphan_cells = np.where(face_counts == 0)[0]
    n_orphans = len(orphan_cells)
    orphan_percentage = (n_orphans / n_cells) * 100 if n_cells > 0 else 0

    # Calculate statistics on face ownership
    face_count_stats = {
        "min": int(np.min(face_counts)),
        "max": int(np.max(face_counts)),
        "mean": float(np.mean(face_counts)),
        "median": float(np.median(face_counts)),
    }

    # Assess quality
    if orphan_percentage == 0:
        quality = "excellent"
    elif orphan_percentage < 1:
        quality = "good"
    elif orphan_percentage < 5:
        quality = "fair"
    elif orphan_percentage < 10:
        quality = "poor"
    else:
        quality = "very poor"

    return {
        "orphan_cells": orphan_cells,
        "orphan_count": n_orphans,
        "orphan_percentage": orphan_percentage,
        "face_count_stats": face_count_stats,
        "quality": quality,
    }


def load_msh_file(filename, suppress_warnings=False):
    """
    Load a .msh file and return a MeshData2D instance.

    Parameters:
    -----------
    filename : str
        Path to the .msh file to load
    suppress_warnings : bool, optional
        If True, suppress warnings about mesh quality. Default is False.

    Returns:
    --------
    MeshData2D
        Mesh data in the format required by NaviFlow solver
    dict
        Mesh quality metrics
    """
    # Initialize gmsh if not already initialized
    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.clear()

    # Load the mesh file
    try:
        gmsh.open(filename)
    except Exception as e:
        raise RuntimeError(f"Failed to open mesh file {filename}: {e}")

    # Get nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(coords).reshape(-1, 3)[:, :2]  # Extract only x,y coordinates
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    # Get elements
    elem_types, elem_tags_nested, elem_nodes_nested = gmsh.model.mesh.getElements()

    # Find 2D elements (triangles=2, quads=3)
    cells = []
    cell_tags = []

    # Track if we're loading a structured mesh (all quads)
    is_structured = True

    for i, elem_type in enumerate(elem_types):
        if elem_type in [2, 3]:  # Triangle or Quad
            nodes_per_elem = 3 if elem_type == 2 else 4
            if elem_type == 2:  # Triangle
                is_structured = False

            elem_node_indices = [node_map[int(n)] for n in elem_nodes_nested[i]]
            cell_data = np.array(elem_node_indices, dtype=np.int64).reshape(
                -1, nodes_per_elem
            )

            for j, cell in enumerate(cell_data):
                cells.append(cell)  # Store as numpy array, not list
                tag = elem_tags_nested[i][j] if j < len(elem_tags_nested[i]) else 0
                cell_tags.append(int(tag))

    if not cells:
        raise RuntimeError(
            "No valid 2D elements (triangles or quads) found in the mesh file."
        )

    # Convert cells to numpy array for faster processing
    cells = np.array(cells)
    n_cells = len(cells)

    # Calculate cell centers and volumes
    cell_centers = calculate_cell_centers(points, cells)
    cell_volumes = calculate_cell_volumes_custom(points, cells)

    # Extract edges (faces) from cells and track which cells own which faces
    all_edges = []
    cell_to_faces = [[] for _ in range(n_cells)]  # To map cells to their faces
    edge_to_cells = {}  # Map each edge to cells containing it

    for cell_idx, cell in enumerate(cells):
        n_vertices = len(cell)
        for i in range(n_vertices):
            v1 = int(cell[i])  # Convert to int to avoid any numpy int64 issues
            v2 = int(cell[(i + 1) % n_vertices])
            # Use ordered vertex pairs for edge identification
            edge = tuple(sorted([v1, v2]))

            # Track which cells contain this edge
            if edge not in edge_to_cells:
                edge_to_cells[edge] = []
            edge_to_cells[edge].append(cell_idx)

            # Add to the main edge list if not already present
            if edge not in all_edges:
                all_edges.append(edge)

    # Create edges as numpy arrays for geometric calculations
    edges_np = np.array([[e[0], e[1]] for e in all_edges], dtype=np.int64)

    # Identify boundary edges (faces)
    boundary_faces = []
    boundary_types = []
    boundary_patch_map = {}

    # Find boundary faces using gmsh physical groups
    for dim, entity in gmsh.model.getEntities(1):  # 1D entities (lines)
        phys_groups = gmsh.model.getPhysicalGroupsForEntity(dim, entity)
        if not phys_groups:
            continue

        # Get physical group name and tag
        phys_tag = phys_groups[0]

        # Get nodes of this entity
        entity_nodes = gmsh.model.mesh.getNodes(1, entity)[0]
        entity_node_indices = [node_map[int(n)] for n in entity_nodes]

        # Find corresponding edges
        for i in range(0, len(entity_node_indices), 2):
            if i + 1 < len(entity_node_indices):
                n1, n2 = entity_node_indices[i], entity_node_indices[i + 1]
                edge = tuple(sorted([n1, n2]))
                if edge in all_edges:
                    edge_idx = all_edges.index(edge)
                    boundary_faces.append(edge_idx)
                    boundary_types.append(phys_tag)
                    boundary_patch_map[edge_idx] = phys_tag

    # If no boundary faces were found via physical groups, identify them topologically
    if not boundary_faces:
        for edge_idx, edge in enumerate(all_edges):
            if (
                len(edge_to_cells.get(edge, [])) == 1
            ):  # Boundary face appears in only one cell
                boundary_faces.append(edge_idx)
                # Default boundary type if no physical groups were found
                boundary_types.append(1)
                boundary_patch_map[edge_idx] = 1

    # Convert to numpy arrays
    boundary_faces = np.array(boundary_faces, dtype=np.int64)
    boundary_types = np.array(boundary_types, dtype=np.int64)

    # Build owner and neighbor arrays based on edge_to_cells mapping
    n_faces = len(all_edges)
    owner = np.full(n_faces, -1, dtype=np.int64)
    neighbor = np.full(n_faces, -1, dtype=np.int64)

    # ---- MOUKALLED-BASED ALGORITHM FOR FACE OWNERSHIP ----
    # Initialize tracking structures
    cell_to_faces = [[] for _ in range(n_cells)]

    # STEP 1: For structured meshes, use the "lower index rule" per Moukalled
    # For unstructured meshes, use a bandwidth-minimizing strategy
    if is_structured:
        # For structured meshes, Moukalled recommends the lower index rule
        # This ensures consistent face normal orientation and matrix structure
        for face_idx, edge in enumerate(all_edges):
            cells_with_edge = edge_to_cells.get(edge, [])
            if not cells_with_edge:
                continue

            if len(cells_with_edge) == 1:
                # Boundary face with only one cell
                owner[face_idx] = cells_with_edge[0]
                neighbor[face_idx] = -1  # No neighbor (boundary)
            else:
                # Internal face - apply lower index rule per Moukalled
                owner_idx = min(cells_with_edge)
                neigh_idx = max(cells_with_edge)
                owner[face_idx] = owner_idx
                neighbor[face_idx] = neigh_idx

            # Update cell-to-faces mapping
            cell_to_faces[owner[face_idx]].append(face_idx)
    else:
        # For unstructured meshes, Moukalled recommends minimizing matrix bandwidth
        # and ensuring no orphan cells for numerical stability

        # First pass: Basic lower-index assignment as starting point
        for face_idx, edge in enumerate(all_edges):
            cells_with_edge = edge_to_cells.get(edge, [])
            if not cells_with_edge:
                continue

            if len(cells_with_edge) == 1:
                # Boundary face with only one cell
                owner[face_idx] = cells_with_edge[0]
                neighbor[face_idx] = -1  # No neighbor (boundary)
            else:
                # Internal face - start with lower index rule
                owner_idx = min(cells_with_edge)
                neigh_idx = max(cells_with_edge)
                owner[face_idx] = owner_idx
                neighbor[face_idx] = neigh_idx

            # Update cell-to-faces mapping
            cell_to_faces[owner[face_idx]].append(face_idx)

        # Second pass: Identify orphan cells (cells that don't own any faces)
        orphan_cells = [i for i in range(n_cells) if len(cell_to_faces[i]) == 0]

        # Minimize bandwidth while ensuring no orphan cells
        # This is a practical enhancement aligned with Moukalled's principles
        for cell_idx in orphan_cells:
            # Find faces where this cell is a neighbor but not an owner
            for face_idx in range(n_faces):
                if neighbor[face_idx] == cell_idx:
                    # Swap owner and neighbor to ensure this cell owns at least one face
                    # This maintains face normal consistency while eliminating orphans
                    current_owner = owner[face_idx]
                    owner[face_idx] = cell_idx
                    neighbor[face_idx] = current_owner

                    # Update cell-to-faces mapping
                    cell_to_faces[cell_idx].append(face_idx)
                    cell_to_faces[current_owner].remove(face_idx)
                    break

            # If still an orphan (no neighbor faces found), try more aggressive approach
            # while maintaining matrix structure integrity
            if len(cell_to_faces[cell_idx]) == 0:
                for edge, cells in edge_to_cells.items():
                    if cell_idx in cells:
                        face_idx = all_edges.index(edge)
                        current_owner = owner[face_idx]

                        # Only reassign if the current owner has multiple faces
                        # This ensures every cell owns at least one face (no orphans)
                        # while maintaining overall matrix structure
                        if len(cell_to_faces[current_owner]) > 1:
                            owner[face_idx] = cell_idx

                            # Update cell-to-faces mapping
                            cell_to_faces[cell_idx].append(face_idx)
                            cell_to_faces[current_owner].remove(face_idx)
                            break

    # STEP 2: Per Moukalled, enforce consistent face normal orientation
    # Ensure normals point from owner to neighbor for internal faces
    # and outward from the domain for boundary faces

    # This is addressed in calculate_face_normals_outward(), which explicitly
    # ensures that face normals point outward from the owner cell

    # STEP 3: Verify no orphan cells remain for numerical stability
    # If any orphan cells still remain (extremely rare),
    # make a final pass to ensure matrix consistency
    orphan_cells = [i for i in range(n_cells) if len(cell_to_faces[i]) == 0]
    if orphan_cells:
        print(
            f"Warning: {len(orphan_cells)} orphan cells remain after primary assignment."
        )
        print("Applying final Moukalled-consistent reassignment.")

        # Last resort: find any available face and assign ownership
        # This is a practical enhancement to Moukalled's approach for stability
        for cell_idx in orphan_cells:
            for edge, cells in edge_to_cells.items():
                if cell_idx in cells:
                    face_idx = all_edges.index(edge)

                    # Assign this face to the orphan cell
                    # While maintaining normal orientation consistency
                    owner[face_idx] = cell_idx
                    cell_to_faces[cell_idx].append(face_idx)
                    break

    # Build geometric quantities - Moukalled emphasizes proper computation
    # of these quantities for accurate discretization
    face_centers = calculate_face_centers(points, edges_np)
    face_areas = calculate_face_areas(points, edges_np)
    face_normals = calculate_face_normals_outward(points, edges_np, owner, cell_centers)

    # Calculate distance vectors and interpolation factors
    d_cf = np.zeros((n_faces, 2))
    internal_faces = neighbor != -1
    for f in np.where(internal_faces)[0]:
        d_cf[f] = cell_centers[neighbor[f]] - cell_centers[owner[f]]

    # Calculate interpolation factors for internal faces
    fx = np.full(n_faces, 0.5)  # Default to 0.5
    for f in np.where(internal_faces)[0]:
        o_cell = owner[f]
        n_cell = neighbor[f]

        # Vector from owner to face center
        d_cf_face = face_centers[f] - cell_centers[o_cell]

        # Vector from owner to neighbor
        d_CF = cell_centers[n_cell] - cell_centers[o_cell]
        d_CF_norm = np.linalg.norm(d_CF)

        if d_CF_norm > 1e-12:
            # Project d_cf_face onto d_CF to find interpolation factor
            dot_product = np.dot(d_cf_face, d_CF)
            fx[f] = dot_product / (d_CF_norm * d_CF_norm)

    # Clamp fx to [0, 1]
    fx = np.clip(fx, 0.0, 1.0)

    # Calculate non-orthogonality correction vectors
    non_ortho_correction = np.zeros_like(d_cf)
    if np.any(internal_faces):
        # For internal faces, calculate non-orthogonality correction
        face_normals_unit = face_normals.copy()

        # Unit vector from owner to neighbor
        d_cf_unit = np.zeros_like(d_cf)
        d_cf_norms = np.linalg.norm(d_cf, axis=1)
        valid_d_cf = d_cf_norms > 1e-12
        d_cf_unit[valid_d_cf] = d_cf[valid_d_cf] / d_cf_norms[valid_d_cf, np.newaxis]

        # Correction vector is orthogonal to face normal
        dot_products = np.einsum("ij,ij->i", face_normals_unit, d_cf_unit)
        for i in np.where(internal_faces & valid_d_cf)[0]:
            non_ortho_correction[i] = (
                face_normals_unit[i] - dot_products[i] * d_cf_unit[i]
            )

    # Determine mesh properties
    is_orthogonal = np.allclose(non_ortho_correction, 0, atol=1e-8)
    is_conforming = True  # Assume conforming by default

    # Default boundary values (zero)
    boundary_values = np.zeros((len(boundary_faces), 2))

    # Create boundary patches array
    boundary_patches = np.zeros(n_faces, dtype=np.int64)
    for face_idx, patch_id in boundary_patch_map.items():
        boundary_patches[face_idx] = patch_id

    # Debug info
    print("Loaded mesh statistics:")
    print(f"  - Points: {len(points)}")
    print(f"  - Cells: {n_cells}")
    print(f"  - Faces: {n_faces}")
    print(f"  - Boundary faces: {len(boundary_faces)}")
    print(f"  - Structured: {is_structured}")

    # Assess mesh quality based on cell-face ownership
    # Recalculate face_counts to include any adjustments made during orphan fixing
    face_counts = np.bincount(owner[owner >= 0], minlength=n_cells)
    quality = assess_mesh_quality(n_cells, face_counts)

    # Provide detailed warnings about mesh quality
    if quality["orphan_count"] > 0 and not suppress_warnings:
        orphan_percentage = quality["orphan_percentage"]
        quality_assessment = quality["quality"].upper()
        stats = quality["face_count_stats"]

        warning_msg = (
            f"\nMESH QUALITY WARNING: {quality_assessment}\n"
            f"  - {quality['orphan_count']} cells ({orphan_percentage:.2f}%) do not own any faces\n"
            f"  - Face ownership: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}, median={stats['median']:.1f}\n"
            f"  - This can lead to poor solution quality, solver instability, or numerical artifacts\n"
            f"  - Consider refining or regenerating the mesh for better quality"
        )

        if orphan_percentage > 5:
            warnings.warn(warning_msg, RuntimeWarning)
        else:
            print(warning_msg)

    # Close gmsh instance
    gmsh.finalize()

    # Create mesh data instance
    mesh = MeshData2D(
        cell_volumes=cell_volumes,
        face_areas=face_areas,
        face_normals=face_normals,
        face_centers=face_centers,
        cell_centers=cell_centers,
        owner_cells=owner,
        neighbor_cells=neighbor,
        boundary_faces=boundary_faces,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        boundary_patches=boundary_patches,
        face_interp_factors=fx,
        d_CF=d_cf,
        non_ortho_correction=non_ortho_correction,
        is_structured=is_structured,
        is_orthogonal=is_orthogonal,
        is_conforming=is_conforming,
    )

    return mesh, quality
