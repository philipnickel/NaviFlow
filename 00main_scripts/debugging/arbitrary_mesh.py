import numpy as np
import matplotlib.pyplot as plt
from naviflow_staggered.preprocessing.mesh.mesh import Mesh
def verify_mesh(mesh: Mesh):
    """
    Comprehensive mesh verification for both structured and unstructured meshes.
    """
    # --- Basic Checks ---
    assert mesh.n_cells > 0, "Mesh has no cells."
    assert mesh.n_faces > 0, "Mesh has no faces."
    assert mesh.n_nodes > 0, "Mesh has no nodes."

    # --- Owner/Neighbor Consistency ---
    owners, neighbors = mesh.get_owner_neighbor()
    for i, (own, nei) in enumerate(zip(owners, neighbors)):
        assert not (own == -1 and nei == -1), f"Face {i} has no owner or neighbor."
        if own != -1 and nei != -1:
            assert own != nei, f"Face {i} has identical owner and neighbor."

    # --- Normals ---
    normals = mesh.get_face_normals()
    assert normals.shape == (mesh.n_faces, 2), "Normals shape mismatch."
    norms = np.linalg.norm(normals, axis=1)
    for i, n in enumerate(norms):
        assert np.all(np.isfinite(normals[i])), f"Face {i} normal has NaN or Inf."
        if n > 1e-10:
            assert np.isclose(n, 1.0, atol=1e-6), f"Face {i} normal not unit length: {n}"
        else:
            print(f"Warning: Face {i} normal has near-zero length: {n}")

    # --- Areas & Volumes ---
    areas = mesh.get_face_areas()
    volumes = mesh.get_cell_volumes()
    assert np.all(np.isfinite(areas)) and np.all(areas >= 0), "Invalid face areas."
    assert np.all(np.isfinite(volumes)) and np.all(volumes > 0), "Invalid cell volumes."

    # --- Centers ---
    cell_centers = mesh.get_cell_centers()
    face_centers = mesh.get_face_centers()
    node_coords = mesh.get_node_positions()
    for arr, name in [(cell_centers, "cell centers"), (face_centers, "face centers"), (node_coords, "node coords")]:
        assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf."

    # --- Normal Direction ---
    tol = 1e-9
    for i in range(mesh.n_faces):
        owner = owners[i]
        if owner != -1:
            vec_owner_to_face = face_centers[i] - cell_centers[owner]
            if np.linalg.norm(vec_owner_to_face) > tol:
                dot_product = np.dot(normals[i], vec_owner_to_face)
                
                    
                assert dot_product >= -tol, (
                    f"Face {i} normal not pointing outward from owner {owner}. "
                    f"Dot product: {dot_product:.2f}"
                )

    # --- Interpolation Weights ---
    for face_idx in range(mesh.n_faces):
        g_C, g_F = mesh.get_face_interpolation_factors(face_idx)
        assert np.isfinite(g_C) and np.isfinite(g_F), f"Face {face_idx} interpolation has NaN or Inf."
        assert np.isclose(g_C + g_F, 1.0, atol=1e-6), f"Face {face_idx} interpolation weights do not sum to 1."

    # --- Face Distances ---
    face_dists = mesh.get_face_distances()
    assert np.all(np.isfinite(face_dists)), "Face distances contain NaN or Inf."
    assert np.all(face_dists > 0), "Zero or negative face-to-cell distance found."

    # --- Duplicate Cells ---
    unique_cell_centers = np.unique(cell_centers.round(decimals=10), axis=0)
    assert len(unique_cell_centers) == mesh.n_cells, "Duplicate cell centers detected."

    # --- Hanging Faces Check ---
    for i, face_nodes in enumerate(mesh._faces):  # Accessing private member for debugging
        assert len(face_nodes) >= 2, f"Face {i} has insufficient node connectivity (Nodes: {face_nodes})"

    # --- Orphan Faces Check ---
    referenced_faces = set()
    for cell_faces in mesh._cells:  # Accessing private member for debugging
        referenced_faces.update(cell_faces)
    assert len(referenced_faces) == mesh.n_faces, (
        f"Mismatch between number of faces ({mesh.n_faces}) and referenced faces ({len(referenced_faces)}). "
        f"Orphan faces might exist."
    )

    # --- Connectivity Map (Cell-to-Cell Adjacency via Shared Faces) ---
    from collections import defaultdict

    connectivity_map = defaultdict(set)
    for face_idx, (own, nei) in enumerate(zip(owners, neighbors)):
         # Check if owner and neighbor indices are valid
         if 0 <= own < mesh.n_cells and 0 <= nei < mesh.n_cells:
             # Interior face: should connect two distinct cells
             assert own != nei, f"Face {face_idx} connects cell {own} to itself."
             connectivity_map[own].add(nei)
             connectivity_map[nei].add(own)
         elif own != -1 and nei != -1: # Case where indices might be valid but out of range (shouldn't happen ideally)
              print(f"Warning: Face {face_idx} has owner {own} or neighbor {nei} outside valid cell range [0, {mesh.n_cells-1}]. Skipping connectivity check for this face.")

    # Verify mutual neighbor consistency
    for cell, neighbors_set in connectivity_map.items():
        for nei in neighbors_set:
            # Ensure the neighbor cell exists in the map before checking its neighbors
            if nei in connectivity_map:
                assert cell in connectivity_map[nei], (
                    f"Inconsistent connectivity: cell {cell} considers {nei} a neighbor, "
                    f"but {nei} does not list {cell} as a neighbor (Neighbors of {nei}: {connectivity_map[nei]})."
                )
            else:
                # This case implies an interior face connected a valid cell to an invalid one, which shouldn't occur if owner/neighbor logic is sound.
                print(f"Warning: Neighbor cell {nei} (neighbor of {cell}) not found in connectivity map keys. This might indicate an issue with owner/neighbor assignment.")

    # --- Boundary labeling checks ---
    expected_boundaries = {'left', 'right', 'top', 'bottom'}
    actual_boundaries = set(mesh.boundary_face_to_name.values())
    assert expected_boundaries == actual_boundaries, (
        f"Boundary labeling mismatch. Expected {expected_boundaries}, got {actual_boundaries}"
    )

    # Check boundary face normals directions:
    for face_idx, boundary_name in mesh.boundary_face_to_name.items():
        normal = mesh.get_face_normals()[face_idx]
        if boundary_name == 'left':
            assert np.allclose(normal, [-1, 0]), f"Left boundary normal incorrect at face {face_idx}"
        elif boundary_name == 'right':
            assert np.allclose(normal, [1, 0]), f"Right boundary normal incorrect at face {face_idx}"
        elif boundary_name == 'bottom':
            assert np.allclose(normal, [0, -1]), f"Bottom boundary normal incorrect at face {face_idx}"
        elif boundary_name == 'top':
            assert np.allclose(normal, [0, 1]), f"Top boundary normal incorrect at face {face_idx}"

    # --- Face ↔ Cell Consistency ---
    face_usage = {i: set() for i in range(mesh.n_faces)}
    for cell_idx, cell_faces in enumerate(mesh._cells):
        for f in cell_faces:
            if f != -1: # Check if face index is valid
                face_usage[f].add(cell_idx)

    for f, users in face_usage.items():
        own, nei = owners[f], neighbors[f]
        expected_users = {own} if nei == -1 else {own, nei}
        # Filter out -1 owner/neighbor indices before comparing sets
        expected_users = {u for u in expected_users if u != -1}
        assert users == expected_users, (
            f"Face {f} is used by cells {users}, "
            f"but owner/neighbor are {own}, {nei}. Expected {expected_users}"
        )

    # --- Gauss Closure Check (divergence theorem) ---
    # Need to import UnstructuredMesh for isinstance check
    from naviflow_staggered.preprocessing.mesh.unstructured import UnstructuredMesh
    closure_tolerance = 5e-2 if isinstance(mesh, UnstructuredMesh) else 1e-10
    worst_norm = 0
    worst_cell = -1
    worst_vector = None

    try:
        for cell_idx, face_ids in enumerate(mesh._cells):
            net_flux_vector = np.zeros(2)
            valid_face_ids = [f for f in face_ids if f != -1] # Ensure valid face indices
            if not valid_face_ids:
                 print(f"Warning: Cell {cell_idx} has no valid faces. Skipping Gauss check.")
                 continue # Skip cells with no valid faces

            for f in valid_face_ids:
                 # Ensure owner index is valid before accessing arrays
                 owner_idx = owners[f]
                 if owner_idx != -1 and 0 <= owner_idx < mesh.n_cells:
                     sign = 1.0 if owner_idx == cell_idx else -1.0
                     # Ensure normal and area indices are valid
                     if 0 <= f < mesh.n_faces:
                         area_vector = normals[f] * areas[f]
                         net_flux_vector += sign * area_vector
                     else:
                         print(f"Warning: Invalid face index {f} encountered for cell {cell_idx}. Skipping face.")
                 else:
                      print(f"Warning: Invalid owner index {owner_idx} for face {f} in cell {cell_idx}. Skipping face.")

            closure_norm = np.linalg.norm(net_flux_vector)
            if closure_norm > worst_norm:
                worst_norm = closure_norm
                worst_cell = cell_idx
                worst_vector = net_flux_vector

            assert closure_norm < closure_tolerance, (
                f"Gauss closure failed for cell {cell_idx}. Residual norm = {closure_norm:.2e}, "
                f"Vector = {net_flux_vector}. Tolerance = {closure_tolerance:.1e}"
            )
    except AssertionError as e:
        print(f"🛑 Gauss Closure Failed: {e}")
        if worst_cell != -1 and worst_vector is not None: # Check if a worst cell was identified
            print(f"   Visualizing problematic cell {worst_cell} with residual norm {worst_norm:.2e}")

            # --- Optional: visualize the problematic cell and its flux vectors ---
            fig, ax = plt.subplots(figsize=(8, 7)) # Increased figure size slightly
            ax.set_title(f"Gauss Closure Failure in Cell {worst_cell} (Norm: {worst_norm:.2e})")
            ax.set_aspect("equal")

            cell_faces = [f for f in mesh._cells[worst_cell] if f != -1] # Filter invalid faces
            if not cell_faces:
                 print(f"   Error: Cannot visualize cell {worst_cell}, no valid faces found.")
                 raise # Re-raise the original assertion if visualization isn't possible

            cell_node_indices = set()
            valid_cell_faces_for_nodes = []
            for f in cell_faces:
                 if 0 <= f < len(mesh._faces):
                     face_node_list = mesh._faces[f]
                     # Ensure all node indices within the face list are valid
                     if all(0 <= node_idx < mesh.n_nodes for node_idx in face_node_list):
                         cell_node_indices.update(face_node_list)
                         valid_cell_faces_for_nodes.append(f)
                     else:
                          print(f"   Warning: Invalid node indices in face {f} ({face_node_list}) of cell {worst_cell}. Skipping face for node collection.")
                 else:
                     print(f"   Warning: Invalid face index {f} in cell {worst_cell}. Skipping face for node collection.")

            if not cell_node_indices:
                print(f"   Error: Cannot visualize cell {worst_cell}, no valid nodes found for its faces.")
                raise # Re-raise if no valid nodes

            cell_nodes_list = list(cell_node_indices)
            coords = mesh.get_node_positions()[cell_nodes_list]

            # Plot cell outline using faces (handles potential non-convexity better)
            plotted_edges = set()
            for f_idx in valid_cell_faces_for_nodes:
                face_nodes = mesh._faces[f_idx]
                if len(face_nodes) >= 2:
                     # Ensure nodes are valid before accessing coords
                     valid_face_nodes = [n for n in face_nodes if 0 <= n < mesh.n_nodes]
                     if len(valid_face_nodes) >= 2:
                         node_coords_for_face = mesh.get_node_positions()[valid_face_nodes]
                         # Plot edges of the face
                         for i in range(len(valid_face_nodes)):
                             p1_idx = valid_face_nodes[i]
                             p2_idx = valid_face_nodes[(i + 1) % len(valid_face_nodes)] # Wrap around
                             edge = tuple(sorted((p1_idx, p2_idx)))
                             if edge not in plotted_edges:
                                 p1 = mesh.get_node_positions()[p1_idx]
                                 p2 = mesh.get_node_positions()[p2_idx]
                                 ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.0)
                                 plotted_edges.add(edge)

            # Plot nodes
            ax.plot(coords[:, 0], coords[:, 1], 'ko', markersize=5, label="Nodes")

            # Plot face centers, normals, and labels
            face_centers = mesh.get_face_centers()
            face_normals = mesh.get_face_normals()
            face_areas = mesh.get_face_areas() # Needed for scaling? No, just for calculation above.
            for f in cell_faces: # Use original cell_faces list for quiver plot
                 if 0 <= f < mesh.n_faces: # Check index validity
                    fc = face_centers[f]
                    fn = face_normals[f]
                    # Scale normal by area for visual flux representation? Optional.
                    # quiver_vec = fn * face_areas[f]
                    quiver_vec = fn
                    ax.quiver(fc[0], fc[1], quiver_vec[0], quiver_vec[1], scale=10, color='r', alpha=0.7, width=0.005)
                    ax.text(fc[0] + 0.02 * quiver_vec[0], fc[1] + 0.02 * quiver_vec[1], f"f{f}", color='blue', fontsize=8)
                 else:
                      print(f"   Warning: Skipping quiver/text for invalid face index {f} in cell {worst_cell}.")


            # Plot cell center and net flux vector
            cc = mesh.get_cell_centers()[worst_cell]
            ax.plot(cc[0], cc[1], 'gx', markersize=8, label="Cell center")
            ax.quiver(cc[0], cc[1], worst_vector[0], worst_vector[1], angles='xy', scale_units='xy', scale=1, color='g', label=f'Net Flux (Norm: {worst_norm:.2e})', width=0.008)

            # Adjust plot limits for better visualization
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            center = (min_coords + max_coords) / 2
            extent = np.max(max_coords - min_coords)
            ax.set_xlim(center[0] - extent * 0.7, center[0] + extent * 0.7)
            ax.set_ylim(center[1] - extent * 0.7, center[1] + extent * 0.7)

            ax.legend()
            plt.show()

        raise # re-raise the original assertion

    # --- Face Normal Orthogonality to Edge Vector (for 2D line segments) ---
    # This check assumes faces are straight line segments in 2D
    for i, face_nodes in enumerate(mesh._faces):
        if len(face_nodes) == 2: # Ensure it's a simple line segment face
            p0_idx, p1_idx = face_nodes[0], face_nodes[1]
            # Ensure node indices are valid
            if 0 <= p0_idx < mesh.n_nodes and 0 <= p1_idx < mesh.n_nodes:
                p0, p1 = node_coords[p0_idx], node_coords[p1_idx]
                edge_vec = p1 - p0
                edge_norm = np.linalg.norm(edge_vec)
                if edge_norm > 1e-12: # Avoid division by zero for coincident nodes
                    # Calculate expected normal based on counter-clockwise node ordering convention (pointing "out" of edge)
                    expected_normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_norm
                    face_normal = normals[i]

                    # Check if the calculated normal is parallel to the expected normal
                    # Dot product should be close to +1 or -1
                    dot_product = np.dot(face_normal, expected_normal)
                    # Allow for normals pointing in the opposite direction as well
                    assert np.isclose(abs(dot_product), 1.0, atol=1e-6), (
                            f"Face {i} normal {face_normal} not parallel to edge vector's perpendicular {expected_normal}. Dot = {dot_product:.4f}"
                    )
            else:
                    print(f"Warning: Skipping orthogonality check for face {i} due to invalid node indices {face_nodes}")
        # else: # Optional: Handle faces with more than 2 nodes if necessary
        #    print(f"Warning: Skipping orthogonality check for face {i} with {len(face_nodes)} nodes.")

    """
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh.plot(ax=ax, title="Mesh Verification")
    ax.quiver(face_centers[:, 0], face_centers[:, 1], normals[:, 0], normals[:, 1], color='r', label='Normals')
    for i, (x, y) in enumerate(face_centers):
        ax.text(x, y, str(i), color='blue', fontsize=8)
    for i, (x, y) in enumerate(cell_centers):
        ax.text(x, y, f'C{i}', color='green', fontsize=8)
    ax.legend()
    plt.show()
    """
    print("✅ Mesh verification complete — all tests passed.")

if __name__ == '__main__':
    # Example usage:
    from naviflow_staggered.preprocessing.mesh.unstructured import UnstructuredUniform
    from naviflow_staggered.preprocessing.mesh.structured_mesh import StructuredMesh   
    from naviflow_staggered.preprocessing.mesh.unstructured import UnstructuredRefined

    mesh = UnstructuredUniform(
        mesh_size=0.5,
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1
    )
    print(f"Validating unstructured uniform mesh...")
    verify_mesh(mesh)
  # unstructured refined mesh
    mesh = UnstructuredRefined(
        mesh_size_walls=0.5,
        mesh_size_lid=0.2,
        mesh_size_center=0.1,
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1
    )
    print(f"Validating unstructured refined mesh...")
    verify_mesh(mesh)
    # --- Structured Mesh ---
    mesh = StructuredMesh(
        n_cells_x=10,
        n_cells_y=10,
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1
    )
    print(f"Validating structured mesh...")
    verify_mesh(mesh)
   