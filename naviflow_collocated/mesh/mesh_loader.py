import numpy as np
import meshio
from numba.typed import List
from collections import defaultdict
from naviflow_collocated.mesh.mesh_data import MeshData2D
import yaml
from numba import njit # Import njit
from .helpers.mesh_loader_helpers import (
    _evaluate_bc_value_at_face,
    parse_physical_names,
    _load_unstructured_mesh,
    _load_structured_mesh,
    _calculate_cell_volumes,
    _compute_detailed_geometry_kernel,
    _compute_d_Cb_kernel,
    _count_faces_per_cell_kernel,
    _populate_cell_faces_kernel
)

# Define boundary condition type mapping constants
# These should be used consistently across the codebase
BC_WALL = 0
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_ZEROGRADIENT = 3
BC_CONVECTIVE = 4
BC_SYMMETRY = 5

# Map from string names to type constants
BC_TYPE_MAP = {
    "wall": BC_WALL,
    "dirichlet": BC_DIRICHLET,
    "neumann": BC_NEUMANN,
    "zerogradient": BC_ZEROGRADIENT,
    "convective": BC_CONVECTIVE,
    "symmetry": BC_SYMMETRY,
}

def load_mesh(filename, bc_config_file=None):
    """
    Load a 2D mesh (structured or unstructured) from a Gmsh .msh file
    and return a MeshData2D object with boundary tagging.
    """
    physical_names = parse_physical_names(filename)
    boundary_conditions = {}
    if bc_config_file is not None:
        with open(bc_config_file, "r") as f:
            boundary_config = yaml.safe_load(f)
        boundary_conditions = boundary_config.get("boundaries", {})

    mesh = meshio.read(filename)
    points = mesh.points[:, :2]

    if "triangle" in mesh.cells_dict:
        return _load_unstructured_mesh(
            mesh, points, physical_names, boundary_conditions, _build_meshdata2d, np.array
        )
    elif "quad" in mesh.cells_dict:
        return _load_structured_mesh(mesh, points, physical_names, boundary_conditions, _build_meshdata2d, np.array)
    else:
        raise ValueError("Unsupported mesh type: must contain triangle or quad cells")


def _build_meshdata2d(
    points,
    cells,
    boundary_lines,
    boundary_tags,
    physical_id_to_name,
    boundary_conditions,
):
    # --- Phase 1: Initial Cell Properties & Basic Face Geometry/Connectivity ---
    def sorted_edge(a, b):
        return tuple(sorted((a, b)))

    n_cells = len(cells)
    cell_centers = np.mean(points[cells], axis=1)
    
    # Pre-check cell shape before calling JITted _calculate_cell_volumes
    if cells.shape[1] not in [3, 4]:
        raise ValueError("Unsupported cell shape for volume calculation: cells must be triangles or quads.")
    cell_volumes = _calculate_cell_volumes(points, cells)

    face_map = defaultdict(list)
    for cid, cell in enumerate(cells):
        for i in range(len(cell)):
            edge = sorted_edge(cell[i], cell[(i + 1) % len(cell)])
            face_map[edge].append(cid)

    # Preallocate lists for initial face properties
    face_centers_list, face_normals_list, edge_lengths_list = [], [], []
    face_vertices_list, owner_cells_list, neighbor_cells_list = [], [], []

    for edge, cids in face_map.items():
        v0_idx, v1_idx = edge
        v0, v1 = points[v0_idx], points[v1_idx]
        center = 0.5 * (v0 + v1)

        edge_vec = v1 - v0
        edge_len = np.linalg.norm(edge_vec)
        
        n_hat = np.zeros(2)
        if edge_len > 1e-12:
            n_hat = np.array([edge_vec[1], -edge_vec[0]]) / edge_len
        
        owner = cids[0]
        if len(cids) == 2:
            neighbor = cids[1]
            if np.dot(n_hat, cell_centers[neighbor] - cell_centers[owner]) < 0:
                n_hat *= -1
        else: # Boundary face
            neighbor = -1
            if np.dot(n_hat, center - cell_centers[owner]) < 0:
                n_hat *= -1

        face_vertices_list.append(edge)
        owner_cells_list.append(owner)
        neighbor_cells_list.append(neighbor)
        face_centers_list.append(center)
        face_normals_list.append(n_hat * edge_len) # S_f = n_hat * Area_f
        edge_lengths_list.append(edge_len)

    # Convert to NumPy arrays
    face_centers   = np.array(face_centers_list)
    vector_S_f     = np.array(face_normals_list)
    initial_edge_lengths = np.array(edge_lengths_list)
    face_areas     = np.linalg.norm(vector_S_f, axis=1)
    face_vertices  = np.array(face_vertices_list)
    owner_cells    = np.array(owner_cells_list)
    neighbor_cells = np.array(neighbor_cells_list)
    n_faces = len(face_areas)

    assert np.allclose(face_areas, initial_edge_lengths, rtol=1e-10, atol=1e-12), \
        "Magnitude of S_f must equal edge length."

    internal_faces = np.where(neighbor_cells >= 0)[0]
    edge_to_face = {tuple(sorted(edge)): i for i, edge in enumerate(face_vertices)}

    # --- Phase 2: Detailed Geometric Calculations (Main Consolidated Loop) ---
    # Preallocate derived geometric arrays (Moukalled Ch. 3, 6, 8)
    vector_d_CE   = np.zeros((n_faces, 2))  # P->N centroid-to-centroid vector (d_PN)
    unit_vector_e = np.zeros((n_faces, 2))  # Unit vector along d_CE (e_PN)
    vector_E_f    = np.zeros((n_faces, 2))  # S_f component along d_CE (Eq. 8.53)
    vector_T_f    = np.zeros((n_faces, 2))  # S_f component perpendicular to d_CE (Eq. 8.54)
    delta_Pf      = np.zeros(n_faces)       # Distance owner P to face center f
    delta_fN      = np.zeros(n_faces)       # Distance face center f to neighbor N
    face_interp_factors = np.zeros(n_faces) # Geometric interpolation factor alpha_f
    skewness_vectors = np.zeros_like(face_centers) # Skewness vector s_f

    # Call the Numba kernel for detailed geometry
    _compute_detailed_geometry_kernel(
        n_faces, owner_cells, neighbor_cells, cell_centers, face_centers,
        vector_S_f, face_areas,
        vector_d_CE, unit_vector_e, vector_E_f, vector_T_f,
        delta_Pf, delta_fN, face_interp_factors, skewness_vectors
    )

    if len(internal_faces) > 0:
        unit_e_mags_internal = np.linalg.norm(unit_vector_e[internal_faces], axis=1)
        assert np.allclose(unit_e_mags_internal, 1.0, rtol=1e-10, atol=1e-12), \
            "Internal face unit_vector_e are not unit vectors."
    if n_faces > 0:
        tangential_check_val = np.abs(np.sum(skewness_vectors * vector_S_f, axis=1))
        assert np.all(tangential_check_val < 1e-9), \
            f"Skewness vectors not purely tangential. Max dot: {np.max(tangential_check_val)}"

    # Rhie-Chow interpolation weights (1 / (alpha_f * (1-alpha_f) * |d_PN|))
    epsilon = 1e-12
    rc_interp_weights = np.zeros(n_faces)
    # delta_CE_mag_full uses vector_d_CE which is populated by the Numba kernel
    delta_CE_mag_full = np.linalg.norm(vector_d_CE, axis=1) 

    # Mask for valid geometric conditions for interpolation
    valid_interp_mask = (
        (neighbor_cells >= 0) &
        (delta_CE_mag_full > epsilon) &
        (face_interp_factors > epsilon) & # face_interp_factors is populated by Numba kernel
        (face_interp_factors < (1.0 - epsilon))
    )
    
    if np.any(valid_interp_mask):
        # Calculate denominator only for these valid faces
        denominator_on_valid = (
            face_interp_factors[valid_interp_mask] *
            (1.0 - face_interp_factors[valid_interp_mask]) *
            delta_CE_mag_full[valid_interp_mask]
        )

        # Create final mask where denominator is also non-zero
        final_mask = np.zeros(n_faces, dtype=bool)
        # Apply denominator_on_valid's non-zero condition back to the original n_faces shape via valid_interp_mask
        final_mask_indices_from_valid = np.where(valid_interp_mask)[0]
        valid_denominator_indices = final_mask_indices_from_valid[np.abs(denominator_on_valid) > epsilon]
        final_mask[valid_denominator_indices] = True
        
        if np.any(final_mask):
            fif_final = face_interp_factors[final_mask]
            delta_CE_mag_final = delta_CE_mag_full[final_mask]
            rc_interp_weights[final_mask] = 1.0 / (fif_final * (1.0 - fif_final) * delta_CE_mag_final)

    # --- Phase 3: Boundary Tagging, d_Cb, and Final Cell-Face Connectivity ---
    tag_to_bc = { tag: boundary_conditions.get(name, {}) for tag, name in physical_id_to_name.items() }

    boundary_faces_list = []
    boundary_patches = np.full(n_faces, -1, dtype=np.int64)
    boundary_types = np.full((n_faces, 2), -1, dtype=np.int64)
    boundary_values = np.full((n_faces, 3), 0.0, dtype=np.float64)

    for i, line in enumerate(boundary_lines):
        edge = sorted_edge(*line)
        if edge not in edge_to_face: continue

        face_id = edge_to_face[edge]
        patch_tag = boundary_tags[i]
        patch_name = physical_id_to_name.get(patch_tag, f"UnnamedPatch_{patch_tag}")
        bc_config_for_patch = tag_to_bc.get(patch_tag, {})
        x_f_coords = face_centers[face_id]

        vel_bc_spec = bc_config_for_patch.get("velocity", {})
        vel_type = BC_TYPE_MAP.get(vel_bc_spec.get("bc", "zerogradient").lower(), BC_ZEROGRADIENT)
        vel_val_raw = vel_bc_spec.get("value", [0.0, 0.0])
        eval_vel = _evaluate_bc_value_at_face(vel_val_raw, x_f_coords, "velocity", patch_name)

        p_bc_spec = bc_config_for_patch.get("pressure", {})
        p_type = BC_TYPE_MAP.get(p_bc_spec.get("bc", "zerogradient").lower(), BC_ZEROGRADIENT)
        p_val_raw = p_bc_spec.get("value", 0.0)
        eval_p = _evaluate_bc_value_at_face(p_val_raw, x_f_coords, "pressure", patch_name)

        boundary_faces_list.append(face_id)
        boundary_patches[face_id] = patch_tag
        boundary_types[face_id] = [vel_type, p_type]

        if isinstance(eval_vel, (list, np.ndarray)) and len(eval_vel) >= 2:
            boundary_values[face_id, 0:2] = eval_vel[0:2]
        elif isinstance(eval_vel, (int, float)):
            boundary_values[face_id, 0] = eval_vel
        if isinstance(eval_p, (int, float, np.number)):
            boundary_values[face_id, 2] = eval_p
    boundary_values = boundary_values #* -1

    boundary_faces = np.array(sorted(list(set(boundary_faces_list))), dtype=np.int64)
    face_boundary_mask = np.zeros(n_faces, dtype=np.int64)
    face_flux_mask = np.ones(n_faces, dtype=np.int64)
    if len(boundary_faces) > 0: # Avoid error if no boundary faces
        face_boundary_mask[boundary_faces] = 1

    d_Cb = np.zeros(n_faces, dtype=np.float64) # Distance P to boundary face f (Moukalled 8.6.8)
    if len(boundary_faces) > 0:
        # Call Numba kernel for d_Cb
        _compute_d_Cb_kernel(d_Cb, boundary_faces, owner_cells, face_centers, cell_centers)

    # Cell-to-face connectivity using Numba kernels
    num_faces_for_cell = _count_faces_per_cell_kernel(n_cells, owner_cells, neighbor_cells, n_faces)
    
    max_faces_per_cell = 0
    if n_cells > 0 and num_faces_for_cell.size > 0: # Check if num_faces_for_cell is not empty
         max_faces_per_cell = np.max(num_faces_for_cell)

    cell_faces = -np.ones((n_cells, max_faces_per_cell), dtype=np.int64)
    # current_idx_for_cell needs to be reset for the populate kernel
    current_idx_for_cell = np.zeros(n_cells, dtype=np.int64)
    _populate_cell_faces_kernel(cell_faces, owner_cells, neighbor_cells, current_idx_for_cell, n_faces)

    return MeshData2D(
        cell_volumes, cell_centers,
        face_areas, face_centers,
        owner_cells, neighbor_cells, cell_faces, face_vertices, points,
        vector_S_f, vector_d_CE, vector_S_f / (face_areas[:, None] + 1e-12), # unit_vector_n
        unit_vector_e, vector_E_f, vector_T_f, skewness_vectors,
        face_interp_factors, rc_interp_weights,
        internal_faces, boundary_faces, boundary_patches,
        boundary_types, boundary_values, d_Cb,
        face_boundary_mask, face_flux_mask,
    )
