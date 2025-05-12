import numpy as np
from numba import njit

def _evaluate_bc_value_at_face(raw_value, x_f, field_name, bc_name_for_error_msg):
    """
    Evaluates a boundary condition value which can be a scalar, list, or string expression.
    String expressions are evaluated with 'np' (NumPy) and 'x' (face center coordinates) available.
    """
    if isinstance(raw_value, str):
        try:
            return eval(raw_value, {"np": np, "x": x_f})
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate {field_name} BC expression for '{bc_name_for_error_msg}': "
                f"'{raw_value}' at x = {x_f}: {e}"
            )
    elif isinstance(raw_value, list):
        evaluated_list = []
        for i, item in enumerate(raw_value):
            if isinstance(item, str):
                try:
                    evaluated_list.append(eval(item, {"np": np, "x": x_f}))
                except Exception as e:
                    raise ValueError(
                        f"Failed to evaluate {field_name} BC expression (item {i}) for '{bc_name_for_error_msg}': "
                        f"'{item}' at x = {x_f}: {e}"
                    )
            else:
                evaluated_list.append(item)
        return evaluated_list
    elif callable(raw_value): # Although current config parsing might not produce callables directly
        return raw_value(x_f)
    return raw_value


def parse_physical_names(msh_filename):
    with open(msh_filename, "r") as f:
        lines = f.readlines()

    phys_names = {}
    inside = False
    for line in lines:
        if line.strip() == "$PhysicalNames":
            inside = True
            continue
        if line.strip() == "$EndPhysicalNames":
            break
        if inside:
            parts = line.strip().split()
            if len(parts) >= 3:
                dim, tag, *name_parts = parts
                name = " ".join(name_parts).strip('"')
                phys_names[int(tag)] = name
    return phys_names


def _load_unstructured_mesh(mesh, points, physical_names, boundary_conditions, _build_meshdata2d_func, np_array_func):
    triangles = np_array_func(mesh.cells_dict["triangle"], dtype=np.int64)
    boundary_lines = np_array_func(mesh.cells_dict.get("line", []), dtype=np.int64)
    boundary_tags = np_array_func(
        mesh.cell_data_dict["gmsh:physical"].get("line", []), dtype=np.int64
    )
    return _build_meshdata2d_func(
        points,
        triangles,
        boundary_lines,
        boundary_tags,
        physical_id_to_name=physical_names,
        boundary_conditions=boundary_conditions,
    )


def _load_structured_mesh(mesh, points, physical_names, boundary_conditions, _build_meshdata2d_func, np_array_func):
    quads = np_array_func(mesh.cells_dict["quad"], dtype=np.int64)
    boundary_lines = np_array_func(mesh.cells_dict.get("line", []), dtype=np.int64)
    boundary_tags = np_array_func(
        mesh.cell_data_dict["gmsh:physical"].get("line", []), dtype=np.int64
    )
    return _build_meshdata2d_func(
        points,
        quads,
        boundary_lines,
        boundary_tags,
        physical_id_to_name=physical_names,
        boundary_conditions=boundary_conditions,
    )


@njit(fastmath=True)
def _calculate_cell_volumes(points, cells):
    # This function assumes cells is an np.array and points is an np.array
    # and cells.shape[1] is either 3 or 4, checked by the caller.
    if cells.shape[1] == 3:  # triangles
        a = points[cells[:, 0]]
        b = points[cells[:, 1]]
        c = points[cells[:, 2]]
        return 0.5 * np.abs(
            a[:, 0] * (b[:, 1] - c[:, 1])
            + b[:, 0] * (c[:, 1] - a[:, 1])
            + c[:, 0] * (a[:, 1] - b[:, 1])
        )
    elif cells.shape[1] == 4:  # quads (Shoelace formula for two triangles: ABD and BCD)
        a = points[cells[:, 0]]
        b = points[cells[:, 1]]
        c = points[cells[:, 2]]
        d = points[cells[:, 3]]
        area_abd = 0.5 * np.abs(
            a[:, 0] * (b[:, 1] - d[:, 1]) +
            b[:, 0] * (d[:, 1] - a[:, 1]) +
            d[:, 0] * (a[:, 1] - b[:, 1])
        )
        area_bcd = 0.5 * np.abs(
            b[:, 0] * (c[:, 1] - d[:, 1]) +
            c[:, 0] * (d[:, 1] - b[:, 1]) +
            d[:, 0] * (b[:, 1] - c[:, 1])
        )
        return area_abd + area_bcd
    # The case for unsupported cell shapes is handled by the caller before this function is invoked.
    return np.empty(0, dtype=np.float64) # Should not be reached if caller validates. Added for Numba typing if somehow reached.


@njit(parallel=False, fastmath=True)
def _compute_detailed_geometry_kernel(
    n_faces, owner_cells, neighbor_cells, cell_centers, face_centers, 
    vector_S_f, face_areas, 
    # Output arrays (modified in-place):
    vector_d_CE, unit_vector_e, vector_E_f, vector_T_f, 
    delta_Pf, delta_fN, face_interp_factors, skewness_vectors
):
    for f in range(n_faces): # Numba will parallelize this loop if possible
        P = owner_cells[f]
        N = neighbor_cells[f]
        x_f_x, x_f_y = face_centers[f, 0], face_centers[f, 1]
        x_P_x, x_P_y = cell_centers[P, 0], cell_centers[P, 1]

        vec_Pf_x = x_f_x - x_P_x
        vec_Pf_y = x_f_y - x_P_y
        delta_Pf[f] = (vec_Pf_x**2 + vec_Pf_y**2)**0.5
        
        s_f_x, s_f_y = vector_S_f[f,0], vector_S_f[f,1]
        area_f = face_areas[f]
        n_hat_f_x, n_hat_f_y = 0.0, 0.0
        if area_f > 1e-12:
            n_hat_f_x = s_f_x / area_f
            n_hat_f_y = s_f_y / area_f

        if N >= 0: # Internal face
            x_N_x, x_N_y = cell_centers[N, 0], cell_centers[N, 1]
            vec_PN_x = x_N_x - x_P_x
            vec_PN_y = x_N_y - x_P_y
            vector_d_CE[f, 0], vector_d_CE[f, 1] = vec_PN_x, vec_PN_y
            
            delta_CE = (vec_PN_x**2 + vec_PN_y**2)**0.5
            if delta_CE > 1e-12:
                unit_vector_e[f, 0] = vec_PN_x / delta_CE
                unit_vector_e[f, 1] = vec_PN_y / delta_CE

            delta_fN[f] = ((x_N_x - x_f_x)**2 + (x_N_y - x_f_y)**2)**0.5

            Sf_dot_e = s_f_x * unit_vector_e[f, 0] + s_f_y * unit_vector_e[f, 1]   # |S_f| cosθ
            Sf_sq    = s_f_x * s_f_x + s_f_y * s_f_y                               # |S_f|²
            scale    = 0.0
            if abs(Sf_dot_e) > 1e-12:
                scale = Sf_sq / Sf_dot_e                                           # over‑relaxed factor
            vector_E_f[f, 0] = scale * unit_vector_e[f, 0]
            vector_E_f[f, 1] = scale * unit_vector_e[f, 1]
            vector_T_f[f, 0] = s_f_x - vector_E_f[f, 0]
            vector_T_f[f, 1] = s_f_y - vector_E_f[f, 1]

            denom_alpha = n_hat_f_x * vec_PN_x + n_hat_f_y * vec_PN_y
            alpha_f = 0.5
            if abs(denom_alpha) > 1e-12:
                alpha_f = (n_hat_f_x * vec_Pf_x + n_hat_f_y * vec_Pf_y) / denom_alpha
            face_interp_factors[f] = min(max(alpha_f, 0.0), 1.0) # np.clip

            x_i_on_PN_line_x = x_P_x + face_interp_factors[f] * vec_PN_x
            x_i_on_PN_line_y = x_P_y + face_interp_factors[f] * vec_PN_y
            skew_vec_raw_x = x_f_x - x_i_on_PN_line_x
            skew_vec_raw_y = x_f_y - x_i_on_PN_line_y
            
            dot_skew_n = skew_vec_raw_x * n_hat_f_x + skew_vec_raw_y * n_hat_f_y
            skewness_vectors[f, 0] = skew_vec_raw_x - dot_skew_n * n_hat_f_x
            skewness_vectors[f, 1] = skew_vec_raw_y - dot_skew_n * n_hat_f_y
        else: # Boundary face
            if delta_Pf[f] > 1e-12:
                unit_vector_e[f, 0] = vec_Pf_x / delta_Pf[f]
                unit_vector_e[f, 1] = vec_Pf_y / delta_Pf[f]
            
            Sf_dot_e = s_f_x * unit_vector_e[f, 0] + s_f_y * unit_vector_e[f, 1]   # |S_f| cosθ
            Sf_sq    = s_f_x * s_f_x + s_f_y * s_f_y                               # |S_f|²
            scale    = 0.0
            if abs(Sf_dot_e) > 1e-12:
                scale = Sf_sq / Sf_dot_e                                           # over‑relaxed factor
            vector_E_f[f, 0] = scale * unit_vector_e[f, 0]
            vector_E_f[f, 1] = scale * unit_vector_e[f, 1]
            vector_T_f[f, 0] = s_f_x - vector_E_f[f, 0]
            vector_T_f[f, 1] = s_f_y - vector_E_f[f, 1]
            
            face_interp_factors[f] = 1.0
            # skewness_vectors for boundary is already [0,0] by preallocation

@njit(fastmath=True)
def _compute_d_Cb_kernel(d_Cb, boundary_faces, owner_cells, face_centers, cell_centers):
    for i in range(boundary_faces.shape[0]):
        f = boundary_faces[i]
        P = owner_cells[f]
        dist = ((face_centers[f,0] - cell_centers[P,0])**2 + 
                (face_centers[f,1] - cell_centers[P,1])**2)**0.5
        if dist == dist: # Check for NaN
            d_Cb[f] = dist

@njit(fastmath=True)
def _count_faces_per_cell_kernel(n_cells, owner_cells, neighbor_cells, n_faces):
    num_faces_for_cell = np.zeros(n_cells, dtype=np.int64)
    for f in range(n_faces):
        own = owner_cells[f]
        num_faces_for_cell[own] += 1
        neigh = neighbor_cells[f]
        if neigh >= 0:
            num_faces_for_cell[neigh] += 1
    return num_faces_for_cell

@njit(fastmath=True)
def _populate_cell_faces_kernel(cell_faces, owner_cells, neighbor_cells, current_idx_for_cell, n_faces):
    # Assumes current_idx_for_cell is pre-filled with zeros
    for f in range(n_faces):
        own = owner_cells[f]
        cell_faces[own, current_idx_for_cell[own]] = f
        current_idx_for_cell[own] += 1
        
        neigh = neighbor_cells[f]
        if neigh >= 0:
            cell_faces[neigh, current_idx_for_cell[neigh]] = f
            current_idx_for_cell[neigh] += 1
