import numpy as np
from numba import types
from numba.experimental import jitclass

mesh_data_spec = [
    # --- Geometry ---
    ("cell_volumes", types.float64[:]),
    ("cell_centers", types.float64[:, :]),
    ("face_areas", types.float64[:]),
    ("face_normals", types.float64[:, :]),
    ("face_centers", types.float64[:, :]),
    # --- Connectivity ---
    ("owner_cells", types.int64[:]),
    ("neighbor_cells", types.int64[:]),
    ("cell_faces", types.int64[:, :]),
    ("face_vertices", types.int64[:, :]),
    ("vertices", types.float64[:, :]),
    # --- Precomputed metrics ---
    ("d_PN", types.float64[:, :]),
    ("e_f", types.float64[:, :]),
    ("delta_PN", types.float64[:]),
    ("delta_Pf", types.float64[:]),
    ("delta_fN", types.float64[:]),
    ("non_ortho_correction", types.float64[:, :]),
    ("face_interp_factors", types.float64[:]),
    # --- Topological masks ---
    ("internal_faces", types.int64[:]),
    ("boundary_faces", types.int64[:]),
    ("boundary_patches", types.int64[:]),
    ("boundary_types", types.int64[:]),
    ("boundary_values", types.float64[:, :]),
    # --- Face-level variables ---
    ("face_fluxes", types.float64[:]),
    ("face_velocities", types.float64[:, :]),
    ("grad_p_f", types.float64[:, :]),
    # --- Cell-level solver data ---
    ("D_f", types.float64[:]),
    ("H_f", types.float64[:, :]),
    ("p_corr_coeffs", types.float64[:, :]),
    # --- Mesh metadata ---
    ("is_structured", types.boolean),
    ("is_orthogonal", types.boolean),
    ("is_conforming", types.boolean),
]


@jitclass(mesh_data_spec)
class MeshData2D:
    def __init__(
        self,
        cell_volumes,
        cell_centers,
        face_areas,
        face_normals,
        face_centers,
        owner_cells,
        neighbor_cells,
        cell_faces,
        face_vertices,
        vertices,
        d_PN,
        e_f,
        delta_PN,
        delta_Pf,
        delta_fN,
        non_ortho_correction,
        face_interp_factors,
        internal_faces,
        boundary_faces,
        boundary_patches,
        boundary_types,
        boundary_values,
        is_structured,
        is_orthogonal,
        is_conforming,
    ):
        # --- Geometry ---
        self.cell_volumes = cell_volumes
        self.cell_centers = cell_centers
        self.face_areas = face_areas
        self.face_normals = face_normals
        self.face_centers = face_centers

        # --- Connectivity ---
        self.owner_cells = owner_cells
        self.neighbor_cells = neighbor_cells
        self.cell_faces = cell_faces
        self.face_vertices = face_vertices
        self.vertices = vertices

        # --- Precomputed metrics ---
        self.d_PN = d_PN
        self.e_f = e_f
        self.delta_PN = delta_PN
        self.delta_Pf = delta_Pf
        self.delta_fN = delta_fN
        self.non_ortho_correction = non_ortho_correction
        self.face_interp_factors = face_interp_factors

        # --- Topological masks ---
        self.internal_faces = internal_faces
        self.boundary_faces = boundary_faces
        self.boundary_patches = boundary_patches
        self.boundary_types = boundary_types
        self.boundary_values = boundary_values

        # --- Face-level variables (initialized to zero) ---
        n_faces = len(face_areas)
        self.face_fluxes = np.zeros(n_faces, dtype=np.float64)
        self.face_velocities = np.zeros((n_faces, 2), dtype=np.float64)
        self.grad_p_f = np.zeros((n_faces, 2), dtype=np.float64)

        # --- Cell-level solver data (initialized to zero) ---
        n_cells = len(cell_volumes)
        self.D_f = np.zeros(n_faces, dtype=np.float64)
        self.H_f = np.zeros((n_cells, 2), dtype=np.float64)
        self.p_corr_coeffs = np.zeros((n_cells, 2), dtype=np.float64)

        # --- Mesh metadata ---
        self.is_structured = is_structured
        self.is_orthogonal = is_orthogonal
        self.is_conforming = is_conforming
