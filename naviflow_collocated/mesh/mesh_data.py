"""
MeshData2D: Core data layout for finite volume CFD (2D, collocated).

This class defines static geometry, connectivity, boundary tagging, and precomputed metrics,
following Moukalled's finite volume formulation.

Indexing Conventions:
- All face-based arrays (e.g., face_normals, owner_cells) use face indexing (0 to n_faces-1).
- All cell-based arrays (e.g., cell_volumes, cell_centers) use cell indexing (0 to n_cells-1).
- Boundary-related arrays (e.g., boundary_values, boundary_types, d_PB) have full-face length (n_faces).
    * Internal faces use sentinel defaults: boundary_types = [-1, -1], boundary_values = [0, 0, 0], d_PB = 0.0

Boundary Condition Metadata:
- boundary_values[f, :] = [u_BC, v_BC, p_BC] for face f. Zero for internal.
- boundary_types[f, :] = [vel_type, p_type] with:
    * 0 = Wall
    * 1 = Dirichlet
    * 2 = Neumann
    * 3 = zeroGradient
- d_Cb[f] = distance from cell center to boundary face center (used for one-sided gradients)

Fast Boolean Masks:
- face_boundary_mask[f] = 1 if face is boundary, 0 otherwise
- face_flux_mask[f] = 1 if face is active in flux computation, 0 otherwise
"""

from numba import types
from numba.experimental import jitclass

mesh_data_spec = [
    # --- Cell Geometry ---
    ("cell_volumes", types.float64[:]),          # Cell volumes V_C
    ("cell_centers", types.float64[:, :]),       # Cell centroids x_C (shape: [n_cells, 2])

    # --- Face Geometry ---
    ("face_areas", types.float64[:]),            # Face area magnitudes |S_f| (lengths in 2D)
    ("face_centers", types.float64[:, :]),       # Face centroids x_f [n_faces, 2]

    # --- Connectivity --- 
    ("owner_cells", types.int64[:]),             # Owner cell index for each face
    ("neighbor_cells", types.int64[:]),          # Neighbor cell index (–1 for boundary faces)
    ("cell_faces", types.int64[:, :]),           # Padded list of face indices for each cell
    ("face_vertices", types.int64[:, :]),        # Vertex indices (2 per face)
    ("vertices", types.float64[:, :]),           # Vertex coordinates (shape: [n_vertices, 2])

    # --- Vector Geometry --- Computed for Over-relaxed approach Moukalled (8.6.4)
    ("vector_S_f", types.float64[:, :]),         # surface vectors S_f. unit vector n scaled by the face area 
    ("vector_d_CE", types.float64[:, :]),        # distance vector between centroids of elements C and E (also CF in Moukalled)
    ("unit_vector_n", types.float64[:, :]),      # unit vector normal to the surface
    ("unit_vector_e", types.float64[:, :]),      # unit vector e along the direction of d_CE
    ("vector_E_f", types.float64[:, :]),         # used for the over-relaxed approach
    ("vector_T_f", types.float64[:, :]),         # used for the over-relaxed approach
    ("vector_skewness", types.float64[:, :]),    # vector from the intersection point f' (of d_CE and S_f) to the face center f

    # --- Interpolation Factors ---
    ("face_interp_factors", types.float64[:]),   # g_f = delta_Pf / delta_PN
    ("rc_interp_weights", types.float64[:]),     # 1 / (g_f * (1 - g_f) * delta_PN) — used in gradient recon

    # --- Topological Masks ---
    ("internal_faces", types.int64[:]),          # Indices of faces with valid neighbor (N >= 0)
    ("boundary_faces", types.int64[:]),          # Indices of faces with N = –1
    ("boundary_patches", types.int64[:]),        # Patch ID per boundary face (–1 for internal)

    # --- Boundary Conditions ---
    ("boundary_types", types.int64[:, :]),       # BC type per face: [vel_type, p_type]
    ("boundary_values", types.float64[:, :]),    # BC values per face: [u_BC, v_BC, p_BC]
    ("d_Cb", types.float64[:]),                  # Distance from cell center to boundary face center (Moukalled 8.6.8)

    # --- Binary Masks (for fast Numba filtering etc.) ---
    ("face_boundary_mask", types.int64[:]),      # 1 if face is a boundary, 0 otherwise
    ("face_flux_mask", types.int64[:]),          # 1 if face is active in flux loops, 0 otherwise
]



@jitclass(mesh_data_spec)
class MeshData2D:
    def __init__(
        self,
        cell_volumes,
        cell_centers,
        face_areas,
        face_centers,
        owner_cells,
        neighbor_cells,
        cell_faces,
        face_vertices,
        vertices,
        vector_S_f,
        vector_d_CE,
        unit_vector_n,
        unit_vector_e,
        vector_E_f,
        vector_T_f,
        vector_skewness,
        face_interp_factors,
        rc_interp_weights,
        internal_faces,
        boundary_faces,
        boundary_patches,
        boundary_types,
        boundary_values,
        d_Cb,
        face_boundary_mask,
        face_flux_mask,
    ):
        # --- Geometry ---
        self.cell_volumes = cell_volumes
        self.cell_centers = cell_centers
        self.face_areas = face_areas
        self.face_centers = face_centers

        # --- Connectivity ---
        self.owner_cells = owner_cells
        self.neighbor_cells = neighbor_cells
        self.cell_faces = cell_faces
        self.face_vertices = face_vertices
        self.vertices = vertices

        # --- Vector Geometry ---
        self.vector_S_f = vector_S_f
        self.vector_d_CE = vector_d_CE
        self.unit_vector_n = unit_vector_n
        self.unit_vector_e = unit_vector_e
        self.vector_E_f = vector_E_f
        self.vector_T_f = vector_T_f
        self.vector_skewness = vector_skewness

        # --- Interpolation Factors ---
        self.face_interp_factors = face_interp_factors
        self.rc_interp_weights = rc_interp_weights

        # --- Topological Info ---
        self.internal_faces = internal_faces
        self.boundary_faces = boundary_faces
        self.boundary_patches = boundary_patches

        # --- BCs ---
        self.boundary_types = boundary_types
        self.boundary_values = boundary_values
        self.d_Cb = d_Cb

        # --- Masks ---
        self.face_boundary_mask = face_boundary_mask
        self.face_flux_mask = face_flux_mask
