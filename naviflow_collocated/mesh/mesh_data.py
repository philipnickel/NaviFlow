"""
MeshData2D: Core structured data layout for finite volume CFD.

This class contains static mesh geometry, connectivity, boundary tagging, and derived metrics.

Indexing Logic:
- All face-level arrays (e.g., face_normals, face_areas, owner_cells) are indexed by face ID (0 to n_faces-1).
- All cell-level arrays (e.g., cell_volumes, cell_centers) are indexed by cell ID (0 to n_cells-1).
- Boundary-related arrays (e.g., boundary_types, boundary_values, d_PB) have full-face indexing (length = n_faces):
    * For internal faces: values are default (e.g., -1 for boundary_types, 0 for d_PB).
    * For boundary faces: entries are meaningful and correspond to boundary_faces indices.
- This ensures all face-based loops (internal and boundary) access consistent data structures.

"""

from numba import types
from numba.experimental import jitclass

mesh_data_spec = [
    # --- Geometry ---
    ("cell_volumes", types.float64[:]),  # Volume of each cell
    ("cell_centers", types.float64[:, :]),  # Coordinates of cell centers
    ("face_areas", types.float64[:]),  # Area of each face
    ("face_normals", types.float64[:, :]),  # Normal vectors of faces
    ("face_centers", types.float64[:, :]),  # Coordinates of face centers
    # --- Connectivity ---
    ("owner_cells", types.int64[:]),  # Indices of owner cells for each face
    ("neighbor_cells", types.int64[:]),  # Indices of neighbor cells for each face
    ("cell_faces", types.int64[:, :]),  # Faces belonging to each cell
    ("face_vertices", types.int64[:, :]),  # Vertices defining each face
    ("vertices", types.float64[:, :]),  # Coordinates of mesh vertices
    # --- Precomputed metrics ---
    (
        "d_PN",
        types.float64[:, :],
    ),  # Distance vectors between owner and neighbor cell centers
    ("e_f", types.float64[:, :]),  # Unit vectors along face normals
    (
        "delta_PN",
        types.float64[:],
    ),  # Distance magnitudes between owner and neighbor cell centers
    ("delta_Pf", types.float64[:]),  # Distance from owner cell center to face center
    ("delta_fN", types.float64[:]),  # Distance from face center to neighbor cell center
    ("non_ortho_correction", types.float64[:, :]),  # Non-orthogonal correction vectors
    ("face_interp_factors", types.float64[:]),  # Interpolation factors for faces
    # --- Topological masks ---
    ("internal_faces", types.int64[:]),  # Indices of internal faces
    ("boundary_faces", types.int64[:]),  # Indices of boundary faces
    ("boundary_patches", types.int64[:]),  # Indices of boundary patches
    (
        "boundary_types",
        types.int64[:],
    ),  # Type of BC at each face (0=Dirichlet, 1=Neumann, etc.); -1 for internal
    (
        "boundary_values",
        types.float64[:, :],
    ),  # Prescribed BC values (e.g., [u_BC, ...]) at each face; [0, 0] for internal
    ("d_PB", types.float64[:]),  # Distance from owner to boundary face; 0 for internal
    # --- Mesh metadata ---
    ("is_structured", types.boolean),  # Flag indicating if mesh is structured
    ("is_orthogonal", types.boolean),  # Flag indicating if mesh is orthogonal
    ("is_conforming", types.boolean),  # Flag indicating if mesh is conforming
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
        boundary_dists,
        d_PB,
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

        self.d_PB = d_PB
        self.boundary_types = boundary_types
        self.boundary_values = boundary_values

        # --- Mesh metadata ---
        self.is_structured = is_structured
        self.is_orthogonal = is_orthogonal
        self.is_conforming = is_conforming
