"""
Numba-accelerated 2D Mesh Data Structure for Finite Volume CFD Simulations

Specialized for 2D simulations while maintaining compatibility with Moukalled's FVM methodology.
All spatial data uses 2D coordinates (x,y) with z=0 implied.
"""

from numba import types
from numba.experimental import jitclass


mesh_data_spec = [
    # Existing geometric properties
    ("cell_volumes", types.float64[:]),  # Cell areas (m²)
    ("face_areas", types.float64[:]),  # Face lengths (m)
    ("face_normals", types.float64[:, :]),  # Face normal vectors
    ("face_centers", types.float64[:, :]),  # Face centroids
    ("cell_centers", types.float64[:, :]),  # Cell centroids
    # Core connectivity (existing)
    ("owner_cells", types.int64[:]),  # Owner cell indices
    ("neighbor_cells", types.int64[:]),  # Neighbor cell indices (-1=boundary)
    # Additional connectivity (required)
    ("vertices", types.float64[:, :]),  # Vertex coordinates
    ("faces", types.int64[:, :]),  # Vertices forming each face
    ("cell_faces", types.int64[:, :]),  # Faces forming each cell
    ("boundary_owners", types.int64[:]),  # Owner cell for each boundary face
    # Boundary data (existing)
    ("boundary_faces", types.int64[:]),  # Boundary face indices
    ("boundary_types", types.int64[:]),  # BC types
    ("boundary_values", types.float64[:, :]),  # BC values
    ("boundary_patches", types.int64[:]),  # Patch IDs
    # Diffusion & convection support
    ("face_interp_factors", types.float64[:]),  # Interpolation factors
    ("d_CF", types.float64[:, :]),  # Owner-to-neighbor vectors
    ("e_f", types.float64[:, :]),  # Unit vector from owner to neighbor
    ("delta_CF", types.float64[:]),  # Distance between cell centers
    ("delta_Cf", types.float64[:]),  # Distance from owner center to face
    ("delta_fF", types.float64[:]),  # Distance from face to neighbor center
    ("non_ortho_correction", types.float64[:, :]),  # Non-orthogonality correction
    # SIMPLE algorithm specific
    ("face_fluxes", types.float64[:]),  # Mass fluxes through faces
    ("face_velocities", types.float64[:, :]),  # Velocity vectors at faces
    ("D_f", types.float64[:]),  # Diffusion coefficients
    ("H_f", types.float64[:, :]),  # Sum of neighbor coefficients and sources
    ("grad_p_f", types.float64[:, :]),  # Pressure gradient at faces
    ("pressure_correction_coeffs", types.float64[:, :]),  # p' equation coefficients
    # Under-relaxation
    ("alpha_u", types.float64),  # Velocity under-relaxation
    ("alpha_p", types.float64),  # Pressure under-relaxation
    # Mesh metadata (existing)
    ("is_structured", types.boolean),  # Structured grid flag
    ("is_orthogonal", types.boolean),  # Orthogonality flag
    ("is_conforming", types.boolean),  # Conformity flag
]


@jitclass(mesh_data_spec)
class MeshData2D:
    def __init__(
        self,
        cell_volumes,
        face_areas,
        face_normals,
        face_centers,
        cell_centers,
        owner_cells,
        neighbor_cells,
        boundary_faces,
        boundary_types,
        boundary_values,
        boundary_patches,
        face_interp_factors,
        d_CF,
        non_ortho_correction,
        is_structured,
        is_orthogonal,
        is_conforming,
    ):
        """
        Initialize 2D mesh data structure

        Parameters:
        -----------
        All arrays must be C-contiguous and properly typed (float64/int64)
        Spatial data must be 2D (shape [n, 2])
        """
        # Validate 2D shape consistency
        assert face_normals.shape[1] == 2, "Face normals must be 2D vectors"
        assert face_centers.shape[1] == 2, "Face centers must be 2D coordinates"
        assert cell_centers.shape[1] == 2, "Cell centers must be 2D coordinates"
        assert d_CF.shape[1] == 2, "d_CF vectors must be 2D"
        assert non_ortho_correction.shape[1] == 2, "Non-ortho correction must be 2D"
        assert boundary_values.shape[1] == 2, "Boundary values must be 2D"

        # Geometric properties
        self.cell_volumes = cell_volumes  # Cell areas [n_cells]
        self.face_areas = face_areas  # Face lengths [n_faces]
        self.face_normals = face_normals  # Face normal vectors [n_faces, 2]
        self.face_centers = face_centers  # Face centroids [n_faces, 2]
        self.cell_centers = cell_centers  # Cell centroids [n_cells, 2]

        # Connectivity maps
        self.owner_cells = owner_cells  # Owner cell indices [n_faces]
        self.neighbor_cells = neighbor_cells  # Neighbor cell indices [n_faces]

        # Boundary condition data
        self.boundary_faces = boundary_faces  # Boundary face indices [n_boundary_faces]
        self.boundary_types = boundary_types  # BC type codes [n_boundary_faces]
        self.boundary_values = boundary_values  # BC values [n_boundary_faces, 2]
        self.boundary_patches = boundary_patches  # Patch IDs [n_boundary_faces]

        # Interpolation factors
        self.face_interp_factors = face_interp_factors  # [n_faces]
        self.d_CF = d_CF  # Owner-to-neighbor vectors [n_faces, 2]
        self.non_ortho_correction = non_ortho_correction  # T_f vectors [n_faces, 2]

        # Mesh metadata
        self.is_structured = is_structured  # Structured grid flag
        self.is_orthogonal = is_orthogonal  # Orthogonality flag
        self.is_conforming = is_conforming  # Conformity flag


# Validation Checklist ---------------------------------------------------------
"""
1. 2D Shape Enforcement:
   - All vector arrays have shape [n, 2]
   - No z-component storage

2. Geometric Integrity:
   - cell_volumes (areas) > 0
   - face_areas (lengths) > 0
   - face_normals magnitude ≈ face_areas

3. Boundary Conditions:
   - Moving wall BCs have non-zero tangential components
   - Pressure BCs use normal gradient conditions
"""

# Performance Notes ------------------------------------------------------------
"""
- 2D storage reduces memory usage by 33% compared to 3D arrays
- Numba can optimize 2D loops more effectively
- Use np.linalg.norm(..., axis=1) for 2D vector magnitudes
"""
