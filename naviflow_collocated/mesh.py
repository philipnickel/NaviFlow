import numpy as np  # Ensure numpy is imported

# from naviflow_collocated.mesh_utils import analyze_mesh_quality # F401: Import analyze_mesh_quality (Commented out as usage is commented)
from dataclasses import dataclass, field

# Define BC constants at module level or import them if defined elsewhere
BC_WALL_NO_SLIP = 1
BC_INLET_VELOCITY = 3  # Use inlet type for moving lid


@dataclass
class MeshData2D:
    """Holds all geometric and connectivity information for a 2D mesh."""

    n_cells: int = 0
    n_faces: int = 0
    n_boundary_faces: int = 0
    cell_centers: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=float)
    )
    face_centers: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=float)
    )
    face_normals: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=float)
    )
    face_areas: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    cell_volumes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    face_owner_cells: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    face_neighbor_cells: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )  # -1 for boundary faces
    boundary_faces: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )  # Indices into the global face list
    boundary_types: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    boundary_values: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=float)
    )
    boundary_conditions: list = field(
        default_factory=list
    )  # List of BC dicts (may deprecate)


def load_msh_file(filename: str) -> MeshData2D:
    # --- Dummy implementation for now ---
    # Replace with actual mesh loading logic (e.g., using meshio)
    print(f"Warning: Using dummy mesh data from {filename}")
    # Simulate loading some data
    n_cells = 100
    n_faces = 200  # Internal + boundary
    n_boundary_faces = 40
    # internal_faces = n_faces - n_boundary_faces # F841: Unused variable

    # Instantiate the mesh object HERE
    mesh = MeshData2D()

    # Populate with dummy data
    mesh.cell_centers = np.random.rand(n_cells, 2)
    mesh.face_centers = np.random.rand(n_faces, 2)
    mesh.face_normals = np.random.rand(n_faces, 2)
    mesh.face_normals /= np.linalg.norm(mesh.face_normals, axis=1)[
        :, None
    ]  # E702: Split statement
    mesh.face_areas = np.random.rand(n_faces) * 0.1
    mesh.cell_volumes = np.random.rand(n_cells) * 0.01
    # Dummy connectivity: owner=cell index, neighbor=next cell index or -1
    mesh.face_owner_cells = np.random.randint(0, n_cells, n_faces)
    mesh.face_neighbor_cells = np.random.randint(0, n_cells, n_faces)
    # Mark some faces as boundary by setting neighbor to -1
    boundary_indices_in_faces = np.random.choice(
        n_faces, n_boundary_faces, replace=False
    )
    mesh.face_neighbor_cells[boundary_indices_in_faces] = -1
    mesh.boundary_faces = boundary_indices_in_faces
    # --- End of Dummy implementation ---

    # ... existing code to load mesh geometry ...
    # ... code populating mesh.cell_centers, mesh.face_centers, mesh.boundary_faces etc. ...

    # Store counts (Now assigned to the instantiated mesh object)
    mesh.n_cells = n_cells
    mesh.n_faces = n_faces
    mesh.n_boundary_faces = n_boundary_faces

    # --- Assign Default Boundary Conditions (for testing) ---
    # TODO: Load actual BCs from config file in a real run

    mesh.boundary_types = np.zeros(n_boundary_faces, dtype=int)
    mesh.boundary_values = np.zeros((n_boundary_faces, 2), dtype=float)  # Store U, V

    # Find the top boundary faces (highest y-coordinate)
    if n_boundary_faces > 0:  # Avoid errors on meshes with no boundaries
        # Ensure we only use boundary faces for max_y calculation
        boundary_face_centers = mesh.face_centers[mesh.boundary_faces, :]
        if boundary_face_centers.shape[0] > 0:
            max_y = np.max(boundary_face_centers[:, 1])
            tolerance = 1e-6  # Tolerance for floating point comparison

            top_face_indices_in_boundary_list = []
            for i, f_idx in enumerate(mesh.boundary_faces):
                if abs(mesh.face_centers[f_idx, 1] - max_y) < tolerance:
                    top_face_indices_in_boundary_list.append(
                        i
                    )  # Store index within the boundary_faces array
        else:
            top_face_indices_in_boundary_list = []  # No boundary faces

        # Assign BCs
        for i in range(n_boundary_faces):
            if i in top_face_indices_in_boundary_list:
                mesh.boundary_types[i] = BC_INLET_VELOCITY
                mesh.boundary_values[i, 0] = 1.0  # U = 1 on lid
                mesh.boundary_values[i, 1] = 0.0  # V = 0 on lid
            else:
                mesh.boundary_types[i] = BC_WALL_NO_SLIP
                mesh.boundary_values[i, 0] = 0.0  # U = 0 on other walls
                mesh.boundary_values[i, 1] = 0.0  # V = 0 on other walls
    else:
        # Handle case with no boundary faces if necessary
        print(
            "Warning: Mesh has no boundary faces. Skipping boundary condition assignment."
        )
        pass

    # Assign the list of boundary condition dictionaries as well for compatibility
    # (This can be removed if all functions switch to using mesh.boundary_types/values directly)
    mesh.boundary_conditions = []
    if n_boundary_faces > 0:
        for i in range(n_boundary_faces):
            mesh.boundary_conditions.append(
                {
                    "face_indices": [
                        mesh.boundary_faces[i]
                    ],  # Assuming one face per BC dict for now
                    "type_code": mesh.boundary_types[i],
                    "value": mesh.boundary_values[i, :].tolist(),
                }
            )

    # Analyze mesh quality
    # Assuming analyze_mesh_quality exists and accepts MeshData2D
    # quality_info = analyze_mesh_quality(mesh) # Make sure this function is imported or defined
    # quality_info = {} # F841: Unused variable (Commented out as usage is commented)

    print("Loaded mesh statistics:")  # Removed extraneous f
    # ... rest of the print statements ...
    print(f"  Num Cells: {mesh.n_cells}")
    print(f"  Num Faces: {mesh.n_faces}")
    print(f"  Num Boundary Faces: {mesh.n_boundary_faces}")
    # print(f"  Mesh Quality: {quality_info}") # Uncomment when analyze_mesh_quality is ready

    return mesh  # , quality_info # Return only mesh for now


# ... other functions like generate_structured_uniform etc. ...
