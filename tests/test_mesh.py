import numpy as np


def test_mesh_basic_properties(mesh_instance):
    """Test basic mesh properties and geometry."""
    # Check cell and face geometry
    assert len(mesh_instance.cell_volumes) > 0, "Mesh has no cells"
    assert len(mesh_instance.face_areas) > 0, "Mesh has no faces"
    assert np.all(mesh_instance.face_areas > 0), "Mesh has zero or negative face areas"
    assert np.all(mesh_instance.cell_volumes > 0), (
        "Mesh has zero or negative cell volumes"
    )

    # Check array dimensions
    assert mesh_instance.face_normals.shape[1] == 2, "Face normals aren't 2D vectors"
    assert mesh_instance.cell_centers.shape[1] == 2, "Cell centers aren't 2D points"
    assert mesh_instance.face_centers.shape[1] == 2, "Face centers aren't 2D points"


def test_mesh_connectivity(mesh_instance):
    """Test mesh connectivity and owner-neighbor relationships."""
    # Test owner cells are valid indices
    face_counts = np.bincount(
        mesh_instance.owner_cells, minlength=len(mesh_instance.cell_volumes)
    )
    print("Cells with no owned faces:", np.where(face_counts == 0)[0])
    assert np.all(mesh_instance.owner_cells >= 0), "Invalid negative owner cell indices"
    assert np.max(mesh_instance.owner_cells) < len(mesh_instance.cell_volumes), (
        "Owner index out of range"
    )

    # Test neighbor cells are valid (when not boundary)
    internal_faces = mesh_instance.neighbor_cells != -1
    if np.any(internal_faces):
        assert np.all(mesh_instance.neighbor_cells[internal_faces] >= 0), (
            "Invalid negative neighbor indices"
        )
        assert np.max(mesh_instance.neighbor_cells[internal_faces]) < len(
            mesh_instance.cell_volumes
        ), "Neighbor index out of range"

    # Test boundary faces
    assert len(mesh_instance.boundary_faces) > 0, "No boundary faces found"
    assert np.all(mesh_instance.boundary_faces >= 0), (
        "Invalid negative boundary face indices"
    )
    assert np.all(mesh_instance.boundary_faces < len(mesh_instance.face_areas)), (
        "Boundary face index out of range"
    )

    # Check all boundary faces have no neighbors
    for bf in mesh_instance.boundary_faces:
        assert mesh_instance.neighbor_cells[bf] == -1, (
            "Boundary face has a neighbor cell"
        )

    # Test most cells own at least one face
    # Note: Some meshes may have orphan cells depending on how they were generated
    n_cells = len(mesh_instance.cell_volumes)
    face_counts = np.bincount(mesh_instance.owner_cells, minlength=n_cells)
    orphan_cells = np.where(face_counts == 0)[0]
    assert len(orphan_cells) < n_cells / 2, (
        f"Too many cells ({len(orphan_cells)}/{n_cells}) own no faces"
    )


def test_mesh_face_normals(mesh_instance):
    """Test face normal vectors properties."""
    # Test face normals are unit vectors
    normal_magnitudes = np.linalg.norm(mesh_instance.face_normals, axis=1)
    assert np.allclose(normal_magnitudes, 1.0, atol=1e-10), (
        "Face normals are not unit vectors"
    )

    # Test face normals point outward from owner cell
    for f in range(len(mesh_instance.face_areas)):
        owner = mesh_instance.owner_cells[f]
        if owner == -1:  # Skip if no owner (shouldn't happen)
            continue

        face_center = mesh_instance.face_centers[f]
        cell_center = mesh_instance.cell_centers[owner]
        normal = mesh_instance.face_normals[f]

        # Vector from cell center to face center
        cf_vector = face_center - cell_center

        # For properly oriented meshes, the dot product should be positive or very close to zero
        # (very close to zero can happen for orthogonal meshes where face is exactly perpendicular)
        assert np.dot(normal, cf_vector) > -1e-8, (
            f"Face {f} normal points inward to owner cell"
        )


def test_mesh_interpolation_factors(mesh_instance):
    """Test face interpolation factors."""
    internal_faces = mesh_instance.neighbor_cells != -1
    factors = mesh_instance.face_interp_factors[internal_faces]

    # Check factors are in valid range [0, 1]
    assert np.all(factors >= 0) and np.all(factors <= 1), (
        "Interpolation factors outside valid range [0, 1]"
    )


def test_mesh_orthogonality(mesh_instance):
    """Test mesh orthogonality."""
    # For a structured mesh, we might want to test that all cells are orthogonal
    if mesh_instance.is_structured:
        assert mesh_instance.is_orthogonal, "Structured mesh should be orthogonal"

    # Test non-orthogonality correction vectors
    internal_faces = mesh_instance.neighbor_cells != -1
    corrections = mesh_instance.non_ortho_correction[internal_faces]

    # Check non-orthogonality vectors are properly sized
    assert corrections.shape[1] == 2, "Non-orthogonality vectors aren't 2D"


def test_mesh_boundary_patches(mesh_instance):
    """Test boundary patch IDs."""
    # Test all boundary patches have valid IDs (non-negative)
    assert np.all(mesh_instance.boundary_patches >= 0), "Negative boundary patch IDs"

    # Test boundary_types match boundary_patches[boundary_faces]
    for i, bf in enumerate(mesh_instance.boundary_faces):
        assert mesh_instance.boundary_patches[bf] == mesh_instance.boundary_types[i], (
            f"Mismatched boundary type for face {bf}"
        )


def test_mesh_non_degenerate(mesh_instance):
    """Test mesh has no degenerate elements."""
    # Check face areas are not too small
    assert np.min(mesh_instance.face_areas) > 1e-10, "Faces with near-zero area found"

    # Check cell volumes are not too small
    assert np.min(mesh_instance.cell_volumes) > 1e-10, (
        "Cells with near-zero volume found"
    )
