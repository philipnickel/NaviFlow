import pytest
import numpy as np
from pathlib import Path

from naviflow_collocated.mesh import load_msh_file


@pytest.fixture
def msh_file_path():
    """Fixture to provide a path to a sample .msh file."""
    # Use the same hardcoded mesh file as in conftest.py
    base_dir = Path(__file__).parent.parent
    mesh_file = (
        base_dir
        / "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh"
    )

    if mesh_file.exists():
        return str(mesh_file)
    else:
        pytest.skip(f"Mesh file not found at {mesh_file}")


def test_msh_loader_basic(msh_file_path):
    """Test that the mesh loader can load a .msh file without errors."""
    mesh, quality = load_msh_file(msh_file_path)

    # Basic validation of mesh properties
    assert mesh is not None
    assert hasattr(mesh, "cell_volumes")
    assert hasattr(mesh, "face_areas")
    assert hasattr(mesh, "face_normals")

    # Check that quality metrics are returned
    assert isinstance(quality, dict)
    assert "quality" in quality
    assert "orphan_count" in quality
    assert "orphan_percentage" in quality


def test_msh_loader_geometric_properties(msh_file_path):
    """Test geometric properties of the loaded mesh."""
    mesh, _ = load_msh_file(msh_file_path)

    # Geometric properties should be valid
    assert np.all(mesh.cell_volumes > 0)
    assert np.all(mesh.face_areas > 0)

    # Normals should be unit vectors
    normal_magnitudes = np.linalg.norm(mesh.face_normals, axis=1)
    assert np.allclose(normal_magnitudes, 1.0, atol=1e-10)


def test_msh_loader_connectivity(msh_file_path):
    """Test the mesh connectivity."""
    mesh, quality = load_msh_file(msh_file_path)

    # Owner cell indices should be valid
    assert np.all(mesh.owner_cells >= 0)
    assert np.max(mesh.owner_cells) < len(mesh.cell_volumes)

    # Boundary faces should be properly marked
    assert len(mesh.boundary_faces) > 0
    for bf in mesh.boundary_faces:
        assert mesh.neighbor_cells[bf] == -1

    # Print quality stats for reference
    print(f"Mesh quality assessment: {quality['quality'].upper()}")
    print(
        f"Orphan cells: {quality['orphan_count']} ({quality['orphan_percentage']:.2f}%)"
    )
    print(f"Face count stats: {quality['face_count_stats']}")


def test_msh_loader_non_orthogonality(msh_file_path):
    """Test the non-orthogonality correction."""
    mesh, _ = load_msh_file(msh_file_path)

    # Non-orthogonality correction should be computed
    assert mesh.non_ortho_correction.shape == mesh.d_CF.shape

    # Orthogonality property should be set
    assert isinstance(mesh.is_orthogonal, bool)


def test_msh_loader_suppress_warnings(msh_file_path):
    """Test that warnings can be suppressed."""
    # With suppress_warnings=True, we should not get warning output
    mesh, quality = load_msh_file(msh_file_path, suppress_warnings=True)

    # But quality metrics should still be returned
    assert "quality" in quality
    assert "orphan_count" in quality

    # Mesh should still be valid
    assert mesh is not None
    assert np.all(mesh.cell_volumes > 0)
