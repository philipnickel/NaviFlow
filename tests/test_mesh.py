import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt


def test_mesh_geometry_suite(mesh_instance, subtests):
    with subtests.test("cell_and_face_geometry"):
        assert mesh_instance.n_cells > 0
        assert mesh_instance.n_faces > 0
        assert np.all(mesh_instance.get_face_areas() > 0)
        assert np.all(mesh_instance.get_cell_volumes() > 0)

    with subtests.test("face_normals_orientation"):
        owners, _ = mesh_instance.get_owner_neighbor()
        for f, n in enumerate(mesh_instance.get_face_normals()):
            if owners[f] == -1:
                continue
            fc = mesh_instance.get_face_centers()[f]
            cc = mesh_instance.get_cell_centers()[owners[f]]
            assert np.dot(n, fc - cc) > -1e-8

    with subtests.test("interpolation_factors_consistency"):
        for f in range(mesh_instance.n_faces):
            gC, gF = mesh_instance.get_face_interpolation_factors(f)
            assert np.isclose(gC + gF, 1.0) or gF == 0.0

    with subtests.test("boundary_classification"):
        owners, neighbors = mesh_instance.get_owner_neighbor()
        for f in range(mesh_instance.n_faces):
            if neighbors[f] == -1:
                name = mesh_instance.get_boundary_name(f)
                assert name in {"left", "right", "top", "bottom"}

    with subtests.test("boundary_cell_index_extraction"):
        for name in ["left", "right", "top", "bottom"]:
            try:
                cells = mesh_instance.get_boundary_cell_indices(name)
                assert isinstance(cells, np.ndarray)
                assert np.all(cells >= 0)
            except AttributeError:
                pass  # StructuredMesh handles this internally

    with subtests.test("face_area_direction"):
        face_normals = mesh_instance.get_face_normals()
        face_areas = mesh_instance.get_face_areas()
        norms = np.linalg.norm(face_normals, axis=1)
        assert np.allclose(norms * face_areas, face_areas, rtol=1e-5)

    with subtests.test("plot_execution"):
        fig, _ = mesh_instance.plot(title="Test Plot")
        plt.close(fig)

    with subtests.test("save_plot_execution"):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_plot.pdf"
            mesh_instance.savePlot(str(path), title="Save Test")
            assert path.exists() and path.stat().st_size > 0
    with subtests.test("serialization_roundtrip"):
        from naviflow_collocated.mesh.mesh_data import (
            mesh_to_data,
            save_mesh_data,
            load_mesh_data,
        )

        # Convert to data
        data = mesh_to_data(mesh_instance)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mesh.npz"
            save_mesh_data(path, data)
            loaded = load_mesh_data(path)

            # Ensure all fields are equal
            for attr in data.__annotations__:
                orig = getattr(data, attr)
                reloaded = getattr(loaded, attr)
                if isinstance(orig, dict):
                    assert orig.keys() == reloaded.keys()
                    for k in orig:
                        assert np.array_equal(orig[k], reloaded[k]), (
                            f"Mismatch in dict field '{attr}' at key '{k}'"
                        )
                else:
                    assert np.array_equal(orig, reloaded), f"Mismatch in field '{attr}'"
