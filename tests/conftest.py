# conftest.py

import pytest
from pathlib import Path
from naviflow_collocated.mesh.mesh_loader import load_mesh


def find_test_meshes():
    """Find mesh files for testing in the meshing/experiments directory."""
    base_dir = Path(__file__).parent.parent

    structured_file = (
        base_dir
        / "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh"
    )
    unstructured_file = (
        base_dir
        / "meshing/experiments/lidDrivenCavity/unstructured/coarse/lidDrivenCavity_unstructured_coarse.msh"
    )
    cylinder_file = (
        base_dir
        / "meshing/experiments/cylinderFlow/unstructured/coarse/cylinderFlow_unstructured_coarse.msh"
    )
    sanity_check_uni_form_file = (
        base_dir
        / "meshing/experiments/sanityCheck/structuredUniform/coarse/sanityCheck_uniform_coarse.msh"
    )
    sanity_check_unstructured_file = (
        base_dir
        / "meshing/experiments/sanityCheck/unstructured/coarse/sanityCheck_unstructured_coarse.msh"
    )

    mesh_files = {}

    if structured_file.exists():
        mesh_files["structured_uniform"] = str(structured_file)
    else:
        print(f"Warning: Structured mesh file not found at {structured_file}")

    if unstructured_file.exists():
        mesh_files["unstructured_refined"] = str(unstructured_file)
    else:
        print(f"Warning: Unstructured mesh file not found at {unstructured_file}")

    if cylinder_file.exists():
        mesh_files["cylinder_flow"] = str(cylinder_file)
    else:
        print(f"Warning: Cylinder flow mesh file not found at {cylinder_file}")

    if sanity_check_uni_form_file.exists():
        mesh_files["sanity_check_uniform"] = str(sanity_check_uni_form_file)
    else:
        print(
            f"Warning: Sanity check uniform mesh file not found at {sanity_check_uni_form_file}"
        )

    if sanity_check_unstructured_file.exists():
        mesh_files["sanity_check_unstructured"] = str(sanity_check_unstructured_file)
    else:
        print(
            f"Warning: Sanity check unstructured mesh file not found at {sanity_check_unstructured_file}"
        )

    return mesh_files


@pytest.fixture
def mesh_instance(mesh_label):
    mesh_file_map = find_test_meshes()
    mesh_file = mesh_file_map[mesh_label]
    return load_mesh(mesh_file)


def pytest_generate_tests(metafunc):
    if "mesh_label" in metafunc.fixturenames:
        metafunc.parametrize(
            "mesh_label",
            [
                "structured_uniform",
                "unstructured_refined",
                "sanity_check_uniform",
                "sanity_check_unstructured",
            ],  # , "cylinder_flow"],
        )
