# conftest.py

import pytest
import os
from pathlib import Path
from naviflow_collocated.mesh.structured_uniform import (
    generate as generate_structured_uniform,
)
from naviflow_collocated.mesh.unstructured import generate as generate_unstructured
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

    return mesh_files


def generate_test_mesh(mesh_type):
    """Generate a test mesh if no .msh files found."""
    output_file = f"test_{mesh_type}.msh"

    if mesh_type == "structured_uniform":
        generate_structured_uniform(L=1.0, nx=15, ny=15, output_filename=output_file)
    elif mesh_type == "unstructured_refined":
        generate_unstructured(
            Lx=1.0, Ly=1.0, n_cells=100, ratio=2.5, output_filename=output_file
        )
    elif mesh_type == "cylinder_flow":
        generate_unstructured(
            Lx=4.0,
            Ly=1.0,
            n_cells=200,
            ratio=2.5,
            obstacle={"type": "circle", "center": (1.0, 0.5), "radius": 0.1},
            output_filename=output_file,
        )
    return output_file


@pytest.fixture
def mesh_instance(request):
    """Provide a mesh instance for testing using the new mesh loader."""
    test_meshes = find_test_meshes()

    if request.param in test_meshes:
        mesh_file = test_meshes[request.param]
        print(f"Loading {request.param} mesh from: {mesh_file}")
        mesh = load_mesh(mesh_file)
        return mesh

    print(f"No pre-existing mesh found for {request.param}, generating one...")
    mesh_file = generate_test_mesh(request.param)
    mesh = load_mesh(mesh_file)

    if os.path.exists(mesh_file):
        try:
            os.remove(mesh_file)
        except FileNotFoundError:
            pass

    return mesh


def pytest_generate_tests(metafunc):
    if "mesh_instance" in metafunc.fixturenames:
        meshes = [
            "structured_uniform",
            "unstructured_refined",
            # "cylinder_flow",
        ]
        metafunc.parametrize("mesh_instance", meshes, indirect=True)
