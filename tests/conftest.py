# conftest.py

import pytest
import os
from pathlib import Path
from naviflow_collocated.mesh import (
    load_msh_file,
    generate_structured_uniform,
    generate_unstructured,
)


def find_lid_driven_cavity_meshes():
    """Find lid driven cavity mesh files in the meshing/experiments directory."""
    base_dir = Path(__file__).parent.parent

    # Hardcode the specific mesh file paths
    structured_file = (
        base_dir
        / "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh"
    )
    structured_file_medium = (
        base_dir
        / "meshing/experiments/lidDrivenCavity/structuredUniform/medium/lidDrivenCavity_uniform_medium.msh"
    )
    structured_file_fine = (
        base_dir
        / "meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh"
    )
    unstructured_file = (
        base_dir
        / "meshing/experiments/lidDrivenCavity/unstructured/coarse/lidDrivenCavity_unstructured_coarse.msh"
    )
    unstructured_file_medium = (
        base_dir
        / "meshing/experiments/lidDrivenCavity/unstructured/medium/lidDrivenCavity_unstructured_medium.msh"
    )
    unstructured_file_fine = (
        base_dir
        / "meshing/experiments/lidDrivenCavity/unstructured/fine/lidDrivenCavity_unstructured_fine.msh"
    )

    mesh_files = {}

    if structured_file.exists():
        mesh_files["structured_uniform"] = str(structured_file)
    if structured_file_medium.exists():
        mesh_files["structured_uniform_medium"] = str(structured_file_medium)
    if structured_file_fine.exists():
        mesh_files["structured_uniform_fine"] = str(structured_file_fine)
    else:
        print(f"Warning: Structured mesh file not found at {structured_file}")

    if unstructured_file.exists():
        mesh_files["unstructured_refined"] = str(unstructured_file)
    if unstructured_file_medium.exists():
        mesh_files["unstructured_refined_medium"] = str(unstructured_file_medium)
    if unstructured_file_fine.exists():
        mesh_files["unstructured_refined_fine"] = str(unstructured_file_fine)
    else:
        print(f"Warning: Unstructured mesh file not found at {unstructured_file}")

    return mesh_files


def generate_test_mesh(mesh_type):
    """Generate a test mesh if no .msh files found."""
    output_file = f"test_{mesh_type}.msh"

    if mesh_type == "structured_uniform":
        generate_structured_uniform(L=1.0, nx=15, ny=15, output_filename=output_file)
    elif mesh_type == "structured_uniform_medium":
        generate_structured_uniform(L=1.0, nx=10, ny=10, output_filename=output_file)
    elif mesh_type == "structured_uniform_fine":
        generate_structured_uniform(L=1.0, nx=5, ny=5, output_filename=output_file)
    elif mesh_type == "structured_clustered":
        generate_structured_uniform(
            L=1.0, nx=15, ny=15, ratio=2.0, output_filename=output_file
        )
    elif mesh_type == "unstructured_uniform":
        generate_unstructured(
            Lx=1.0, Ly=1.0, n_cells=100, ratio=1.0, output_filename=output_file
        )
    elif mesh_type == "unstructured_refined":
        generate_unstructured(
            Lx=1.0, Ly=1.0, n_cells=100, ratio=2.5, output_filename=output_file
        )
    elif mesh_type == "unstructured_refined_medium":
        generate_unstructured(
            Lx=1.0, Ly=1.0, n_cells=100, ratio=2.5, output_filename=output_file
        )
    elif mesh_type == "unstructured_refined_fine":
        generate_unstructured(
            Lx=1.0, Ly=1.0, n_cells=100, ratio=2.5, output_filename=output_file
        )
    return output_file


@pytest.fixture
def mesh_instance(request):
    """Provide a mesh instance for testing using the new mesh loader."""
    # Look for lid-driven cavity mesh files
    lid_cavity_meshes = find_lid_driven_cavity_meshes()

    # If we have a specific mesh file for the requested type, load it
    if request.param in lid_cavity_meshes:
        mesh_file = lid_cavity_meshes[request.param]
        print(f"Loading {request.param} mesh from: {mesh_file}")
        mesh, quality = load_msh_file(mesh_file)

        # Print a summary of mesh quality
        print(f"Mesh quality: {quality['quality'].upper()}")
        print(
            f"Orphan cells: {quality['orphan_count']} ({quality['orphan_percentage']:.2f}%)"
        )

        return mesh

    # Otherwise, generate a mesh based on the requested type
    print(f"No pre-existing mesh found for {request.param}, generating one...")
    mesh_file = generate_test_mesh(request.param)
    mesh, quality = load_msh_file(mesh_file)

    # Clean up the generated file after test
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
            "structured_uniform_medium",
            "structured_uniform_fine",
            "unstructured_refined",
            "unstructured_refined_medium",
            "unstructured_refined_fine",
        ]
        metafunc.parametrize("mesh_instance", meshes, indirect=True)
