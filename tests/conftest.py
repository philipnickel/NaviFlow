# conftest.py

import pytest
from naviflow_collocated.mesh.structured import StructuredMesh
from naviflow_collocated.mesh.unstructured_LEGACY import (
    # UnstructuredUniform,
    UnstructuredRefined,
)


@pytest.fixture
def mesh_instance(request):
    if request.param == "structured_uniform":
        return StructuredMesh(15, 15, is_uniform=True)
    elif request.param == "structured_clustered":
        return StructuredMesh(15, 15, refine=True)
    # elif request.param == "unstructured_uniform":
    #    return UnstructuredUniform(mesh_size=0.05)
    elif request.param == "unstructured_refined":
        return UnstructuredRefined(0.03, 0.02, 0.01)
    else:
        raise ValueError(f"Unknown mesh type: {request.param}")


def pytest_generate_tests(metafunc):
    if "mesh_instance" in metafunc.fixturenames:
        meshes = [
            "structured_uniform",
            "structured_clustered",
            # "unstructured_uniform",
            "unstructured_refined",
        ]
        metafunc.parametrize("mesh_instance", meshes, indirect=True)
