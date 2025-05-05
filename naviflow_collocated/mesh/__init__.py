"""
Mesh module for NaviFlow collocated CFD solver.

Provides functionality for mesh generation, loading, and representation
for finite volume discretization.
"""

from .mesh_data import MeshData2D
from .structured_uniform import generate as generate_structured_uniform
from .unstructured import generate as generate_unstructured
from .mesh_loader import load_msh_file

__all__ = [
    "MeshData2D",
    "generate_structured_uniform",
    "generate_unstructured",
    "load_msh_file",
]
