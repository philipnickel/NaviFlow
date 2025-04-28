"""
Mesh module for 2D computational fluid dynamics simulations.
Provides structured and unstructured mesh capabilities.
"""

from .mesh import Mesh, UnstructuredMesh
from .structured import (
    StructuredMesh,
    StructuredUniform,
    StructuredNonUniform
)
from .unstructured import (
    UnstructuredUniform,
    UnstructuredRefined,
    Unstructured  # Kept for backwards compatibility
)
from .plot_utils import plot_mesh, plot_structured_mesh, plot_unstructured_mesh 