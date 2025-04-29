"""
Mesh module for 2D computational fluid dynamics simulations.
Provides structured and unstructured mesh capabilities.
"""

# from .mesh import Mesh, UnstructuredMesh # Base Mesh class is imported below
from .mesh import Mesh # Import base Mesh class

# Import the consolidated structured mesh class
from .structured_mesh import StructuredMesh

# Remove imports from the deleted .structured module
# from .structured import (
#     StructuredMesh,
#     StructuredUniform,
#     StructuredNonUniform
# )

# Keep unstructured imports
from .unstructured import (
    UnstructuredUniform,
    UnstructuredRefined,
    Unstructured  # Kept for backwards compatibility
)
from .plot_utils import plot_mesh, plot_structured_mesh, plot_unstructured_mesh 