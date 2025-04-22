# Algorithms module initialization 

from .base_algorithm import BaseAlgorithm
from .simple import SimpleSolver
# Import the deprecated solver class for backward compatibility
from .simple_with_dict import SimpleSolverDict  # Deprecated: Use SimpleSolver instead