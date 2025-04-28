"""
Mesh-agnostic solvers for NaviFlow.

This module contains implementations of CFD solvers that can work with arbitrary mesh topologies,
not just structured grids. These solvers are designed to work with the base Mesh class and its
subclasses, providing a unified interface for solving flow problems on different mesh types.
"""

from .power_law import MeshAgnosticPowerLaw
from .mesh_amg_solver import MeshAgnosticAMGSolver
from .mesh_direct_pressure import MeshAgnosticDirectPressureSolver
from .mesh_simple import MeshAgnosticSimpleSolver

__all__ = [
    'MeshAgnosticPowerLaw',
    'MeshAgnosticAMGSolver',
    'MeshAgnosticDirectPressureSolver',
    'MeshAgnosticSimpleSolver',
] 