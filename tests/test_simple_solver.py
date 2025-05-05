"""
Tests for the SimpleSolver class.
"""

import numpy as np
import pytest

from naviflow_collocated.solvers.simple_solver import SimpleSolver


# Mock Linear Solver class for testing initialization
class MockLinearSolver:
    def solve(self, A, b, x0):
        # Doesn't actually solve, just part of the interface
        print(f"Mock solver called with matrix shape {A.shape}")
        return np.zeros_like(b)


# Mock config dictionary
def get_mock_config():
    return {
        "physical_properties": {"rho": 1.0, "mu": 0.01},
        "relaxation_factors": {"u": 0.7, "v": 0.7, "p": 0.3},
        "boundary_conditions": [],
    }


def test_simple_solver_initialization(mesh_instance):
    """Test that the SimpleSolver initializes correctly."""
    config = get_mock_config()
    mock_solver = MockLinearSolver()
    n_cells = len(mesh_instance.cell_volumes)

    try:
        solver = SimpleSolver(
            mesh=mesh_instance,
            config=config,
            linear_solver_u=mock_solver,
            linear_solver_v=mock_solver,
            linear_solver_p=mock_solver,
        )

        # Check if attributes are set correctly
        assert solver.mesh is mesh_instance
        assert solver.config is config
        assert solver.rho == 1.0
        assert solver.mu == 0.01
        assert solver.alpha_u == 0.7
        assert solver.alpha_v == 0.7
        assert solver.alpha_p == 0.3
        assert solver.linear_solver_u is mock_solver
        assert solver.linear_solver_v is mock_solver
        assert solver.linear_solver_p is mock_solver

        # Check field initialization
        assert isinstance(solver.u, np.ndarray) and len(solver.u) == n_cells
        assert isinstance(solver.v, np.ndarray) and len(solver.v) == n_cells
        assert isinstance(solver.p, np.ndarray) and len(solver.p) == n_cells
        assert np.all(solver.u == 0)
        assert np.all(solver.v == 0)
        assert np.all(solver.p == 0)
        assert (
            isinstance(solver.Ap_u_unrelaxed, np.ndarray)
            and len(solver.Ap_u_unrelaxed) == n_cells
        )
        assert (
            isinstance(solver.Ap_v_unrelaxed, np.ndarray)
            and len(solver.Ap_v_unrelaxed) == n_cells
        )

    except Exception as e:
        pytest.fail(f"SimpleSolver initialization failed: {e}")


# TODO: Add test for run_iteration (requires more setup or mocking)
