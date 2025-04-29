"""
Tests for the MatrixMomentumSolver.
"""

import unittest
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the solver
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from naviflow_oo.solver.momentum_solver.matrix_momentum_solver import MatrixMomentumSolver
from naviflow_oo.preprocessing.mesh.structured import StructuredUniform
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.constructor.boundary_conditions import BoundaryConditionManager


class TestMatrixMomentumSolver(unittest.TestCase):
    
    def setUp(self):
        # Create a small mesh for testing
        self.nx = 10
        self.ny = 10
        self.lx = 1.0
        self.ly = 1.0
        self.mesh = StructuredUniform(self.nx, self.ny, self.lx, self.ly)
        
        # Set fluid properties
        self.fluid = FluidProperties(density=1.0, reynolds_number=100, characteristic_velocity=1.0)
        
        # Initialize fields
        imax, jmax = self.nx, self.ny
        self.u = np.zeros((imax+1, jmax))
        self.v = np.zeros((imax, jmax+1))
        self.p = np.zeros((imax, jmax))
        
        # Set boundary conditions (lid-driven cavity)
        self.bc_manager = BoundaryConditionManager()
        self.bc_manager.set_condition('north', 'u', {'type': 'dirichlet', 'value': 1.0})
        self.bc_manager.set_condition('north', 'v', {'type': 'dirichlet', 'value': 0.0})
        self.bc_manager.set_condition('south', 'u', {'type': 'dirichlet', 'value': 0.0})
        self.bc_manager.set_condition('south', 'v', {'type': 'dirichlet', 'value': 0.0})
        self.bc_manager.set_condition('east', 'u', {'type': 'dirichlet', 'value': 0.0})
        self.bc_manager.set_condition('east', 'v', {'type': 'dirichlet', 'value': 0.0})
        self.bc_manager.set_condition('west', 'u', {'type': 'dirichlet', 'value': 0.0})
        self.bc_manager.set_condition('west', 'v', {'type': 'dirichlet', 'value': 0.0})
        
        # Apply BCs to initial fields
        self.u, self.v = self.bc_manager.apply_velocity_boundary_conditions(
            self.u, self.v, imax, jmax
        )
    
    def test_bicgstab_solver(self):
        """Test that BiCGSTAB solver works correctly."""
        
        # Create solver
        solver = MatrixMomentumSolver(
            solver_type='bicgstab',
            discretization_scheme='power_law',
            tolerance=1e-5,
            max_iterations=100,
            use_preconditioner=False,
            print_its=False
        )
        
        # Solve u-momentum equation
        u_star, d_u, residual_info_u = solver.solve_u_momentum(
            self.mesh, self.fluid, self.u, self.v, self.p,
            relaxation_factor=0.7, boundary_conditions=self.bc_manager
        )
        
        # Solve v-momentum equation
        v_star, d_v, residual_info_v = solver.solve_v_momentum(
            self.mesh, self.fluid, self.u, self.v, self.p,
            relaxation_factor=0.7, boundary_conditions=self.bc_manager
        )
        
        # Verify solver returns expected data types
        self.assertIsInstance(u_star, np.ndarray)
        self.assertIsInstance(d_u, np.ndarray)
        self.assertIsInstance(residual_info_u, dict)
        self.assertIn('rel_norm', residual_info_u)
        self.assertIn('field', residual_info_u)
        self.assertIn('iterations', residual_info_u)
        
        # Check boundary conditions are applied correctly
        self.assertTrue(np.all(u_star[:, self.ny-1] == 1.0))  # North boundary
        self.assertTrue(np.all(u_star[:, 0] == 0.0))          # South boundary
        self.assertTrue(np.all(u_star[self.nx, :] == 0.0))    # East boundary
        self.assertTrue(np.all(u_star[0, :] == 0.0))          # West boundary
    
    def test_gmres_solver(self):
        """Test that GMRES solver works correctly."""
        
        # Create solver
        solver = MatrixMomentumSolver(
            solver_type='gmres',
            discretization_scheme='power_law',
            tolerance=1e-5,
            max_iterations=100,
            use_preconditioner=False,
            print_its=False,
            restart=20
        )
        
        # Solve u-momentum equation
        u_star, d_u, residual_info_u = solver.solve_u_momentum(
            self.mesh, self.fluid, self.u, self.v, self.p,
            relaxation_factor=0.7, boundary_conditions=self.bc_manager
        )
        
        # Solve v-momentum equation
        v_star, d_v, residual_info_v = solver.solve_v_momentum(
            self.mesh, self.fluid, self.u, self.v, self.p,
            relaxation_factor=0.7, boundary_conditions=self.bc_manager
        )
        
        # Verify solver returns expected data types
        self.assertIsInstance(u_star, np.ndarray)
        self.assertIsInstance(d_u, np.ndarray)
        self.assertIsInstance(residual_info_u, dict)
        self.assertIn('rel_norm', residual_info_u)
        self.assertIn('field', residual_info_u)
        self.assertIn('iterations', residual_info_u)
        
        # Check boundary conditions are applied correctly
        self.assertTrue(np.all(u_star[:, self.ny-1] == 1.0))  # North boundary
        self.assertTrue(np.all(u_star[:, 0] == 0.0))          # South boundary
        self.assertTrue(np.all(u_star[self.nx, :] == 0.0))    # East boundary
        self.assertTrue(np.all(u_star[0, :] == 0.0))          # West boundary
    
    def test_amg_solver(self):
        """Test that AMG solver works correctly."""
        
        # Skip test if pyamg is not available
        try:
            import pyamg
        except ImportError:
            self.skipTest("PyAMG not installed, skipping AMG solver test")
            
        # Create solver
        solver = MatrixMomentumSolver(
            solver_type='amg',
            discretization_scheme='power_law',
            tolerance=1e-5,
            max_iterations=100,
            print_its=False,
            amg_cycles=1,
            amg_cycle_type='V'
        )
        
        # Solve u-momentum equation
        u_star, d_u, residual_info_u = solver.solve_u_momentum(
            self.mesh, self.fluid, self.u, self.v, self.p,
            relaxation_factor=0.7, boundary_conditions=self.bc_manager
        )
        
        # Solve v-momentum equation
        v_star, d_v, residual_info_v = solver.solve_v_momentum(
            self.mesh, self.fluid, self.u, self.v, self.p,
            relaxation_factor=0.7, boundary_conditions=self.bc_manager
        )
        
        # Verify solver returns expected data types
        self.assertIsInstance(u_star, np.ndarray)
        self.assertIsInstance(d_u, np.ndarray)
        self.assertIsInstance(residual_info_u, dict)
        self.assertIn('rel_norm', residual_info_u)
        self.assertIn('field', residual_info_u)
        self.assertIn('iterations', residual_info_u)
        
        # Check boundary conditions are applied correctly
        self.assertTrue(np.all(u_star[:, self.ny-1] == 1.0))  # North boundary
        self.assertTrue(np.all(u_star[:, 0] == 0.0))          # South boundary
        self.assertTrue(np.all(u_star[self.nx, :] == 0.0))    # East boundary
        self.assertTrue(np.all(u_star[0, :] == 0.0))          # West boundary
    
    def test_with_preconditioning(self):
        """Test both solvers with preconditioning."""
        
        # BiCGSTAB with preconditioning
        solver_bicg_prec = MatrixMomentumSolver(
            solver_type='bicgstab',
            discretization_scheme='power_law',
            tolerance=1e-5,
            max_iterations=100,
            use_preconditioner=True,
            print_its=False
        )
        
        u_bicg_prec, _, _ = solver_bicg_prec.solve_u_momentum(
            self.mesh, self.fluid, self.u, self.v, self.p,
            relaxation_factor=0.7, boundary_conditions=self.bc_manager
        )
        
        # GMRES with preconditioning
        solver_gmres_prec = MatrixMomentumSolver(
            solver_type='gmres',
            discretization_scheme='power_law',
            tolerance=1e-5,
            max_iterations=100,
            use_preconditioner=True,
            print_its=False
        )
        
        u_gmres_prec, _, _ = solver_gmres_prec.solve_u_momentum(
            self.mesh, self.fluid, self.u, self.v, self.p,
            relaxation_factor=0.7, boundary_conditions=self.bc_manager
        )
        
        # AMG solver (AMG doesn't use ILU preconditioning)
        try:
            import pyamg
            
            solver_amg = MatrixMomentumSolver(
                solver_type='amg',
                discretization_scheme='power_law',
                tolerance=1e-5,
                max_iterations=100,
                print_its=False
            )
            
            u_amg, _, _ = solver_amg.solve_u_momentum(
                self.mesh, self.fluid, self.u, self.v, self.p,
                relaxation_factor=0.7, boundary_conditions=self.bc_manager
            )
            
            # Compare all solvers
            self.assertTrue(np.allclose(u_bicg_prec, u_gmres_prec, atol=1e-4))
            self.assertTrue(np.allclose(u_bicg_prec, u_amg, atol=1e-4))
            
        except ImportError:
            # Just compare BiCGSTAB and GMRES if PyAMG is not available
            self.assertTrue(np.allclose(u_bicg_prec, u_gmres_prec, atol=1e-4))


if __name__ == '__main__':
    unittest.main() 