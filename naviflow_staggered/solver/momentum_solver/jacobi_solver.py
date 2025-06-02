"""
Jacobi-method momentum solver that can use different discretization schemes.
"""

import numpy as np
from .base_momentum_solver import MomentumSolver
from .discretization import power_law
from ...constructor.boundary_conditions import BoundaryConditionManager

class JacobiMomentumSolver(MomentumSolver):
    """
    Momentum solver that uses Jacobi iterations to solve the momentum equations.
    Can use different discretization schemes.
    """

    def __init__(self, discretization_scheme='power_law', n_jacobi_sweeps=1):
        super().__init__()
        self.n_jacobi_sweeps = n_jacobi_sweeps

        if discretization_scheme == 'power_law':
            self.discretization = power_law.PowerLawDiscretization()
        else:
            raise ValueError(f"Unsupported discretization scheme: {discretization_scheme}")

        self.u_a_e = None
        self.u_a_w = None
        self.u_a_n = None
        self.u_a_s = None
        self.u_a_p = None
        self.u_source = None
        self.u_a_p_unrelaxed = None
        self.u_source_unrelaxed = None

        self.v_a_e = None
        self.v_a_w = None
        self.v_a_n = None
        self.v_a_s = None
        self.v_a_p = None
        self.v_source = None
        self.v_a_p_unrelaxed = None
        self.v_source_unrelaxed = None

    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        u_star = np.zeros((imax+1, jmax))
        d_u = np.zeros((imax+1, jmax))

        # Apply BCs before coefficient generation
        if boundary_conditions:
            if isinstance(boundary_conditions, BoundaryConditionManager):
                bc_manager = boundary_conditions
            else:
                bc_manager = BoundaryConditionManager()
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)
            u, v = bc_manager.apply_velocity_boundary_conditions(u.copy(), v.copy(), imax, jmax)

        self.calculate_coefficients(mesh, fluid, u, v, p, boundary_conditions, relaxation_factor)

        u_star_unrelaxed = u.copy()
        i_range = np.arange(1, imax)
        j_range = np.arange(1, jmax-1)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')

        for _ in range(self.n_jacobi_sweeps):
            u_old = u_star_unrelaxed.copy()
            correction = (
                (self.u_a_e[i_grid, j_grid] * u_old[i_grid + 1, j_grid] +
                 self.u_a_w[i_grid, j_grid] * u_old[i_grid - 1, j_grid] +
                 self.u_a_n[i_grid, j_grid] * u_old[i_grid, j_grid + 1] +
                 self.u_a_s[i_grid, j_grid] * u_old[i_grid, j_grid - 1] + 
                 self.u_source[i_grid, j_grid]) / self.u_a_p[i_grid, j_grid] - u_old[i_grid, j_grid]
            )
            u_star_unrelaxed[i_grid, j_grid] += correction

        u_star[i_grid, j_grid] = alpha * u_star_unrelaxed[i_grid, j_grid] + (1-alpha)*u[i_grid, j_grid]
        d_u[i_grid, j_grid] = alpha * mesh.get_cell_sizes()[1] / self.u_a_p[i_grid, j_grid]

        j = 0
        i_bottom = np.arange(1, imax)
        d_u[i_bottom, j] = alpha * mesh.get_cell_sizes()[1] / self.u_a_p[i_bottom, j]

        j = jmax-1
        i_top = np.arange(1, imax)
        d_u[i_top, j] = alpha * mesh.get_cell_sizes()[1] / self.u_a_p[i_top, j]

        if boundary_conditions:
            u_star, _ = bc_manager.apply_velocity_boundary_conditions(u_star, v.copy(), imax, jmax)

        u_residual = self.calculate_u_algebraic_residual(u_star)

        return u_star, d_u, u_residual

    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, boundary_conditions=None):
        nx, ny = mesh.get_dimensions()
        imax, jmax = nx, ny
        alpha = relaxation_factor
        v_star = np.zeros((imax, jmax+1))
        d_v = np.zeros((imax, jmax+1))

        if boundary_conditions:
            if isinstance(boundary_conditions, BoundaryConditionManager):
                bc_manager = boundary_conditions
            else:
                bc_manager = BoundaryConditionManager()
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)
            u, v = bc_manager.apply_velocity_boundary_conditions(u.copy(), v.copy(), imax, jmax)

        if self.v_a_p is None:
            self.calculate_coefficients(mesh, fluid, u, v, p, boundary_conditions, relaxation_factor)

        v_star_unrelaxed = v.copy()
        i_range = np.arange(1, imax-1)
        j_range = np.arange(1, jmax)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')

        for _ in range(self.n_jacobi_sweeps):
            v_old = v_star_unrelaxed.copy()
            correction = (
                (self.v_a_e[i_grid, j_grid] * v_old[i_grid + 1, j_grid] +
                 self.v_a_w[i_grid, j_grid] * v_old[i_grid - 1, j_grid] +
                 self.v_a_n[i_grid, j_grid] * v_old[i_grid, j_grid + 1] +
                 self.v_a_s[i_grid, j_grid] * v_old[i_grid, j_grid - 1] + 
                 self.v_source[i_grid, j_grid]) / self.v_a_p[i_grid, j_grid] - v_old[i_grid, j_grid]
            )
            v_star_unrelaxed[i_grid, j_grid] += correction

        v_star[i_grid, j_grid] = alpha * v_star_unrelaxed[i_grid, j_grid] + (1-alpha)*v[i_grid, j_grid]
        d_v[i_grid, j_grid] = alpha * mesh.get_cell_sizes()[0] / self.v_a_p[i_grid, j_grid]

        i = 0
        j_left = np.arange(1, jmax)
        d_v[i, j_left] = alpha * mesh.get_cell_sizes()[0] / self.v_a_p[i, j_left]

        i = imax-1
        j_right = np.arange(1, jmax)
        d_v[i, j_right] = alpha * mesh.get_cell_sizes()[0] / self.v_a_p[i, j_right]

        if boundary_conditions:
            _, v_star = bc_manager.apply_velocity_boundary_conditions(u.copy(), v_star, imax, jmax)

        v_residual = self.calculate_v_algebraic_residual(v_star)

        return v_star, d_v, v_residual