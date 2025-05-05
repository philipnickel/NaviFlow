"""
Implements the SIMPLE (Semi-Implicit Method for Pressure Linked Equations) algorithm.
"""

import numpy as np

from naviflow_collocated.mesh import MeshData2D as Mesh
# from naviflow_collocated.linear_solvers.solver_base import LinearSolverBase # Placeholder

from naviflow_collocated.assembly.momentum_eq_assembly import (
    assemble_momentum_matrix_csr,
)
from naviflow_collocated.assembly.pressure_correction_eq_assembly import (
    assemble_pressure_correction_matrix_csr,
)
from naviflow_collocated.solvers.correction_steps import (
    update_pressure,
    correct_velocity,
)
from naviflow_collocated.gradients.gradients import calculate_gradient_gauss_linear


class SimpleSolver:
    """
    Orchestrates the SIMPLE algorithm steps for solving incompressible flow.

    Uses dependency injection for mesh, configuration, and linear solvers.
    """

    def __init__(
        self,
        mesh: Mesh,
        config: dict,
        linear_solver_u,
        linear_solver_v,
        linear_solver_p,
    ):
        """
        Initializes the SIMPLE solver.

        Args:
            mesh: The computational mesh.
            config: Dictionary containing simulation parameters (rho, mu, relaxation factors, etc.).
            linear_solver_u: Solver instance for the u-momentum equation.
            linear_solver_v: Solver instance for the v-momentum equation.
            linear_solver_p: Solver instance for the pressure correction equation.
        """
        self.mesh = mesh
        self.config = config
        self.linear_solver_u = linear_solver_u
        self.linear_solver_v = linear_solver_v
        self.linear_solver_p = linear_solver_p

        # Extract physical properties and relaxation factors
        self.rho = config["physical_properties"]["rho"]
        self.mu = config["physical_properties"]["mu"]
        self.alpha_u = config["relaxation_factors"]["u"]
        self.alpha_v = config["relaxation_factors"]["v"]
        self.alpha_p = config["relaxation_factors"]["p"]

        # Initialize fields (consider moving initialization outside later)
        n_cells = len(mesh.cell_volumes)
        self.u = np.zeros(n_cells)
        self.v = np.zeros(n_cells)
        self.p = np.zeros(n_cells)

        # Store unrelaxed Ap values needed for correction steps
        self.Ap_u_unrelaxed = np.ones(n_cells)  # Initialize
        self.Ap_v_unrelaxed = np.ones(n_cells)  # Initialize

    def run_iteration(self):
        """Runs a single iteration of the SIMPLE algorithm."""

        # --- 1. Solve Momentum Equations (U*, V*) ---
        # Assemble U-momentum
        A_u, b_u = assemble_momentum_matrix_csr(
            self.mesh,
            self.u,
            self.v,
            self.p,
            self.mu,
            self.rho,
            None,  # No boundary_conditions needed
            component=0,
            under_relax=self.alpha_u,
        )
        # Store unrelaxed diagonal (handle potential division by zero if alpha_u is 0)
        self.Ap_u_unrelaxed = (
            A_u.diagonal() / self.alpha_u if self.alpha_u != 0 else A_u.diagonal()
        )

        # Solve U-momentum (placeholder)
        # u_star = self.linear_solver_u.solve(A_u, b_u, self.u) # Pass current u as guess
        u_star = np.linalg.solve(A_u.toarray(), b_u)  # TEMPORARY: Dense solve
        self.u = u_star  # Update u field for next step

        # Assemble V-momentum
        A_v, b_v = assemble_momentum_matrix_csr(
            self.mesh,
            self.u,
            self.v,
            self.p,
            self.mu,
            self.rho,
            None,  # No boundary_conditions needed
            component=1,
            under_relax=self.alpha_v,
        )
        # Store unrelaxed diagonal (handle potential division by zero if alpha_v is 0)
        self.Ap_v_unrelaxed = (
            A_v.diagonal() / self.alpha_v if self.alpha_v != 0 else A_v.diagonal()
        )

        # Solve V-momentum (placeholder)
        # v_star = self.linear_solver_v.solve(A_v, b_v, self.v) # Pass current v as guess
        v_star = np.linalg.solve(A_v.toarray(), b_v)  # TEMPORARY: Dense solve
        self.v = v_star  # Update v field for next step

        # --- 2. Solve Pressure Correction Equation (p') ---
        # Assemble p' equation
        A_p_prime, b_p_prime = assemble_pressure_correction_matrix_csr(
            self.mesh,
            self.u,
            self.v,
            self.p,
            self.rho,
            self.Ap_u_unrelaxed,
            self.Ap_v_unrelaxed,
            None,  # No boundary_conditions needed
            # Pressure relaxation is applied *after* solve
        )

        # Solve p' equation (placeholder)
        # p_prime = self.linear_solver_p.solve(A_p_prime, b_p_prime, np.zeros_like(self.p))
        # Need initial guess for p_prime, zeros is common
        p_prime = np.linalg.solve(
            A_p_prime.toarray(), b_p_prime
        )  # TEMPORARY: Dense solve

        # --- 3. Update Pressure (p) ---
        self.p = update_pressure(self.p, p_prime, self.alpha_p)

        # --- 4. Correct Velocities (U, V) ---
        # Velocity correction needs gradient of p_prime
        p_prime_gradient = calculate_gradient_gauss_linear(
            self.mesh,
            p_prime,
            None,  # No boundary_conditions needed
        )
        # Pass the pre-calculated gradient to correct_velocity
        self.u, self.v = correct_velocity(
            self.mesh,
            self.u,
            self.v,
            p_prime,
            self.Ap_u_unrelaxed,
            self.Ap_v_unrelaxed,
            p_prime_gradient,
        )

        # --- 5. Update Boundary Conditions (if needed) ---
        # Some BCs might need explicit updates after correction (e.g., derived outlets)
        # Add logic here if required later.

        # --- 6. Calculate Residuals / Check Convergence (Optional Here) ---
        # Typically done outside the single iteration method.

        print("SIMPLE iteration completed (using temporary dense solves).")
        # Return fields or residuals if needed
        # return self.u, self.v, self.p
