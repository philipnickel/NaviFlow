from abc import ABC, abstractmethod
import numpy as np

# For full type hinting if needed later, and to avoid circular imports:
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from naviflow_collocated.mesh import MeshData2D
#     from naviflow_collocated.linear_solvers.base_solver import BaseLinearSolver


class MomentumSolverBase(ABC):
    """
    Abstract Base Class for momentum equation solvers.
    """

    def __init__(
        self,
        mesh,  # naviflow_collocated.mesh.MeshData2D
        config: dict,
        phys_props: dict,
        linear_solver,  # naviflow_collocated.linear_solvers.base_solver.BaseLinearSolver
    ):
        """
        Initializes the base momentum solver.

        Parameters:
        - mesh: MeshData2D object.
        - config: Dictionary with specific configurations for this momentum equation (e.g., schemes).
        - phys_props: Dictionary with global physical properties (e.g., {"rho": 1.0, "mu": 0.01}).
        - linear_solver: An instance of a linear solver (e.g., ScipyDirectSolver) that has a .solve(A,b) method.
        """
        self.mesh = mesh
        self.config_specific = config
        self.phys_props = phys_props
        self.linear_solver = linear_solver

        self.rho = phys_props.get("rho")
        self.mu = phys_props.get("mu")

        if self.rho is None or self.mu is None:
            raise ValueError(
                "Physical properties 'rho' and 'mu' must be provided in phys_props."
            )

    @abstractmethod
    def solve(
        self, p_star_field: np.ndarray, u_old_iter: np.ndarray, v_old_iter: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assembles and solves the momentum equation for one component.

        This method must be implemented by subclasses to define the specific discretization
        and solution process for a momentum component (e.g., U or V).

        Parameters:
        - p_star_field: Pressure field from the previous SIMPLE iteration (p_k), used for gradient calculation.
        - u_old_iter: U-velocity field from the end of the previous SIMPLE iteration (u_k).
        - v_old_iter: V-velocity field from the end of the previous SIMPLE iteration (v_k).

        Returns:
            A tuple containing:
            - provisional_velocity_component (np.ndarray): The unrelaxed solution for the velocity component
                                                          being solved (e.g., u_provisional or v_provisional).
            - Ap_coefficients (np.ndarray): Diagonal coefficients from the unrelaxed matrix assembly (Ap_cell).
        """
        pass


# Placeholder for SolverInterface, if this file was meant for more interfaces.
# class SolverInterface(ABC):
#     @abstractmethod
#     def solve(self):
#         pass
