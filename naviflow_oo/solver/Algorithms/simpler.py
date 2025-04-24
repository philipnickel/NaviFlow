"""
SIMPLER (SIMPLE-Revised) algorithm implementation – **gauge-fixed**.

Author: ChatGPT (corrected 2025-04-23)
"""

from __future__ import annotations

import os
import numpy as np

from .base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult
from ...postprocessing.validation.cavity_flow import (
    calculate_infinity_norm_error,
    calculate_l2_norm_error,
)

SMALL = 1.0e-30  # one reusable tiny number


class SimplerSolver(BaseAlgorithm):
    """
    Patankar’s SIMPLER algorithm.

    Outer loop:
        1. Momentum prediction with previous pressure  →  u*, v*
        2. Pressure Poisson (from u*, v*)              →  p̄
        3. Momentum re-solve with p̄                   →  new u*, v*
        4. Pressure-correction equation                →  p′
        5. p ← p̄ + α_p·p′   (remove mean ⇒ fixed gauge)
        6. Velocity correction with p′
    """

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        mesh,
        fluid,
        pressure_solver=None,
        momentum_solver=None,
        velocity_updater=None,
        boundary_conditions=None,
        *,
        alpha_p: float = 0.3,
        alpha_u: float = 0.7,
    ):
        super().__init__(
            mesh,
            fluid,
            pressure_solver,
            momentum_solver,
            velocity_updater,
            boundary_conditions,
        )

        self.alpha_p = alpha_p
        self.alpha_u = alpha_u

        # ­histories ----------------------------------------------------
        self.u_rel_hist: list[float] = []
        self.v_rel_hist: list[float] = []
        self.p_rel_hist: list[float] = []
        self.outer_hist: list[float] = []
        self.infinity_hist: list[float] = []

        # fields for final plotting
        self._final_u_res_field = None
        self._final_v_res_field = None
        self._final_p_res_field = None

    # ------------------------------------------------------------------ #
    # solver                                                             #
    # ------------------------------------------------------------------ #
    def solve(
        self,
        *,
        max_iterations: int = 1000,
        tolerance: float = 1.0e-6,
        save_profile: bool = True,
        profile_dir: str = "results/profiles",
        track_infinity_norm: bool = False,
        infinity_norm_interval: int = 10,
        use_l2_norm: bool = False,
    ) -> SimulationResult:

        self.profiler.start()
        nx, ny = self.mesh.get_dimensions()
        n_cells = nx * ny

        print(f"Using α_p = {self.alpha_p}, α_u = {self.alpha_u}")

        # ----------------------------------------------------------------
        iteration = 1
        outer_residual = 1.0  # initialise larger than tol

        while iteration <= max_iterations and outer_residual > tolerance:
            # -- store previous step ----------------------------------
            u_old = self.u.copy()
            v_old = self.v.copy()
            p_old = self.p.copy()

            # === 1. momentum prediction (old p) =====================
            u_star, d_u, u_info = self.momentum_solver.solve_u_momentum(
                self.mesh,
                self.fluid,
                self.u,
                self.v,
                self.p,
                relaxation_factor=self.alpha_u,
                boundary_conditions=self.bc_manager,
                return_dict=True,
            )
            v_star, d_v, v_info = self.momentum_solver.solve_v_momentum(
                self.mesh,
                self.fluid,
                self.u,
                self.v,
                self.p,
                relaxation_factor=self.alpha_u,
                boundary_conditions=self.bc_manager,
                return_dict=True,
            )

            # === 2. intermediate pressure p̄ ========================
            p_bar, _ = self.pressure_solver.solve(
                self.mesh, u_star, v_star, d_u, d_v, self.p
            )
            self.p += p_bar
            self._enforce_pressure_boundary_conditions()

            # === 3. momentum with p̄ =================================
            u_star, d_u, _ = self.momentum_solver.solve_u_momentum(
                self.mesh,
                self.fluid,
                self.u,
                self.v,
                self.p,
                relaxation_factor=self.alpha_u,
                boundary_conditions=self.bc_manager,
                return_dict=True,
            )
            v_star, d_v, _ = self.momentum_solver.solve_v_momentum(
                self.mesh,
                self.fluid,
                self.u,
                self.v,
                self.p,
                relaxation_factor=self.alpha_u,
                boundary_conditions=self.bc_manager,
                return_dict=True,
            )

            # === 4. correction pressure p′ ==========================
            p_prime, p_info = self.pressure_solver.solve(
                self.mesh, u_star, v_star, d_u, d_v, self.p
            )

            # === 5. final pressure & velocity =======================
            self.p += self.alpha_p * p_prime
            self._enforce_pressure_boundary_conditions()

            self.u, self.v = self.velocity_updater.update_velocity(
                self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
            )

            # -- residuals ------------------------------------------
            u_rel = u_info["rel_norm"]
            v_rel = v_info["rel_norm"]
            p_rel = np.linalg.norm(self.p - p_old) / (np.sqrt(n_cells) + SMALL)
            outer_residual = max(u_rel, v_rel)

            # -- histories & fields ---------------------------------
            self.u_rel_hist.append(u_rel)
            self.v_rel_hist.append(v_rel)
            self.p_rel_hist.append(p_rel)
            self.outer_hist.append(outer_residual)

            self._final_u_res_field = u_info["field"]
            self._final_v_res_field = v_info["field"]
            self._final_p_res_field = p_info["field"]

            # -- optional Ghia error --------------------------------
            if track_infinity_norm and (
                iteration % infinity_norm_interval == 0
                or outer_residual < tolerance
            ):
                try:
                    inf_err = calculate_infinity_norm_error(
                        self.u, self.v, self.mesh, self.fluid.get_reynolds_number()
                    )
                    l2_err = calculate_l2_norm_error(
                        self.u, self.v, self.mesh, self.fluid.get_reynolds_number()
                    )
                    self.infinity_hist.append(l2_err if use_l2_norm else inf_err)
                    print(
                        f"Iter {iteration:4d}  ∞-norm err = {inf_err:.3e}  "
                        f"L2 err = {l2_err:.3e}"
                    )
                except Exception as exc:
                    print(f"Iter {iteration}: could not compute Ghia error – {exc}")

            # -- console line ---------------------------------------
            print(
                f"[{iteration:4d}]  "
                f"u-rel {u_rel:.3e}  v-rel {v_rel:.3e}  p-rel {p_rel:.3e}"
            )

            iteration += 1

        # ----------------------------------------------------------------
        # profiling & result                                             #
        # ----------------------------------------------------------------
        self.profiler.set_iterations(iteration - 1)
        self.profiler.set_convergence_info(
            tolerance=tolerance,
            final_residual=outer_residual,
            residual_history=self.outer_hist,
            converged=outer_residual < tolerance,
        )
        if hasattr(self.pressure_solver, "get_solver_info"):
            info = self.pressure_solver.get_solver_info()
            self.profiler.set_pressure_solver_info(
                solver_name=info.get("name", "unknown"),
                inner_iterations=info.get("inner_iterations_history"),
                convergence_rate=info.get("convergence_rate"),
                solver_specific=info.get("solver_specific"),
            )
        self.profiler.end()

        # -- assemble SimulationResult -----------------------------------
        result = SimulationResult(
            self.u,
            self.v,
            self.p,
            self.mesh,
            iterations=iteration - 1,
            residuals=self.outer_hist,
            reynolds=self.fluid.get_reynolds_number(),
            u_residual_field=self._final_u_res_field,
            v_residual_field=self._final_v_res_field,
            p_residual_field=self._final_p_res_field,
        )
        result.add_history("u_rel_norm", self.u_rel_hist)
        result.add_history("v_rel_norm", self.v_rel_hist)
        result.add_history("p_rel_norm", self.p_rel_hist)
        result.add_history("total_rel_norm", self.outer_hist)

        # -- optional profile on disk ------------------------------------
        if save_profile:
            os.makedirs(profile_dir, exist_ok=True)
            fname = os.path.join(
                profile_dir,
                f"SIMPLER_Re{int(self.fluid.get_reynolds_number())}"
                f"_mesh{nx}x{ny}_profile.h5",
            )
            print(f"Saved profile to {self.save_profiling_data(fname)}")

        return result
