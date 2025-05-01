import numpy as np
import os
import matplotlib.pyplot as plt
from .base_algorithm import BaseAlgorithm
from ...postprocessing.simulation_result import SimulationResult
from ...postprocessing.validation.cavity_flow import calculate_infinity_norm_error, calculate_l2_norm_error
from ...postprocessing.visualization import plot_final_residuals

class PisoSolver(BaseAlgorithm):
    """
    PISO (Pressure Implicit with Splitting of Operators) algorithm implementation.

    The PISO (Pressure Implicit with Splitting of Operators) algorithm
    is a widely used method for solving the Navier-Stokes equations for incompressible flows.
    It uses a predictor-corrector approach with multiple pressure corrections per iteration.
    """
    def __init__(self, mesh, fluid, pressure_solver=None, momentum_solver=None, 
                 velocity_updater=None, boundary_conditions=None, 
                 alpha_p=0.3, alpha_u=0.7, fix_lid_corners=False, n_corrections=2):
        self.alpha_p = alpha_p
        self.alpha_u = alpha_u
        self.fix_lid_corners = fix_lid_corners
        self.n_corrections = n_corrections
        self.u_old = None
        self.v_old = None
        self.p_old = None

        self.x_momentum_rel_norms = []
        self.y_momentum_rel_norms = []
        self.pressure_rel_norms = []
        self.residual_history = []
        self.infinity_norm_history = []

        self._final_u_residual_field = None
        self._final_v_residual_field = None
        self._final_p_residual_field = None

        super().__init__(mesh, fluid, pressure_solver, momentum_solver, 
                         velocity_updater, boundary_conditions)

    def solve(self, max_iterations=1000, tolerance=1e-6, save_profile=True, profile_dir='results/profiles', 
              track_infinity_norm=False, infinity_norm_interval=10, use_l2_norm=False):
        self.profiler.start()
        nx, ny = self.mesh.get_dimensions()
        p_star = self.p.copy()
        p_prime = np.zeros((nx, ny))

        iteration = 1
        u_rel_norm = v_rel_norm = p_rel_norm = 1.0
        total_res_check = max(u_rel_norm, v_rel_norm, p_rel_norm)

        print(f"Using α_p = {self.alpha_p}, α_u = {self.alpha_u}")

        while iteration <= max_iterations and total_res_check > tolerance:
            self.u_old = self.u.copy()
            self.v_old = self.v.copy()
            self.p_old = self.p.copy()

            u_star, d_u, u_res_info = self.momentum_solver.solve_u_momentum(
                self.mesh, self.fluid, self.u, self.v, p_star,
                relaxation_factor=self.alpha_u,
                boundary_conditions=self.bc_manager,
                return_dict=True
            )

            v_star, d_v, v_res_info = self.momentum_solver.solve_v_momentum(
                self.mesh, self.fluid, self.u, self.v, p_star,
                relaxation_factor=self.alpha_u,
                boundary_conditions=self.bc_manager,
                return_dict=True
            )

            for correction in range(self.n_corrections):
                p_prime, p_res_info = self.pressure_solver.solve(
                    self.mesh, u_star, v_star, d_u, d_v, p_star,
                    return_dict=True
                )
                
                self.p = p_star + self.alpha_p * p_prime
                #self.p = p_prime + self.p
                self._enforce_pressure_boundary_conditions()
                p_star = self.p.copy()

                self.u, self.v = self.velocity_updater.update_velocity(
                    self.mesh, u_star, v_star, p_prime, d_u, d_v, self.bc_manager
                )
                u_star = self.u.copy()
                v_star = self.v.copy()

                if correction < self.n_corrections - 1:
                    # Recompute new u*, v*, and coefficients for the updated pressure
                    u_star, d_u, _ = self.momentum_solver.solve_u_momentum(
                        self.mesh, self.fluid, self.u, self.v, self.p,
                        relaxation_factor=1,  # no relaxation in corrections
                        boundary_conditions=self.bc_manager,
                        return_dict=True
                    )
                    v_star, d_v, _ = self.momentum_solver.solve_v_momentum(
                        self.mesh, self.fluid, self.u, self.v, self.p,
                        relaxation_factor=1,
                        boundary_conditions=self.bc_manager,
                        return_dict=True
                    )


            u_rel_norm = u_res_info['rel_norm']
            v_rel_norm = v_res_info['rel_norm']
            p_rel_norm = p_res_info['rel_norm']

            self._final_u_residual_field = u_res_info['field']
            self._final_v_residual_field = v_res_info['field']
            self._final_p_residual_field = p_res_info['field']

            self.x_momentum_rel_norms.append(u_rel_norm)
            self.y_momentum_rel_norms.append(v_rel_norm)
            self.pressure_rel_norms.append(p_rel_norm)

            total_res_check = max(u_rel_norm, v_rel_norm)
            self.residual_history.append(total_res_check)

            if track_infinity_norm and (iteration % infinity_norm_interval == 0 or total_res_check < tolerance):
                try:
                    inf_err = calculate_infinity_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                    l2_err = calculate_l2_norm_error(self.u, self.v, self.mesh, self.fluid.get_reynolds_number())
                    self.infinity_norm_history.append(l2_err if use_l2_norm else inf_err)
                    print(f"Iteration {iteration}: ∞-norm error = {inf_err:.3e}, L2 error = {l2_err:.3e}")
                except Exception as e:
                    print(f"Error calc failed: {e}")

            print(f"[{iteration}] Relative L2 norms: u: {u_rel_norm:.3e}, "
                  f"v: {v_rel_norm:.3e}, p: {p_rel_norm:.3e}")
            iteration += 1

        self.profiler.set_iterations(iteration - 1)
        self.profiler.set_convergence_info(
            tolerance=tolerance,
            final_residual=total_res_check,
            residual_history=self.residual_history,
            converged=(total_res_check < tolerance)
        )

        if hasattr(self.pressure_solver, 'get_solver_info'):
            info = self.pressure_solver.get_solver_info()
            self.profiler.set_pressure_solver_info(
                solver_name=info.get('name', 'unknown'),
                inner_iterations=info.get('inner_iterations_history'),
                convergence_rate=info.get('convergence_rate'),
                solver_specific=info.get('solver_specific')
            )

        self.profiler.end()

        result = SimulationResult(
            self.u, self.v, self.p, self.mesh,
            iterations=iteration-1,
            residuals=self.residual_history,
            reynolds=self.fluid.get_reynolds_number(),
            u_residual_field=self._final_u_residual_field,
            v_residual_field=self._final_v_residual_field,
            p_residual_field=self._final_p_residual_field
        )

        result.add_history('u_rel_norm', self.x_momentum_rel_norms)
        result.add_history('v_rel_norm', self.y_momentum_rel_norms)
        result.add_history('p_rel_norm', self.pressure_rel_norms)
        result.add_history('total_rel_norm', self.residual_history)

        self.profiler.start_section()
        if save_profile:
            os.makedirs(profile_dir, exist_ok=True)
            filename = os.path.join(profile_dir, f"PISO_Re{int(self.fluid.get_reynolds_number())}_mesh{nx}x{ny}_profile.h5")
            print(f"Saved profile to {self.save_profiling_data(filename)}")
        self.profiler.end_section("Finalization")

        return result
