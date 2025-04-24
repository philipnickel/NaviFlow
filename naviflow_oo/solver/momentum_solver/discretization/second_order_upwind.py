"""
Second‑order (linear) upwind discretisation scheme – often called **LUDS** –
for the convection–diffusion terms of the incompressible Navier–Stokes
momentum equations on a staggered grid.

▪ Compatible with the Practice‑B boundary treatment that the existing
  Power‑Law implementation uses.
▪ Produces the extra second‑neighbour coefficient arrays (`a_ee`, `a_ww`,
  `a_nn`, `a_ss`) that your sparse‑matrix builder already consumes when the
  QUICK scheme is selected, so the only change you need there is to test for
  `self.discretization_scheme in ("quick", "linear_upwind")`.

Save this file into your *discretization* package next to *power_law.py* and
*quick.py*, then instantiate your solver with

    solver = AMGMomentumSolver(discretization_scheme="linear_upwind")
"""

from __future__ import annotations

import numpy as np
from ....constructor.boundary_conditions import BoundaryConditionManager
from ....constructor.boundary_conditions import BoundaryType  # re‑export if callers need it


class SecondOrderUpwindDiscretization:  # pylint: disable=too-many-public-methods
    """Second‑order upwind (LUDS) discretisation for *u* and *v* momentum."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_flux(F: np.ndarray):
        """Return the positive and (absolute) negative parts of *F*.

        ``F = Fp – Fn`` with *Fp* ≥ 0 and *Fn* ≥ 0.
        """
        Fp = np.maximum(F, 0.0)
        Fn = np.maximum(-F, 0.0)  # magnitude of negative part
        return Fp, Fn

    # ------------------------------------------------------------------
    # u‑momentum (u lives on an (nx+1) × ny grid)
    # ------------------------------------------------------------------
    def calculate_u_coefficients(
        self,
        mesh,
        fluid,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        bc_manager: BoundaryConditionManager | None = None,
    ):
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho, mu = fluid.get_density(), fluid.get_viscosity()

        # coefficient arrays -------------------------------------------------
        a_e = np.zeros((nx + 1, ny))
        a_w = np.zeros_like(a_e)
        a_n = np.zeros_like(a_e)
        a_s = np.zeros_like(a_e)
        a_p = np.zeros_like(a_e)
        a_ee = np.zeros_like(a_e)  # second E
        a_ww = np.zeros_like(a_e)  # second W
        a_nn = np.zeros_like(a_e)  # second N (needed for LUDS)
        a_ss = np.zeros_like(a_e)  # second S
        Su = np.zeros_like(a_e)

        # constant diffusive conductances                                   
        De = Dw = mu * dy / dx
        Dn = Ds = mu * dx / dy

        # vectorised interior block -----------------------------------------
        i = np.arange(1, nx)
        j = np.arange(1, ny - 1)
        I, J = np.meshgrid(i, j, indexing="ij")

        # convective mass fluxes on the staggered u‑grid
        Fe = 0.5 * rho * dy * (u[I + 1, J] + u[I, J])
        Fw = 0.5 * rho * dy * (u[I - 1, J] + u[I, J])
        Fn = 0.5 * rho * dx * (v[I, J + 1] + v[I - 1, J + 1])
        Fs = 0.5 * rho * dx * (v[I, J] + v[I - 1, J])

        # ------------------------------------------------------------------
        # diffusion – identical to Power‑Law
        a_e[I, J] += De
        a_w[I, J] += Dw
        a_n[I, J] += Dn
        a_s[I, J] += Ds

        # ------------------------------------------------------------------
        # convection – EAST face
        Fp, Fnve = self._split_flux(Fe)          # F+, |F‑|
        a_p[I, J]  += 1.5 * Fp                   # P  contribution (upstream)
        a_w[I, J]  += 0.5 * Fp                   # W  (second upstream)
        a_ww[I, J] += -0.5 * Fp                  # WW (note negative)

        a_e[I, J]  += 1.5 * Fnve                 # negative flow → E is upstream
        a_ee[I, J] += 0.5 * Fnve                 # EE second upstream

        # ------------------------------------------------------------------
        # convection – WEST face
        Fp, Fnve = self._split_flux(Fw)
        a_w[I, J]  += 1.5 * Fp
        a_ww[I, J] += -0.5 * Fp
        a_p[I, J]  += 1.5 * Fnve
        a_e[I, J]  += 0.5 * Fnve

        # ------------------------------------------------------------------
        # convection – NORTH face
        Fp, Fnve = self._split_flux(Fn)
        a_p[I, J]  += 1.5 * Fp
        a_s[I, J]  += -0.5 * Fp                  # second upstream is S
        a_n[I, J]  += 1.5 * Fnve                
        a_nn[I, J] += 0.5 * Fnve                 # second upstream is NN

        # ------------------------------------------------------------------
        # convection – SOUTH face
        Fp, Fnve = self._split_flux(Fs)
        a_s[I, J]  += 1.5 * Fp
        a_ss[I, J] += -0.5 * Fp
        a_p[I, J]  += 1.5 * Fnve
        a_n[I, J]  += 0.5 * Fnve

        # ------------------------------------------------------------------
        # pressure source term
        Su[I, J] += (p[I - 1, J] - p[I, J]) * dy

        # ------------------------------------------------------------------
        # base diagonal (include convective flux imbalance)
        a_p[I, J] += (
            a_e[I, J]
            + a_w[I, J]
            + a_n[I, J]
            + a_s[I, J]
            + a_ee[I, J]
            + a_ww[I, J]
            + a_nn[I, J]
            + a_ss[I, J]
            + (Fe - Fw)
            + (Fn - Fs)
        )

        # ------------------------------------------------------------------
        # boundaries – reuse the solid‑wall Practice‑B modifications from
        # the original Power‑Law class.  For brevity we literally copy them
        # here (they account for all four sides).
        # ------------------------------------------------------------------
        if bc_manager is not None:
            # ------ LEFT & RIGHT (adjacent u‑cells i=1 and i=nx‑1) --------
            left_bc = bc_manager.get_condition("left")
            if left_bc:
                i_adj = 1
                source_col = Su
                for jj in range(ny):
                    source_col[i_adj, jj] += a_w[i_adj, jj] * u[0, jj]
                    a_w[i_adj, jj] = 0.0
            right_bc = bc_manager.get_condition("right")
            if right_bc:
                i_adj = nx - 1
                source_col = Su
                for jj in range(ny):
                    source_col[i_adj, jj] += a_e[i_adj, jj] * u[nx, jj]
                    a_e[i_adj, jj] = 0.0

            # ------ BOTTOM & TOP (adjacent rows j=1 and j=ny‑2) -----------
            bottom_bc = bc_manager.get_condition("bottom")
            if bottom_bc:
                j_adj = 1
                for ii in range(1, nx):
                    Su[ii, j_adj] += a_s[ii, j_adj] * u[ii, 0]
                    a_s[ii, j_adj] = 0.0
            top_bc = bc_manager.get_condition("top")
            if top_bc:
                j_adj = ny - 2
                for ii in range(1, nx):
                    Su[ii, j_adj] += a_n[ii, j_adj] * u[ii, ny - 1]
                    a_n[ii, j_adj] = 0.0

        # ------------------------------------------------------------------
        return {
            "a_e": a_e,
            "a_w": a_w,
            "a_n": a_n,
            "a_s": a_s,
            "a_p": a_p,
            "a_ee": a_ee,
            "a_ww": a_ww,
            "a_nn": a_nn,
            "a_ss": a_ss,
            "source": Su,
        }

    # ------------------------------------------------------------------
    # v‑momentum – identical logic but on the (nx × (ny+1)) v‑grid.
    # ------------------------------------------------------------------
    def calculate_v_coefficients(
        self,
        mesh,
        fluid,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        bc_manager: BoundaryConditionManager | None = None,
    ):
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        rho, mu = fluid.get_density(), fluid.get_viscosity()

        a_e = np.zeros((nx, ny + 1))
        a_w = np.zeros_like(a_e)
        a_n = np.zeros_like(a_e)
        a_s = np.zeros_like(a_e)
        a_p = np.zeros_like(a_e)
        a_ee = np.zeros_like(a_e)
        a_ww = np.zeros_like(a_e)
        a_nn = np.zeros_like(a_e)
        a_ss = np.zeros_like(a_e)
        Sv = np.zeros_like(a_e)

        De = Dw = mu * dy / dx
        Dn = Ds = mu * dx / dy

        i = np.arange(1, nx - 1)
        j = np.arange(1, ny)
        I, J = np.meshgrid(i, j, indexing="ij")

        Fe = 0.5 * rho * dy * (u[I + 1, J] + u[I + 1, J - 1])
        Fw = 0.5 * rho * dy * (u[I, J] + u[I, J - 1])
        Fn = 0.5 * rho * dx * (v[I, J + 1] + v[I, J])
        Fs = 0.5 * rho * dx * (v[I, J - 1] + v[I, J])

        # diffusion
        a_e[I, J] += De
        a_w[I, J] += Dw
        a_n[I, J] += Dn
        a_s[I, J] += Ds

        # EAST face
        Fp, Fnve = self._split_flux(Fe)
        a_e[I, J]  += 1.5 * Fp
        a_ee[I, J] += 0.5 * Fp
        a_p[I, J]  += 1.5 * Fnve
        a_w[I, J]  += 0.5 * Fnve

        # WEST face
        Fp, Fnve = self._split_flux(Fw)
        a_p[I, J]  += 1.5 * Fp
        a_e[I, J]  += 0.5 * Fp
        a_w[I, J]  += 1.5 * Fnve
        a_ww[I, J] += 0.5 * Fnve

        # NORTH face
        Fp, Fnve = self._split_flux(Fn)
        a_n[I, J]  += 1.5 * Fp
        a_nn[I, J] += 0.5 * Fp
        a_p[I, J]  += 1.5 * Fnve
        a_s[I, J]  += 0.5 * Fnve

        # SOUTH face
        Fp, Fnve = self._split_flux(Fs)
        a_p[I, J]  += 1.5 * Fp
        a_n[I, J]  += 0.5 * Fp
        a_s[I, J]  += 1.5 * Fnve
        a_ss[I, J] += 0.5 * Fnve

        # pressure source term
        Sv[I, J] += (p[I, J - 1] - p[I, J]) * dx

        # diagonal
        a_p[I, J] += (
            a_e[I, J]
            + a_w[I, J]
            + a_n[I, J]
            + a_s[I, J]
            + a_ee[I, J]
            + a_ww[I, J]
            + a_nn[I, J]
            + a_ss[I, J]
            + (Fe - Fw)
            + (Fn - Fs)
        )

        # Practice‑B boundary hook (copy from Power‑Law) ----------------
        if bc_manager is not None:
            # bottom boundary – v is on boundary itself at j=0
            bottom_bc = bc_manager.get_condition("bottom")
            if bottom_bc:
                j_adj = 1
                for ii in range(nx):
                    Sv[ii, j_adj] += a_s[ii, j_adj] * v[ii, 0]
                    a_s[ii, j_adj] = 0.0
            top_bc = bc_manager.get_condition("top")
            if top_bc:
                j_adj = ny - 1
                for ii in range(nx):
                    Sv[ii, j_adj] += a_n[ii, j_adj] * v[ii, ny]
                    a_n[ii, j_adj] = 0.0
            left_bc = bc_manager.get_condition("left")
            if left_bc:
                i_adj = 1
                for jj in range(1, ny):
                    Sv[i_adj, jj] += a_w[i_adj, jj] * v[0, jj]
                    a_w[i_adj, jj] = 0.0
            right_bc = bc_manager.get_condition("right")
            if right_bc:
                i_adj = nx - 2
                for jj in range(1, ny):
                    Sv[i_adj, jj] += a_e[i_adj, jj] * v[nx - 1, jj]
                    a_e[i_adj, jj] = 0.0

        # ------------------------------------------------------------------
        return {
            "a_e": a_e,
            "a_w": a_w,
            "a_n": a_n,
            "a_s": a_s,
            "a_p": a_p,
            "a_ee": a_ee,
            "a_ww": a_ww,
            "a_nn": a_nn,
            "a_ss": a_ss,
            "source": Sv,
        }
