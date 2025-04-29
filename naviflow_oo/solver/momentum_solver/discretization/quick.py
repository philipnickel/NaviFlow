"""
Fully–implicit QUICK discretisation for staggered u- and v-momentum
equations (2-D, uniform grid).

 ─────────────────────────────────────────────────────────────────────
  • True QUICK interpolation is placed directly into the matrix
    → second-neighbour coefficients a_ee, a_ww, a_nn, a_ss
  • First cell next to every wall falls back to 1st-order up-wind
    (no extended stencil available there).
  • Boundary values handled with Practice-B (move into source term).
  • Returns the nine coefficient arrays that the extended AMG builder
    understands.
 ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
from ....constructor.boundary_conditions import BoundaryConditionManager
# from .base_discretization import MomentumDiscretization # Removed import
# import scipy.sparse as sparse # Removed import


# ─────────── helpers ────────────
def _c_pos(F): return np.maximum( F, 0.0)        # positive part
def _c_neg(F): return np.maximum(-F, 0.0)         # negative part


# ─────────── discretisation class ────────────
# class QuickDiscretization(MomentumDiscretization): # Reverted class definition
class QUICKDiscretization:
    # """Implements the QUICK discretization scheme.""" # Removed docstring

    # Removed the discretize method added previously

    # =======================  u-momentum  ========================= #
    def calculate_u_coefficients(self, mesh, fluid, u, v, p,
                                 bc: BoundaryConditionManager | None = None):

        nx, ny         = mesh.get_dimensions()
        dx, dy         = mesh.get_cell_sizes()
        rho, mu        = fluid.get_density(), fluid.get_viscosity()

        shp_u = (nx+1, ny)                   # staggered u grid

        # allocate 2-nd–neighbour arrays as well
        a_e  = np.zeros(shp_u); a_w  = np.zeros_like(a_e)
        a_n  = np.zeros_like(a_e); a_s  = np.zeros_like(a_e)
        a_ee = np.zeros_like(a_e); a_ww = np.zeros_like(a_e)
        a_nn = np.zeros_like(a_e); a_ss = np.zeros_like(a_e)
        a_p  = np.zeros_like(a_e); Su   = np.zeros_like(a_e)

        De, Dn = mu*dy/dx, mu*dx/dy                       # diffusion

        # index arrays
        I = np.arange(1, nx)          # 1 … nx-1
        J = np.arange(1, ny-1)        # 1 … ny-2
        iG, jG = np.meshgrid(I, J, indexing='ij')

        # mass fluxes ------------------------------------------------
        Fe = 0.5*rho*dy*(u[iG+1, jG] + u[iG, jG])
        Fw = 0.5*rho*dy*(u[iG-1, jG] + u[iG, jG])
        Fn = 0.5*rho*dx*(v[iG,   jG+1] + v[iG-1, jG+1])
        Fs = 0.5*rho*dx*(v[iG,   jG  ] + v[iG-1, jG  ])

        # ========= east / west QUICK coefficients =========
        # cells that have both E and EE neighbours ( i = 1 … nx-2 )
        mask_EE = (iG <= nx-2)
        if np.any(mask_EE):
            ie, je = iG[mask_EE], jG[mask_EE]

            # --- east face (φ_e) ---
            #  positive Fe → up-wind is P
            a_e [ie, je] += (+6./8.)*_c_pos(Fe[mask_EE]) + De
            a_p [ie, je] += (+3./8.)*_c_pos(Fe[mask_EE])
            a_ee[ie, je] += (-1./8.)*_c_pos(Fe[mask_EE])
            #  negative Fe → up-wind is E
            a_e [ie, je] += (+3./8.)*_c_neg(Fe[mask_EE]) + De
            a_p [ie, je] += (+6./8.)*_c_neg(Fe[mask_EE])
            a_w [ie, je] += (-1./8.)*_c_neg(Fe[mask_EE])  # actually W but we
                                                          # store in a_w of P

        # west face (φ_w) – needs WW : i = 2 … nx-1
        mask_WW = (iG >= 2)
        if np.any(mask_WW):
            iw, jw = iG[mask_WW], jG[mask_WW]
            # positive Fw → up-wind = W
            a_w [iw, jw] += (+6./8.)*_c_pos(Fw[mask_WW]) + De
            a_p [iw, jw] += (+3./8.)*_c_pos(Fw[mask_WW])
            a_ww[iw, jw] += (-1./8.)*_c_pos(Fw[mask_WW])
            # negative Fw → up-wind = P
            a_w [iw, jw] += (+3./8.)*_c_neg(Fw[mask_WW]) + De
            a_p [iw, jw] += (+6./8.)*_c_neg(Fw[mask_WW])
            a_e [iw, jw] += (-1./8.)*_c_neg(Fw[mask_WW])

        # ========= north / south QUICK coefficients =========
        mask_NN = (jG <= ny-3)      # needs NN
        if np.any(mask_NN):
            in_, jn = iG[mask_NN], jG[mask_NN]
            a_n [in_, jn] += (+6./8.)*_c_pos(Fn[mask_NN]) + Dn
            a_p [in_, jn] += (+3./8.)*_c_pos(Fn[mask_NN])
            a_nn[in_, jn] += (-1./8.)*_c_pos(Fn[mask_NN])

            a_n [in_, jn] += (+3./8.)*_c_neg(Fn[mask_NN]) + Dn
            a_p [in_, jn] += (+6./8.)*_c_neg(Fn[mask_NN])
            a_s [in_, jn] += (-1./8.)*_c_neg(Fn[mask_NN])

        mask_SS = (jG >= 2)         # needs SS
        if np.any(mask_SS):
            is_, js = iG[mask_SS], jG[mask_SS]
            a_s [is_, js] += (+6./8.)*_c_pos(Fs[mask_SS]) + Dn
            a_p [is_, js] += (+3./8.)*_c_pos(Fs[mask_SS])
            a_ss[is_, js] += (-1./8.)*_c_pos(Fs[mask_SS])

            a_s [is_, js] += (+3./8.)*_c_neg(Fs[mask_SS]) + Dn
            a_p [is_, js] += (+6./8.)*_c_neg(Fs[mask_SS])
            a_n [is_, js] += (-1./8.)*_c_neg(Fs[mask_SS])

        # pressure gradient ------------------------------------------------
        Su[iG, jG] += (p[iG-1, jG] - p[iG, jG])*dy

        # ================= boundary handling =================
        if bc is not None:
            self._practiceB_u(a_e, a_w, a_n, a_s, a_p, Su, u, nx, ny, bc)

        return dict(a_e=a_e, a_w=a_w, a_n=a_n, a_s=a_s,
                    a_ee=a_ee, a_ww=a_ww, a_nn=a_nn, a_ss=a_ss,
                    a_p=a_p,  source=Su)

    # =======================  v-momentum  ========================= #
    def calculate_v_coefficients(self, mesh, fluid, u, v, p,
                                 bc: BoundaryConditionManager | None = None):

        nx, ny        = mesh.get_dimensions()
        dx, dy        = mesh.get_cell_sizes()
        rho, mu       = fluid.get_density(), fluid.get_viscosity()

        shp_v = (nx, ny+1)

        a_e  = np.zeros(shp_v); a_w  = np.zeros_like(a_e)
        a_n  = np.zeros_like(a_e);   a_s  = np.zeros_like(a_e)
        a_ee = np.zeros_like(a_e);   a_ww = np.zeros_like(a_e)
        a_nn = np.zeros_like(a_e);   a_ss = np.zeros_like(a_e)
        a_p  = np.zeros_like(a_e);   Sv  = np.zeros_like(a_e)

        De, Dn = mu*dy/dx, mu*dx/dy
        I = np.arange(1, nx-1); J = np.arange(1, ny)
        iG, jG = np.meshgrid(I, J, indexing='ij')

        Fe = 0.5*rho*dy*(u[iG+1, jG] + u[iG+1, jG-1])
        Fw = 0.5*rho*dy*(u[iG,   jG] + u[iG,   jG-1])
        Fn = 0.5*rho*dx*(v[iG, jG+1] + v[iG, jG])
        Fs = 0.5*rho*dx*(v[iG, jG]   + v[iG, jG-1])

        # east / west QUICK (needs EE, WW)
        mask_EE = (iG <= nx-3)
        mask_WW = (iG >= 2)
        if np.any(mask_EE):
            ie, je = iG[mask_EE], jG[mask_EE]
            a_e [ie, je] += (+6./8.)*_c_pos(Fe[mask_EE]) + De
            a_p [ie, je] += (+3./8.)*_c_pos(Fe[mask_EE])
            a_ee[ie, je] += (-1./8.)*_c_pos(Fe[mask_EE])
            a_e [ie, je] += (+3./8.)*_c_neg(Fe[mask_EE]) + De
            a_p [ie, je] += (+6./8.)*_c_neg(Fe[mask_EE])
            a_w [ie, je] += (-1./8.)*_c_neg(Fe[mask_EE])

        if np.any(mask_WW):
            iw, jw = iG[mask_WW], jG[mask_WW]
            a_w [iw, jw] += (+6./8.)*_c_pos(Fw[mask_WW]) + De
            a_p [iw, jw] += (+3./8.)*_c_pos(Fw[mask_WW])
            a_ww[iw, jw] += (-1./8.)*_c_pos(Fw[mask_WW])
            a_w [iw, jw] += (+3./8.)*_c_neg(Fw[mask_WW]) + De
            a_p [iw, jw] += (+6./8.)*_c_neg(Fw[mask_WW])
            a_e [iw, jw] += (-1./8.)*_c_neg(Fw[mask_WW])

        # north / south QUICK (needs NN, SS)
        mask_NN = (jG <= ny-2)
        mask_SS = (jG >= 2)
        if np.any(mask_NN):
            in_, jn = iG[mask_NN], jG[mask_NN]
            a_n [in_, jn] += (+6./8.)*_c_pos(Fn[mask_NN]) + Dn
            a_p [in_, jn] += (+3./8.)*_c_pos(Fn[mask_NN])
            a_nn[in_, jn] += (-1./8.)*_c_pos(Fn[mask_NN])
            a_n [in_, jn] += (+3./8.)*_c_neg(Fn[mask_NN]) + Dn
            a_p [in_, jn] += (+6./8.)*_c_neg(Fn[mask_NN])
            a_s [in_, jn] += (-1./8.)*_c_neg(Fn[mask_NN])

        if np.any(mask_SS):
            is_, js = iG[mask_SS], jG[mask_SS]
            a_s [is_, js] += (+6./8.)*_c_pos(Fs[mask_SS]) + Dn
            a_p [is_, js] += (+3./8.)*_c_pos(Fs[mask_SS])
            a_ss[is_, js] += (-1./8.)*_c_pos(Fs[mask_SS])
            a_s [is_, js] += (+3./8.)*_c_neg(Fs[mask_SS]) + Dn
            a_p [is_, js] += (+6./8.)*_c_neg(Fs[mask_SS])
            a_n [is_, js] += (-1./8.)*_c_neg(Fs[mask_SS])

        Sv[iG, jG] += (p[iG, jG-1] - p[iG, jG])*dx

        if bc is not None:
            self._practiceB_v(a_e,a_w,a_n,a_s,a_p,Sv,v,nx,ny,bc)

        return dict(a_e=a_e, a_w=a_w, a_n=a_n, a_s=a_s,
                    a_ee=a_ee, a_ww=a_ww, a_nn=a_nn, a_ss=a_ss,
                    a_p=a_p,  source=Sv)

    # ---------- Practice-B helpers (exactly as in your previous files) ---- #
    @staticmethod
    def _practiceB_u(a_e,a_w,a_n,a_s,a_p,Su,u,nx,ny,bc):
        if bc.get_condition("left"):
            for j in range(ny):   Su[1, j]   += a_w[1, j]*u[0, j];   a_w[1, j]=0
        if bc.get_condition("right"):
            for j in range(ny):   Su[nx-1,j]+= a_e[nx-1,j]*u[nx,j];  a_e[nx-1,j]=0
        if bc.get_condition("bottom"):
            for i in range(1,nx): Su[i,1]    += a_s[i,1]*u[i,0];     a_s[i,1]=0
        if bc.get_condition("top"):
            for i in range(1,nx): Su[i,ny-2]+= a_n[i,ny-2]*u[i,ny-1];a_n[i,ny-2]=0

    @staticmethod
    def _practiceB_v(a_e,a_w,a_n,a_s,a_p,Sv,v,nx,ny,bc):
        if bc.get_condition("bottom"):
            for i in range(nx):   Sv[i,1]     += a_s[i,1]*v[i,0];     a_s[i,1]=0
        if bc.get_condition("top"):
            for i in range(nx):   Sv[i,ny-1]  += a_n[i,ny-1]*v[i,ny]; a_n[i,ny-1]=0
        if bc.get_condition("left"):
            for j in range(1,ny): Sv[1,j]     += a_w[1,j]*v[0,j];     a_w[1,j]=0
        if bc.get_condition("right"):
            for j in range(1,ny): Sv[nx-2,j]  += a_e[nx-2,j]*v[nx-1,j];a_e[nx-2,j]=0
