import numpy as np
from .base_momentum_solver import MomentumSolver
from ...constructor.boundary_conditions import BoundaryConditionManager

def van_leer_limiter(r):
    """
    Van Leer flux limiter function.
    r can be a scalar or ndarray.
    """
    return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-20)

def compute_slope_ratio(phi_left, phi_center, phi_right):
    """
    Compute the slope ratio for the flux limiter:
       r = (phi_center - phi_left) / (phi_right - phi_center)

    We add a small epsilon to avoid division by zero.
    """
    numerator = (phi_center - phi_left)
    denominator = (phi_right - phi_center) + 1e-20
    return numerator / denominator

def compute_tvd_face_value(phi_left, phi_center, phi_right, limiter_func):
    """
    Given three neighboring values (left, center, right),
    compute a limited face value between 'center' and 'right'.

    'phi_left' is the cell upstream of 'center' (for slope calculation).
    """
    r = compute_slope_ratio(phi_left, phi_center, phi_right)
    psi = limiter_func(r)
    return phi_center + 0.5 * psi * (phi_right - phi_center)

class TVDMomentumSolver(MomentumSolver):
    """
    Momentum solver using a TVD (flux-limiter) scheme for convection.

    This replaces the power-law logic by explicitly computing face-centered
    fluxes with a flux-limiter approach.
    """

    def solve_u_momentum(self, mesh, fluid, u, v, p,
                         relaxation_factor=0.7,
                         boundary_conditions=None,
                         limiter_func=van_leer_limiter):
        """
        Solve the u-momentum equation (dimensions: u.shape = (imax+1, jmax)).
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()

        rho = fluid.get_density()
        mu = fluid.get_viscosity()

        imax, jmax = nx, ny   # so u has shape (imax+1, jmax)
        alpha = relaxation_factor

        u_star = np.zeros((imax+1, jmax))
        d_u    = np.zeros((imax+1, jmax))

        # Diffusion terms
        De = mu * dy / dx
        Dw = mu * dy / dx
        Dn = mu * dx / dy
        Ds = mu * dx / dy

        # ----------------------------------------------------
        # 1) Horizontal face fluxes for u (Fe_array)
        #    We'll do i in [1 .. imax-1], j in [0 .. jmax-1].
        # ----------------------------------------------------
        Fe_array = np.zeros((imax+1, jmax))
        for j in range(jmax):  # jmax is valid up to jmax-1
            for i in range(1, imax):  # so i+1 goes up to imax, in-bounds for u
                u_face = 0.5 * (u[i, j] + u[i+1, j])
                if u_face >= 0.0:
                    # Upwind = cell i
                    phi_center = u[i, j]
                    phi_right  = u[i+1, j]
                    # For slope ratio, we need i-1 as 'left' if valid
                    if i-1 >= 0:
                        phi_left = u[i-1, j]
                    else:
                        phi_left = phi_center
                else:
                    # Upwind = cell i+1
                    phi_center = u[i+1, j]
                    phi_right  = u[i, j]
                    # 'left' is i+2 if valid
                    if (i+2) <= imax:
                        phi_left = u[i+2, j]
                    else:
                        phi_left = phi_center

                face_phi = compute_tvd_face_value(phi_left, phi_center, phi_right, limiter_func)
                Fe_array[i, j] = rho * u_face * dy * face_phi

        # For west flux (Fw), we can just shift indices or store explicitly:
        # Fw_array[i,j] = Fe_array[i-1, j], if i-1 >= 0. We'll do direct access later.

        # ----------------------------------------------------
        # 2) Vertical face fluxes for u (Fn_array)
        #    We'll do i in [1 .. imax-1], j in [1 .. jmax-2]
        #    so that j+1 is valid. We fallback if j+2 is out-of-bounds.
        # ----------------------------------------------------
        Fn_array = np.zeros((imax+1, jmax+1))
        for i in range(1, imax):          
            for j in range(1, jmax-1):    
                # v_face is the vertical velocity between (i,j) and (i-1,j) for u's control volume:
                # But your existing code might do something like:
                v_face = 0.5 * (v[i, j] + v[i-1, j])  # check if i-1 is valid
                if v_face >= 0.0:
                    phi_center = u[i, j]
                    phi_right  = u[i, j+1]
                    if (j-1) >= 0:
                        phi_left = u[i, j-1]
                    else:
                        phi_left = phi_center
                else:
                    # upwind is cell j+1
                    phi_center = u[i, j+1]
                    phi_right  = u[i, j]
                    if (j+2) < jmax:
                        phi_left = u[i, j+2]
                    else:
                        phi_left = phi_center

                face_phi = compute_tvd_face_value(phi_left, phi_center, phi_right, limiter_func)
                Fn_array[i, j] = rho * v_face * dx * face_phi

        # We'll define Fs = Fn_array[i, j-1] if valid, else 0
        Fs_array = np.zeros_like(Fn_array)
        for i in range(1, imax):
            for j in range(1, jmax):
                Fs_array[i, j] = Fn_array[i, j-1]

        # ----------------------------------------------------
        # 3) Assemble aE, aW, aN, aS, aP
        #    We'll loop interior cells i in [1..imax-1], j in [1..jmax-2]
        # ----------------------------------------------------
        for i in range(1, imax):
            for j in range(1, jmax-1):
                Fe = Fe_array[i, j]
                Fw = Fe_array[i-1, j] if i-1 >= 0 else 0.0
                Fn = Fn_array[i, j+1]
                Fs = Fn_array[i, j]

                aE = De + max(-Fe, 0.0)
                aW = Dw + max(Fw, 0.0)
                aN = Dn + max(-Fn, 0.0)
                aS = Ds + max(Fs, 0.0)

                aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

                # Pressure gradient
                # p.shape likely (imax, jmax). For index i in [1..imax-1], p[i,j] is valid.
                # Typically: pressure_term = (p[i-1,j] - p[i,j]) * dy
                pressure_term = (p[i-1, j] - p[i, j]) * dy

                u_star[i, j] = (alpha / aP) * (
                    aE * u[i+1, j] +
                    aW * u[i-1, j] +
                    aN * u[i, j+1] +
                    aS * u[i, j-1] +
                    pressure_term
                ) + (1.0 - alpha)*u[i, j]

                d_u[i, j] = alpha * dy / aP

        # ----------------------------------------------------
        # 4) Handle boundary lines (j=0, j=jmax-1, i=0, i=imax)
        # ----------------------------------------------------
        # For example, if it's a cavity flow with no-slip walls except top:
        # You can do something like:
        for i in range(1, imax):
            # bottom boundary: j=0
            u_star[i, 0] = 0.0
            d_u[i, 0]    = 0.0
            # top boundary: j=jmax-1
            u_star[i, jmax-1] = 0.0
            d_u[i, jmax-1]    = 0.0

        # left & right boundaries (i=0, i=imax)
        for j in range(jmax):
            u_star[0, j] = 0.0
            d_u[0, j]    = 0.0
            u_star[imax, j] = 0.0
            d_u[imax, j]    = 0.0

        # ----------------------------------------------------
        # 5) Apply boundary conditions from BC manager
        # ----------------------------------------------------
        if boundary_conditions:
            if isinstance(boundary_conditions, BoundaryConditionManager):
                bc_manager = boundary_conditions
            else:
                bc_manager = BoundaryConditionManager()
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)

            u_star, _ = bc_manager.apply_velocity_boundary_conditions(u_star, v.copy(), imax, jmax)
        else:
            # default: zero at all boundaries
            pass

        return u_star, d_u

    def solve_v_momentum(self, mesh, fluid, u, v, p,
                         relaxation_factor=0.7,
                         boundary_conditions=None,
                         limiter_func=van_leer_limiter):
        """
        Solve the v-momentum equation (dimensions: v.shape = (imax, jmax+1)).
        """
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()

        rho = fluid.get_density()
        mu = fluid.get_viscosity()

        imax, jmax = nx, ny   # so v has shape (imax, jmax+1)
        alpha = relaxation_factor

        v_star = np.zeros((imax, jmax+1))
        d_v    = np.zeros((imax, jmax+1))

        # Diffusion terms
        De = mu * dy / dx
        Dw = mu * dy / dx
        Dn = mu * dx / dy
        Ds = mu * dx / dy

        # ----------------------------------------------------
        # 1) Horizontal fluxes for v (Fe_array).
        #    Now the "transport velocity" is u, and the advected quantity is v[i,j].
        #    We'll do i in [1..imax-1], j in [1..jmax], etc. 
        #    But watch for out-of-bounds in v when i+1, i+2 are used.
        # ----------------------------------------------------
        Fe_array = np.zeros((imax+1, jmax+1))
        for j in range(1, jmax):
            for i in range(1, imax-1):
                # For v's control volume, face velocity = average of u at [i,j] and [i,j-1], etc.
                # This can vary depending on how your grid is arranged. Example:
                u_face = 0.5 * (u[i, j] + u[i, j-1])
                if u_face >= 0.0:
                    phi_center = v[i, j]
                    if (i+1) < imax:
                        phi_right  = v[i+1, j]
                    else:
                        phi_right  = phi_center
                    phi_left   = v[i-1, j] if (i-1) >= 0 else phi_center
                else:
                    # upwind = i+1
                    if (i+1) < imax:
                        phi_center = v[i+1, j]
                    else:
                        phi_center = v[i, j]
                    phi_right  = v[i, j]
                    if (i+2) < imax:
                        phi_left = v[i+2, j]
                    else:
                        phi_left = phi_center

                face_phi = compute_tvd_face_value(phi_left, phi_center, phi_right, limiter_func)
                Fe_array[i, j] = rho * u_face * dy * face_phi

        # West flux array
        Fw_array = np.zeros_like(Fe_array)
        for j in range(1, jmax):
            for i in range(1, imax):
                Fw_array[i, j] = Fe_array[i-1, j]

        # ----------------------------------------------------
        # 2) Vertical fluxes for v (Fn_array).
        #    This depends on v at faces in the y-direction.
        #    We'll do i in [0..imax-1], j in [1..jmax] so j+1 => jmax+1 is out of range,
        #    but v.shape is (imax, jmax+1), so last valid j = jmax. 
        # ----------------------------------------------------
        Fn_array = np.zeros((imax, jmax+1))
        for i in range(imax):
            for j in range(1, jmax):
                v_face = 0.5 * (v[i, j] + v[i, j+1])  # between j and j+1
                if v_face >= 0.0:
                    phi_center = v[i, j]
                    phi_right  = v[i, j+1]
                    phi_left   = v[i, j-1] if (j-1) >= 0 else phi_center
                else:
                    phi_center = v[i, j+1]
                    phi_right  = v[i, j]
                    if (j+2) <= jmax:
                        phi_left = v[i, j+2]
                    else:
                        phi_left = phi_center

                face_phi = compute_tvd_face_value(phi_left, phi_center, phi_right, limiter_func)
                Fn_array[i, j] = rho * v_face * dx * face_phi

        Fs_array = np.zeros_like(Fn_array)
        for i in range(imax):
            for j in range(1, jmax+1):
                Fs_array[i, j] = Fn_array[i, j-1] if (j-1) >= 0 else 0.0

        # ----------------------------------------------------
        # 3) Assemble coefficients for v
        #    Typically i in [1..imax-2], j in [1..jmax], etc.
        #    Adjust as needed to avoid out-of-bounds.
        # ----------------------------------------------------
        for j in range(1, jmax):
            for i in range(1, imax-1):
                Fe = Fe_array[i, j]
                Fw = Fw_array[i, j]
                Fn = Fn_array[i, j]
                Fs = Fs_array[i, j]

                aE = De + max(-Fe, 0.0)
                aW = Dw + max(Fw, 0.0)
                aN = Dn + max(-Fn, 0.0)
                aS = Ds + max(Fs, 0.0)

                aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

                # Pressure gradient for v:
                # Usually (p[i, j-1] - p[i, j]) * dx if p.shape = (imax+1, jmax) or (imax, jmax).
                # Carefully match indices to your actual p array:
                pressure_term = (p[i, j-1] - p[i, j]) * dx

                v_star[i, j] = (alpha / aP) * (
                    aE * v[i+1, j] +
                    aW * v[i-1, j] +
                    aN * v[i, j+1] +
                    aS * v[i, j-1] +
                    pressure_term
                ) + (1.0 - alpha)*v[i, j]

                d_v[i, j] = alpha * dx / aP

        # ----------------------------------------------------
        # 4) Boundary lines for v
        # ----------------------------------------------------
        for i in range(imax):
            # bottom boundary j=0
            v_star[i, 0] = 0.0
            d_v[i, 0]    = 0.0
            # top boundary j=jmax
            v_star[i, jmax] = 0.0
            d_v[i, jmax]    = 0.0

        for j in range(jmax+1):
            # left boundary i=0
            v_star[0, j] = 0.0
            d_v[0, j]    = 0.0
            # right boundary i=imax-1 or i=imax? depends on your code
            if imax > 1:
                v_star[imax-1, j] = 0.0
                d_v[imax-1, j]    = 0.0

        # ----------------------------------------------------
        # 5) Apply boundary conditions if provided
        # ----------------------------------------------------
        if boundary_conditions:
            if isinstance(boundary_conditions, BoundaryConditionManager):
                bc_manager = boundary_conditions
            else:
                bc_manager = BoundaryConditionManager()
                for boundary, conditions in boundary_conditions.items():
                    for field_type, values in conditions.items():
                        bc_manager.set_condition(boundary, field_type, values)

            # apply BC for velocity
            _, v_star = bc_manager.apply_velocity_boundary_conditions(u.copy(), v_star, imax, jmax)
        else:
            # default: zero at boundaries
            pass

        return v_star, d_v
