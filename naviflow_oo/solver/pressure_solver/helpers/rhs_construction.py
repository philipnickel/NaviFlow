import numpy as np

def get_rhs(imax, jmax, dx, dy, rho, u_star, v_star):
   # right hand side of the pressure correction equation
    # Vectorized implementation
    bp = np.zeros(imax*jmax)
    
    # Create 2D matrix first - easier to work with
    bp_2d = np.zeros((imax, jmax))
    
    # Compute entire array at once
    bp_2d = rho * (u_star[:-1, :] * dy - u_star[1:, :] * dy + 
                   v_star[:, :-1] * dx - v_star[:, 1:] * dx)
    
    # Flatten to 1D array in correct order
    bp = bp_2d.flatten('F')  # Fortran-style order (column-major)
    
    # Modify for p_prime(0,0) - pressure at first node is fixed
    bp[0] = 0
    
    return bp


# helpers/rhs_construction.py  – NEW VERSION
import numpy as np


def get_rhs2(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    rho: float,
    u_star: np.ndarray,
    v_star: np.ndarray,
) -> np.ndarray:
    """
    Build the right-hand side vector bp for the pressure-correction equation
    such that  A · p' = bp   (matrix built by `get_coeff_mat`).

    Sign convention matches the coefficient matrix *and* the Rhie–Chow
    velocity correction with  u = u* + d_u ∂p'/∂x   (note the **plus** sign).
    """
    # continuity defect on cell faces (vectorised)
    bp_2d = rho * (
        (u_star[1:, :] - u_star[:-1, :]) * dy
        + (v_star[:, 1:] - v_star[:, :-1]) * dx
    )

    bp = bp_2d.flatten("F")        # Fortran order
    bp[0] = 0.0                    # consistency with pinned pressure node
    return bp
