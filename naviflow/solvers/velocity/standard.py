import numpy as np 
from numba import njit, prange
"""
This file contains the velocity solver for the SIMPLE algorithm.
"""

@njit(parallel=True)
def update_velocity_numba(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity):
    """Numba-accelerated parallel version of velocity update."""
    v = np.zeros((imax, jmax+1))
    u = np.zeros((imax+1, jmax))
    
    # Update u velocity for interior nodes in parallel
    for i in prange(1, imax):
        for j in range(1, jmax-1):
            u[i, j] = u_star[i, j] + d_u[i, j] * (p_prime[i-1, j] - p_prime[i, j])
    
    # Update v velocity for interior nodes in parallel
    for i in prange(1, imax-1):
        for j in range(1, jmax):
            v[i, j] = v_star[i, j] + d_v[i, j] * (p_prime[i, j-1] - p_prime[i, j])
    
    # Apply BCs
    v[0, :] = 0.0                      # left wall
    v[imax-1, :] = 0.0                 # right wall
    v[:, 0] = -v[:, 1]                 # bottom wall
    v[:, jmax] = -v[:, jmax-1]         # top wall
    
    u[0, :] = -u[1, :]                 # left wall
    u[imax, :] = -u[imax-1, :]         # right wall
    u[:, 0] = 0.0                      # bottom wall
    u[:, jmax-1] = velocity            # top wall
    
    return u, v

def update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity, use_numba=False):
    """Update velocities based on pressure correction."""
    if use_numba:
        return update_velocity_numba(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity)
    
    v = np.zeros((imax, jmax+1))
    u = np.zeros((imax+1, jmax))
    
    # Vectorized u velocity update for interior nodes
    i_range = np.arange(1, imax)
    j_range = np.arange(1, jmax-1)
    i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
    
    u[i_grid, j_grid] = u_star[i_grid, j_grid] + d_u[i_grid, j_grid] * (p_prime[i_grid-1, j_grid] - p_prime[i_grid, j_grid])
    
    # Vectorized v velocity update for interior nodes
    i_range = np.arange(1, imax-1)
    j_range = np.arange(1, jmax)
    i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
    
    v[i_grid, j_grid] = v_star[i_grid, j_grid] + d_v[i_grid, j_grid] * (p_prime[i_grid, j_grid-1] - p_prime[i_grid, j_grid])
    
    # Apply BCs
    v[0, :] = 0.0                      # left wall
    v[imax-1, :] = 0.0                 # right wall
    v[:, 0] = -v[:, 1]                 # bottom wall
    v[:, jmax] = -v[:, jmax-1]         # top wall
    
    u[0, :] = -u[1, :]                 # left wall
    u[imax, :] = -u[imax-1, :]         # right wall
    u[:, 0] = 0.0                      # bottom wall
    u[:, jmax-1] = velocity            # top wall
    
    return u, v

