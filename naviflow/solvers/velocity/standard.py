import numpy as np 
"""
This file contains the velocity solver for the SIMPLE algorithm.
"""

def update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity):
    """Update velocities based on pressure correction."""
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

