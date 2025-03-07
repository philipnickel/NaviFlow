import numpy as np

def gauss_seidel(solver_params):
    """
    Solve the linear system Ax = b using the Jacobi iterative method with vectorized operations.
    
    Parameters:
    -----------
    solver_params : dict
        Dictionary containing parameters for matrix-free operations
        
    Returns:
    --------
    x : numpy.ndarray
        Solution vector
    """
    # Extract needed parameters
    b = solver_params['b']
    params = solver_params['params']
    imax = solver_params['imax']
    jmax = solver_params['jmax']
    
    # Extract parameters for matrix-free operations
    dx = params['dx']
    dy = params['dy']
    rho = params['rho']
    d_u = params['d_u']  # Shape (imax+1, jmax)
    d_v = params['d_v']  # Shape (imax, jmax+1)
    
    # Extract solver control parameters (with defaults)
    max_iter = solver_params.get('max_iter', 500)
    tolerance = solver_params.get('tolerance', 1e-4)
    omega = solver_params.get('omega', 0.5)  # Relaxation factor
    
    # Initial guess
    x = np.zeros_like(b)
    
    # Reshape to 2D for easier manipulation
    x_2d = x.reshape((imax, jmax), order='F')
    b_2d = b.reshape((imax, jmax), order='F')
    
    # Set reference pressure point
    b_2d[0, 0] = 0.0
    
    # Ensure d_u and d_v have the correct shapes for this grid
    d_u_padded = np.zeros((imax+1, jmax))
    d_v_padded = np.zeros((imax, jmax+1))
    
    # Copy available values from d_u and d_v
    d_u_i_max = min(d_u.shape[0], imax+1)
    d_u_j_max = min(d_u.shape[1], jmax)
    d_v_i_max = min(d_v.shape[0], imax)
    d_v_j_max = min(d_v.shape[1], jmax+1)
    
    d_u_padded[:d_u_i_max, :d_u_j_max] = d_u[:d_u_i_max, :d_u_j_max]
    d_v_padded[:d_v_i_max, :d_v_j_max] = d_v[:d_v_i_max, :d_v_j_max]
    
    # Precompute coefficient arrays
    # East coefficients
    aE = np.zeros((imax, jmax))
    if imax > 1:
        aE[:-1, :] = rho * d_u_padded[1:-1, :] * dy
    
    # West coefficients
    aW = np.zeros((imax, jmax))
    if imax > 1:
        aW[1:, :] = rho * d_u_padded[1:-1, :] * dy
    
    # North coefficients
    aN = np.zeros((imax, jmax))
    if jmax > 1:
        aN[:, :-1] = rho * d_v_padded[:, 1:-1] * dx
    
    # South coefficients
    aS = np.zeros((imax, jmax))
    if jmax > 1:
        aS[:, 1:] = rho * d_v_padded[:, 1:-1] * dx
    
    # Diagonal coefficients
    aP = aE + aW + aN + aS
    
    # Special handling for reference pressure point
    aP[0, 0] = 1.0
    
    # Ensure diagonal is non-zero
    aP = np.maximum(aP, 1e-10)
    
    # Gauss-Seidel iteration
    for iter in range(max_iter):
        # Store previous values for convergence check
        x_old = x_2d.copy()

        # Shifted versions of x_2d for neighboring values
        p_east = np.roll(x_2d, shift=-1, axis=0)  # Shift left (i+1)
        p_west = np.roll(x_2d, shift=1, axis=0)   # Shift right (i-1)
        p_north = np.roll(x_2d, shift=-1, axis=1) # Shift up (j+1)
        p_south = np.roll(x_2d, shift=1, axis=1)  # Shift down (j-1)

        # Apply zero boundary conditions (avoiding boundaries)
        if imax > 1:
            p_east[-1, :] = 0  # Right boundary
            p_west[0, :] = 0   # Left boundary

        if jmax > 1:
            p_north[:, -1] = 0  # Top boundary
            p_south[:, 0] = 0   # Bottom boundary

        # Gauss-Seidel update (vectorized)
        # Creating shifted versions of x_2d with boundary conditions
        xE = np.roll(x_2d, shift=-1, axis=0)  # Shift left (i+1)
        xW = np.roll(x_2d, shift=1, axis=0)   # Shift right (i-1)
        xN = np.roll(x_2d, shift=-1, axis=1)  # Shift up (j+1)
        xS = np.roll(x_2d, shift=1, axis=1)   # Shift down (j-1)

        # Applying boundary conditions (zero-padding at edges)
        xE[-1, :] = 0
        xW[0, :] = 0
        xN[:, -1] = 0
        xS[:, 0] = 0

        # Computing the updated x_2d values in a vectorized manner
        x_2d = (1 - omega) * x_2d + omega * (
            (b_2d + (aE * xE + aW * xW + aN * xN + aS * xS)) / aP
        )


        # Fix reference pressure point
        x_2d[0, 0] = 0.0

        # Compute residual using NumPy (maximum absolute change)
        residual = np.max(np.abs(x_2d - x_old))

        # Check convergence
        if residual < tolerance:
            break

    # Flatten back to 1D
    return x_2d.flatten(order='F')
