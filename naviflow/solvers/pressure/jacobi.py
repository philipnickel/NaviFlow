import numpy as np

def jacobi(solver_params):
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
    
    # Jacobi iteration
    for iter in range(max_iter):
        # Save previous solution for convergence check
        x_old = x_2d.copy()
        
        # Compute new solution using vectorized operations
        # Create shifted arrays for each direction
        p_east = np.zeros_like(x_old)
        p_west = np.zeros_like(x_old)
        p_north = np.zeros_like(x_old)
        p_south = np.zeros_like(x_old)
        
        # Fill shifted arrays safely
        if imax > 1:
            p_east[:-1, :] = x_old[1:, :]
            p_west[1:, :] = x_old[:-1, :]
        
        if jmax > 1:
            p_north[:, :-1] = x_old[:, 1:]
            p_south[:, 1:] = x_old[:, :-1]
        
        # Compute new solution with relaxation
        x_2d = (1 - omega) * x_old + omega * (b_2d + 
                                             aE * p_east + 
                                             aW * p_west + 
                                             aN * p_north + 
                                             aS * p_south) / aP
        
        # Fix reference pressure point
        x_2d[0, 0] = 0.0
        
        # Check convergence
        residual = np.max(np.abs(x_2d - x_old))
        if residual < tolerance:
            break
    
    # Flatten back to 1D
    return x_2d.flatten(order='F')
