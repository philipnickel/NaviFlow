def pres_correct(solver, imax, jmax, rhsp, solver_params, p, alpha):
    """Solve for pressure correction and update pressure."""
    # Solve for pressure correction with a single parameter
    p_prime_interior = solver(solver_params)
    
    # Reshape to 2D array using vectorized operation
    p_prime = p_prime_interior.reshape((imax, jmax), order='F')
    
    # Vectorized pressure update
    pressure = p + alpha * p_prime
    
    # Fix pressure at a reference point
    pressure[0, 0] = 0.0
    
    return pressure, p_prime

