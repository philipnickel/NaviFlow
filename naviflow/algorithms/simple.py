import numpy as np

def simple_algorithm(imax, jmax, dx, dy, rho, mu, u, v, p, 
                    velocity, alphaU, alphaP, max_iteration, tol,
                    momentum_solver=None,
                    pressure_solver=None,
                    velocity_updater=None,
                    use_numba=False,
                    solver_params=None,
                    callback=None):
    """
    SIMPLE algorithm for solving the Navier-Stokes equations.
    
    Parameters:
    -----------
    imax, jmax : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    rho : float
        Fluid density
    mu : float
        Fluid viscosity
    u, v : numpy.ndarray
        Initial velocity fields
    p : numpy.ndarray
        Initial pressure field
    velocity : float
        Lid velocity
    alphaU, alphaP : float
        Relaxation factors for velocity and pressure
    max_iteration : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    momentum_solver : function, optional
        Function to solve momentum equations
    pressure_solver : str or function, optional
        Function or name of method to solve pressure correction equation
    velocity_updater : function, optional
        Function to update velocities
    use_numba : bool, optional
        Whether to use Numba acceleration
    solver_params : dict, optional
        Additional parameters for the pressure solver
    callback : function, optional
        Function called at each iteration with parameters (iteration, u, v, p, maxRes)
        If the callback returns True, the iteration will stop
        
    Returns:
    --------
    u, v : numpy.ndarray
        Final velocity fields
    p : numpy.ndarray
        Final pressure field
    iteration : int
        Number of iterations performed
    maxRes : float
        Final maximum residual
    divergence : numpy.ndarray
        Final divergence field
    """
    # Initialize solver_params if not provided
    if solver_params is None:
        solver_params = {}
    
    # Import default implementations if none provided
    if momentum_solver is None:
        from ..solvers.momentum import u_momentum, v_momentum
        u_momentum_solver = u_momentum
        v_momentum_solver = v_momentum
    else:
        u_momentum_solver = momentum_solver
        v_momentum_solver = momentum_solver
    
    if pressure_solver is None:
        from ..solvers.pressure.helpers import get_rhs, get_coeff_mat, pres_correct
        from ..solvers.pressure.direct import penta_diag_solve
        pressure_solver = pres_correct
        get_rhs_func = get_rhs
        get_coeff_mat_func = get_coeff_mat
        solver = penta_diag_solve
    elif pressure_solver == "pres_correct_matrix_free":
        from ..solvers.pressure.helpers import get_rhs, get_coeff_mat_matrix_free, pres_correct
        from ..solvers.pressure.conjugent_gradient import cg_matrix_free
        pressure_solver = pres_correct
        get_rhs_func = get_rhs
        get_coeff_mat_func = get_coeff_mat_matrix_free
        solver = cg_matrix_free
    elif pressure_solver == "jacobi":
        from ..solvers.pressure.helpers import get_rhs, get_coeff_mat_matrix_free, pres_correct
        from ..solvers.pressure.jacobi import jacobi
        pressure_solver = pres_correct
        get_rhs_func = get_rhs
        get_coeff_mat_func = get_coeff_mat_matrix_free
        solver = jacobi
    elif pressure_solver == "multigrid":
        from ..solvers.pressure.helpers import get_rhs, get_coeff_mat_matrix_free, pres_correct
        from ..solvers.pressure.multigrid_vcycle import multigrid_vcycle_solver
        pressure_solver = pres_correct
        get_rhs_func = get_rhs
        get_coeff_mat_func = get_coeff_mat_matrix_free
        solver = multigrid_vcycle_solver
    
    if velocity_updater is None:
        from ..solvers.velocity.standard import update_velocity
        velocity_updater = update_velocity
    
    # Import validation utilities
    from ..utils.validation import check_divergence_free, calculate_divergence

    # Initialize variables
    p_star = p.copy()
    p_prime = np.zeros((imax, jmax))
    rhsp = np.zeros(imax*jmax)
    divergence = np.zeros((imax, jmax))

    # Vertical velocity
    v_star = np.zeros((imax, jmax+1))
    vold = np.zeros((imax, jmax+1))
    vRes = np.zeros((imax, jmax+1))
    d_v = np.zeros((imax, jmax+1))

    # Horizontal Velocity
    u_star = np.zeros((imax+1, jmax))
    uold = np.zeros((imax+1, jmax))
    uRes = np.zeros((imax+1, jmax))
    d_u = np.zeros((imax+1, jmax))

    # Boundary condition: Lid velocity (Top wall is moving with 1m/s)
    u_star[:, jmax-1] = velocity
    u[:, jmax-1] = velocity
    
    # ---------- iterations -------------------
    iteration = 1
    maxRes = 1000
    
    while (iteration <= max_iteration) and (maxRes > tol):
        
        # Solve u-momentum equation for intermediate velocity u_star
        u_star, d_u = u_momentum_solver(imax, jmax, dx, dy, rho, mu, u, v, p_star, velocity, alphaU, use_numba=use_numba)
        
        # Solve v-momentum equation for intermediate velocity v_star
        v_star, d_v = v_momentum_solver(imax, jmax, dx, dy, rho, mu, u, v, p_star, alphaU, use_numba=use_numba)
        
        uold = u.copy()
        vold = v.copy()

        # Calculate rhs vector of the Pressure Poisson matrix
        rhsp = get_rhs_func(imax, jmax, dx, dy, rho, u_star, v_star)
        
        # Form the Pressure Poisson coefficient matrix
        Ap = get_coeff_mat_func(imax, jmax, dx, dy, rho, d_u, d_v)
        
        # Create solver parameter object with all necessary data
        solver_param_dict = {
            'A': Ap,  # The coefficient matrix or matrix-free parameters
            'b': rhsp,  # The right-hand side vector
            'params': {  # Additional parameters for matrix-free solvers
                'dx': dx,
                'dy': dy,
                'rho': rho,
                'd_u': d_u,
                'd_v': d_v
            },
            'imax': imax,
            'jmax': jmax,
            'use_numba': use_numba  # Pass the Numba flag
        }
        
        # Add any additional solver parameters
        solver_param_dict.update(solver_params)

        # Solve pressure correction equation
        p, p_prime = pressure_solver(solver, imax, jmax, rhsp, solver_param_dict, p_star, alphaP)
        
        # Update velocity based on pressure correction
        u, v = velocity_updater(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity, use_numba=use_numba)
        
        # Calculate residuals
        uRes = np.abs(u - uold)
        vRes = np.abs(v - vold)
        
        # Maximum residual
        maxRes = max(np.max(uRes), np.max(vRes))
        
        # Update pressure for next iteration
        p_star = p.copy()
        
        # Calculate divergence for monitoring
        divergence = calculate_divergence(u, v, dx, dy)
        
        # Print progress
        print(f"Iteration {iteration}, Residual: {maxRes:.6e}, Max Divergence: {np.max(np.abs(divergence)):.6e}")
        
        # Call the callback function if provided
        if callback is not None:
            should_stop = callback(iteration, u, v, p, maxRes)
            if should_stop:
                break
        
        iteration += 1
    
    return u, v, p, iteration, maxRes, divergence
