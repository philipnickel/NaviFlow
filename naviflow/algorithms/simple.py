import numpy as np

def simple_algorithm(imax, jmax, dx, dy, rho, mu, u, v, p, 
                    velocity, alphaU, alphaP, max_iteration, tol,
                    momentum_solver=None,
                    pressure_solver=None,
                    velocity_updater=None):
    """
    SIMPLE (Semi-Implicit Method for Pressure Linked Equations) algorithm
    for solving the Navier-Stokes equations.
    
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
        Lid velocity (boundary condition)
    alphaU : float
        Velocity under-relaxation factor
    alphaP : float
        Pressure under-relaxation factor
    max_iteration : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    momentum_solver : function, optional
        Function to solve the momentum equation.
        If None, the default solve_momentum function will be used.
    pressure_solver : function, optional
        Function to solve the pressure correction equation.
        If None, the default solve_pressure_correction function will be used.
    velocity_updater : function, optional
        Function to update the velocity field based on pressure correction.
        If None, the default update_velocity function will be used.
        
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
    # Import default implementations if none provided
    if momentum_solver is None:
        from ..solvers.momentum.standard import u_momentum, v_momentum
        u_momentum_solver = u_momentum
        v_momentum_solver = v_momentum
    else:
        u_momentum_solver = momentum_solver
        v_momentum_solver = momentum_solver
    
    if pressure_solver is None:
        from ..solvers.pressure.direct import get_rhs, get_coeff_mat, pres_correct
        pressure_solver = pres_correct
        get_rhs_func = get_rhs
        get_coeff_mat_func = get_coeff_mat
    elif pressure_solver is "pres_correct_matrix_free":
        from ..solvers.pressure.direct import get_rhs
        from ..solvers.pressure.matrix_free import get_coeff_mat_matrix_free, pres_correct_matrix_free
        pressure_solver = pres_correct_matrix_free
        get_rhs_func = get_rhs
        get_coeff_mat_func = get_coeff_mat_matrix_free
    
    if velocity_updater is None:
        from ..solvers.velocity.standard import update_velocity
        velocity_updater = update_velocity
    
    # Import validation utilities
    from ..utils.validation import check_divergence_free

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
        u_star, d_u = u_momentum_solver(imax, jmax, dx, dy, rho, mu, u, v, p_star, velocity, alphaU)
        
        # Solve v-momentum equation for intermediate velocity v_star
        v_star, d_v = v_momentum_solver(imax, jmax, dx, dy, rho, mu, u, v, p_star, alphaU)
        
        uold = u.copy()
        vold = v.copy()

        # Calculate rhs vector of the Pressure Poisson matrix
        rhsp = get_rhs_func(imax, jmax, dx, dy, rho, u_star, v_star)
        
        # Form the Pressure Poisson coefficient matrix
        Ap = get_coeff_mat_func(imax, jmax, dx, dy, rho, d_u, d_v)
        
        # Solve pressure correction using the selected solver
        p, p_prime = pressure_solver(imax, jmax, rhsp, Ap, p_star, alphaP)
        
        # Update velocity based on pressure correction
        u, v = velocity_updater(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity)
        
        # Check if velocity field is divergence free
        divergence = check_divergence_free(imax, jmax, dx, dy, u, v)
        
        # Use p as p_star for the next iteration
        p_star = p.copy()
        
        # Find maximum residual in the domain
        vRes = np.abs(v - vold)
        uRes = np.abs(u - uold)
        maxRes_u = np.max(uRes)
        maxRes_v = np.max(vRes)
        maxRes = max(maxRes_u, maxRes_v)
        
        print(f"It = {iteration}; Res = {maxRes}")
        
        if maxRes > 2:
            print("Not going to converge!")
            break
            
        iteration += 1

    return u, v, p, iteration, maxRes, divergence
