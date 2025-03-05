import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def u_momentum(imax, jmax, dx, dy, rho, mu, u, v, p, velocity, alpha):
    """Solve the u-momentum equation for the intermediate velocity u_star."""
    u_star = np.zeros((imax+1, jmax))
    d_u = np.zeros((imax+1, jmax))
    
    De = mu * dy / dx   # convective coefficients
    Dw = mu * dy / dx
    Dn = mu * dx / dy
    Ds = mu * dx / dy
    
    # Define the power-law function
    def A(F, D):
        return max(0, (1 - 0.1 * abs(F/D))**5) if D != 0 else 0
    
    # Compute u_star for interior points
    for i in range(1, imax):
        for j in range(1, jmax-1):
            Fe = 0.5 * rho * dy * (u[i+1,j] + u[i,j])
            Fw = 0.5 * rho * dy * (u[i-1,j] + u[i,j])
            Fn = 0.5 * rho * dx * (v[i,j+1] + v[i-1,j+1])
            Fs = 0.5 * rho * dx * (v[i,j] + v[i-1,j])
            
            aE = De * A(Fe, De) + max(-Fe, 0)
            aW = Dw * A(Fw, Dw) + max(Fw, 0)
            aN = Dn * A(Fn, Dn) + max(-Fn, 0)
            aS = Ds * A(Fs, Ds) + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
            
            pressure_term = (p[i-1,j] - p[i,j]) * dy
            
            u_star[i,j] = alpha/aP * ((aE*u[i+1,j] + aW*u[i-1,j] + aN*u[i,j+1] + aS*u[i,j-1]) + pressure_term) + (1-alpha)*u[i,j]
            
            d_u[i,j] = alpha * dy / aP   # Velocity correction coefficient
    
    # Compute u_star for the bottom boundary (j=0)
    j = 0
    for i in range(1, imax):
        Fe = 0.5 * rho * dy * (u[i+1,j] + u[i,j])
        Fw = 0.5 * rho * dy * (u[i-1,j] + u[i,j])
        Fn = 0.5 * rho * dx * (v[i,j+1] + v[i-1,j+1])
        Fs = 0.0  # No flow through bottom wall
        
        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = 0.0  # No southern neighbor
        aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
        
        d_u[i,j] = alpha * dy / aP   # Velocity correction coefficient
    
    # Compute u_star for the top boundary (j=jmax-1)
    j = jmax-1
    for i in range(1, imax):
        Fe = 0.5 * rho * dy * (u[i+1,j] + u[i,j])
        Fw = 0.5 * rho * dy * (u[i-1,j] + u[i,j])
        Fn = 0.0  # No flow through top wall
        Fs = 0.5 * rho * dx * (v[i,j] + v[i-1,j])
        
        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = 0.0  # No northern neighbor
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
        
        d_u[i,j] = alpha * dy / aP   # Velocity correction coefficient
    
    # Apply BCs
    u_star[0, :] = -u_star[1, :]                # left wall
    u_star[imax, :] = -u_star[imax-1, :]        # right wall
    u_star[:, 0] = 0.0                          # bottom wall
    u_star[:, jmax-1] = velocity                # top wall (lid)
    
    return u_star, d_u

def v_momentum(imax, jmax, dx, dy, rho, mu, u, v, p, alpha):
    """Solve the v-momentum equation for the intermediate velocity v_star."""
    v_star = np.zeros((imax, jmax+1))
    d_v = np.zeros((imax, jmax+1))
    
    De = mu * dy / dx   # convective coefficients
    Dw = mu * dy / dx
    Dn = mu * dx / dy
    Ds = mu * dx / dy
    
    # Define the power-law function
    def A(F, D):
        return max(0, (1 - 0.1 * abs(F/D))**5) if D != 0 else 0
    
    # Compute v_star for interior points
    for i in range(1, imax-1):
        for j in range(1, jmax):
            Fe = 0.5 * rho * dy * (u[i+1,j] + u[i+1,j-1])
            Fw = 0.5 * rho * dy * (u[i,j] + u[i,j-1])
            Fn = 0.5 * rho * dx * (v[i,j] + v[i,j+1])
            Fs = 0.5 * rho * dx * (v[i,j-1] + v[i,j])
            
            aE = De * A(Fe, De) + max(-Fe, 0)
            aW = Dw * A(Fw, Dw) + max(Fw, 0)
            aN = Dn * A(Fn, Dn) + max(-Fn, 0)
            aS = Ds * A(Fs, Ds) + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
            
            pressure_term = (p[i,j-1] - p[i,j]) * dx
            
            v_star[i,j] = alpha/aP * ((aE*v[i+1,j] + aW*v[i-1,j] + aN*v[i,j+1] + aS*v[i,j-1]) + pressure_term) + (1-alpha)*v[i,j]
            
            d_v[i,j] = alpha * dx / aP   # Velocity correction coefficient
    
    # Compute v_star for the left boundary (i=0)
    i = 0
    for j in range(1, jmax):
        Fe = 0.5 * rho * dy * (u[i+1,j] + u[i+1,j-1])
        Fw = 0.0  # No flow through left wall
        Fn = 0.5 * rho * dx * (v[i,j] + v[i,j+1])
        Fs = 0.5 * rho * dx * (v[i,j-1] + v[i,j])
        
        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = 0.0  # No western neighbor
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
        
        d_v[i,j] = alpha * dx / aP   # Velocity correction coefficient
    
    # Compute v_star for the right boundary (i=imax-1)
    i = imax-1
    for j in range(1, jmax):
        Fe = 0.0  # No flow through right wall
        Fw = 0.5 * rho * dy * (u[i,j] + u[i,j-1])
        Fn = 0.5 * rho * dx * (v[i,j] + v[i,j+1])
        Fs = 0.5 * rho * dx * (v[i,j-1] + v[i,j])
        
        aE = 0.0  # No eastern neighbor
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
        
        d_v[i,j] = alpha * dx / aP   # Velocity correction coefficient
    
    # Apply BCs
    v_star[0, :] = 0.0                      # left wall
    v_star[imax-1, :] = 0.0                 # right wall
    v_star[:, 0] = -v_star[:, 1]            # bottom wall
    v_star[:, jmax] = -v_star[:, jmax-1]    # top wall
    
    return v_star, d_v

def get_rhs(imax, jmax, dx, dy, rho, u_star, v_star):
    """Calculate RHS vector of the pressure correction equation."""
    bp = np.zeros(imax*jmax)
    
    for j in range(jmax):
        for i in range(imax):
            position = i + j*imax
            
            if i == 0 and j == 0:
                # Fix pressure at first node
                bp[position] = 0
            else:
                # Get imbalance in continuity equation
                bp[position] = rho * (u_star[i,j]*dy - u_star[i+1,j]*dy + v_star[i,j]*dx - v_star[i,j+1]*dx)
    
    return bp

def get_coeff_mat(imax, jmax, dx, dy, rho, d_u, d_v):
    """Form the coefficient matrix for the pressure correction equation."""
    N = imax * jmax
    Ap = np.zeros((N, N))
    
    # Fix pressure at first node
    Ap[0, 0] = 1.0
    
    for j in range(jmax):
        for i in range(imax):
            position = i + j*imax
            
            if i == 0 and j == 0:
                continue  # Skip reference pressure point
            
            # Set diagonal coefficient
            aP = 0.0
            
            # East coefficient
            if i < imax-1:
                aE = rho * d_u[i+1, j] * dy
                Ap[position, position+1] = -aE
                aP += aE
            
            # West coefficient
            if i > 0:
                aW = rho * d_u[i, j] * dy
                Ap[position, position-1] = -aW
                aP += aW
            
            # North coefficient
            if j < jmax-1:
                aN = rho * d_v[i, j+1] * dx
                Ap[position, position+imax] = -aN
                aP += aN
            
            # South coefficient
            if j > 0:
                aS = rho * d_v[i, j] * dx
                Ap[position, position-imax] = -aS
                aP += aS
            
            # Set diagonal coefficient
            Ap[position, position] = aP
    
    return Ap

def penta_diag_solve(A, b):
    """Solve the pentadiagonal system Ax = b."""
    # Convert to sparse format for efficiency
    A_sparse = sparse.csr_matrix(A)
    x = spsolve(A_sparse, b)
    return x

def pres_correct(imax, jmax, rhsp, Ap, p, alpha):
    """Solve for pressure correction and update pressure."""
    # Solve for pressure correction
    p_prime_interior = penta_diag_solve(Ap, rhsp)
    
    # Initialize pressure correction matrix
    p_prime = np.zeros((imax, jmax))
    pressure = np.zeros((imax, jmax))
    
    # Convert 1D solution to 2D array and update pressure
    for j in range(jmax):
        for i in range(imax):
            idx = i + j*imax
            p_prime[i,j] = p_prime_interior[idx]
            pressure[i,j] = p[i,j] + alpha*p_prime[i,j]
    
    # Fix reference pressure
    pressure[0,0] = 0.0
    
    return pressure, p_prime

def update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity):
    """Update velocities based on pressure correction."""
    v = np.zeros((imax, jmax+1))
    u = np.zeros((imax+1, jmax))
    
    # Update interior nodes of u
    for i in range(1, imax):
        for j in range(1, jmax-1):
            u[i,j] = u_star[i,j] + d_u[i,j]*(p_prime[i-1,j] - p_prime[i,j])
    
    # Update interior nodes of v
    for i in range(1, imax-1):
        for j in range(1, jmax):
            v[i,j] = v_star[i,j] + d_v[i,j]*(p_prime[i,j-1] - p_prime[i,j])
    
    # Apply BCs
    v[0, :] = 0.0                      # left wall
    v[imax-1, :] = 0.0                 # right wall
    v[:, 0] = -v[:, 1]                 # bottom wall
    v[:, jmax] = -v[:, jmax-1]         # top wall
    
    u[0, :] = -u[1, :]                 # left wall
    u[imax, :] = -u[imax-1, :]         # right wall
    u[:, 0] = 0.0                      # bottom wall
    u[:, jmax-1] = velocity            # top wall (lid)
    
    return u, v

def check_divergence_free(imax, jmax, dx, dy, u, v):
    """Check if velocity field is divergence free."""
    divergence = np.zeros((imax, jmax))
    
    for j in range(jmax):
        for i in range(imax):
            divergence[i,j] = (u[i+1,j] - u[i,j])/dx + (v[i,j+1] - v[i,j])/dy
    
    return divergence