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
    
    # Define the power-law function A (vectorized version)
    def A(F, D):
        return np.maximum(0, (1 - 0.1 * np.abs(F/D))**5)
    
    # Interior points - vectorized computation
    i_range = np.arange(1, imax)
    j_range = np.arange(1, jmax-1)
    i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
    
    # Calculate flow terms
    Fe = 0.5 * rho * dy * (u[i_grid+1, j_grid] + u[i_grid, j_grid])
    Fw = 0.5 * rho * dy * (u[i_grid-1, j_grid] + u[i_grid, j_grid])
    Fn = 0.5 * rho * dx * (v[i_grid, j_grid+1] + v[i_grid-1, j_grid+1])
    Fs = 0.5 * rho * dx * (v[i_grid, j_grid] + v[i_grid-1, j_grid])
    
    # Calculate coefficients
    aE = De * A(Fe, De) + np.maximum(-Fe, 0)
    aW = Dw * A(Fw, Dw) + np.maximum(Fw, 0)
    aN = Dn * A(Fn, Dn) + np.maximum(-Fn, 0)
    aS = Ds * A(Fs, Ds) + np.maximum(Fs, 0)
    aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
    
    pressure_term = (p[i_grid-1, j_grid] - p[i_grid, j_grid]) * dy
    
    # Calculate u_star and d_u
    u_star[i_grid, j_grid] = alpha/aP * ((aE*u[i_grid+1, j_grid] + 
                                          aW*u[i_grid-1, j_grid] + 
                                          aN*u[i_grid, j_grid+1] + 
                                          aS*u[i_grid, j_grid-1]) + 
                                         pressure_term) + (1-alpha)*u[i_grid, j_grid]
    
    d_u[i_grid, j_grid] = alpha * dy / aP
    
    # Bottom boundary (j=0) - can also be vectorized
    j = 0
    i_bottom = np.arange(1, imax)
    Fe_bottom = 0.5 * rho * dy * (u[i_bottom+1, j] + u[i_bottom, j])
    Fw_bottom = 0.5 * rho * dy * (u[i_bottom-1, j] + u[i_bottom, j])
    Fn_bottom = 0.5 * rho * dx * (v[i_bottom, j+1] + v[i_bottom-1, j+1])
    Fs_bottom = 0
    
    aE_bottom = De * A(Fe_bottom, De) + np.maximum(-Fe_bottom, 0)
    aW_bottom = Dw * A(Fw_bottom, Dw) + np.maximum(Fw_bottom, 0)
    aN_bottom = Dn * A(Fn_bottom, Dn) + np.maximum(-Fn_bottom, 0)
    aS_bottom = 0
    aP_bottom = aE_bottom + aW_bottom + aN_bottom + aS_bottom + (Fe_bottom-Fw_bottom) + (Fn_bottom-Fs_bottom)
    d_u[i_bottom, j] = alpha * dy / aP_bottom
    
    # Top boundary (j=jmax-1) - vectorized
    j = jmax-1
    i_top = np.arange(1, imax)
    Fe_top = 0.5 * rho * dy * (u[i_top+1, j] + u[i_top, j])
    Fw_top = 0.5 * rho * dy * (u[i_top-1, j] + u[i_top, j])
    Fn_top = 0
    Fs_top = 0.5 * rho * dx * (v[i_top, j] + v[i_top-1, j])
    
    aE_top = De * A(Fe_top, De) + np.maximum(-Fe_top, 0)
    aW_top = Dw * A(Fw_top, Dw) + np.maximum(Fw_top, 0)
    aN_top = 0
    aS_top = Ds * A(Fs_top, Ds) + np.maximum(Fs_top, 0)
    aP_top = aE_top + aW_top + aN_top + aS_top + (Fe_top-Fw_top) + (Fn_top-Fs_top)
    d_u[i_top, j] = alpha * dy / aP_top
    
    # Apply BCs
    u_star[0, :] = -u_star[1, :]                # left wall
    u_star[imax, :] = -u_star[imax-1, :]        # right wall
    u_star[:, 0] = 0.0                          # bottom wall
    u_star[:, jmax-1] = velocity                # top wall
    
    return u_star, d_u

def v_momentum(imax, jmax, dx, dy, rho, mu, u, v, p, alpha):
    """Solve the v-momentum equation for the intermediate velocity v_star."""
    v_star = np.zeros((imax, jmax+1))
    d_v = np.zeros((imax, jmax+1))
    
    De = mu * dy / dx   # convective coefficients
    Dw = mu * dy / dx
    Dn = mu * dx / dy
    Ds = mu * dx / dy
    
    # Define the power-law function A (vectorized version)
    def A(F, D):
        return np.maximum(0, (1 - 0.1 * np.abs(F/D))**5)
    
    # Interior points - vectorized computation
    i_range = np.arange(1, imax-1)
    j_range = np.arange(1, jmax)
    i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
    
    # Calculate flow terms
    Fe = 0.5 * rho * dy * (u[i_grid+1, j_grid] + u[i_grid+1, j_grid-1])
    Fw = 0.5 * rho * dy * (u[i_grid, j_grid] + u[i_grid, j_grid-1])
    Fn = 0.5 * rho * dx * (v[i_grid, j_grid] + v[i_grid, j_grid+1])
    Fs = 0.5 * rho * dx * (v[i_grid, j_grid-1] + v[i_grid, j_grid])
    
    # Calculate coefficients
    aE = De * A(Fe, De) + np.maximum(-Fe, 0)
    aW = Dw * A(Fw, Dw) + np.maximum(Fw, 0)
    aN = Dn * A(Fn, Dn) + np.maximum(-Fn, 0)
    aS = Ds * A(Fs, Ds) + np.maximum(Fs, 0)
    aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)
    
    pressure_term = (p[i_grid, j_grid-1] - p[i_grid, j_grid]) * dx
    
    # Calculate v_star and d_v
    v_star[i_grid, j_grid] = alpha/aP * ((aE*v[i_grid+1, j_grid] + 
                                          aW*v[i_grid-1, j_grid] + 
                                          aN*v[i_grid, j_grid+1] + 
                                          aS*v[i_grid, j_grid-1]) + 
                                         pressure_term) + (1-alpha)*v[i_grid, j_grid]
    
    d_v[i_grid, j_grid] = alpha * dx / aP
    
    # Left boundary (i=0) - vectorized
    i = 0
    j_left = np.arange(1, jmax)
    Fe_left = 0.5 * rho * dy * (u[i+1, j_left] + u[i+1, j_left-1])
    Fw_left = 0
    Fn_left = 0.5 * rho * dx * (v[i, j_left] + v[i, j_left+1])
    Fs_left = 0.5 * rho * dx * (v[i, j_left-1] + v[i, j_left])
    
    aE_left = De * A(Fe_left, De) + np.maximum(-Fe_left, 0)
    aW_left = 0
    aN_left = Dn * A(Fn_left, Dn) + np.maximum(-Fn_left, 0)
    aS_left = Ds * A(Fs_left, Ds) + np.maximum(Fs_left, 0)
    aP_left = aE_left + aW_left + aN_left + aS_left + (Fe_left-Fw_left) + (Fn_left-Fs_left)
    d_v[i, j_left] = alpha * dx / aP_left
    
    # Right boundary (i=imax-1) - vectorized
    i = imax-1
    j_right = np.arange(1, jmax)
    Fe_right = 0
    Fw_right = 0.5 * rho * dy * (u[i, j_right] + u[i, j_right-1])
    Fn_right = 0.5 * rho * dx * (v[i, j_right] + v[i, j_right+1])
    Fs_right = 0.5 * rho * dx * (v[i, j_right-1] + v[i, j_right])
    
    aE_right = 0
    aW_right = Dw * A(Fw_right, Dw) + np.maximum(Fw_right, 0)
    aN_right = Dn * A(Fn_right, Dn) + np.maximum(-Fn_right, 0)
    aS_right = Ds * A(Fs_right, Ds) + np.maximum(Fs_right, 0)
    aP_right = aE_right + aW_right + aN_right + aS_right + (Fe_right-Fw_right) + (Fn_right-Fs_right)
    d_v[i, j_right] = alpha * dx / aP_right
    
    # Apply BCs
    v_star[0, :] = 0.0                      # left wall
    v_star[imax-1, :] = 0.0                 # right wall
    v_star[:, 0] = -v_star[:, 1]            # bottom wall
    v_star[:, jmax] = -v_star[:, jmax-1]    # top wall
    
    return v_star, d_v

def get_rhs(imax, jmax, dx, dy, rho, u_star, v_star):
    """Calculate RHS vector of the pressure correction equation."""
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

def get_coeff_mat(imax, jmax, dx, dy, rho, d_u, d_v):
    """Form the coefficient matrix for the pressure correction equation using diagonal construction."""
    N = imax * jmax
    
    # Initialize data structures for sparse matrix in COO format
    row_indices = []
    col_indices = []
    data_values = []
    
    # Set reference pressure point
    row_indices.append(0)
    col_indices.append(0)
    data_values.append(1.0)
    
    # Prepare arrays for the five diagonals
    # These will be used to efficiently build the coefficient matrix
    for j in range(jmax):
        for i in range(imax):
            idx = i + j*imax
            
            # Skip reference pressure point
            if i == 0 and j == 0:
                continue
                
            # Initialize diagonal coefficient
            aP = 0.0
            
            # East connection (if not at right boundary)
            if i < imax-1:
                aE = rho * d_u[i+1, j] * dy
                # Add east connection (idx, idx+1)
                row_indices.append(idx)
                col_indices.append(idx+1)
                data_values.append(-aE)
                aP += aE
            
            # West connection (if not at left boundary)
            if i > 0:
                aW = rho * d_u[i, j] * dy
                # Add west connection (idx, idx-1)
                row_indices.append(idx)
                col_indices.append(idx-1)
                data_values.append(-aW)
                aP += aW
            
            # North connection (if not at top boundary)
            if j < jmax-1:
                aN = rho * d_v[i, j+1] * dx
                # Add north connection (idx, idx+imax)
                row_indices.append(idx)
                col_indices.append(idx+imax)
                data_values.append(-aN)
                aP += aN
            
            # South connection (if not at bottom boundary)
            if j > 0:
                aS = rho * d_v[i, j] * dx
                # Add south connection (idx, idx-imax)
                row_indices.append(idx)
                col_indices.append(idx-imax)
                data_values.append(-aS)
                aP += aS
            
            # Add diagonal entry
            row_indices.append(idx)
            col_indices.append(idx)
            data_values.append(aP)
    
    # Create sparse matrix directly in CSR format for efficient solver
    Ap = sparse.csr_matrix((data_values, (row_indices, col_indices)), shape=(N, N))
    
    return Ap

def penta_diag_solve(A, b):
    """Solve the pentadiagonal system Ax = b."""
    # A is already a sparse matrix, so we can use spsolve directly
    x = spsolve(A, b)
    return x

def pres_correct(imax, jmax, rhsp, Ap, p, alpha):
    """Solve for pressure correction and update pressure."""
    # Solve for pressure correction
    p_prime_interior = penta_diag_solve(Ap, rhsp)
    
    # Reshape to 2D array using vectorized operation
    p_prime = p_prime_interior.reshape((imax, jmax), order='F')
    
    # Vectorized pressure update
    pressure = p + alpha * p_prime
    
    # Fix pressure at a reference point
    pressure[0, 0] = 0.0
    
    return pressure, p_prime

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

def check_divergence_free(imax, jmax, dx, dy, u, v):
    """Check if velocity field is divergence free."""
    # Fully vectorized implementation
    i_indices = np.arange(imax)
    j_indices = np.arange(jmax)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    divergence = (u[i_grid+1, j_grid] - u[i_grid, j_grid])/dx + \
                 (v[i_grid, j_grid+1] - v[i_grid, j_grid])/dy
    
    return divergence