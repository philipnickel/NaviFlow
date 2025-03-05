import numpy as np
import matplotlib.pyplot as plt
from naviflow import *


# Grid size and other parameters
imax = 65                       # grid size in x-direction
jmax = 65                       # grid size in y-direction
max_iteration = 100
maxRes = 1000
iteration = 1
mu = 0.01                       # viscosity
rho = 1                         # density
velocity = 1                    # lid velocity
dx = 1/(imax-1)                 # dx,dy cell sizes along x and y directions
dy = 1/(jmax-1)
x = np.arange(dx/2, 1, dx)      # cell centers in x
y = np.arange(0, 1+dy, dy)      # cell centers in y
alphaP = 0.1                    # pressure under-relaxation
alphaU = 0.7                    # velocity under-relaxation
tol = 1e-5

# Calculate Reynolds number based on lid velocity, cavity length, and fluid properties
Re = rho * velocity * 1.0 / mu  # Length = 1.0 (cavity dimension)
print(f"Reynolds number: {Re}")



# Initialize variables
p = np.zeros((imax, jmax))          # p = Pressure
p_star = np.zeros((imax, jmax))
p_prime = np.zeros((imax, jmax))    # pressure correction
rhsp = np.zeros(imax*jmax)          # RHS vector of pressure correction equation
divergence = np.zeros((imax, jmax))

# Vertical velocity
v_star = np.zeros((imax, jmax+1))
vold = np.zeros((imax, jmax+1))
vRes = np.zeros((imax, jmax+1))
v = np.zeros((imax, jmax+1))
d_v = np.zeros((imax, jmax+1))      # velocity correction coefficient

# Horizontal Velocity
u_star = np.zeros((imax+1, jmax))
uold = np.zeros((imax+1, jmax))
uRes = np.zeros((imax+1, jmax))
u = np.zeros((imax+1, jmax))
d_u = np.zeros((imax+1, jmax))      # velocity correction coefficient

# Boundary condition: Lid velocity (Top wall is moving with 1m/s)
u_star[:, jmax-1] = velocity
u[:, jmax-1] = velocity

# ---------- iterations -------------------
while (iteration <= max_iteration) and (maxRes > tol):
    iteration += 1
    
    # Solve u-momentum equation for intermediate velocity u_star
    u_star, d_u = u_momentum(imax, jmax, dx, dy, rho, mu, u, v, p_star, velocity, alphaU)
    
    # Solve v-momentum equation for intermediate velocity v_star
    v_star, d_v = v_momentum(imax, jmax, dx, dy, rho, mu, u, v, p_star, alphaU)
    
    uold = u.copy()
    vold = v.copy()
    
    # Calculate rhs vector of the Pressure Poisson matrix
    rhsp = get_rhs(imax, jmax, dx, dy, rho, u_star, v_star)
    
    # Form the Pressure Poisson coefficient matrix
    Ap = get_coeff_mat(imax, jmax, dx, dy, rho, d_u, d_v)
    
    # Solve pressure correction implicitly and update pressure
    p, p_prime = pres_correct(imax, jmax, rhsp, Ap, p_star, alphaP)
    
    # Update velocity based on pressure correction
    u, v = update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity)
    
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
        print("not going to converge!")
        break

# 1. Plot velocity field
plot_velocity_field(u, v, x, y, 
                   title=f'Velocity Field (Re={Re})',
                   filename=f'velocity_field_Re{Re}.png', show=False)

# 2. Plot streamlines with pressure as background
plot_streamlines(u, v, x, y, 
                title=f'Streamlines (Re={Re})',
                filename=f'streamlines_Re{Re}.png',
                background_field=p, show=False)

# 3. Validate against Ghia benchmark
error_metrics = compare_with_ghia(v, x, Re, 
                                filename=f'validation_Re{Re}.png',
                                save_data=True, show=False)

# 4. Check for mass conservation
div = calculate_divergence(u, v, dx, dy)
max_div = np.max(np.abs(div))
print(f"Maximum absolute divergence: {max_div:.6e}")