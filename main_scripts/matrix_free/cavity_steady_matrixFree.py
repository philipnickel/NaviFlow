import numpy as np
import matplotlib.pyplot as plt
from naviflow.helpers_cavity_matrixFree import (u_momentum, v_momentum, get_rhs, get_coeff_mat, 
                           pres_correct, update_velocity, check_divergence_free)

# Grid size and other parameters
imax = 129                       # grid size in x-direction
jmax = 129                       # grid size in y-direction
max_iteration = 10000
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
tol = 1e-7

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
    
    # Form the parameters needed for matrix-free operations
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

print(f"Total Iterations = {iteration}")

# Extract v-velocity along horizontal centerline for comparison with Ghia et al.
j_center = jmax // 2  # Center index for y-direction (horizontal centerline)
v_centerline = np.zeros(imax)

# Interpolate v-velocity to cell centers along the centerline
for i in range(imax):
    v_centerline[i] = 0.5 * (v[i, j_center] + v[i, j_center+1])

# Define the Ghia et al. data for Re = 100
ghia_x = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                   0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
ghia_v = np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, 
                  -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 
                  0.10091, 0.09233, 0.00000])

# Normalize x-coordinates for comparison
x_normalized = np.linspace(0, 1, imax)

# Plot the regular contour plot first
plt.figure(figsize=(8, 6))
x_centers = x
y_centers = y  # Use all points in y
X, Y = np.meshgrid(x_centers, y_centers)
plt.contourf(X, Y, u[1:imax, :].T, 50, cmap='jet')
plt.colorbar()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title(f'Steady Ux (Re = {Re:.0f})')
plt.savefig(f'cavity_ux_Re{Re:.0f}_matrixFree.png')
plt.show()

# Now create the comparison plot with Ghia et al. data
plt.figure(figsize=(10, 6))
plt.plot(x_normalized, v_centerline, 'b-', linewidth=2, label='Current Simulation')
plt.plot(ghia_x, ghia_v, 'ro--', markersize=6, label='Ghia et al. (1982)')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('v-velocity')
plt.title(f'Comparison with Ghia et al. Benchmark Data (Re = {Re:.0f})')
plt.legend()
plt.savefig(f'v_centerline_comparison_Re{Re:.0f}_matrixFree.png')
plt.show()

# Calculate error metrics
from scipy.interpolate import interp1d

# Interpolate simulation data to Ghia x-positions
interp_func = interp1d(x_normalized, v_centerline, kind='cubic', bounds_error=False, fill_value='extrapolate')
v_interp = interp_func(ghia_x)

# Calculate error metrics
abs_error = np.abs(v_interp - ghia_v)
mean_abs_error = np.mean(abs_error)
max_abs_error = np.max(abs_error)

print(f"Validation against Ghia et al. (1982) for Re = 100:")
print(f"Mean Absolute Error: {mean_abs_error:.6f}")
print(f"Maximum Absolute Error: {max_abs_error:.6f}")

# Optional: save the data for future reference
comparison_data = np.column_stack((ghia_x, ghia_v, v_interp, abs_error))
np.savetxt('ghia_comparison_Re100_matrixFree.txt', comparison_data, 
           header='x Ghia_v Simulation_v Abs_Error', delimiter='\t')
