import numpy as np
from naviflow import *


# Grid size and other parameters
imax = 65                      # grid size in x-direction
jmax = 65                       # grid size in y-direction
max_iteration = 2000 
maxRes = 1000
iteration = 1
Re = 100                        # Reynolds number
velocity = 1                    # lid velocity
rho = 1                         # density
mu = rho * velocity * 1.0 / Re  # viscosity calculated from Reynolds number
dx = 1/(imax-1)                 # dx,dy cell sizes along x and y directions
dy = 1/(jmax-1)
x = np.arange(dx/2, 1, dx)      # cell centers in x
y = np.arange(0, 1+dy, dy)      # cell centers in y
alphaP = 0.1                    # pressure under-relaxation
alphaU = 0.7                    # velocity under-relaxation
tol = 1e-5

print(f"Reynolds number: {Re}")
print(f"Calculated viscosity: {mu}")


# Initialize variables
p = np.zeros((imax, jmax))          # p = Pressure

# Vertical velocity
v = np.zeros((imax, jmax+1))

# Horizontal Velocity
u = np.zeros((imax+1, jmax))

# Boundary condition: Lid velocity (Top wall is moving with 1m/s)
u[:, jmax-1] = velocity

# Run the SIMPLE algorithm
u, v, p, iteration, maxRes, divergence = simple_algorithm(
    imax, jmax, dx, dy, rho, mu, u, v, p, 
    velocity, alphaU, alphaP, max_iteration, tol
    # Use default solvers by not specifying momentum_solver, pressure_solver, or velocity_updater
)

print(f"Total Iterations = {iteration}")

# Visualization and validation using utility functions
# 1. Plot velocity field
plot_velocity_field(u, v, x, y, 
                   title=f'Velocity Field (Re={Re})',
                   filename=f'velocity_field_Re{Re}.png',
                   show=False)

# 2. Plot streamlines with pressure as background
plot_streamlines(u, v, x, y, 
                title=f'Streamlines (Re={Re})',
                filename=f'streamlines_Re{Re}.png',
                background_field=p,
                show=False)

# 3. Validate against Ghia benchmark
error_metrics = compare_with_ghia(v, x, Re, 
                                filename=f'validation_Re{Re}.png',
                                save_data=True,
                                show=False)

# 4. Check for mass conservation
div = calculate_divergence(u, v, dx, dy)
max_div = np.max(np.abs(div))
print(f"Maximum absolute divergence: {max_div:.6e}")
