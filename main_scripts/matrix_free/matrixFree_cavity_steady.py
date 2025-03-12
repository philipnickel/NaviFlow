import numpy as np
from naviflow import *
from numba import njit


# Grid size and other parameters
imax = 129                      # grid size in x-direction
jmax = 129                      # grid size in y-direction
max_iteration = 10000
maxRes = 1000
iteration = 0
Re = 100                        # Reynolds number
velocity = 1                    # lid velocity
rho = 1                         # density
mu = rho * velocity * 1.0 / Re  # viscosity calculated from Reynolds number
dx = 1/(imax-1)                 # dx,dy cell sizes along x and y directions
dy = 1/(jmax-1)
x = np.linspace(dx/2, 1-dx/2, imax)  # cell centers in x with exactly imax points
y = np.linspace(dy/2, 1-dy/2, jmax)  # cell centers in y with exactly jmax points
alphaP = 0.4                    # pressure under-relaxation
alphaU = 0.9                    # velocity under-relaxation
tol = 1e-7

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

# Run the SIMPLE algorithm with matrix-free pressure solver
u, v, p, iteration, maxRes, divergence = simple_algorithm(
    imax, jmax, dx, dy, rho, mu, u, v, p, 
    velocity, alphaU, alphaP, max_iteration, tol,
    pressure_solver="pres_correct_matrix_free",
    use_numba=True  # Enable Numba acceleration
)


print(f"Total Iterations = {iteration}")
# Visualization and validation using utility functions
# 1. Plot combined results
plot_combined_results_matrix(u, v, p, x, y, Re,
                           title=f'Cavity Flow Results (Re={Re})',
                           filename=f'cavity_Re{Re}_matrix_results.pdf',
                           cmap='coolwarm',
                           show=False)

# 4. Check for mass conservation
div = calculate_divergence(u, v, dx, dy)
max_div = np.max(np.abs(div))
print(f"Maximum absolute divergence: {max_div:.6e}")

