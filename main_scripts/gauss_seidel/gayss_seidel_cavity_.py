import numpy as np
from naviflow import *
import matplotlib.pyplot as plt


# Grid size and other parameters
imax = 516                     # grid size in x-direction
jmax = 516                      # grid size in y-direction
max_iteration = 7500           # Reduced for faster animation
maxRes = 1000
iteration = 0
Re = 7500                      # Reynolds number
velocity = 1                    # lid velocity
rho = 1                         # density
mu = rho * velocity * 1.0 / Re  # viscosity calculated from Reynolds number
dx = 1/(imax-1)                 # dx,dy cell sizes along x and y directions
dy = 1/(jmax-1)
x = np.linspace(0, 1, imax)     # Correct x coordinates
y = np.linspace(0, 1, jmax)     # Correct y coordinates
alphaP = 0.1                    # pressure under-relaxation 
alphaU = 0.9                    # velocity under-relaxation 
tol = 1e-7                      # Relaxed tolerance for animation

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

# Lists to store fields for animation
u_list = []
v_list = []
iterations = []
residuals = []

# Create callback function for saving fields during iterations
save_fields_callback = create_callback_for_animation(
    u_list, v_list, 
    iterations=iterations, 
    residuals=residuals,
    save_interval=20, 
    tol=tol
)

# Run the SIMPLE algorithm with Gauss_Seidel pressure solver
u, v, p, iteration, maxRes, divergence = simple_algorithm(
    imax, jmax, dx, dy, rho, mu, u, v, p, 
    velocity, alphaU, alphaP, max_iteration, tol,
    pressure_solver="gauss-seidel",  # Use Gauss_seidel solver
    solver_params={
        'max_iter': 1000,     # Maximum Gauss_seidel iterations
        'tolerance': 1e-5,    # Convergence tolerance
        'omega': 0.8          # Relaxation factor
    },
    callback=save_fields_callback  # Add the callback function
)

print(f"Total Iterations = {iteration}")
print(f"Number of saved frames: {len(u_list)}")

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

# 5. Create side-by-side animation with velocity magnitude and streamlines
create_side_by_side_animation(u_list, v_list, x, y, 
                            title=f'Gauss_seidel Cavity Flow (Re={Re})',
                            filename=f'gauss_seidel_cavity_Re{Re}_combined_animation.mp4',
                            fps=15, dpi=150,
                            cmap='coolwarm',
                            output_dir=None)

