import numpy as np
from naviflow import *
import matplotlib.pyplot as plt


# Grid size and other parameters
imax = 129                      # grid size in x-direction
jmax = 129                      # grid size in y-direction
max_iteration = 5000           # Reduced for faster animation
maxRes = 1000
iteration = 0
Re = 3200                       # Reynolds number
velocity = 1                    # lid velocity
rho = 1                         # density
mu = rho * velocity * 1.0 / Re  # viscosity calculated from Reynolds number
dx = 1/(imax-1)                 # dx,dy cell sizes along x and y directions
dy = 1/(jmax-1)
x = np.linspace(0, 1, imax)     # Correct x coordinates
y = np.linspace(0, 1, jmax)     # Correct y coordinates
alphaP = 0.1                    # pressure under-relaxation 
alphaU = 0.9                    # velocity under-relaxation 
tol = 1e-5                      # Relaxed tolerance for animation

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
    save_interval=10, 
    tol=tol
)

# Run the SIMPLE algorithm with Jacobi pressure solver
u, v, p, iteration, maxRes, divergence = simple_algorithm(
    imax, jmax, dx, dy, rho, mu, u, v, p, 
    velocity, alphaU, alphaP, max_iteration, tol,
    pressure_solver="jacobi",  # Use Jacobi solver
    solver_params={
        'max_iter': 1000,     # Maximum Jacobi iterations
        'tolerance': 1e-5,    # Convergence tolerance
        'omega': 0.8          # Relaxation factor
    },
    callback=save_fields_callback  # Add the callback function
)

print(f"Total Iterations = {iteration}")
print(f"Number of saved frames: {len(u_list)}")

# Visualization and validation using utility functions
# 1. Plot velocity field
plot_velocity_field(u, v, x, y, 
                   title=f'Velocity Field (Re={Re})',
                   filename=f'velocity_field_Re{Re}_jacobi.png',
                   show=False)

# 2. Plot streamlines with pressure as background
plot_streamlines(u, v, x, y, 
                title=f'Streamlines (Re={Re})',
                filename=f'streamlines_Re{Re}_jacobi.png',
                background_field=p,
                show=False)

# 3. Validate against Ghia benchmark
error_metrics = compare_with_ghia(v, x, Re, 
                                filename=f'validation_Re{Re}_jacobi.png',
                                save_data=False,
                                show=False)

# 4. Check for mass conservation
div = calculate_divergence(u, v, dx, dy)
max_div = np.max(np.abs(div))
print(f"Maximum absolute divergence: {max_div:.6e}")

# 5. Create velocity magnitude animation
create_animation(u_list, v_list, x, y, 
                title=f'Jacobi Cavity Flow (Re={Re})',
                filename=f'jacobi_cavity_Re{Re}_velocity_magnitude.mp4',
                fps=20, dpi=150,
                field_type='magnitude',
                output_dir=None)

# 6. Create streamline animation
create_streamline_animation(u_list, v_list, x, y, 
                          title=f'Jacobi Cavity Flow (Re={Re})',
                          filename=f'velocity_streamlines_Re{Re}_jacobi_animation.mp4',
                          fps=20, dpi=150,
                          output_dir=None)

