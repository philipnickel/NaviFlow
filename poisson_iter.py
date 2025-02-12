# %% imports
import numpy as np
import matplotlib.pyplot as plt

#%% functions
def uexact(x, y):
    return np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)

def f(x, y):
    return -16 * np.pi**2 * ((x**2 + y**2) * np.cos(4 * np.pi * x * y) + 2 * np.sin(4 * np.pi * (x + y)))


# %% Problem setup
# Domain and grid parameters
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 50, 50    # Number of cells in x and y directions
dx, dy = Lx / Nx, Ly / Ny  # Grid spacing

# Grid coordinates
x = np.linspace(0, Lx, Nx + 1)
y = np.linspace(0, Ly, Ny + 1)
j, Y = np.meshgrid(x, y)

# Initialize solution array
u = np.zeros((Nx + 1, Ny + 1))

# Apply Dirichlet boundary conditions
u[0, :] = uexact(0, y)       # Left boundary
u[-1, :] = uexact(Lx, y)     # Right boundary
u[:, 0] = uexact(x, 0)       # Bottom boundary
u[:, -1] = uexact(x, Ly)     # Top boundary

# %% Solve

# Iterative solver (Gauss-Seidel)
max_iter = 10000
tolerance = 1e-6
for iteration in range(max_iter):
    u_old = u.copy()
    for i in range(1, Nx):
        for j in range(1, Ny):
            u[i, j] = 0.25 * (
                u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - dx**2 * f(x[i], y[j])
            )
    
    # Check for convergence
    residual = np.linalg.norm(u - u_old)
    if residual < tolerance:
        print(f"Converged after {iteration} iterations.")
        break


# %% Plots 

# Compute exact solution for comparison
u_exact = uexact(X, Y)

# Plot numerical solution, exact solution and difference
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.contourf(X, Y, u, levels=50, cmap="viridis")
plt.colorbar(label="u(x, y)")
plt.title("Numerical Solution")

plt.subplot(1, 3, 2)
plt.contourf(X, Y, u_exact, levels=50, cmap="viridis")
plt.colorbar(label="u_exact(x, y)")
plt.title("Exact Solution")

plt.subplot(1, 3, 3)
plt.contourf(X, Y, u - u_exact, levels=50, cmap="viridis")
plt.colorbar(label="Difference")
plt.title("Error")

plt.tight_layout()
plt.show()

# %%
