# %% imports
import numpy as np
import matplotlib.pyplot as plt

# %% functions
# Exact solution and source term
def uexact(x, y):
    return np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)

def f(x, y):
    return -16 * np.pi**2 * ((x**2 + y**2) * np.cos(4 * np.pi * x * y) + 2 * np.sin(4 * np.pi * (x + y)))

# %% Problem setup
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 50, 50    # Number of cells in x and y directions
dx, dy = Lx / Nx, Ly / Ny  # Grid spacing

# Grid coordinates
x = np.linspace(0, Lx, Nx + 1)
y = np.linspace(0, Ly, Ny + 1)
X, Y = np.meshgrid(x, y)

# Initialize solution array
u = np.zeros((Nx + 1, Ny + 1))

# Apply Dirichlet boundary conditions
u[0, :] = uexact(0, y)       # Left boundary
u[-1, :] = uexact(Lx, y)     # Right boundary
u[:, 0] = uexact(x, 0)       # Bottom boundary
u[:, -1] = uexact(x, Ly)     # Top boundary

# Map 2D grid indices to 1D vector indices
def get_index(i, j):
    return i * (Ny + 1) + j

# Assemble the coefficient matrix A and right-hand side vector b
size = (Nx + 1) * (Ny + 1)  # Total number of unknowns
A = np.zeros((size, size))
b = np.zeros(size)

# Fill A and b
for i in range(1, Nx):
    for j in range(1, Ny):
        k = get_index(i, j)
        A[k, k] = -2 / dx**2 - 2 / dy**2  # Diagonal
        A[k, get_index(i + 1, j)] = 1 / dx**2  # Right neighbor
        A[k, get_index(i - 1, j)] = 1 / dx**2  # Left neighbor
        A[k, get_index(i, j + 1)] = 1 / dy**2  # Top neighbor
        A[k, get_index(i, j - 1)] = 1 / dy**2  # Bottom neighbor
        b[k] = f(x[i], y[j])

# Apply boundary conditions to A and b
for i in range(Nx + 1):
    for j in range(Ny + 1):
        if i == 0 or i == Nx or j == 0 or j == Ny:
            k = get_index(i, j)
            A[k, :] = 0  # Zero out the row
            A[k, k] = 1  # Set diagonal to 1
            b[k] = u[i, j]  # Set RHS to boundary value

# %% Solve
# Solve the linear system
u_flat = np.linalg.solve(A, b)

# Reshape the solution back to 2D
u = u_flat.reshape((Nx + 1, Ny + 1))

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




