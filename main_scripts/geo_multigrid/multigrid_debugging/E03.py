# %%

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Callable, Tuple

# %% A)

def calc_Au(u: np.ndarray) -> np.ndarray:
    """Computes the matrix-vector product Au for the discrete Laplacian operator.
    
    Implements the 5-point stencil discretization of the Laplacian operator
    without explicitly forming the matrix.
    
    Args:
        u (np.ndarray): Input vector representing the solution on a 2D grid
        
    Returns:
        np.ndarray: Result of applying the discrete Laplacian operator to u
    """
    n = np.size(u)
    grid_inner_size = int(np.sqrt(n))
    
    # Reshape u into a 2D grid
    u_grid = u.reshape((grid_inner_size, grid_inner_size))
    
    # Initialize Au with -4*u
    Au = -4 * u_grid
    
    # Add contributions from neighboring cells
    Au[1:, :] += u_grid[:-1, :]  # from below
    Au[:-1, :] += u_grid[1:, :]  # from above
    Au[:, 1:] += u_grid[:, :-1]  # from left
    Au[:, :-1] += u_grid[:, 1:]  # from right
    
    # Flatten the result back to 1D
    h = 1 / (grid_inner_size + 1)
    return -Au.flatten()/(h**2)

def u_exact(x_cord: np.ndarray, y_cord: np.ndarray) -> np.ndarray:
    """Computes the exact solution of the PDE.
    
    Args:
        x_cord (np.ndarray): x-coordinates
        y_cord (np.ndarray): y-coordinates
        
    Returns:
        np.ndarray: Exact solution values at the given coordinates
    """
    return np.sin(4*np.pi*(x_cord + y_cord)) + np.cos(4*np.pi*x_cord*y_cord)

def laplacian(x_cord: np.ndarray, y_cord: np.ndarray) -> np.ndarray:
    """Computes the Laplacian of the exact solution.
    
    Args:
        x_cord (np.ndarray): x-coordinates
        y_cord (np.ndarray): y-coordinates
        
    Returns:
        np.ndarray: Laplacian values at the given coordinates
    """
    return ( -32*np.pi**2*np.sin(4*np.pi*(x_cord + y_cord))
             - 16*np.pi**2*(y_cord**2)*np.cos(4*np.pi*x_cord*y_cord)
             - 16*np.pi**2*(x_cord**2)*np.cos(4*np.pi*x_cord*y_cord) )

def calc_b(grid_inner_size: int) -> np.ndarray:
    """Computes the right-hand side vector including boundary conditions.
    
    Args:
        grid_inner_size (int): Number of interior points in each dimension
        
    Returns:
        np.ndarray: Right-hand side vector incorporating boundary conditions
    """
    n = grid_inner_size**2
    grid_outer_size = grid_inner_size + 2
    h = 1 / (grid_outer_size - 1)

    # Create meshgrid for inner points
    x_inner, y_inner = np.meshgrid(np.arange(1, grid_inner_size + 1), np.arange(1, grid_inner_size + 1))
    x_inner = x_inner.flatten()
    y_inner = y_inner.flatten()

    # Calculate b for all inner points
    b = (h**2) * laplacian(x_inner * h, y_inner * h)

    # Create masks for boundary conditions
    left_mask = (x_inner == 1)
    right_mask = (x_inner == grid_inner_size)
    bottom_mask = (y_inner == 1)
    top_mask = (y_inner == grid_inner_size)

    # Apply boundary conditions
    b[left_mask] -= u_exact((x_inner[left_mask] - 1) * h, y_inner[left_mask] * h)
    b[right_mask] -= u_exact((x_inner[right_mask] + 1) * h, y_inner[right_mask] * h)
    b[bottom_mask] -= u_exact(x_inner[bottom_mask] * h, (y_inner[bottom_mask] - 1) * h)
    b[top_mask] -= u_exact(x_inner[top_mask] * h, (y_inner[top_mask] + 1) * h)

    return b/(h**2)

def calc_eigenvalue(omega: float, h: float, p: float, q: float) -> float:
    """Computes eigenvalue for the iteration matrix of the relaxed Jacobi method.
    
    Args:
        omega (float): Relaxation parameter
        h (float): Grid spacing
        p (float) 
        q (float) 
        
    Returns:
        float: Eigenvalue for the given parameters
    """
    return (1 - omega) + omega * (((np.cos(p * np.pi * h) ) + np.cos(q * np.pi * h) ))

def calc_max_eigenvalue(grid_inner_size: int, omegas: np.ndarray) -> np.ndarray:
    """Computes maximum eigenvalues for different relaxation parameters.
    
    Args:
        grid_inner_size (int): Number of interior points in each dimension
        omegas (np.ndarray): Array of relaxation parameters to test
        
    Returns:
        np.ndarray: Maximum eigenvalue for each omega value
    """
    h = 1 / (grid_inner_size + 1)
    max_eigenvalues = np.zeros((omegas.size))
    ps = np.arange(int(grid_inner_size / 2), grid_inner_size + 1)[:, np.newaxis]
    qs = np.arange(int(grid_inner_size / 2), grid_inner_size + 1)[np.newaxis, :]
    for i, omega in np.ndenumerate(omegas):
        max_eigenvalues[i] = np.max(np.abs(calc_eigenvalue(omega, h, ps, qs)))
    
    return max_eigenvalues

def calc_Gu(u: np.ndarray) -> np.ndarray:
    """Computes the Jacobi iteration matrix applied to a vector.
    
    Args:
        u (np.ndarray): Input vector
        
    Returns:
        np.ndarray: Result of applying the Jacobi iteration matrix
    """
    n = np.size(u)
    grid_inner_size = int(np.sqrt(n))
    u_reshaped = u.reshape(grid_inner_size, grid_inner_size)
    
    Gu = np.zeros_like(u_reshaped)
    
    # Add contributions from neighbors
    Gu[1:, :] += u_reshaped[:-1, :] / 4
    Gu[:-1, :] += u_reshaped[1:, :] / 4
    Gu[:, 1:] += u_reshaped[:, :-1] / 4
    Gu[:, :-1] += u_reshaped[:, 1:] / 4
    
    h = 1 / (grid_inner_size + 1)
    return -Gu.flatten() / (h ** 2)

def smooth(u: np.ndarray, 
          omega: float, 
          b: np.ndarray, 
          nsmooth: int) -> np.ndarray:
    """Applies multiple iterations of relaxed Jacobi smoothing.
    
    Args:
        u (np.ndarray): Initial guess
        omega (float): Relaxation parameter (0 < omega < 2)
        b (np.ndarray): Right-hand side vector
        nsmooth (int): Number of smoothing iterations
        
    Returns:
        np.ndarray: Smoothed solution
    """
    n = np.size(u)
    grid_inner_size = int(np.sqrt(n))
    h = 1 / (grid_inner_size + 1)
    u_new = u.reshape(grid_inner_size, grid_inner_size).copy()
    b_reshaped = b.reshape(grid_inner_size, grid_inner_size)
    
    for iteration in range(nsmooth):
        # Create a new array for the next iteration, similar to jacobi_smooth
        u_next = (1 - omega) * u_new
        
        # Add contributions from neighbors
        u_next[:, :-1] += omega / 4 * u_new[:, 1:]
        u_next[:, 1:] += omega / 4 * u_new[:, :-1]
        u_next[1:, :] += omega / 4 * u_new[:-1, :]
        u_next[:-1, :] += omega / 4 * u_new[1:, :]
        
        # Update with the scaled right-hand side (F)
        u_next -= omega * b_reshaped * (h ** 2) / 4
        
        # Calculate residuals
        residual = np.linalg.norm(
            (-4 * u_new + np.roll(u_new, 1, axis=0) + np.roll(u_new, -1, axis=0) +
             np.roll(u_new, 1, axis=1) + np.roll(u_new, -1, axis=1)) / (h ** 2) - b_reshaped,
            ord=np.inf
        )
        
        # Move to the next state
        u_new = u_next
    
    return u_new.flatten()

def main() -> None:
 
    # Problem settings
    grid_inner_size = 62
    grid_outer_size = grid_inner_size + 2
    n = grid_inner_size**2
    h = 1 / (grid_outer_size - 1)

    iterations = []
    def save_iteration(xk):
        iterations.append(np.copy(xk))
    
    Au = LinearOperator((n, n), matvec=calc_Au)
    b = calc_b(grid_inner_size)
    u_sol, info = cg(-Au, b,tol=1e-12, callback=save_iteration)

    # Calculate residuals
    residuals = [np.linalg.norm(b + calc_Au(x), ord=np.inf) for x in iterations]

    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(residuals) + 1), residuals, 'b-', label='Residual norm')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title('Convergence History of Conjugate Gradient Method')
    plt.legend()
    plt.grid(True)
    plt.savefig("Convergence_History_Conjugate_Gradient_Method.pdf")
    plt.show()

    # Estimate order of convergence
    orders = []
    for i in range(2, len(residuals)):
        order = np.log(residuals[i] / residuals[i-1]) / np.log(residuals[i-1] / residuals[i-2])
        orders.append(order)

    avg_order = np.mean(orders)
    print(f"Estimated average order of convergence: {avg_order:.4f}")
    print(f"Rate of convergence is exponential")

    # Plots 
    xs_inner = np.linspace(h, 1 - h, grid_inner_size)
    ys_inner = np.linspace(h, 1 - h, grid_inner_size)
    X_inner, Y_inner = np.meshgrid(xs_inner, ys_inner)
    Z_approx = u_sol.reshape(grid_inner_size, grid_inner_size)
    Z_exact = u_exact(X_inner, Y_inner)

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.contourf(X_inner, Y_inner, Z_approx)
    plt.colorbar()
    plt.title("Approx CG")
    plt.subplot(1,3,2)
    plt.contourf(X_inner, Y_inner, Z_exact)
    plt.colorbar()
    plt.title("Exact")
    plt.subplot(1,3,3)
    plt.contourf(X_inner, Y_inner, Z_exact - Z_approx)
    plt.colorbar()
    plt.title("Difference")
    plt.tight_layout()
    plt.show()

    # Eigenvalue calculations
    omegas = np.linspace(0, 2, 100)
    max_eigenvalues = calc_max_eigenvalue(grid_inner_size, omegas)
    plt.plot(omegas, max_eigenvalues)
    index_min_eigenvalue = np.argmin(max_eigenvalues)
    print(f"min = {max_eigenvalues[index_min_eigenvalue]} at omega = {omegas[index_min_eigenvalue]}")
    plt.title("Optimal relaxation parameter")
    plt.xlabel("$\omega$")
    plt.ylabel("$\lambda_{max}$")
    plt.savefig("figures/E03b_optimal_relax.pdf")
    plt.show()

    # Matrix-free relaxed Jacobi iteration
    #iteration_ranges = [1, 2, 3, 10, 20, 30]
    iteration_ranges = np.arange(1, 200, 1)
    max_differences = []
    for iterations in iteration_ranges:
        u = np.zeros(n)  # Reset u for each iteration range
        u = smooth(u, 0.5, b, iterations)  
            
        U_approx = u.reshape(grid_inner_size, grid_inner_size)
        difference = np.abs(Z_exact - U_approx)
        max_differences.append(np.linalg.norm(difference, ord=np.inf))

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_ranges, max_differences, 'r-', label='Inf Norm')
    plt.xlabel('Number of Iterations')
    plt.ylabel('$||u - u_h||_{\infty}$')
    plt.title('Convergence of Solution')
    plt.legend()
    plt.grid(True)
    plt.savefig("Convergence_of_Jacobi.pdf")
    plt.show()

if __name__ == "__main__":
    main()





# %%
