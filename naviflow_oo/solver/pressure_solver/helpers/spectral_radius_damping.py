"""
Helper module for finding optimal damping parameter in Jacobi iteration using spectral radius.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the parent directory to the Python path when running directly
if __name__ == "__main__": 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
    from naviflow_oo.solver.pressure_solver.jacobi import JacobiSolver
    from naviflow_oo.solver.pressure_solver.gauss_seidel import GaussSeidelSolver
    from naviflow_oo.solver.pressure_solver.helpers.matrix_free import compute_Ap_product
    from naviflow_oo.solver.pressure_solver.helpers.coeff_matrix import get_coeff_mat
else:
    from ..jacobi import JacobiSolver
    from ..gauss_seidel import GaussSeidelSolver
    from ..helpers.matrix_free import compute_Ap_product
    from ..helpers.coeff_matrix import get_coeff_mat


def find_optimal_gauss_seidel_omega_matrix_free(nx, ny, dx, dy, rho, d_u, d_v, num_iterations=20, num_random_vectors=5):
    """
    Find the optimal omega value for matrix-free Gauss-Seidel iteration using spectral radius analysis.
    
    This function uses power iteration with multiple random vectors to estimate the spectral radius 
    of the iteration matrix for different omega values and finds the one that minimizes it.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    rho : float
        Fluid density
    d_u, d_v : ndarray
        Momentum equation coefficients
    num_iterations : int, optional
        Number of power iterations to perform for spectral radius estimation
    num_random_vectors : int, optional
        Number of random vectors to use for power iteration
        
    Returns:
    --------
    optimal_omega : float
        Optimal omega value that minimizes the spectral radius
    spectral_radius : float
        The spectral radius at the optimal omega value
    """
    print("Setting up for matrix-free spectral radius analysis...")
    
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    print(f"Using {num_random_vectors} random vectors for robust spectral radius estimation")
    
    # Create multiple random vectors for power iteration
    n = nx * ny
    random_vectors = []
    for i in range(num_random_vectors):
        x = np.random.rand(n)
        x = x / np.linalg.norm(x)
        random_vectors.append(x)
    
    # Search for optimal omega
    # MODIFIED RANGE: Change these values to adjust the search range
    omega_min = 0  # Minimum omega value to test
    omega_max = 1   # Maximum omega value to test
    num_points = 100   # Number of points to test in that range
    
    omega_range = np.linspace(omega_min, omega_max, num_points)
    print(f"Searching for optimal omega in range: [{omega_min}, {omega_max}] with {num_points} points")
    
    min_spectral_radius = float('inf')
    optimal_omega = (omega_min + omega_max) / 2  # Start with the middle of the range
    
    # Store results for plotting
    spectral_radii = []
    
    print("Searching for optimal omega...")
    for omega in tqdm(omega_range, desc="Testing omega values"):
        # Create a Gauss-Seidel solver with the current omega
        solver = GaussSeidelSolver(tolerance=1e-10, max_iterations=1, omega=omega)
        
        # Create a zero right-hand side for power iteration
        b = np.zeros(n)
        
        # Use power iteration with multiple random vectors
        max_eig = 0.0
        
        for x in random_vectors:
            # Start with a random vector
            p = x.copy()
            
            # Apply one iteration of Gauss-Seidel
            p_2d = p.reshape((nx, ny), order='F')
            p_2d = solver.solve(p=p_2d, b=b, nx=nx, ny=ny, dx=dx, dy=dy, rho=rho, 
                               d_u=d_u, d_v=d_v, num_iterations=1, track_residuals=False)
            
            # Compute the action of the iteration matrix on x
            # This is equivalent to M*x where M is the iteration matrix
            Mx = p_2d.flatten('F')
            
            # Compute the spectral radius as the norm of Mx
            eig = np.linalg.norm(Mx)
            
            # Update max_eig if this vector gives a larger eigenvalue
            max_eig = max(max_eig, eig)
        
        # Store spectral radius for plotting
        spectral_radii.append(max_eig)
        
        # Update optimal omega if we found a smaller spectral radius
        if max_eig < min_spectral_radius:
            min_spectral_radius = max_eig
            optimal_omega = omega
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(omega_range, spectral_radii, 'b-', label='Spectral Radius')
    plt.plot(optimal_omega, min_spectral_radius, 'ro', label='Optimal Point')
    plt.xlabel('Omega (ω)')
    plt.ylabel('Spectral Radius')
    plt.title('Spectral Radius vs Omega (Matrix-Free)')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, 'spectral_radius_minimization_matrix_free.pdf')
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Display the plot
    plt.show()
    plt.close()
    
    return optimal_omega, min_spectral_radius

def find_optimal_jacobi_omega_matrix_free(nx, ny, dx, dy, rho, d_u, d_v, num_iterations=20, num_random_vectors=5):
    """
    Find the optimal omega value for matrix-free Jacobi iteration using spectral radius analysis.
    
    This function uses power iteration with multiple random vectors to estimate the spectral radius 
    of the Jacobi iteration matrix for different omega values and finds the one that minimizes it.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    rho : float
        Fluid density
    d_u, d_v : ndarray
        Momentum equation coefficients
    num_iterations : int, optional
        Number of power iterations to perform for spectral radius estimation
    num_random_vectors : int, optional
        Number of random vectors to use for power iteration
        
    Returns:
    --------
    optimal_omega : float
        Optimal omega value that minimizes the spectral radius
    spectral_radius : float
        The spectral radius at the optimal omega value
    """
    print("Setting up for matrix-free Jacobi spectral radius analysis...")
    
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    print(f"Using {num_random_vectors} random vectors for robust spectral radius estimation")
    
    # Create multiple random vectors for power iteration
    n = nx * ny
    random_vectors = []
    for i in range(num_random_vectors):
        x = np.random.rand(n)
        x = x / np.linalg.norm(x)
        random_vectors.append(x)
    
    # Search for optimal omega
    # MODIFIED RANGE: Change these values to adjust the search range
    omega_min = 0.75  # Minimum omega value to test
    omega_max = 0.8  # Maximum omega value to test
    num_points = 100000   # Number of points to test in that range
    
    omega_range = np.linspace(omega_min, omega_max, num_points)
    print(f"Searching for optimal omega in range: [{omega_min}, {omega_max}] with {num_points} points")
    
    min_spectral_radius = float('inf')
    optimal_omega = (omega_min + omega_max) / 2  # Start with the middle of the range
    
    # Store results for plotting
    spectral_radii = []
    
    print("Searching for optimal omega...")
    for omega in tqdm(omega_range, desc="Testing omega values"):
        # Create a Jacobi solver with the current omega
        solver = JacobiSolver(tolerance=1e-10, max_iterations=1, omega=omega)
        
        # Create a zero right-hand side for power iteration
        b = np.zeros(n)
        
        # Use power iteration with multiple random vectors
        max_eig = 0.0
        
        for x in random_vectors:
            # Start with a random vector
            p = x.copy()
            
            # Apply one iteration of Jacobi
            p_2d = p.reshape((nx, ny), order='F')
            p_2d = solver.solve(p=p_2d, b=b, nx=nx, ny=ny, dx=dx, dy=dy, rho=rho, 
                               d_u=d_u, d_v=d_v, num_iterations=1, track_residuals=False)
            
            # Compute the action of the iteration matrix on x
            # This is equivalent to M*x where M is the iteration matrix
            Mx = p_2d.flatten('F')
            
            # Compute the spectral radius as the norm of Mx
            eig = np.linalg.norm(Mx)
            
            # Update max_eig if this vector gives a larger eigenvalue
            max_eig = max(max_eig, eig)
        
        # Store spectral radius for plotting
        spectral_radii.append(max_eig)
        
        # Update optimal omega if we found a smaller spectral radius
        if max_eig < min_spectral_radius:
            min_spectral_radius = max_eig
            optimal_omega = omega
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(omega_range, spectral_radii, 'b-', label='Spectral Radius')
    plt.plot(optimal_omega, min_spectral_radius, 'ro', label='Optimal Point')
    plt.xlabel('Omega (ω)')
    plt.ylabel('Spectral Radius')
    plt.title('Spectral Radius vs Omega (Jacobi)')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, 'spectral_radius_minimization_jacobi.pdf')
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Display the plot
    plt.show()
    plt.close()
    
    return optimal_omega, min_spectral_radius

# Example usage:
if __name__ == "__main__":
    print("Starting spectral radius analysis...")
    
    # Example parameters
    nx, ny = 127, 127
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    rho = 1.0
    d_u = np.ones((nx, ny))
    d_v = np.ones((nx, ny))
    
    # Find optimal omega using matrix-free spectral radius analysis for Gauss-Seidel
    print("\nComputing matrix-free spectral radius analysis for Gauss-Seidel...")
    optimal_omega_gs, spectral_radius_gs = find_optimal_gauss_seidel_omega_matrix_free(nx, ny, dx, dy, rho, d_u, d_v, num_random_vectors=5)
    print(f"\nOptimal omega (Gauss-Seidel): {optimal_omega_gs:.4f}")
    print(f"Spectral radius (Gauss-Seidel): {spectral_radius_gs:.4f}")