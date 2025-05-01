"""
Helper module for finding optimal damping parameter in Jacobi iteration using spectral radius.
"""

import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots

# Add the parent directory to the Python path when running directly
if __name__ == "__main__": 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
    from naviflow_staggered.solver.pressure_solver.jacobi import JacobiSolver
    from naviflow_staggered.solver.pressure_solver.gauss_seidel import GaussSeidelSolver
else:
    from ..jacobi import JacobiSolver
    from ..gauss_seidel import GaussSeidelSolver


def find_optimal_gauss_seidel_omega_matrix_free(nx, ny, dx, dy, rho, d_u, d_v, num_iterations=20, num_random_vectors=5, method_type='red_black'):
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
    omega_min = 0.5  # Minimum omega value to test
    omega_max = 2.5   # Maximum omega value to test
    num_points = 50   # Number of points to test in that range
    
    omega_range = np.linspace(omega_min, omega_max, num_points)
    print(f"Searching for optimal omega in range: [{omega_min}, {omega_max}] with {num_points} points")
    
    min_spectral_radius = float('inf')
    optimal_omega = (omega_min + omega_max) / 2  # Start with the middle of the range
    
    # Store results for plotting
    spectral_radii = []
    
    print("Searching for optimal omega...")
    for omega in tqdm(omega_range, desc="Testing omega values"):
        # Create a Gauss-Seidel solver with the current omega
        solver = GaussSeidelSolver(tolerance=1e-10, max_iterations=1, omega=omega, method_type=method_type)
        
        # Create a zero right-hand side for power iteration
        b = np.zeros(n)
        
        # Use power iteration with multiple random vectors
        max_eig = 0.0
        
        for x in random_vectors:
            # Start with a random vector
            v = x.copy()
            
            # Power iteration to estimate the dominant eigenvalue
            for _ in range(num_iterations):
                # Apply one iteration of Gauss-Seidel
                v_2d = v.reshape((nx, ny), order='F')
                result = solver.solve(p=v_2d, b=b, nx=nx, ny=ny, dx=dx, dy=dy, rho=rho, 
                                   d_u=d_u, d_v=d_v, num_iterations=1, track_residuals=False)
                # Extract the pressure array from the result (which is a tuple)
                v_new_2d = result[0] if isinstance(result, tuple) else result
                v_new = v_new_2d.flatten('F')
                
                # Normalize to prevent overflow/underflow
                v_norm = np.linalg.norm(v_new)
                if v_norm > 0:
                    v = v_new / v_norm
            
            # Apply one final iteration to estimate eigenvalue
            v_2d = v.reshape((nx, ny), order='F')
            result = solver.solve(p=v_2d, b=b, nx=nx, ny=ny, dx=dx, dy=dy, rho=rho, 
                               d_u=d_u, d_v=d_v, num_iterations=1, track_residuals=False)
            v_new_2d = result[0] if isinstance(result, tuple) else result
            v_new = v_new_2d.flatten('F')
            
            # Calculate Rayleigh quotient for better eigenvalue approximation
            # For non-symmetric matrices, use the Rayleigh quotient with the current eigenvector
            eig = np.abs(np.dot(v_new, v) / np.dot(v, v))
            
            # Update max_eig if this vector gives a larger eigenvalue
            max_eig = max(max_eig, eig)
        
        # Store spectral radius for plotting
        spectral_radii.append(max_eig)
        
        # Update optimal omega if we found a smaller spectral radius
        if max_eig < min_spectral_radius:
            min_spectral_radius = max_eig
            optimal_omega = omega
    
    # Plot the results using standard matplotlib
    plt.style.use(['science', 'grid'])
    plt.figure(figsize=(12, 8))
    plt.plot(omega_range, spectral_radii, 'b-', label=f'Spectral Radius')
    plt.plot(optimal_omega, min_spectral_radius, 'ro', label=f'Optimal Point, $\\omega$ = {optimal_omega:.4f}, SR = {min_spectral_radius:.4f}')
    plt.xlabel('Omega ($\\omega$)')
    plt.ylabel(f'Spectral Radius')
    plt.title(f'Spectral Radius vs Omega (Matrix-Free), resolution = {nx}x{ny}, GS-type = {method_type}')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limits with some padding
    y_min = min(spectral_radii) * 0.95
    y_max = max(spectral_radii) * 1.05
    plt.ylim(y_min, y_max)
    
    plt.tight_layout(pad=2.0)  # Add more padding around the plot
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, f'SR_{method_type}_resolution_{nx}x{ny}.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
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
    omega_min = 0.5  # Minimum omega value to test
    omega_max = 2.5  # Maximum omega value to test
    num_points = 50  # Number of points to test in that range
    
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
            v = x.copy()
            
            # Power iteration to estimate the dominant eigenvalue
            for _ in range(num_iterations):
                # Apply one iteration of Jacobi
                v_2d = v.reshape((nx, ny), order='F')
                result = solver.solve(p=v_2d, b=b, nx=nx, ny=ny, dx=dx, dy=dy, rho=rho, 
                                   d_u=d_u, d_v=d_v, num_iterations=1, track_residuals=False)
                # Extract the pressure array from the result (which is a tuple)
                v_new_2d = result[0] if isinstance(result, tuple) else result
                v_new = v_new_2d.flatten('F')
                
                # Normalize to prevent overflow/underflow
                v_norm = np.linalg.norm(v_new)
                if v_norm > 0:
                    v = v_new / v_norm
            
            # Apply one final iteration to estimate eigenvalue
            v_2d = v.reshape((nx, ny), order='F')
            result = solver.solve(p=v_2d, b=b, nx=nx, ny=ny, dx=dx, dy=dy, rho=rho, 
                               d_u=d_u, d_v=d_v, num_iterations=1, track_residuals=False)
            v_new_2d = result[0] if isinstance(result, tuple) else result
            v_new = v_new_2d.flatten('F')
            
            # Calculate Rayleigh quotient for better eigenvalue approximation
            # For Jacobi, we can use this simpler approach since the matrix is closer to symmetric
            eig = np.abs(np.dot(v_new, v) / np.dot(v, v))
            
            # Update max_eig if this vector gives a larger eigenvalue
            max_eig = max(max_eig, eig)
        
        # Store spectral radius for plotting
        spectral_radii.append(max_eig)
        
        # Update optimal omega if we found a smaller spectral radius
        if max_eig < min_spectral_radius:
            min_spectral_radius = max_eig
            optimal_omega = omega
    
    # Plot the results using standard matplotlib
    plt.style.use(['science', 'grid'])
    plt.figure(figsize=(12, 8))
    plt.plot(omega_range, spectral_radii, 'b-', label='Spectral Radius')
    plt.plot(optimal_omega, min_spectral_radius, 'ro', label=f'Optimal Point, $\\omega$ = {optimal_omega:.4f}, SR = {min_spectral_radius:.4f}')
    plt.xlabel('Omega ($\\omega$)')
    plt.ylabel('Spectral Radius')
    plt.title(f'Spectral Radius vs Omega (Jacobi), resolution = {nx}x{ny}')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limits with some padding
    y_min = min(spectral_radii) * 0.95
    y_max = max(spectral_radii) * 1.05
    plt.ylim(y_min, y_max)
    
    plt.tight_layout(pad=2.0)  # Add more padding around the plot
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, f'SR_Jacobi_resolution_{nx}x{ny}.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"\nPlot saved to: {plot_path}")
    
    # Display the plot
    plt.show()
    plt.close()
    
    return optimal_omega, min_spectral_radius


# Example usage:
if __name__ == "__main__":
    print("Starting spectral radius analysis...")
    
    # Example parameters
    nx, ny = 2**7-1, 2**7-1 
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    rho = 1.0
    d_u = np.ones((nx, ny))
    d_v = np.ones((nx, ny))


    method_type = 'standard'
    # Find optimal omega using matrix-free spectral radius analysis for Gauss-Seidel
    print("\nComputing matrix-free spectral radius analysis for Gauss-Seidel...")
    optimal_omega_gs, spectral_radius_gs = find_optimal_gauss_seidel_omega_matrix_free(nx, ny, dx, dy, rho, d_u, d_v, num_random_vectors=5, method_type=method_type)
    print(f"\nOptimal omega (Gauss-Seidel): {optimal_omega_gs:.4f}")
    print(f"Spectral radius (Gauss-Seidel): {spectral_radius_gs:.4f}")
    """
    # Find optimal omega using matrix-free spectral radius analysis for Jacobi
    print("\nComputing matrix-free spectral radius analysis for Jacobi...")
    optimal_omega_jacobi, spectral_radius_jacobi = find_optimal_jacobi_omega_matrix_free(nx, ny, dx, dy, rho, d_u, d_v, num_random_vectors=5)
    print(f"\nOptimal omega (Jacobi): {optimal_omega_jacobi:.4f}")
    print(f"Spectral radius (Jacobi): {spectral_radius_jacobi:.4f}")

    """