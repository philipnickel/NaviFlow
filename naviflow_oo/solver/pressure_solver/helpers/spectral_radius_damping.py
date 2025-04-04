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
    from naviflow_oo.solver.pressure_solver.helpers.matrix_free import compute_Ap_product
    from naviflow_oo.solver.pressure_solver.helpers.coeff_matrix import get_coeff_mat
else:
    from ..jacobi import JacobiSolver
    from ..helpers.matrix_free import compute_Ap_product
    from ..helpers.coeff_matrix import get_coeff_mat

def find_optimal_gauss_seidel_omega(nx, ny, dx, dy, rho, d_u, d_v, num_eigenvalues=10):
    """
    Find the optimal omega value for Gauss-Seidel iteration using spectral radius analysis.
    
    This function computes the spectral radius of the Gauss-Seidel iteration matrix
    for different omega values and finds the one that minimizes it.
    
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
    num_eigenvalues : int, optional
        Number of eigenvalues to compute for spectral radius estimation
        
    Returns:
    --------
    optimal_omega : float
        Optimal omega value that minimizes the spectral radius
    spectral_radius : float
        The spectral radius at the optimal omega value
    """
    print("Computing coefficient matrix...")
    # Get the coefficient matrix
    A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
    
    # Create Gauss-Seidel iteration matrix
    def get_gauss_seidel_matrix(omega):
        """Helper function to get Gauss-Seidel iteration matrix for a given omega."""
        # Split A into D (diagonal), L (lower triangular), and U (upper triangular)
        D = A.diagonal()
        L = np.tril(A.toarray(), k=-1)
        U = np.triu(A.toarray(), k=1)
        
        # Add small regularization to avoid singular matrices
        epsilon = 1e-10
        D_reg = D + epsilon
        
        # Create the iteration matrix using a more stable approach
        def matvec(x):
            # Solve (D + omega*L)y = ((1-omega)*D - omega*U)x
            y = np.zeros_like(x)
            for i in range(len(x)):
                # Forward substitution
                sum_term = 0
                for j in range(i):
                    sum_term += L[i,j] * y[j]
                y[i] = ((1-omega)*D_reg[i]*x[i] - omega*np.sum(U[i,:]*x) - omega*sum_term) / (D_reg[i] + omega*L[i,i])
            return y
        
        return matvec
    
    # Search for optimal omega
    omega_range = np.linspace(0.0015, 0.007, 10)  # Range of omega values to try
    min_spectral_radius = float('inf')
    optimal_omega = 0.00165
    
    # Create a random vector for power iteration
    n = nx * ny
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    # Store results for plotting
    spectral_radii = []
    
    print("Searching for optimal omega...")
    for omega in tqdm(omega_range, desc="Testing omega values"):
        # Get Gauss-Seidel matrix for current omega
        M = get_gauss_seidel_matrix(omega)
        
        # Use power iteration to estimate spectral radius
        max_eig = 0
        for _ in range(20):  # Reduced number of iterations
            x_new = M(x)
            max_eig = np.linalg.norm(x_new)
            x = x_new / max_eig
        
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
    plt.title('Spectral Radius vs Omega')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, 'spectral_radius_minimization.pdf')
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Display the plot
    plt.show()
    plt.close()
    
    return optimal_omega, min_spectral_radius

def estimate_optimal_gauss_seidel_omega(nx, ny, dx, dy, rho, d_u, d_v):
    """
    Estimate the optimal omega value for Gauss-Seidel iteration using a simpler approach.
    
    This function provides a quick estimate of the optimal omega value
    based on the grid size and coefficient ratios. It's less accurate than
    find_optimal_gauss_seidel_omega but much faster.
    
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
        
    Returns:
    --------
    estimated_omega : float
        Estimated optimal omega value
    """
    print("Computing quick estimate...")
    # Get the coefficient matrix
    A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
    
    # Calculate average coefficient ratios
    D = A.diagonal()
    L = np.tril(A.toarray(), k=-1)
    U = np.triu(A.toarray(), k=1)
    
    # Calculate average ratio of off-diagonal to diagonal terms
    avg_ratio = np.mean((np.abs(L) + np.abs(U)) / np.abs(D))
    
    # Estimate optimal omega based on the ratio
    estimated_omega = 2.0 / (1.0 + np.sqrt(1.0 + avg_ratio))
    
    # Ensure omega is in a reasonable range
    estimated_omega = np.clip(estimated_omega, 0.5, 1.5)
    
    return estimated_omega

# Example usage:
if __name__ == "__main__":
    print("Starting spectral radius analysis...")
    
    # Example parameters
    nx, ny = 127, 127
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    rho = 1.0
    d_u = np.ones((nx, ny))
    d_v = np.ones((nx, ny))
    
    # Get quick estimate first (much faster)
    estimated_omega = estimate_optimal_gauss_seidel_omega(nx, ny, dx, dy, rho, d_u, d_v)
    print(f"\nQuick estimate of optimal omega: {estimated_omega:.4f}")
    
    # Find optimal omega using spectral radius analysis
    print("\nComputing detailed spectral radius analysis...")
    optimal_omega, spectral_radius = find_optimal_gauss_seidel_omega(nx, ny, dx, dy, rho, d_u, d_v)
    print(f"\nOptimal omega (spectral radius): {optimal_omega:.4f}")
    print(f"Spectral radius: {spectral_radius:.4f}")