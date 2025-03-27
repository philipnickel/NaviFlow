"""
Helper module for finding optimal damping parameter in Jacobi iteration using spectral radius.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from ..jacobi import JacobiSolver
from ..helpers.matrix_free import compute_Ap_product

def find_optimal_damping(nx, ny, dx, dy, rho, d_u, d_v, num_eigenvalues=10):
    """
    Find the optimal damping parameter for Jacobi iteration using spectral radius analysis.
    
    This function computes the spectral radius of the Jacobi iteration matrix
    for different damping parameters and finds the one that minimizes it.
    
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
        Optimal damping parameter that minimizes the spectral radius
    spectral_radius : float
        The spectral radius at the optimal damping parameter
    """
    # Create coefficient arrays for the pressure correction equation
    aE = np.zeros((nx, ny))
    aE[:-1, :] = rho * d_u[1:nx, :] * dy
    
    aW = np.zeros((nx, ny))
    aW[1:, :] = rho * d_u[1:nx, :] * dy
    
    aN = np.zeros((nx, ny))
    aN[:, :-1] = rho * d_v[:, 1:ny] * dx
    
    aS = np.zeros((nx, ny))
    aS[:, 1:] = rho * d_v[:, 1:ny] * dx
    
    # Diagonal coefficients
    aP = aE + aW + aN + aS
    aP[0, 0] = 1.0  # Reference point
    
    # Avoid division by zero
    aP[aP == 0] = 1.0
    
    # Create Jacobi iteration matrix
    def get_jacobi_matrix(omega):
        """Helper function to get Jacobi iteration matrix for a given omega."""
        # Initialize matrix
        n = nx * ny
        J = np.zeros((n, n))
        
        # Fill matrix elements
        for i in range(nx):
            for j in range(ny):
                row = i * ny + j
                
                # Skip reference point
                if i == 0 and j == 0:
                    J[row, row] = 1.0
                    continue
                
                # Center point
                J[row, row] = (1 - omega) + omega * (1 - aP[i, j] / aP[i, j])
                
                # East point
                if i < nx - 1:
                    col = (i + 1) * ny + j
                    J[row, col] = -omega * aE[i, j] / aP[i, j]
                
                # West point
                if i > 0:
                    col = (i - 1) * ny + j
                    J[row, col] = -omega * aW[i, j] / aP[i, j]
                
                # North point
                if j < ny - 1:
                    col = i * ny + (j + 1)
                    J[row, col] = -omega * aN[i, j] / aP[i, j]
                
                # South point
                if j > 0:
                    col = i * ny + (j - 1)
                    J[row, col] = -omega * aS[i, j] / aP[i, j]
        
        return J
    
    # Search for optimal omega
    omega_range = np.linspace(0.1, 2.0, 20)  # Range of omega values to try
    min_spectral_radius = float('inf')
    optimal_omega = 1.0
    
    for omega in omega_range:
        # Get Jacobi matrix for current omega
        J = get_jacobi_matrix(omega)
        
        # Compute eigenvalues
        eigenvalues = eigsh(J, k=num_eigenvalues, which='LM', return_eigenvectors=False)
        
        # Compute spectral radius (maximum absolute eigenvalue)
        spectral_radius = np.max(np.abs(eigenvalues))
        
        # Update optimal omega if we found a smaller spectral radius
        if spectral_radius < min_spectral_radius:
            min_spectral_radius = spectral_radius
            optimal_omega = omega
    
    return optimal_omega, min_spectral_radius

def estimate_optimal_damping(nx, ny, dx, dy, rho, d_u, d_v):
    """
    Estimate the optimal damping parameter using a simpler approach.
    
    This function provides a quick estimate of the optimal damping parameter
    based on the grid size and coefficient ratios. It's less accurate than
    find_optimal_damping but much faster.
    
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
        Estimated optimal damping parameter
    """
    # Calculate average coefficient ratios
    aE = rho * d_u * dy
    aW = rho * d_u * dy
    aN = rho * d_v * dx
    aS = rho * d_v * dx
    aP = aE + aW + aN + aS
    
    # Calculate average ratio of off-diagonal to diagonal terms
    avg_ratio = np.mean((aE + aW + aN + aS) / aP)
    
    # Estimate optimal omega based on the ratio
    # This is a heuristic based on typical values that work well
    estimated_omega = 2.0 / (1.0 + np.sqrt(1.0 + avg_ratio))
    
    # Ensure omega is in a reasonable range
    estimated_omega = np.clip(estimated_omega, 0.5, 1.5)
    
    return estimated_omega