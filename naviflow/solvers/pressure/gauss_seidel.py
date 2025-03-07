import numpy as np

def smooth_gauss_seidel(u: np.ndarray, 
                        omega: float, 
                        b: np.ndarray, 
                        nsmooth: int) -> np.ndarray:
    """Applies multiple iterations of relaxed Gauss-Seidel smoothing.
    
    Args:
        u (np.ndarray): Initial guess.
        omega (float): Relaxation parameter (0 < omega < 2).
        b (np.ndarray): Right-hand side vector.
        nsmooth (int): Number of smoothing iterations.
        
    Returns:
        np.ndarray: Smoothed solution.
    """
    n = np.size(u)
    grid_inner_size = int(np.sqrt(n))
    h = 1 / (grid_inner_size + 1)
    u_new = u.reshape(grid_inner_size, grid_inner_size).copy()
    b_reshaped = b.reshape(grid_inner_size, grid_inner_size)

    for _ in range(nsmooth):
        for i in range(1, u_new.shape[0] - 1):  # Loop over inner grid points (avoid boundaries)
            for j in range(1, u_new.shape[1] - 1):
            # Compute the new value of u_new[i, j] using the latest updates
                u_new[i, j] = (1 - omega) * u_new[i, j] + (omega / 4) * (
                    u_new[i + 1, j] +  # Bottom neighbor
                    u_new[i - 1, j] +  # Top neighbor
                    u_new[i, j + 1] +  # Right neighbor
                    u_new[i, j - 1] -  # Left neighbor
                    b_reshaped[i, j] * (h ** 2)  # Source term
            )

    return u_new.flatten()
