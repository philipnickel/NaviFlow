import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class BenchmarkData:
    """Class to store benchmark data for comparison."""
    
    # Ghia et al. (1982) data for Re = 100
    GHIA_RE_100 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                      0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, 
                      -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 
                      0.10091, 0.09233, 0.00000])
    }
    
    # Ghia et al. (1982) data for Re = 400
    GHIA_RE_400 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                      0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, 
                      -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 
                      0.19713, 0.18360, 0.00000])
    }
    
    # Ghia et al. (1982) data for Re = 1000
    GHIA_RE_1000 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                       0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, 
                       -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 
                       0.29012, 0.27485, 0.00000])
    }
    
    @classmethod
    def get_ghia_data(cls, Re):
        """Get Ghia et al. data for given Reynolds number."""
        if Re <= 100:
            return cls.GHIA_RE_100
        elif Re <= 400:
            return cls.GHIA_RE_400
        else:
            return cls.GHIA_RE_1000


def compare_with_ghia(v, x, Re, j_center=None, title=None, filename=None, 
                     show=True, figsize=(10, 6), save_data=False, data_filename=None):
    """
    Compare v-velocity along horizontal centerline with Ghia et al. benchmark.
    
    Parameters:
    -----------
    v : ndarray
        v-velocity field
    x : ndarray
        x-coordinates
    Re : float
        Reynolds number
    j_center : int, optional
        Index for centerline. If None, uses middle of domain.
    title : str, optional
        Plot title
    filename : str, optional
        If provided, saves the figure to this filename
    show : bool, optional
        Whether to display the plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_data : bool, optional
        Whether to save comparison data to a file
    data_filename : str, optional
        Filename for saving comparison data
        
    Returns:
    --------
    dict : 
        Dictionary with error metrics
    """
    # Get benchmark data
    ghia_data = BenchmarkData.get_ghia_data(Re)
    ghia_x = ghia_data['x']
    ghia_v = ghia_data['v']
    
    # Define centerline if not provided
    if j_center is None:
        j_center = v.shape[1] // 2
    
    # Interpolate v-velocity to cell centers along the centerline
    v_centerline = np.zeros(len(x))
    for i in range(len(x)):
        if i < v.shape[0]:  # Check if index is within bounds
            v_centerline[i] = 0.5 * (v[i, j_center] + v[i, j_center+1])
    
    # Normalize x-coordinates for comparison
    x_normalized = np.linspace(0, 1, len(x))
    
    # Create the comparison plot
    plt.figure(figsize=figsize)
    plt.plot(x_normalized, v_centerline, 'b-', linewidth=2, label='Current Simulation')
    plt.plot(ghia_x, ghia_v, 'ro--', markersize=6, label='Ghia et al. (1982)')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('v-velocity')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Comparison with Ghia et al. Benchmark Data (Re = {Re:.0f})')
    
    plt.legend()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Calculate error metrics
    interp_func = interp1d(x_normalized, v_centerline, kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
    v_interp = interp_func(ghia_x)
    
    abs_error = np.abs(v_interp - ghia_v)
    mean_abs_error = np.mean(abs_error)
    max_abs_error = np.max(abs_error)
    rmse = np.sqrt(np.mean((v_interp - ghia_v)**2))
    
    # Optional: save comparison data
    if save_data:
        if data_filename is None:
            data_filename = f'ghia_comparison_Re{Re:.0f}.txt'
        
        comparison_data = np.column_stack((ghia_x, ghia_v, v_interp, abs_error))
        header = "x_position, ghia_v, simulation_v, abs_error"
        np.savetxt(data_filename, comparison_data, header=header)
        print(f"Comparison data saved to {data_filename}")
    
    # Print and return error metrics
    print(f"Validation against Ghia et al. (1982) for Re = {Re:.0f}:")
    print(f"Mean Absolute Error: {mean_abs_error:.6f}")
    print(f"Maximum Absolute Error: {max_abs_error:.6f}")
    print(f"Root Mean Square Error: {rmse:.6f}")
    
    return {
        'mean_abs_error': mean_abs_error,
        'max_abs_error': max_abs_error,
        'rmse': rmse,
        'v_centerline': v_centerline,
        'v_interp': v_interp,
        'x_positions': x_normalized
    }


def calculate_divergence(u, v, dx, dy):
    """
    Calculate the divergence of the velocity field to check mass conservation.
    
    Parameters:
    -----------
    u : ndarray
        x-velocity component (staggered grid)
    v : ndarray
        y-velocity component (staggered grid)
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    
    Returns:
    --------
    div : ndarray
        Divergence of velocity field at cell centers
    """
    imax, jmax = u.shape[0]-1, v.shape[1]-1
    div = np.zeros((imax, jmax))
    
    for i in range(imax):
        for j in range(jmax):
            div[i, j] = (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dy
    
    return div
