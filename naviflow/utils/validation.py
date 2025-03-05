import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import inspect


def check_divergence_free(imax, jmax, dx, dy, u, v):
    """Check if velocity field is divergence free."""
    # Fully vectorized implementation
    i_indices = np.arange(imax)
    j_indices = np.arange(jmax)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    divergence = (u[i_grid+1, j_grid] - u[i_grid, j_grid])/dx + \
                 (v[i_grid, j_grid+1] - v[i_grid, j_grid])/dy
    
    return divergence


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


def _get_caller_directory():
    """
    Get the directory of the script that called the function.
    
    Returns:
    --------
    str : The directory of the calling script
    """
    # Walk up the stack to find the first frame that's not in this file
    # and not in another utility module
    current_file = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file)
    
    for frame in inspect.stack():
        caller_file = frame.filename
        # Skip frames from this file or other utility modules
        if caller_file != current_file and not caller_file.startswith(utils_dir):
            # Found a frame from outside the utils directory
            caller_dir = os.path.dirname(os.path.abspath(caller_file))
            return caller_dir
    
    # Fallback: use current working directory
    return os.getcwd()


def _ensure_output_directory(filename, output_dir=None):
    """
    Ensure the output directory exists, creating it if necessary.
    If output_dir is None, uses the directory of the calling script.
    
    Parameters:
    -----------
    filename : str
        The filename where the data will be saved
    output_dir : str, optional
        The directory where to save the output
        
    Returns:
    --------
    str : The full path where the file should be saved
    """
    if not filename:
        return None
    
    # If no output_dir is provided, use the caller's directory + 'results'
    if output_dir is None:
        caller_dir = _get_caller_directory()
        output_dir = os.path.join(caller_dir, 'results')
    
    # Create full path
    full_path = os.path.join(output_dir, os.path.basename(filename))
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    return full_path


def compare_with_ghia(v, x, Re, j_center=None, title=None, filename=None, 
                     show=True, figsize=(10, 6), save_data=False, data_filename=None,
                     output_dir=None):
    """
    Compare simulation results with Ghia et al. benchmark data.
    
    Parameters:
    -----------
    v : ndarray
        v-velocity field
    x : ndarray
        x-coordinates
    Re : float
        Reynolds number
    j_center : int, optional
        Index for the horizontal centerline. If None, uses the middle of the domain.
    title : str, optional
        Plot title
    filename : str, optional
        If provided, saves the figure to this filename
    show : bool, optional
        Whether to display the plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_data : bool, optional
        Whether to save the comparison data to a file
    data_filename : str, optional
        Filename for saving comparison data
    output_dir : str, optional
        Directory where to save outputs. If None, uses current directory.
    
    Returns:
    --------
    dict
        Dictionary containing error metrics and comparison data
    """
    # Get the appropriate Ghia data based on Reynolds number
    ghia_data = BenchmarkData.get_ghia_data(Re)
    if ghia_data is None:
        print(f"Warning: No benchmark data available for Re={Re}. Using closest available data.")
        ghia_data = get_closest_ghia_data(Re)
    
    ghia_x = ghia_data['x']
    ghia_v = ghia_data['v']
    
    # Extract v-velocity along horizontal centerline
    imax = v.shape[0]
    jmax = v.shape[1] - 1  # Assuming staggered grid
    
    if j_center is None:
        j_center = jmax // 2  # Center index for y-direction
    
    # Interpolate v-velocity to cell centers along the centerline
    v_centerline = np.zeros(imax)
    for i in range(imax):
        v_centerline[i] = 0.5 * (v[i, j_center] + v[i, j_center+1])
    
    # Normalize x-coordinates for comparison
    x_normalized = np.linspace(0, 1, imax)
    
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
        # Ensure output directory exists and get full path
        full_path = _ensure_output_directory(filename, output_dir)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
    
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
        
        full_data_path = _ensure_output_directory(data_filename, output_dir)
        comparison_data = np.column_stack((ghia_x, ghia_v, v_interp, abs_error))
        header = "x_position, ghia_v, simulation_v, abs_error"
        np.savetxt(full_data_path, comparison_data, header=header)
        print(f"Comparison data saved to {full_data_path}")
    
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
    Calculate the divergence of the velocity field.
    
    Parameters:
    -----------
    u : ndarray
        x-velocity component
    v : ndarray
        y-velocity component
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    
    Returns:
    --------
    ndarray
        Divergence field
    """
    imax = u.shape[0] - 1
    jmax = v.shape[1] - 1
    
    divergence = np.zeros((imax, jmax))
    
    # Vectorized calculation
    i_indices = np.arange(imax)
    j_indices = np.arange(jmax)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    divergence = (u[i_grid+1, j_grid] - u[i_grid, j_grid])/dx + \
                 (v[i_grid, j_grid+1] - v[i_grid, j_grid])/dy
    
    return divergence


def get_ghia_data(Re):
    """
    Get Ghia et al. benchmark data for a specific Reynolds number.
    
    Parameters:
    -----------
    Re : float
        Reynolds number
    
    Returns:
    --------
    dict or None
        Dictionary containing benchmark data or None if not available
    """
    # Ghia data for different Reynolds numbers
    ghia_data = {
        100: {
            'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                           0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
            'v': np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, 
                          -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 
                          0.10091, 0.09233, 0.00000])
        },
        400: {
            'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                           0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
            'v': np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, 
                          -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 
                          0.19713, 0.18360, 0.00000])
        },
        1000: {
            'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                           0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
            'v': np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, 
                          -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 
                          0.29012, 0.27485, 0.00000])
        },
        # Add more Reynolds numbers as needed
    }
    
    # Round to nearest integer for lookup
    Re_int = int(round(Re))
    
    return ghia_data.get(Re_int)


def get_closest_ghia_data(Re):
    """
    Get the closest available Ghia et al. benchmark data.
    
    Parameters:
    -----------
    Re : float
        Reynolds number
    
    Returns:
    --------
    dict
        Dictionary containing benchmark data
    """
    available_Re = [100, 400, 1000]
    closest_Re = min(available_Re, key=lambda x: abs(x - Re))
    print(f"Using benchmark data for Re={closest_Re} (requested Re={Re})")
    return get_ghia_data(closest_Re)
