"""
Tools for plotting fields, streamlines, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import inspect
import scienceplots
from scipy.interpolate import interp1d
from .validation import BenchmarkData, get_closest_ghia_data


plt.style.use('science')

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
        The filename where the plot will be saved
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


def plot_velocity_field(u, v, x, y, title=None, filename=None, cmap='coolwarm', 
                        show=True, figsize=(8, 6), colorbar=True, levels=50,
                        output_dir=None):
    """
    Plot a contour of the velocity magnitude.
    
    Parameters:
    -----------
    u : ndarray
        x-velocity component
    v : ndarray
        y-velocity component
    x : ndarray
        x-coordinates
    y : ndarray
        y-coordinates
    title : str, optional
        Plot title
    filename : str, optional
        If provided, saves the figure to this filename
    cmap : str, optional
        Colormap to use
    show : bool, optional
        Whether to display the plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    colorbar : bool, optional
        Whether to show the colorbar
    levels : int, optional
        Number of contour levels
    output_dir : str, optional
        Directory where to save the output. If None, uses 'results' in the calling script's directory.
    """
    # For staggered grid, u is defined at i+1/2,j and v at i,j+1/2
    # We need to interpolate to get values at cell centers (i,j)
    
    # Get the dimensions for the cell centers
    nx = len(x)
    ny = len(y)
    
    # Create arrays for velocity at cell centers
    u_centers = np.zeros((nx, ny))
    v_centers = np.zeros((nx, ny))
    
    # Interpolate u to cell centers - handle staggered grid
    if u.shape[0] > nx:  # Staggered grid for u
        for j in range(min(ny, u.shape[1])):
            for i in range(nx):
                u_centers[i, j] = 0.5 * (u[i, j] + u[i+1, j])
    else:
        # Copy values directly if dimensions match
        u_centers[:u.shape[0], :u.shape[1]] = u[:nx, :ny]
    
    # Interpolate v to cell centers - handle staggered grid
    if v.shape[1] > ny:  # Staggered grid for v
        for i in range(min(nx, v.shape[0])):
            for j in range(ny):
                v_centers[i, j] = 0.5 * (v[i, j] + v[i, j+1])
    else:
        # Copy values directly if dimensions match
        v_centers[:v.shape[0], :v.shape[1]] = v[:nx, :ny]
    
    # Calculate velocity magnitude at cell centers
    u_mag = np.sqrt(u_centers**2 + v_centers**2)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create mesh grid for plotting
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Plot contour
    contour = plt.contourf(X, Y, u_mag, levels=levels, cmap=cmap)
    
    if colorbar:
        plt.colorbar(label='Velocity magnitude')
    
    plt.xlim(0, max(x))
    plt.ylim(0, max(y))
    
    if title:
        plt.title(title)
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    if filename:
        # Ensure output directory exists and get full path
        full_path = _ensure_output_directory(filename, output_dir)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return plt.gcf()


def plot_streamlines(u, v, x, y, title=None, filename=None, density=1.0, color='black',
                    show=True, figsize=(8, 6), background_field=None, cmap='coolwarm',
                    output_dir=None):
    """
    Plot streamlines of the velocity field.
    
    Parameters:
    -----------
    u : ndarray
        x-velocity component
    v : ndarray
        y-velocity component
    x : ndarray
        x-coordinates
    y : ndarray
        y-coordinates
    title : str, optional
        Plot title
    filename : str, optional
        If provided, saves the figure to this filename
    density : float, optional
        Density of streamlines
    color : str, optional
        Color of streamlines if no background field is provided
    show : bool, optional
        Whether to display the plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    background_field : ndarray, optional
        Field to use as background color (e.g., pressure or vorticity)
    cmap : str, optional
        Colormap to use for background field
    output_dir : str, optional
        Directory where to save the output. If None, uses 'results' in the calling script's directory.
    """
    # Get the dimensions for the cell centers
    nx = len(x)
    ny = len(y)
    
    # Create arrays for velocity at cell centers
    u_centers = np.zeros((nx, ny))
    v_centers = np.zeros((nx, ny))
    
    # Interpolate u to cell centers - handle staggered grid
    if u.shape[0] > nx:  # Staggered grid for u
        for j in range(min(ny, u.shape[1])):
            for i in range(nx):
                u_centers[i, j] = 0.5 * (u[i, j] + u[i+1, j])
    else:
        # Copy values directly if dimensions match
        u_centers[:u.shape[0], :u.shape[1]] = u[:nx, :ny]
    
    # Interpolate v to cell centers - handle staggered grid
    if v.shape[1] > ny:  # Staggered grid for v
        for i in range(min(nx, v.shape[0])):
            for j in range(ny):
                v_centers[i, j] = 0.5 * (v[i, j] + v[i, j+1])
    else:
        # Copy values directly if dimensions match
        v_centers[:v.shape[0], :v.shape[1]] = v[:nx, :ny]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create mesh grid for plotting
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # If background field is provided, plot it as contour
    if background_field is not None:
        # Ensure background field has same dimensions as the mesh grid
        if background_field.shape != X.shape:
            # Create a properly sized background field
            bg = np.zeros_like(X)
            # Copy as much of the original field as possible
            min_i = min(bg.shape[0], background_field.shape[0])
            min_j = min(bg.shape[1], background_field.shape[1])
            bg[:min_i, :min_j] = background_field[:min_i, :min_j]
        else:
            bg = background_field
            
        plt.contourf(X, Y, bg, cmap=cmap, levels=50)
        plt.colorbar()
    
    # Transpose for streamplot (which expects data in (y, x) indexing)
    u_plot = u_centers.T
    v_plot = v_centers.T
    X_plot = X.T
    Y_plot = Y.T
    
    # Plot streamlines
    plt.streamplot(X_plot, Y_plot, u_plot, v_plot, density=density, color=color)
    
    plt.xlim(0, max(x))
    plt.ylim(0, max(y))
    
    if title:
        plt.title(title)
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    if filename:
        # Ensure output directory exists and get full path
        full_path = _ensure_output_directory(filename, output_dir)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return plt.gcf()

def plot_combined_results_matrix(u, v, p, x, y, title=None, filename=None, show=True, Re=None):
    """
    Create a combined plot with three subplots side by side: velocity magnitude, 
    streamlines with pressure background, and validation against benchmark data.
    
    Parameters:
    -----------
    u : ndarray
        x-velocity component
    v : ndarray
        y-velocity component
    p : ndarray
        pressure field
    x : ndarray
        x-coordinates
    y : ndarray
        y-coordinates
    title : str, optional
        Plot title
    filename : str, optional
        If provided, saves the figure to this filename
    show : bool, optional
        Whether to display the plot
    Re : int, optional
        Reynolds number for benchmark comparison
    
    Returns:
    --------
    fig : Figure
        The created figure object
    """
    from matplotlib.gridspec import GridSpec
    from scipy.interpolate import interp1d
    
    # Handle staggered grid dimensions
    nx = len(x)
    ny = len(y)
    
    # Create arrays for velocity at cell centers
    u_centers = np.zeros((nx, ny))
    v_centers = np.zeros((nx, ny))
    
    # Interpolate u to cell centers
    if u.shape[0] > nx:  # Staggered grid for u
        for j in range(min(ny, u.shape[1])):
            for i in range(nx):
                u_centers[i, j] = 0.5 * (u[i, j] + u[i+1, j])
    else:
        # Copy values directly if dimensions match
        u_centers[:u.shape[0], :u.shape[1]] = u[:nx, :ny]
    
    # Interpolate v to cell centers
    if v.shape[1] > ny:  # Staggered grid for v
        for i in range(min(nx, v.shape[0])):
            for j in range(ny):
                v_centers[i, j] = 0.5 * (v[i, j] + v[i, j+1])
    else:
        # Copy values directly if dimensions match
        v_centers[:v.shape[0], :v.shape[1]] = v[:nx, :ny]
    
    # Calculate velocity magnitude
    u_mag = np.sqrt(u_centers**2 + v_centers**2)
    
    # Create a figure with 3 subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Plot velocity magnitude (left)
    im1 = ax1.matshow(u_mag.T, cmap='coolwarm', origin='lower', aspect='auto')
    plt.colorbar(im1, ax=ax1, label='Velocity magnitude')
    
    # Set proper ticks for the axes
    x_ticks = np.linspace(0, nx-1, 5)
    y_ticks = np.linspace(0, ny-1, 5)
    x_labels = np.linspace(0, 1, 5)
    y_labels = np.linspace(0, 1, 5)
    
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.set_xticklabels([f'{x:.1f}' for x in x_labels])
    ax1.set_yticklabels([f'{y:.1f}' for y in y_labels])
    
    ax1.set_title(f'Velocity Magnitude')
    if Re is not None:
        ax1.set_title(f'Velocity Magnitude (Re={Re})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # 2. Plot streamlines with pressure background (middle)
    # Create proper meshgrid for contourf and streamplot
    X, Y = np.meshgrid(x, y)
    
    im2 = ax2.contourf(X, Y, p.T, 50, cmap='coolwarm')
    plt.colorbar(im2, ax=ax2, label='Pressure')
    
    # Plot streamlines - use the same meshgrid for both
    ax2.streamplot(X, Y, u_centers.T, v_centers.T, density=1.0, color='white')
    ax2.set_xlim(0, max(x))
    ax2.set_ylim(0, max(y))
    ax2.set_title(f'Streamlines')
    if Re is not None:
        ax2.set_title(f'Streamlines (Re={Re})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # 3. Validation plot (right)
    if Re is not None:
        try:
            # Get benchmark data - use exact match only
            ghia_data = BenchmarkData.get_ghia_data(Re)
            
            if ghia_data is None:
                # Try to get the closest available Reynolds number data
                ghia_data = get_closest_ghia_data(Re)
                if ghia_data is None:
                    ax3.text(0.5, 0.5, f"Ghia data not available for Re={Re}", 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title("Validation")
                else:
                    # Display a note about using approximate data
                    ax3.text(0.5, 0.95, f"Using closest available benchmark data", 
                            ha='center', va='top', transform=ax3.transAxes, fontsize=8)
            
            if ghia_data is not None:
                # Get centerline data
                ghia_y = ghia_data['y']
                ghia_u = ghia_data['u']
                ghia_x = ghia_data['x']
                ghia_v = ghia_data['v']
                
                # Get dimensions for centerlines
                imax = u.shape[0] - 1 if u.shape[0] > nx else u.shape[0]
                jmax = v.shape[1] - 1 if v.shape[1] > ny else v.shape[1]
                
                # Set center indices
                j_center = jmax // 2  # Center index for y-direction
                i_center = imax // 2  # Center index for x-direction
                
                # Extract centerline data from simulation
                u_centerline = np.zeros(jmax)
                for j in range(jmax):
                    if u.shape[0] > nx:  # Staggered grid
                        u_centerline[j] = 0.5 * (u[i_center, j] + u[i_center+1, j])
                    else:
                        u_centerline[j] = u[i_center, j]
                
                v_centerline = np.zeros(imax)
                for i in range(imax):
                    if v.shape[1] > ny:  # Staggered grid
                        v_centerline[i] = 0.5 * (v[i, j_center] + v[i, j_center+1])
                    else:
                        v_centerline[i] = v[i, j_center]
                
                # Normalize coordinates for comparison
                x_normalized = np.linspace(0, 1, imax)
                y_normalized = np.linspace(0, 1, jmax)
                
                # Create a twin axis for the second plot
                ax3b = ax3.twinx()
                
                # Plot v-velocity on the left y-axis
                line1, = ax3.plot(x_normalized, v_centerline, 'b-', linewidth=2, label='v-velocity (simulation)')
                line2, = ax3.plot(ghia_x, ghia_v, 'bo', markersize=6, label='v-velocity (Ghia et al.)')
                ax3.set_xlabel('Position')
                ax3.set_ylabel('v-velocity', color='blue')
                ax3.tick_params(axis='y', labelcolor='blue')
                
                # Plot u-velocity on the right y-axis
                line3, = ax3b.plot(u_centerline, y_normalized, 'r-', linewidth=2, label='u-velocity (simulation)')
                line4, = ax3b.plot(ghia_u, ghia_y, 'ro', markersize=6, label='u-velocity (Ghia et al.)')
                ax3b.set_ylabel('u-velocity', color='red')
                ax3b.tick_params(axis='y', labelcolor='red')
                
                # Add a title
                ax3.set_title('Velocity Profiles Comparison')
                
                # Add a combined legend
                lines = [line1, line2, line3, line4]
                labels = [l.get_label() for l in lines]
                ax3.legend(lines, labels, loc='best')
                
                # Add grid
                ax3.grid(True)
        except Exception as e:
            # If validation fails, just show a message
            ax3.text(0.5, 0.5, f"Validation unavailable\n{str(e)}", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Validation")
    else:
        ax3.text(0.5, 0.5, "Validation requires Reynolds number", 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Validation")

    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.90)  # Make room for the suptitle
    
    plt.tight_layout()
    
    # Save figure if filename is provided
    if filename:
        # Ensure output directory exists and get full path
        full_path = _ensure_output_directory(filename, None)
        # Make sure the file extension is .pdf
        if not full_path.endswith('.pdf'):
            full_path += '.pdf'
        plt.savefig(full_path, format='pdf', bbox_inches='tight')
        print(f"Combined matrix results saved to {full_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

