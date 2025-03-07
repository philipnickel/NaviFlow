import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import inspect
import scienceplots


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


def plot_velocity_field(u, v, x, y, title=None, filename=None, cmap='jet', 
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
                    show=True, figsize=(8, 6), background_field=None, cmap='jet',
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


def create_callback_for_animation(u_list, v_list, p_list=None, iterations=None, residuals=None, save_interval=50, tol=1e-6):
    """
    Create a callback function for saving fields during SIMPLE algorithm iterations.
    
    Parameters:
    -----------
    u_list : list
        List to store u velocity fields
    v_list : list
        List to store v velocity fields
    p_list : list, optional
        List to store pressure fields
    iterations : list, optional
        List to store iteration numbers
    residuals : list, optional
        List to store residual values
    save_interval : int, optional
        Interval for saving fields
    tol : float, optional
        Tolerance for convergence
        
    Returns:
    --------
    callback : function
        Callback function to pass to simple_algorithm
    """
    def save_fields_callback(iteration, u, v, p, maxRes):
        if iteration % save_interval == 0 or maxRes <= tol:
            u_list.append(u.copy())
            v_list.append(v.copy())
            if p_list is not None:
                p_list.append(p.copy())
            if iterations is not None:
                iterations.append(iteration)
            if residuals is not None:
                residuals.append(maxRes)
            print(f"Saved fields at iteration {iteration}, Residual: {maxRes:.6e}")
        return False  # Return False to continue iterations
    
    return save_fields_callback

def plot_combined_results_matrix(u, v, p, x, y, Re, title=None, filename=None, 
                               figsize=(18, 6), cmap='jet', show=True, output_dir=None):
    """
    Create a combined plot with three subplots side by side: velocity magnitude using matshow, 
    streamlines with pressure background, and a validation plot for u and v velocity
    against Ghia et al. benchmark data.
    
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
    Re : float
        Reynolds number
    title : str, optional
        Base title for the plot
    filename : str, optional
        If provided, saves the figure to this filename (should end with .pdf)
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str, optional
        Colormap to use
    show : bool, optional
        Whether to display the plot
    output_dir : str, optional
        Directory where to save the output. If None, uses 'results' in the calling script's directory.
        
    Returns:
    --------
    fig : Figure
        The created figure object
    """
    from matplotlib.gridspec import GridSpec
    from scipy.interpolate import interp1d
    import numpy as np
    from .validation import BenchmarkData
    
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
    
    # Create mesh grid for plotting
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create a figure with 3 subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Plot velocity magnitude using matshow (left)
    im1 = ax1.matshow(u_mag.T, cmap=cmap, origin='lower', aspect='auto')
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
    
    ax1.set_title(f'Velocity Magnitude (Re={Re})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # 2. Plot streamlines with pressure background (middle)
    # Prepare pressure field for plotting
    p_plot = p.T
    
    # Plot pressure contours
    im2 = ax2.contourf(X.T, Y.T, p_plot, 50, cmap=cmap)
    plt.colorbar(im2, ax=ax2, label='Pressure')
    
    # Prepare velocity components for streamplot
    u_plot = u_centers.T
    v_plot = v_centers.T
    X_plot = X.T
    Y_plot = Y.T
    
    # Plot streamlines
    ax2.streamplot(X_plot, Y_plot, u_plot, v_plot, density=1.0, color='white')
    ax2.set_xlim(0, max(x))
    ax2.set_ylim(0, max(y))
    ax2.set_title(f'Streamlines (Re={Re})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Get the appropriate Ghia data based on Reynolds number
    ghia_data = BenchmarkData.get_ghia_data(Re)
    if ghia_data is None:
        from .validation import get_closest_ghia_data
        ghia_data = get_closest_ghia_data(Re)
    
    # Extract Ghia data
    ghia_x = ghia_data['x']
    ghia_v = ghia_data['v']
    ghia_y = ghia_data['y']
    ghia_u = ghia_data['u']
    
    # Get dimensions for centerlines
    imax = u.shape[0] - 1 if u.shape[0] > nx else u.shape[0]  # Assuming staggered grid for u
    jmax = v.shape[1] - 1 if v.shape[1] > ny else v.shape[1]  # Assuming staggered grid for v
    
    # Set center indices
    j_center = jmax // 2  # Center index for y-direction
    i_center = imax // 2  # Center index for x-direction
    
    # Extract v-velocity along horizontal centerline
    v_centerline = np.zeros(imax)
    for i in range(imax):
        if v.shape[1] > ny:  # Staggered grid
            v_centerline[i] = 0.5 * (v[i, j_center] + v[i, j_center+1])
        else:
            v_centerline[i] = v[i, j_center]
    
    # Extract u-velocity along vertical centerline
    u_centerline = np.zeros(jmax)
    for j in range(jmax):
        if u.shape[0] > nx:  # Staggered grid
            u_centerline[j] = 0.5 * (u[i_center, j] + u[i_center+1, j])
        else:
            u_centerline[j] = u[i_center, j]
    
    # Normalize coordinates for comparison
    x_normalized = np.linspace(0, 1, imax)
    y_normalized = np.linspace(0, 1, jmax)
    
    # 3. Validation plot (right)
    # Create a twin axis for the second plot
    ax3b = ax3.twinx()
    
    # Plot v-velocity on the left y-axis
    line1, = ax3.plot(x_normalized, v_centerline, 'b-', linewidth=2, label='v-velocity (simulation)')
    line2, = ax3.plot(ghia_x, ghia_v, 'bo', markersize=6, label='v-velocity (Ghia et al.)')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('v-velocity', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    
    # Plot u-velocity on the right y-axis
    line3, = ax3b.plot(y_normalized, u_centerline, 'r-', linewidth=2, label='u-velocity (simulation)')
    line4, = ax3b.plot(ghia_y, ghia_u, 'ro', markersize=6, label='u-velocity (Ghia et al.)')
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
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.90)  # Make room for the suptitle
    
    if filename:
        # Ensure output directory exists and get full path
        full_path = _ensure_output_directory(filename, output_dir)
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

def create_side_by_side_animation(u_list, v_list, x, y, title=None, filename=None, fps=10, dpi=150,
                                cmap='jet', figsize=(16, 8), output_dir=None):
    """
    Create a side-by-side animation with velocity magnitude using matshow on the left
    and streamlines on the right.
    
    Parameters:
    -----------
    u_list : list of ndarray
        List of u velocity fields at different time steps
    v_list : list of ndarray
        List of v velocity fields at different time steps
    x : ndarray
        x-coordinates
    y : ndarray
        y-coordinates
    title : str, optional
        Animation title
    filename : str, optional
        Output filename (should end with .mp4)
    fps : int, optional
        Frames per second
    dpi : int, optional
        Resolution of output animation
    cmap : str, optional
        Colormap to use
    figsize : tuple, optional
        Figure size (width, height) in inches
    output_dir : str, optional
        Directory where to save the output. If None, uses 'results' in the calling script's directory.
    
    Returns:
    --------
    animation : FuncAnimation
        The created animation object
    """
    # Create mesh grid
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_plot, Y_plot = X.T, Y.T
    
    # Get the dimensions for the cell centers
    nx = len(x)
    ny = len(y)
    
    # Calculate velocity magnitude fields and center velocities for all frames
    u_centers_list = []
    v_centers_list = []
    u_mag_list = []
    
    for i in range(len(u_list)):
        # Create arrays for velocity at cell centers
        u_centers = np.zeros((nx, ny))
        v_centers = np.zeros((nx, ny))
        
        # Interpolate u to cell centers - handle staggered grid
        if u_list[i].shape[0] > nx:  # Staggered grid for u
            for j in range(min(ny, u_list[i].shape[1])):
                for i_idx in range(nx):
                    u_centers[i_idx, j] = 0.5 * (u_list[i][i_idx, j] + u_list[i][i_idx+1, j])
        else:
            # Copy values directly if dimensions match
            u_centers[:u_list[i].shape[0], :u_list[i].shape[1]] = u_list[i][:nx, :ny]
        
        # Interpolate v to cell centers - handle staggered grid
        if v_list[i].shape[1] > ny:  # Staggered grid for v
            for i_idx in range(min(nx, v_list[i].shape[0])):
                for j in range(ny):
                    v_centers[i_idx, j] = 0.5 * (v_list[i][i_idx, j] + v_list[i][i_idx, j+1])
        else:
            # Copy values directly if dimensions match
            v_centers[:v_list[i].shape[0], :v_list[i].shape[1]] = v_list[i][:nx, :ny]
        
        # Calculate velocity magnitude
        u_mag = np.sqrt(u_centers**2 + v_centers**2)
        
        u_centers_list.append(u_centers.T)
        v_centers_list.append(v_centers.T)
        u_mag_list.append(u_mag.T)
    
    # Determine vmin, vmax for consistent color scaling
    vmin = min(np.min(field) for field in u_mag_list)
    vmax = max(np.max(field) for field in u_mag_list)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Initial plots
    # Left: Velocity magnitude using matshow
    im1 = ax1.matshow(u_mag_list[0], cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    cbar1 = fig.colorbar(im1, ax=ax1, label='Velocity magnitude')
    
    # Set proper ticks for the axes
    x_ticks = np.linspace(0, nx-1, 5)
    y_ticks = np.linspace(0, ny-1, 5)
    x_labels = np.linspace(0, 1, 5)
    y_labels = np.linspace(0, 1, 5)
    
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.set_xticklabels([f'{x:.1f}' for x in x_labels])
    ax1.set_yticklabels([f'{y:.1f}' for y in y_labels])
    
    ax1.set_title('Velocity Magnitude')
    
    # Right: Streamlines with velocity magnitude background
    im2 = ax2.contourf(X_plot, Y_plot, u_mag_list[0], cmap=cmap, levels=50, vmin=vmin, vmax=vmax)
    strm = ax2.streamplot(X_plot, Y_plot, u_centers_list[0], v_centers_list[0], 
                         color='white', linewidth=0.8, density=1.5, arrowsize=0.8)
    cbar2 = fig.colorbar(im2, ax=ax2, label='Velocity magnitude')
    
    ax2.set_xlim(0, max(x))
    ax2.set_ylim(0, max(y))
    ax2.set_title('Streamlines')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    if title:
        fig.suptitle(f"{title} - Iteration: 0", fontsize=14)
    
    # Update function for animation
    def update_frame(i):
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Update matshow plot
        im1 = ax1.matshow(u_mag_list[i], cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        ax1.set_xticks(x_ticks)
        ax1.set_yticks(y_ticks)
        ax1.set_xticklabels([f'{x:.1f}' for x in x_labels])
        ax1.set_yticklabels([f'{y:.1f}' for y in y_labels])
        ax1.set_title('Velocity Magnitude')
        
        # Update streamlines plot
        im2 = ax2.contourf(X_plot, Y_plot, u_mag_list[i], cmap=cmap, levels=50, vmin=vmin, vmax=vmax)
        strm = ax2.streamplot(X_plot, Y_plot, u_centers_list[i], v_centers_list[i], 
                             color='white', linewidth=0.8, density=1.5, arrowsize=0.8)
        
        ax2.set_xlim(0, max(x))
        ax2.set_ylim(0, max(y))
        ax2.set_title('Streamlines')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        if title:
            fig.suptitle(f"{title} - Iteration: {i*len(u_list)//len(u_list)}", fontsize=14)
        
        return [im1, im2, strm.lines]
    
    # Create animation
    animation = FuncAnimation(
        fig,
        update_frame,
        frames=len(u_list),
        interval=1000/fps,  # interval in milliseconds
        blit=False
    )
    
    if filename:
        # Ensure output directory exists and get full path
        full_path = _ensure_output_directory(filename, output_dir)
        animation.save(full_path, writer='ffmpeg', fps=fps, dpi=dpi)
        print(f"Saved side-by-side animation to {full_path}")
    
    plt.close(fig)
    return animation
