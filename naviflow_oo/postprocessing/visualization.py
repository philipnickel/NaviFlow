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


def plot_velocity_field(u, v, x, y, title=None, filename=None, cmap='spring', 
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

def plot_final_residuals(u, v, p, u_old, v_old, p_old, mesh, title=None, filename=None, show=True, output_dir=None):
    """
    Plot final residuals for pressure, momentum, and total fields.
    
    Parameters:
    -----------
    u, v : ndarray
        Current velocity fields
    p : ndarray
        Current pressure field
    u_old, v_old : ndarray
        Previous velocity fields
    p_old : ndarray
        Previous pressure field
    mesh : StructuredMesh
        The computational mesh
    title : str, optional
        Plot title
    filename : str, optional
        If provided, saves the figure to this filename
    show : bool, optional
        Whether to display the plot
    output_dir : str, optional
        Directory where to save the output. If None, uses 'results' in the calling script's directory.
    """
    # Calculate the final residual fields
    final_p_residual = np.abs(p - p_old)  # Final pressure residual
    final_u_residual = np.abs(u - u_old)  # Final u-velocity residual
    final_v_residual = np.abs(v - v_old)  # Final v-velocity residual
    
    # Get mesh dimensions
    nx, ny = mesh.get_dimensions()
    
    # Create cell-centered final u-velocity residual
    u_res_centered = np.zeros_like(p)
    for i in range(nx):
        for j in range(ny):
            if i < nx-1:
                u_res_centered[i,j] = 0.5 * (final_u_residual[i,j] + final_u_residual[i+1,j])
            else:
                u_res_centered[i,j] = final_u_residual[i,j]
    
    # Create cell-centered final v-velocity residual
    v_res_centered = np.zeros_like(p)
    for i in range(nx):
        for j in range(ny):
            if j < ny-1:
                v_res_centered[i,j] = 0.5 * (final_v_residual[i,j] + final_v_residual[i,j+1])
            else:
                v_res_centered[i,j] = final_v_residual[i,j]
    
    # Combined final momentum residual
    final_momentum_res_field = np.sqrt(u_res_centered**2 + v_res_centered**2)
    final_total_res_field = final_momentum_res_field + final_p_residual
    
    # Calculate relative errors
    rel_p_residual = final_p_residual / (np.abs(p) + 1e-10)
    
    # Create cell-centered velocity magnitude for relative momentum residual
    u_mag = np.zeros_like(p)
    v_mag = np.zeros_like(p)
    for i in range(nx):
        for j in range(ny):
            if i < nx-1:
                u_mag[i,j] = 0.5 * (u[i,j] + u[i+1,j])
            else:
                u_mag[i,j] = u[i,j]
            if j < ny-1:
                v_mag[i,j] = 0.5 * (v[i,j] + v[i,j+1])
            else:
                v_mag[i,j] = v[i,j]
    
    rel_momentum_residual = final_momentum_res_field / (np.sqrt(u_mag**2 + v_mag**2) + 1e-10)
    rel_total_residual = final_total_res_field / (np.sqrt(u_mag**2 + v_mag**2 + p**2) + 1e-10)
    
    # Create a figure with 3x4 subplots (3 fields, 4 error types each)
    fig = plt.figure(figsize=(20, 15))
    
    # Function to create subplot with error field
    def plot_error_field(ax, data, title, is_log=False, is_relative=False):
        if is_log:
            data = np.log10(data + 1e-10)
        img = ax.matshow(data, cmap='coolwarm')
        plt.colorbar(img, ax=ax, label=f"{'Log10 ' if is_log else ''}{'Relative' if is_relative else 'Absolute'} Error")
        ax.set_title(f"{title}\n{'Log10 ' if is_log else ''}{'Relative' if is_relative else 'Absolute'} Error")
    
    # Pressure error plots
    plot_error_field(plt.subplot(3, 4, 1), final_p_residual, "Pressure", is_log=False, is_relative=False)
    plot_error_field(plt.subplot(3, 4, 2), final_p_residual, "Pressure", is_log=True, is_relative=False)
    plot_error_field(plt.subplot(3, 4, 3), rel_p_residual, "Pressure", is_log=False, is_relative=True)
    plot_error_field(plt.subplot(3, 4, 4), rel_p_residual, "Pressure", is_log=True, is_relative=True)
    
    # Momentum error plots
    plot_error_field(plt.subplot(3, 4, 5), final_momentum_res_field, "Momentum", is_log=False, is_relative=False)
    plot_error_field(plt.subplot(3, 4, 6), final_momentum_res_field, "Momentum", is_log=True, is_relative=False)
    plot_error_field(plt.subplot(3, 4, 7), rel_momentum_residual, "Momentum", is_log=False, is_relative=True)
    plot_error_field(plt.subplot(3, 4, 8), rel_momentum_residual, "Momentum", is_log=True, is_relative=True)
    
    # Total error plots
    plot_error_field(plt.subplot(3, 4, 9), final_total_res_field, "Total", is_log=False, is_relative=False)
    plot_error_field(plt.subplot(3, 4, 10), final_total_res_field, "Total", is_log=True, is_relative=False)
    plot_error_field(plt.subplot(3, 4, 11), rel_total_residual, "Total", is_log=False, is_relative=True)
    plot_error_field(plt.subplot(3, 4, 12), rel_total_residual, "Total", is_log=True, is_relative=True)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if filename:
        # Ensure output directory exists and get full path
        full_path = _ensure_output_directory(filename, output_dir)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Final residuals plot saved to {full_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_live_residuals(residual_history, momentum_residuals=None, pressure_residuals=None, 
                       title=None, show=True, clear_figure=True):
    """
    Plot residuals in real-time during simulation with autoscaling and stop button.
    
    Parameters:
    -----------
    residual_history : list
        History of total residuals
    momentum_residuals : list, optional
        History of momentum residuals
    pressure_residuals : list, optional
        History of pressure residuals
    title : str, optional
        Plot title
    show : bool, optional
        Whether to display the plot
    clear_figure : bool, optional
        Whether to clear the figure before plotting
        
    Returns:
    --------
    tuple
        (matplotlib.figure.Figure, bool) - The figure and whether to stop the simulation
    """
    if clear_figure:
        plt.clf()
    
    # Create figure if it doesn't exist
    if not plt.get_fignums():
        # Get screen size and create full screen figure
        screen_width, screen_height = plt.rcParams['figure.figsize']
        fig = plt.figure(figsize=(screen_width, screen_height))
        # Add stop button
        ax_stop = plt.axes([0.8, 0.01, 0.1, 0.04])
        stop_button = plt.Button(ax_stop, 'Stop')
        stop_button.on_clicked(lambda x: plt.close())
    else:
        fig = plt.gcf()
    
    # Function to handle window resize
    def on_resize(event):
        if event.inaxes is None:
            return
        # Get the current figure size
        fig_width, fig_height = fig.get_size_inches()
        # Update the figure size to match the window
        fig.set_size_inches(fig_width, fig_height, forward=True)
        # Redraw the figure
        fig.canvas.draw_idle()
    
    # Connect the resize event
    fig.canvas.mpl_connect('resize_event', on_resize)
    
    # Plot total residuals
    iterations = range(1, len(residual_history) + 1)
    plt.semilogy(iterations, residual_history, 'b-', linewidth=2, label='Total')
    
    # Plot component residuals if available
    if momentum_residuals and len(momentum_residuals) == len(residual_history):
        plt.semilogy(iterations, momentum_residuals, 'r--', linewidth=1.5, label='Momentum')
        
    if pressure_residuals and len(pressure_residuals) == len(residual_history):
        plt.semilogy(iterations, pressure_residuals, 'g-.', linewidth=1.5, label='Pressure')
    
    # Autoscale the axes
    plt.autoscale(enable=True, axis='both', tight=True)
    
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    
    if title:
        plt.title(title)
    else:
        plt.title('Residual History')
    
    # Simplify legend to only show the three main types
    plt.legend(loc='upper right')
    
    # Adjust subplot parameters to fill the window
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.tight_layout(pad=0.1)
    
    if show:
        plt.pause(0.001)  # Small pause to allow the plot to update
    
    # Check if the figure is still open
    should_stop = not plt.fignum_exists(fig.number)
    
    return fig, should_stop

