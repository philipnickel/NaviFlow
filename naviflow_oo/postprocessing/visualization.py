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
                
                # Extract centerline data from simulation using the same approach as in validation
                u_centerline = u[nx//2, :]  # u along vertical centerline
                v_centerline = v[:, ny//2]  # v along horizontal centerline
                
                # Normalize coordinates for comparison
                x_normalized = np.linspace(0, 1, imax)
                y_normalized = np.linspace(0, 1, jmax)
                
                # Clear the axis first
                ax3.clear()
                
                # Get colors from coolwarm colormap
                u_color = cm.coolwarm(0.95)  # Blue-ish color from coolwarm
                v_color = cm.coolwarm(0.05)  # Red-ish color from coolwarm
                
                # Plot u-velocity along vertical centerline (y vs u)
                ax3.plot(y_normalized, u_centerline, '-', label='u solution', color=u_color)
                ax3.scatter(ghia_y, ghia_u, marker='o', label='u from Ghia et al.', color=u_color)
                
                # Plot v-velocity along horizontal centerline (x vs v)
                ax3.plot(x_normalized, v_centerline, '-', label='v solution', color=v_color)
                ax3.scatter(ghia_x, ghia_v, marker='o', label='v from Ghia et al.', color=v_color)
                
                # Set labels and grid
                ax3.set_xlabel('y')
                ax3.set_ylabel('u, v')
                ax3.grid(True)
                ax3.set_title(f'Comparison with Ghia et al. (Re={Re})')
                ax3.legend(loc='best')
                
                # Set DPI higher for better quality
                fig = plt.gcf()
                fig.set_dpi(150)
                
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

def plot_final_residuals(u_residual_field, v_residual_field, p_residual_field, mesh, title=None, filename=None, show=True, output_dir=None,
                      u_rel_norms=None, v_rel_norms=None, p_rel_norms=None, history_filename=None):
    """
    Plot final absolute algebraic residuals for pressure, u-momentum, and v-momentum fields on their native grids.
    Shows both linear scale (top row) and logarithmic scale (bottom row).
    
    If residual history arrays are provided, also generates a separate plot showing the history of
    relative residual norms defined as l2(r)/max(l2(r)).
    
    Parameters:
    -----------
    u_residual_field : ndarray
        Final algebraic u-momentum residual field (staggered grid)
    v_residual_field : ndarray
        Final algebraic v-momentum residual field (staggered grid)
    p_residual_field : ndarray
        Final algebraic pressure residual field (cell-centered)
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
    u_rel_norms : list, optional
        History of relative u-velocity residual norms (l2(r)/max(l2(r)))
    v_rel_norms : list, optional
        History of relative v-velocity residual norms (l2(r)/max(l2(r)))
    p_rel_norms : list, optional
        History of relative pressure/continuity residual norms (l2(r)/max(l2(r)))
    history_filename : str, optional
        If provided, saves the residual history plot to this filename
    """
    if u_residual_field is None or v_residual_field is None or p_residual_field is None:
        print("Warning: One or more final residual fields are missing. Skipping plot_final_residuals.")
        return None
        
    final_p_residual = np.abs(p_residual_field)
    final_u_residual = np.abs(u_residual_field)
    final_v_residual = np.abs(v_residual_field)
    
    nx, ny = mesh.get_dimensions()
    dx, dy = mesh.get_cell_sizes()
    
    # --- Slice data and coordinates to get interior points only ---
    # Pressure residuals (cell centers: 1 to nx-2, 1 to ny-2)
    p_res_interior = final_p_residual[1:nx-1, 1:ny-1]
    x_p_interior = np.linspace(dx/2, mesh.length - dx/2, nx)[1:nx-1] # Interior cell centers
    y_p_interior = np.linspace(dy/2, mesh.height - dy/2, ny)[1:ny-1] # Interior cell centers

    # U-momentum residuals (u-faces: i=1 to nx-1, cell centers j=1 to ny-2)
    # Note: u_residual_field has shape (nx+1, ny)
    u_res_interior = final_u_residual[1:nx, 1:ny-1] 
    # Coordinates for u-faces (x) and corresponding cell centers (y)
    x_u_interior = np.linspace(dx, mesh.length - dx, nx-1) # Interior u-face x-locations 
    y_u_interior = np.linspace(dy*1.5, mesh.height - dy*1.5, ny-2) # y-centers for rows 1 to ny-2

    # V-momentum residuals (cell centers i=1 to nx-2, v-faces j=1 to ny-1)
    # Note: v_residual_field has shape (nx, ny+1)
    v_res_interior = final_v_residual[1:nx-1, 1:ny]
    # Coordinates for cell centers (x) and corresponding v-faces (y)
    x_v_interior = np.linspace(dx*1.5, mesh.length - dx*1.5, nx-2) # x-centers for cols 1 to nx-2
    y_v_interior = np.linspace(dy, mesh.height - dy, ny-1) # Interior v-face y-locations
    # --- End slicing ---

    # Create a figure with 2x3 subplots (linear scale top row, log scale bottom row)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Internal helper function to plot a field on its native grid
    def plot_residual_field_native(ax, data, x_coords, y_coords, field_title, log_scale=False):
        label = '|Residual|' if not log_scale else 'log10(|Residual|)'
        plot_data = np.abs(data)
        
        # Apply log transformation if log_scale is True
        if log_scale:
            # Add small value to avoid log(0)
            plot_data = np.log10(plot_data + 1e-20)

        M, N = data.shape
        nx, ny = mesh.get_dimensions() # Get mesh dimensions here
        dx, dy = mesh.get_cell_sizes()
        
        valid_plot = True
        if not (x_coords.size == M and y_coords.size == N):
            print(f"Warning: Coordinate/Data mismatch for {field_title}.")
            print(f"  Data shape: ({M}, {N})")
            print(f"  X coords shape: {x_coords.shape}")
            print(f"  Y coords shape: {y_coords.shape}")
            valid_plot = False

        # Plot using pcolormesh if coordinates and data are valid
        if valid_plot:            
            vmin = np.min(plot_data) if np.isfinite(np.min(plot_data)) else 0
            vmax = np.max(plot_data) if np.isfinite(np.max(plot_data)) else 1
            if vmin == vmax:
                vmin = 0
                vmax = vmin + 1e-6 if vmin > 0 else 1e-6
                
            # Use shading='nearest' - expects coordinates to be centers matching data dimensions
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
            
            # Check if meshgrid dimensions match data dimensions
            if X.shape == data.shape:
                # Use shading='nearest'. Data needs transposing for correct orientation if meshgrid is ij indexed.
                img = ax.pcolormesh(X, Y, plot_data, cmap='coolwarm', shading='nearest', vmin=vmin, vmax=vmax)
                plt.colorbar(img, ax=ax, label=label)
                
                # Calculate and display L2 norm
                data_l2 = np.linalg.norm(np.abs(data), ord=2)
                scale_type = "Log10 " if log_scale else ""
                ax.set_title(f"{field_title}\n{scale_type}Absolute Residual (L2: {data_l2:.2e})")
                
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel('x (Interior)')
                ax.set_ylabel('y (Interior)')
                # Set limits slightly outside the centers for visibility
                # Adjust limits to only show interior region
                ax.set_xlim(dx, mesh.length - dx)
                ax.set_ylim(dy, mesh.height - dy)
            else:
                ax.text(0.5, 0.5, f"Meshgrid/Data mismatch for {field_title}\nData: {data.shape}\nMeshgrid: {X.shape}", 
                         ha='center', va='center', transform=ax.transAxes)
                ax.set_title(field_title)
        else:
            ax.text(0.5, 0.5, f"Plotting failed for {field_title}", 
                     ha='center', va='center', transform=ax.transAxes)
            ax.set_title(field_title)

    # Call the plotting function for absolute residuals (top row)
    plot_residual_field_native(axes[0, 0], p_res_interior, x_p_interior, y_p_interior, "Pressure Residual")
    plot_residual_field_native(axes[0, 1], u_res_interior, x_u_interior, y_u_interior, "U-Momentum Residual")
    plot_residual_field_native(axes[0, 2], v_res_interior, x_v_interior, y_v_interior, "V-Momentum Residual")
    
    # Call the plotting function for log of absolute residuals (bottom row)
    plot_residual_field_native(axes[1, 0], p_res_interior, x_p_interior, y_p_interior, "Pressure Residual", log_scale=True)
    plot_residual_field_native(axes[1, 1], u_res_interior, x_u_interior, y_u_interior, "U-Momentum Residual", log_scale=True)
    plot_residual_field_native(axes[1, 2], v_res_interior, x_v_interior, y_v_interior, "V-Momentum Residual", log_scale=True)
    
    if title:
        fig.suptitle(title + " (Absolute Residuals, Interior Points)", fontsize=16)
        fig.subplots_adjust(top=0.92, hspace=0.3) # Adjust top spacing for suptitle and between rows
    
    # Set row labels
    axes[0, 0].annotate('Linear Scale', xy=(0, 0.5), xytext=(-axes[0, 0].yaxis.labelpad - 15, 0),
                       xycoords='axes fraction', textcoords='offset points',
                       ha='right', va='center', rotation=90, fontsize=12, fontweight='bold')
    axes[1, 0].annotate('Log10 Scale', xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - 15, 0),
                       xycoords='axes fraction', textcoords='offset points',
                       ha='right', va='center', rotation=90, fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92]) # Adjust layout rect to prevent suptitle overlap
    
    if filename:
        full_path = _ensure_output_directory(filename, output_dir)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Final residuals plot saved to {full_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Create and save residual history plot if history data is provided
    if u_rel_norms is not None and v_rel_norms is not None and p_rel_norms is not None and history_filename is not None:
        # Plot relative residual history
        plt.figure(figsize=(10, 5))
        
        # Get colors from coolwarm colormap
        colors = [plt.cm.coolwarm(0.95), plt.cm.coolwarm(0.75), plt.cm.coolwarm(0.2)]  # Get colors at 0.8, 0.5, 0.2 positions
        
        # Plot the relative residual norms
        plt.semilogy(range(len(u_rel_norms)), u_rel_norms, color=colors[0], label='u-momentum')  # Warm color
        plt.semilogy(range(len(v_rel_norms)), v_rel_norms, color=colors[1], label='v-momentum')  # Neutral color
        plt.semilogy(range(len(p_rel_norms)), p_rel_norms, color=colors[2], label='pressure')    # Cool color
        plt.grid(True)
        plt.title(f'Residual History')
        plt.xlabel('Iteration')
        plt.ylabel('Residual Norm')
        plt.legend()
        
        # Add overall title if provided
        if title:
            reynolds_match = None
            if isinstance(title, str):
                import re
                reynolds_match = re.search(r'Re\s*=\s*(\d+)', title)
            
            if reynolds_match:
                reynolds = reynolds_match.group(1)
                plt.suptitle(f'Residual History (Re={reynolds})', fontsize=16)
            else:
                plt.suptitle('Residual History', fontsize=16)
                
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the plot
        hist_full_path = _ensure_output_directory(history_filename, output_dir)
        plt.savefig(hist_full_path, dpi=300, bbox_inches='tight')
        print(f"Relative residual history plot saved to {hist_full_path}")
        
        if not show:
            plt.close()
    
    return fig
