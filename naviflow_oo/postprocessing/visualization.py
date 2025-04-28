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
import matplotlib.tri as tri


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

def plot_final_residuals(u_residual_field, v_residual_field, p_residual_field, mesh, 
                        title=None, filename=None, show=True, output_dir=None,
                        u_rel_norms=None, v_rel_norms=None, p_rel_norms=None, 
                        history_filename=None):
    """
    Plot the final residual fields from the simulation.
    
    Parameters:
    -----------
    u_residual_field : ndarray
        u-momentum residual field
    v_residual_field : ndarray
        v-momentum residual field
    p_residual_field : ndarray
        Pressure residual field
    mesh : StructuredMesh or UnstructuredMesh
        The computational mesh
    title : str, optional
        Plot title
    filename : str, optional
        If provided, saves the figure to this filename
    show : bool, optional
        Whether to display the plot
    output_dir : str, optional
        Directory where to save the output
    u_rel_norms, v_rel_norms, p_rel_norms : list, optional
        Residual history data for plotting
    history_filename : str, optional
        If provided, saves the history plot to this filename
    
    Returns:
    --------
    tuple
        (residual_fig, history_fig) - The generated figures
    """
    # Create figure for residual fields
    is_structured = (len(u_residual_field.shape) == 2)
    
    if is_structured:  # Structured mesh residuals (2D arrays)
        # Create figure for residual fields
        fig_residuals, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Get mesh dimensions
        nx, ny = mesh.get_dimensions()
        dx, dy = mesh.get_cell_sizes()
        
        # Create x and y coordinates
        x = np.linspace(dx/2, 1-dx/2, nx)
        y = np.linspace(dy/2, 1-dy/2, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Interior points only (exclude boundary cells)
        if len(u_residual_field.shape) > 1 and u_residual_field.shape[0] > 2 and u_residual_field.shape[1] > 2:
            u_res_interior = u_residual_field[1:nx-1, 1:ny-1]
        else:
            # Handle 1D or small array
            u_res_interior = u_residual_field
            
        if len(v_residual_field.shape) > 1 and v_residual_field.shape[0] > 2 and v_residual_field.shape[1] > 2:
            v_res_interior = v_residual_field[1:nx-1, 1:ny-1]
        else:
            # Handle 1D or small array
            v_res_interior = v_residual_field
            
        if len(p_residual_field.shape) > 1 and p_residual_field.shape[0] > 2 and p_residual_field.shape[1] > 2:
            p_res_interior = p_residual_field[1:nx-1, 1:ny-1]
        else:
            # Handle 1D or small array
            p_res_interior = p_residual_field
            
        # Handle X and Y for plotting
        if len(X.shape) > 1 and X.shape[0] > 2 and X.shape[1] > 2:
            X_interior = X[1:nx-1, 1:ny-1]
            Y_interior = Y[1:nx-1, 1:ny-1]
        else:
            # For 1D arrays or smaller meshes, keep as is
            X_interior = X
            Y_interior = Y
        
        # Calculate combined residual field
        try:
            # Try to calculate combined residual if the shapes are compatible
            if u_res_interior.shape == v_res_interior.shape == p_res_interior.shape:
                combined_res = np.sqrt(u_res_interior**2 + v_res_interior**2 + p_res_interior**2)
            else:
                # If shapes don't match, try flattening and reshaping
                if hasattr(u_res_interior, 'flatten'):
                    u_flat = u_res_interior.flatten()
                    v_flat = v_res_interior.flatten()
                    p_flat = p_res_interior.flatten()
                    
                    # Get the smallest length to ensure compatibility
                    min_len = min(len(u_flat), len(v_flat), len(p_flat))
                    
                    # Compute with truncated arrays
                    combined_flat = np.sqrt(u_flat[:min_len]**2 + v_flat[:min_len]**2 + p_flat[:min_len]**2)
                    
                    # Use the first array's shape as a template if possible
                    if hasattr(u_res_interior, 'shape') and len(u_res_interior.shape) > 1:
                        # Try to reshape to 2D if possible
                        rows = int(np.sqrt(min_len))
                        cols = min_len // rows
                        combined_res = combined_flat[:rows*cols].reshape(rows, cols)
                    else:
                        # Keep as 1D array
                        combined_res = combined_flat
                else:
                    # Fallback: create a dummy array with same shape as u_res_interior
                    combined_res = np.zeros_like(u_res_interior)
        except Exception as e:
            # If all else fails, create a dummy array
            if hasattr(u_res_interior, 'shape'):
                combined_res = np.zeros_like(u_res_interior)
            else:
                combined_res = np.zeros(10)
        
        # Define a helper function for plotting each residual field
        def plot_residual_field(ax, data, X, Y, field_title, log_scale=False):
            """Plot residual field with option for log scale."""
            # Handle 1D data by reshaping to 2D if needed
            if data.ndim == 1:
                # Create a square-ish 2D array
                rows = int(np.sqrt(len(data)))
                cols = len(data) // rows
                data_2d = data[:rows*cols].reshape(rows, cols)
                
                # Create grid for plotting
                x_grid = np.linspace(0, 1, cols)
                y_grid = np.linspace(0, 1, rows)
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                
                plot_X, plot_Y = X_grid, Y_grid
                plot_data = data_2d
            else:
                # Use original data and coordinates
                plot_X, plot_Y = X, Y
                plot_data = data
                
                # Ensure X and Y have compatible shapes
                if plot_X.shape != plot_data.shape or plot_Y.shape != plot_data.shape:
                    # Create new grid
                    rows, cols = plot_data.shape
                    x_grid = np.linspace(0, 1, cols)
                    y_grid = np.linspace(0, 1, rows)
                    plot_X, plot_Y = np.meshgrid(x_grid, y_grid)
            
            # Apply log transform if needed
            if log_scale:
                # Add small value to avoid log(0)
                plot_data = np.log10(plot_data + 1e-16)
                color_label = f'log10({field_title})'
            else:
                color_label = field_title
            
            # Create the plot
            im = ax.pcolormesh(plot_X, plot_Y, plot_data, cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax, label=color_label)
                
            ax.set_title(field_title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        
        # Plot each residual field
        plot_residual_field(axs[0, 0], u_res_interior, X_interior, Y_interior, 'u-momentum residual', log_scale=True)
        plot_residual_field(axs[0, 1], v_res_interior, X_interior, Y_interior, 'v-momentum residual', log_scale=True)
        plot_residual_field(axs[1, 0], p_res_interior, X_interior, Y_interior, 'Pressure residual', log_scale=True)
        plot_residual_field(axs[1, 1], combined_res, X_interior, Y_interior, 'Combined residual', log_scale=True)
    
    else:  # Unstructured mesh residuals (1D arrays)
        # Create figure for residual fields
        fig_residuals, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Get cell centers for plotting
        if hasattr(mesh, 'mesh'):  # If using MeshAdapter
            centers = mesh.mesh.get_cell_centers()
        else:  # Direct use of the mesh
            centers = mesh.get_cell_centers()
        
        # Extract x and y coordinates
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Create a triangulation of the cell centers
        triang = tri.Triangulation(x, y)
        
        # Exclude boundary cells for better visualization
        # For unstructured meshes, we'll use a simple heuristic
        if hasattr(mesh, 'mesh'):  # If using MeshAdapter
            unstructured_mesh = mesh.mesh
        else:  # Direct use of the mesh
            unstructured_mesh = mesh
            
        owner_cells, neighbor_cells = unstructured_mesh.get_owner_neighbor()
        
        # Identify boundary cells (cells that have a face with no neighbor)
        boundary_cells = set()
        for face_idx, neighbor in enumerate(neighbor_cells):
            if neighbor < 0:  # Boundary face
                boundary_cells.add(owner_cells[face_idx])
        
        # Create a mask for interior cells
        interior_mask = np.ones(len(x), dtype=bool)
        for cell_idx in boundary_cells:
            interior_mask[cell_idx] = False
        
        # Calculate combined residual field
        try:
            # Try to calculate combined residual if the shapes are compatible
            if u_residual_field.shape == v_residual_field.shape == p_residual_field.shape:
                combined_res = np.sqrt(u_residual_field**2 + v_residual_field**2 + p_residual_field**2)
            else:
                # If shapes don't match, try flattening and reshaping
                if hasattr(u_residual_field, 'flatten'):
                    u_flat = u_residual_field.flatten()
                    v_flat = v_residual_field.flatten()
                    p_flat = p_residual_field.flatten()
                    
                    # Get the smallest length to ensure compatibility
                    min_len = min(len(u_flat), len(v_flat), len(p_flat))
                    
                    # Compute with truncated arrays
                    combined_flat = np.sqrt(u_flat[:min_len]**2 + v_flat[:min_len]**2 + p_flat[:min_len]**2)
                    
                    # Use the first array's shape as a template if possible
                    if hasattr(u_residual_field, 'shape') and len(u_residual_field.shape) > 1:
                        # Try to reshape to 2D if possible
                        rows = int(np.sqrt(min_len))
                        cols = min_len // rows
                        combined_res = combined_flat[:rows*cols].reshape(rows, cols)
                    else:
                        # Keep as 1D array
                        combined_res = combined_flat
                else:
                    # Fallback: create a dummy array with same shape as u_residual_field
                    combined_res = np.zeros_like(u_residual_field)
        except Exception as e:
            # If all else fails, create a dummy array
            if hasattr(u_residual_field, 'shape'):
                combined_res = np.zeros_like(u_residual_field)
            else:
                combined_res = np.zeros(10)
        
        # Define a helper function for plotting each residual field
        def plot_residual_field_unstruct(ax, data, mesh, field_title, log_scale=False):
            """Plot residual field for unstructured data with option for log scale."""
            # Handle 1D data
            if data.ndim == 1:
                plot_data = data
            else:
                plot_data = data.flatten()  # Ensure data is flattened
                
            # Get cell centers
            cell_centers = mesh.get_cell_centers()
            x = cell_centers[:, 0]
            y = cell_centers[:, 1]
            
            # Apply log transform if needed
            if log_scale:
                # Add small value to avoid log(0)
                plot_data = np.log10(plot_data + 1e-16)
                color_label = f'log10({field_title})'
            else:
                color_label = field_title
            
            # Create scatter plot
            scatter = ax.scatter(x, y, c=plot_data, cmap='viridis', s=20, marker='s')
            plt.colorbar(scatter, ax=ax, label=color_label)
                
            ax.set_title(field_title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        
        # Plot each residual field
        plot_residual_field_unstruct(axs[0, 0], u_residual_field, mesh, 'u-momentum residual', log_scale=True)
        plot_residual_field_unstruct(axs[0, 1], v_residual_field, mesh, 'v-momentum residual', log_scale=True)
        plot_residual_field_unstruct(axs[1, 0], p_residual_field, mesh, 'Pressure residual', log_scale=True)
        plot_residual_field_unstruct(axs[1, 1], combined_res, mesh, 'Combined residual', log_scale=True)
    
    # Add main title
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the suptitle
    
    # Save figure if filename is provided
    if filename:
        # Ensure output directory exists and get full path
        full_path = _ensure_output_directory(filename, output_dir)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
    
    # Create history plot if residual history is provided
    fig_history = None
    if any(x is not None for x in [u_rel_norms, v_rel_norms, p_rel_norms]):
        fig_history = plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        
        if u_rel_norms is not None:
            ax.semilogy(range(1, len(u_rel_norms)+1), u_rel_norms, 'r-', linewidth=2, label='u-momentum')
        
        if v_rel_norms is not None:
            ax.semilogy(range(1, len(v_rel_norms)+1), v_rel_norms, 'g-', linewidth=2, label='v-momentum')
        
        if p_rel_norms is not None:
            ax.semilogy(range(1, len(p_rel_norms)+1), p_rel_norms, 'b-', linewidth=2, label='pressure')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative L2 Norm')
        ax.set_title('Residual History')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend()
        
        # Save history figure if filename is provided
        if history_filename:
            # Ensure output directory exists and get full path
            history_path = _ensure_output_directory(history_filename, output_dir)
            plt.savefig(history_path, dpi=150, bbox_inches='tight')
    
    # Show plots if requested
    if show:
        plt.show()
    else:
        plt.close(fig_residuals)
        if fig_history:
            plt.close(fig_history)
    
    return (fig_residuals, fig_history)

def plot_combined_results_unstructured(u, v, p, x, y, title=None, filename=None, show=True, Re=None,
                                      output_dir=None, figsize=(15, 9)):
    """
    Plot combined results (velocity, pressure, streamlines) for unstructured mesh data.
    
    Parameters:
    -----------
    u : ndarray, shape (n_cells,)
        x-velocity component at cell centers
    v : ndarray, shape (n_cells,)
        y-velocity component at cell centers
    p : ndarray, shape (n_cells,)
        Pressure field at cell centers
    x : ndarray, shape (n_cells,)
        x-coordinates of cell centers
    y : ndarray, shape (n_cells,)
        y-coordinates of cell centers
    title : str, optional
        Plot title
    filename : str, optional
        If provided, saves the figure to this filename
    show : bool, optional
        Whether to display the plot
    Re : int, optional
        Reynolds number for benchmark comparison
    output_dir : str, optional
        Directory where to save the output
    figsize : tuple, optional
        Figure size (width, height) in inches
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Create a Delaunay triangulation of the cell centers
    triang = tri.Triangulation(x, y)
    
    # Calculate velocity magnitude
    u_mag = np.sqrt(u**2 + v**2)
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Plot u-velocity component
    u_plot = axs[0, 0].tripcolor(triang, u, cmap='RdBu_r', shading='gouraud')
    axs[0, 0].set_title('u-velocity component')
    plt.colorbar(u_plot, ax=axs[0, 0])
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_aspect('equal')
    
    # Plot v-velocity component
    v_plot = axs[0, 1].tripcolor(triang, v, cmap='RdBu_r', shading='gouraud')
    axs[0, 1].set_title('v-velocity component')
    plt.colorbar(v_plot, ax=axs[0, 1])
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_aspect('equal')
    
    # Plot pressure
    p_plot = axs[1, 0].tripcolor(triang, p, cmap='viridis', shading='gouraud')
    axs[1, 0].set_title('Pressure')
    plt.colorbar(p_plot, ax=axs[1, 0])
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_aspect('equal')
    
    # Plot velocity magnitude
    mag_plot = axs[1, 1].tripcolor(triang, u_mag, cmap='plasma', shading='gouraud')
    axs[1, 1].set_title('Velocity Magnitude')
    plt.colorbar(mag_plot, ax=axs[1, 1])
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_aspect('equal')
    
    # Add streamlines or vectors on top of the velocity magnitude plot
    # Use quiver for unstructured data
    skip = max(1, len(x) // 1000)  # Skip some points for clarity
    axs[1, 1].quiver(x[::skip], y[::skip], u[::skip], v[::skip], 
                    color='white', alpha=0.8, scale=20, width=0.003)
    
    # Add main title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the suptitle
    
    # Save figure if filename is provided
    if filename:
        # Ensure output directory exists and get full path
        full_path = _ensure_output_directory(filename, output_dir)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
