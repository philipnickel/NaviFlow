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


def create_animation(u_list, v_list, x, y, title=None, filename=None, fps=10, dpi=150,
                     cmap='jet', figsize=(8, 6), field_type='magnitude', output_dir=None):
    """
    Create an animation of the velocity field evolution.
    
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
    field_type : str, optional
        Type of field to show ('magnitude', 'u', 'v')
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
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine vmin, vmax for consistent color scaling
    if field_type == 'magnitude':
        fields = [np.sqrt(0.25*(u_list[i][:-1,:]**2 + u_list[i][1:,:]**2 + 
                               v_list[i][:,:-1]**2 + v_list[i][:,1:]**2)).T
                 for i in range(len(u_list))]
    elif field_type == 'u':
        fields = [0.5*(u_list[i][:-1,:] + u_list[i][1:,:]).T for i in range(len(u_list))]
    elif field_type == 'v':
        fields = [0.5*(v_list[i][:,:-1] + v_list[i][:,1:]).T for i in range(len(v_list))]
    
    vmin = min(np.min(field) for field in fields)
    vmax = max(np.max(field) for field in fields)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Initial plot
    cont = ax.contourf(X_plot, Y_plot, fields[0], cmap=cmap, levels=50, norm=norm)
    cbar = fig.colorbar(cont, ax=ax)
    
    if title:
        ax.set_title(f"{title} - Frame 0")
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Update function for animation
    def update_frame(i):
        ax.clear()
        cont = ax.contourf(X_plot, Y_plot, fields[i], cmap=cmap, levels=50, norm=norm)
        if title:
            ax.set_title(f"{title} - Frame {i}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, max(x))
        ax.set_ylim(0, max(y))
        return [cont]
    
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
    
    plt.close()
    
    return animation


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


def animate_solution_process(imax, jmax, dx, dy, rho, mu, u, v, p, 
                            velocity, alphaU, alphaP, max_iteration, tol,
                            x, y, save_interval=50, title=None, filename=None,
                            fps=10, dpi=150, cmap='jet', figsize=(8, 6),
                            field_types=None, output_dir=None, **solver_kwargs):
    """
    Run the SIMPLE algorithm and create animations of the solution process.
    
    Parameters:
    -----------
    imax, jmax : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    rho : float
        Fluid density
    mu : float
        Fluid viscosity
    u, v : numpy.ndarray
        Initial velocity fields
    p : numpy.ndarray
        Initial pressure field
    velocity : float
        Lid velocity
    alphaU, alphaP : float
        Relaxation factors for velocity and pressure
    max_iteration : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    x, y : numpy.ndarray
        Grid coordinates
    save_interval : int, optional
        Interval for saving frames
    title : str, optional
        Base title for animations
    filename : str, optional
        Base filename for animations (without extension)
    fps : int, optional
        Frames per second
    dpi : int, optional
        Resolution of output animation
    cmap : str, optional
        Colormap to use
    figsize : tuple, optional
        Figure size (width, height) in inches
    field_types : list, optional
        List of field types to animate ('magnitude', 'u', 'v', 'p')
        If None, animates all types
    output_dir : str, optional
        Directory where to save the output
    **solver_kwargs : dict
        Additional arguments to pass to simple_algorithm
        
    Returns:
    --------
    u, v : numpy.ndarray
        Final velocity fields
    p : numpy.ndarray
        Final pressure field
    iteration : int
        Number of iterations performed
    maxRes : float
        Final maximum residual
    divergence : numpy.ndarray
        Final divergence field
    animations : dict
        Dictionary of created animations
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import Normalize
    import numpy as np
    from ..algorithms.simple import simple_algorithm
    
    # Default field types if none specified
    if field_types is None:
        field_types = ['magnitude', 'u', 'v', 'p']
    
    # Lists to store fields for animation
    u_list = []
    v_list = []
    p_list = []
    iterations = []
    residuals = []
    
    # Custom callback function to save fields during iterations
    save_fields_callback = create_callback_for_animation(u_list, v_list, p_list, iterations, residuals, save_interval, tol)
    
    # Run the SIMPLE algorithm with the callback
    u, v, p, iteration, maxRes, divergence = simple_algorithm(
        imax, jmax, dx, dy, rho, mu, u, v, p, 
        velocity, alphaU, alphaP, max_iteration, tol,
        callback=save_fields_callback,
        **solver_kwargs
    )
    
    print(f"Total Iterations = {iteration}")
    print(f"Number of saved frames: {len(u_list)}")
    
    # Create animations for each requested field type
    animations = {}
    
    # Create mesh grid for plotting
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_plot, Y_plot = X.T, Y.T
    
    # Generate animations for each field type
    for field_type in field_types:
        if field_type == 'magnitude':
            fields = [np.sqrt(0.25*(u_list[i][:-1,:]**2 + u_list[i][1:,:]**2 + 
                                  v_list[i][:,:-1]**2 + v_list[i][:,1:]**2)).T
                     for i in range(len(u_list))]
            field_name = "Velocity Magnitude"
            anim_filename = f"{filename}_velocity_magnitude.mp4" if filename else None
        elif field_type == 'u':
            fields = [0.5*(u_list[i][:-1,:] + u_list[i][1:,:]).T for i in range(len(u_list))]
            field_name = "U-Velocity"
            anim_filename = f"{filename}_u_velocity.mp4" if filename else None
        elif field_type == 'v':
            fields = [0.5*(v_list[i][:,:-1] + v_list[i][:,1:]).T for i in range(len(v_list))]
            field_name = "V-Velocity"
            anim_filename = f"{filename}_v_velocity.mp4" if filename else None
        elif field_type == 'p':
            fields = [p_list[i].T for i in range(len(p_list))]
            field_name = "Pressure"
            anim_filename = f"{filename}_pressure.mp4" if filename else None
        else:
            continue
        
        # Determine min/max for consistent color scaling
        vmin = min(np.min(field) for field in fields)
        vmax = max(np.max(field) for field in fields)
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=figsize)
        cont = ax.contourf(X_plot, Y_plot, fields[0], cmap=cmap, levels=50, norm=norm)
        cbar = fig.colorbar(cont, ax=ax)
        cbar.set_label(field_name)
        
        # Set title with iteration info
        full_title = f"{title} - {field_name}" if title else field_name
        ax.set_title(f"{full_title}\nIteration: {iterations[0]}, Residual: {residuals[0]:.2e}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Update function for animation
        def update_frame(i):
            ax.clear()
            cont = ax.contourf(X_plot, Y_plot, fields[i], cmap=cmap, levels=50, norm=norm)
            ax.set_title(f"{full_title}\nIteration: {iterations[i]}, Residual: {residuals[i]:.2e}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(0, max(x))
            ax.set_ylim(0, max(y))
            return [cont]
        
        # Create animation
        animation = FuncAnimation(
            fig,
            update_frame,
            frames=len(fields),
            interval=1000/fps,  # interval in milliseconds
            blit=False
        )
        
        if anim_filename:
            # Ensure output directory exists and get full path
            full_path = _ensure_output_directory(anim_filename, output_dir)
            animation.save(full_path, writer='ffmpeg', fps=fps, dpi=dpi)
            print(f"Saved animation to {full_path}")
        
        animations[field_type] = animation
        plt.close(fig)
    
    # Create convergence history plot
    if iterations and residuals:
        fig, ax = plt.subplots(figsize=figsize)
        ax.semilogy(iterations, residuals, 'b-', linewidth=2)
        ax.grid(True)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual')
        conv_title = f"{title} - Convergence History" if title else "Convergence History"
        ax.set_title(conv_title)
        
        if filename:
            conv_filename = f"{filename}_convergence_history.png"
            full_path = _ensure_output_directory(conv_filename, output_dir)
            fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved convergence history to {full_path}")
        
        plt.close(fig)
    
    return u, v, p, iteration, maxRes, divergence, animations


def create_streamline_animation(u_list, v_list, x, y, title=None, filename=None, fps=10, dpi=150,
                              cmap='jet', figsize=(10, 8), output_dir=None):
    """
    Create an animation of velocity magnitude with streamlines overlay.
    
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
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import Normalize
    import numpy as np
    
    # Create mesh grid
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_plot, Y_plot = X.T, Y.T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate velocity magnitude fields
    fields = [np.sqrt(0.25*(u_list[i][:-1,:]**2 + u_list[i][1:,:]**2 + 
                           v_list[i][:,:-1]**2 + v_list[i][:,1:]**2)).T
             for i in range(len(u_list))]
    
    # Determine vmin, vmax for consistent color scaling
    vmin = min(np.min(field) for field in fields)
    vmax = max(np.max(field) for field in fields)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Calculate u and v at cell centers for streamlines
    u_centers = [0.5*(u_list[i][:-1,:] + u_list[i][1:,:]).T for i in range(len(u_list))]
    v_centers = [0.5*(v_list[i][:,:-1] + v_list[i][:,1:]).T for i in range(len(v_list))]
    
    # Initial plot
    cont = ax.contourf(X_plot, Y_plot, fields[0], cmap=cmap, levels=50, norm=norm)
    strm = ax.streamplot(X_plot, Y_plot, u_centers[0], v_centers[0], 
                        color='white', linewidth=0.8, density=1.5, arrowsize=0.8)
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label('Velocity Magnitude')
    
    ax.set_title(f"{title}\nIteration: {0}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    
    # Update function for animation
    def update_frame(i):
        ax.clear()
        cont = ax.contourf(X_plot, Y_plot, fields[i], cmap=cmap, levels=50, norm=norm)
        strm = ax.streamplot(X_plot, Y_plot, u_centers[i], v_centers[i], 
                            color='white', linewidth=0.8, density=1.5, arrowsize=0.8)
        ax.set_title(f"{title}\nIteration: {i*len(fields)//len(fields)}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, max(x))
        ax.set_ylim(0, max(y))
        return [cont, strm.lines]
    
    # Create animation
    animation = FuncAnimation(
        fig,
        update_frame,
        frames=len(fields),
        interval=1000/fps,  # interval in milliseconds
        blit=False
    )
    
    if filename:
        # Ensure output directory exists
        full_path = _ensure_output_directory(filename, output_dir)
        animation.save(full_path, writer='ffmpeg', fps=fps, dpi=dpi)
        print(f"Saved streamline animation to {full_path}")
    
    plt.close(fig)
    return animation
