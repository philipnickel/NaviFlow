import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_velocity_field(u, v, x, y, title=None, filename=None, cmap='jet', 
                        show=True, figsize=(8, 6), colorbar=True, levels=50):
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
    """
    # Handle staggered grid: interpolate u, v to cell centers if needed
    if u.shape[0] != len(x):
        # Assumes staggered grid
        u_centers = 0.5 * (u[:-1, :] + u[1:, :])
    else:
        u_centers = u
        
    if v.shape[1] != len(y):
        # Assumes staggered grid
        v_centers = 0.5 * (v[:, :-1] + v[:, 1:])
    else:
        v_centers = v
    
    # Calculate velocity magnitude
    u_mag = np.sqrt(u_centers**2 + v_centers**2)
    
    # Create mesh grid for plotting
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create figure
    plt.figure(figsize=figsize)
    
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
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return plt.gcf()


def plot_streamlines(u, v, x, y, title=None, filename=None, density=1.0, color='black',
                    show=True, figsize=(8, 6), background_field=None, cmap='jet'):
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
    """
    # Interpolate if needed (staggered grid)
    if u.shape[0] != len(x):
        u_centers = 0.5 * (u[:-1, :] + u[1:, :])
    else:
        u_centers = u
        
    if v.shape[1] != len(y):
        v_centers = 0.5 * (v[:, :-1] + v[:, 1:])
    else:
        v_centers = v
    
    # Create mesh grid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Transpose for streamplot (which expects data in (y, x) indexing)
    u_plot = u_centers.T
    v_plot = v_centers.T
    X_plot = X.T
    Y_plot = Y.T
    
    plt.figure(figsize=figsize)
    
    # If background field is provided, plot it as contour
    if background_field is not None:
        bg = background_field.T if background_field.shape == u_centers.shape else background_field
        plt.contourf(X_plot, Y_plot, bg, cmap=cmap, levels=50)
        plt.colorbar()
    
    # Plot streamlines
    plt.streamplot(X_plot, Y_plot, u_plot, v_plot, density=density, color=color)
    
    plt.xlim(0, max(x))
    plt.ylim(0, max(y))
    
    if title:
        plt.title(title)
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return plt.gcf()


def create_animation(u_list, v_list, x, y, title=None, filename=None, fps=10, dpi=150,
                     cmap='jet', figsize=(8, 6), field_type='magnitude'):
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
        animation.save(filename, writer='ffmpeg', fps=fps, dpi=dpi)
        print(f"Animation saved to {filename}")
    
    plt.close()
    
    return animation
