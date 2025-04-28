"""
Plotting utilities for 2D mesh visualization.
"""

import numpy as np


def plot_structured_mesh(mesh, ax, title=None):
    """
    Plots the nodes and cell outlines of a structured mesh.
    
    Parameters:
    -----------
    mesh : StructuredMesh
        The structured mesh to plot
    ax : matplotlib.axes.Axes
        The axes to plot on
    title : str, optional
        Title for the plot
    """
    # Create meshgrid for efficient plotting
    xn, yn = np.meshgrid(mesh.x_nodes, mesh.y_nodes, indexing='ij')
    
    # Plot nodes
    ax.plot(xn.flatten(), yn.flatten(), 'ko', markersize=1, alpha=0.6)
    
    # Plot grid lines (cell outlines) more efficiently
    # Vertical lines
    for i in range(mesh.nx):
        ax.plot([mesh.x_nodes[i], mesh.x_nodes[i]], 
                [mesh.y_nodes[0], mesh.y_nodes[-1]], 
                'k-', linewidth=0.5)
    
    # Horizontal lines
    for j in range(mesh.ny):
        ax.plot([mesh.x_nodes[0], mesh.x_nodes[-1]], 
                [mesh.y_nodes[j], mesh.y_nodes[j]], 
                'k-', linewidth=0.5)
    
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')


def plot_unstructured_mesh(mesh, ax, title=None):
    """
    Plots the nodes and edges (faces in 2D) of an unstructured mesh.
    
    Parameters:
    -----------
    mesh : UnstructuredMesh
        The unstructured mesh to plot
    ax : matplotlib.axes.Axes
        The axes to plot on
    title : str, optional
        Title for the plot
    """
    nodes = mesh.get_node_positions()
    
    # Plot nodes
    ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=1, alpha=0.6)
    
    # Plot edges more efficiently by grouping edges of the same size
    faces_by_size = {}
    for face in mesh.faces:
        size = len(face)
        if size not in faces_by_size:
            faces_by_size[size] = []
        faces_by_size[size].append(face)
    
    # Process each size group
    for size, faces in faces_by_size.items():
        # For size 2 (line segments), we can do efficient plotting
        if size == 2:
            # Extract start and end points
            start_nodes = np.array([nodes[face[0]] for face in faces])
            end_nodes = np.array([nodes[face[1]] for face in faces])
            
            # Plot all segments at once using vectorized approach
            x_coords = np.column_stack([start_nodes[:, 0], end_nodes[:, 0]])
            y_coords = np.column_stack([start_nodes[:, 1], end_nodes[:, 1]])
            
            for i in range(len(faces)):
                ax.plot(x_coords[i], y_coords[i], 'k-', linewidth=0.5)
        else:
            # For polygons, plot each individually
            for face in faces:
                face_nodes = nodes[face]
                # Close the polygon by adding the first point again
                face_nodes_closed = np.vstack([face_nodes, face_nodes[0]])
                ax.plot(face_nodes_closed[:, 0], face_nodes_closed[:, 1], 'k-', linewidth=0.5)
    
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')


def plot_mesh(mesh, ax, title=None):
    """
    Plot any 2D mesh type on the given axes.
    
    Parameters:
    -----------
    mesh : Mesh
        The mesh to plot
    ax : matplotlib.axes.Axes
        The axes to plot on
    title : str, optional
        Title for the plot
    """
    from .structured import StructuredMesh
    from .unstructured import UnstructuredMesh
    
    if isinstance(mesh, StructuredMesh):
        plot_structured_mesh(mesh, ax, title)
    elif isinstance(mesh, UnstructuredMesh):
        plot_unstructured_mesh(mesh, ax, title)
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh)}") 