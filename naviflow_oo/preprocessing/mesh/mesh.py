"""
Base mesh class for CFD simulations.
Designed to support 2D meshes.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Mesh(ABC):
    """
    Abstract base mesh class for 2D CFD simulations.
    Serves as a common interface for all mesh types.
    """
    
    @abstractmethod
    def get_node_positions(self):
        """
        Returns the positions of all nodes in the mesh.
        
        Returns:
        --------
        ndarray : shape (N, 2)
            Coordinates of all mesh nodes
        """
        pass
    
    @abstractmethod
    def get_cell_centers(self):
        """
        Returns the center positions of all cells in the mesh.
        
        Returns:
        --------
        ndarray : shape (M, 2)
            Coordinates of all cell centers
        """
        pass
    
    @abstractmethod
    def get_face_centers(self):
        """
        Returns the center positions of all faces in the mesh.
        
        Returns:
        --------
        ndarray : shape (F, 2)
            Coordinates of all face centers
        """
        pass
    
    @abstractmethod
    def get_face_normals(self):
        """
        Returns the normal vectors for all faces in the mesh.
        
        Returns:
        --------
        ndarray : shape (F, 2)
            Normal vectors for all faces
        """
        pass
    
    @abstractmethod
    def get_face_areas(self):
        """
        Returns the lengths of all faces in the mesh.
        
        Returns:
        --------
        ndarray : shape (F,)
            Lengths of all faces (edges)
        """
        pass
    
    @abstractmethod
    def get_cell_volumes(self):
        """
        Returns the areas of all cells in the mesh.
        
        Returns:
        --------
        ndarray : shape (C,)
            Areas of all cells
        """
        pass
    
    @abstractmethod
    def get_owner_neighbor(self):
        """
        Returns the owner and neighbor cell indices for all faces.
        
        Returns:
        --------
        tuple : (owner, neighbor)
            owner : ndarray, shape (F,)
                Owner cell indices for all faces
            neighbor : ndarray, shape (F,)
                Neighbor cell indices for all faces (or -1 for boundary faces)
        """
        pass
    
    @property
    @abstractmethod
    def n_cells(self):
        """
        Returns the number of cells in the mesh.
        
        Returns:
        --------
        int : Number of cells
        """
        pass
    
    @property
    @abstractmethod
    def n_faces(self):
        """
        Returns the number of faces in the mesh.
        
        Returns:
        --------
        int : Number of faces
        """
        pass
    
    @property
    @abstractmethod
    def n_nodes(self):
        """
        Returns the number of nodes in the mesh.
        
        Returns:
        --------
        int : Number of nodes
        """
        pass
    
    @abstractmethod
    def plot(self, ax=None, title=None):
        """
        Plot the mesh on given axes or create new figure if none provided.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        title : str, optional
            Title for the plot
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
        pass
    
    @abstractmethod
    def savePlot(self, filename, title=None, dpi=150, format='pdf'):
        """
        Create and save a plot of the mesh directly to file.
        
        Parameters:
        -----------
        filename : str
            The name of the file to save the plot to
        title : str, optional
            Title for the plot
        dpi : int, optional
            The resolution in dots per inch
        format : str, optional
            The file format to save in ('pdf', 'png', etc.)
        """
        pass


class UnstructuredMesh(Mesh):
    """
    Base class for unstructured meshes in 2D CFD simulations.
    """
    
    def __init__(self, nodes, faces, cells):
        """
        Initialize an unstructured mesh.
        
        Parameters:
        -----------
        nodes : ndarray, shape (N, 2) or (N, 3)
            Coordinates of all nodes in the mesh. If 3D coordinates are provided,
            only the x and y components will be used for 2D simulations.
        faces : list of list
            Each face is defined by a list of node indices
        cells : list of list
            Each cell is defined by a list of face indices
        """
        # Ensure we use only x and y coordinates for 2D mesh
        if nodes.shape[1] > 2:
            self._nodes = np.asarray(nodes)[:, :2]
        else:
            self._nodes = np.asarray(nodes)
            
        self._faces = faces
        self._cells = cells
        
        # Compute geometric properties
        self._compute_geometry()
    
    def _compute_geometry(self):
        """Compute geometric properties of the mesh."""
        n_cells = len(self._cells)
        n_faces = len(self._faces)
        n_nodes = len(self._nodes)
        
        # Compute face centers and normals
        self._face_centers = np.zeros((n_faces, 2))
        self._face_normals = np.zeros((n_faces, 2))
        self._face_areas = np.zeros(n_faces)
        
        for i, face in enumerate(self._faces):
            # Get node coordinates for this face
            face_nodes = self._nodes[face]
            
            # Face center is the average of node coordinates
            self._face_centers[i] = np.mean(face_nodes, axis=0)
            
            # For a 2D face (line segment), normal is perpendicular to the line
            if len(face) >= 2:
                # For a general polygon, we use the first and last point
                # since they're guaranteed to be on the boundary
                dx = face_nodes[-1, 0] - face_nodes[0, 0]
                dy = face_nodes[-1, 1] - face_nodes[0, 1]
                
                # Normal is (-dy, dx) normalized
                length = np.sqrt(dx*dx + dy*dy)
                self._face_normals[i] = np.array([-dy/length, dx/length])
                self._face_areas[i] = length
        
        # Compute cell centers and volumes
        self._cell_centers = np.zeros((n_cells, 2))
        self._cell_volumes = np.zeros(n_cells)
        
        for i, cell in enumerate(self._cells):
            # Get face indices for this cell
            if len(cell) > 0:
                # Get face centers for this cell
                cell_face_centers = self._face_centers[cell]
                
                # Cell center is the average of face centers
                self._cell_centers[i] = np.mean(cell_face_centers, axis=0)
                
                # For a 2D cell, volume (area) is computed using shoelace formula
                # We'll approximate this using the face centers as vertices
                x = cell_face_centers[:, 0]
                y = cell_face_centers[:, 1]
                
                # Shoelace formula for area of polygon
                self._cell_volumes[i] = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        # Compute owner and neighbor cells for each face
        self._owner_cells = np.full(n_faces, -1, dtype=int)
        self._neighbor_cells = np.full(n_faces, -1, dtype=int)
        
        for i, cell in enumerate(self._cells):
            for face_idx in cell:
                if self._owner_cells[face_idx] == -1:
                    # If owner is not assigned, this cell is the owner
                    self._owner_cells[face_idx] = i
                else:
                    # If owner is already assigned, this cell is the neighbor
                    self._neighbor_cells[face_idx] = i
    
    def get_node_positions(self):
        """Returns all node positions."""
        return self._nodes
    
    def get_cell_centers(self):
        """Returns all cell centers."""
        return self._cell_centers
    
    def get_face_centers(self):
        """Returns all face centers."""
        return self._face_centers
    
    def get_face_normals(self):
        """Returns all face normals."""
        return self._face_normals
    
    def get_face_areas(self):
        """Returns all face areas."""
        return self._face_areas
    
    def get_cell_volumes(self):
        """Returns the volumes (areas in 2D) of all cells."""
        return self._cell_volumes
    
    def get_owner_neighbor(self):
        """Returns owner and neighbor cell indices for all faces."""
        return self._owner_cells, self._neighbor_cells
    
    @property
    def n_cells(self):
        """Returns the number of cells in the mesh."""
        return len(self._cells)
    
    @property
    def n_faces(self):
        """Returns the number of faces in the mesh."""
        return len(self._faces)
    
    @property
    def n_nodes(self):
        """Returns the number of nodes in the mesh."""
        return len(self._nodes)
    
    def plot(self, ax=None, title=None):
        """
        Plot the unstructured mesh on given axes or create new figure if none provided.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        title : str, optional
            Title for the plot
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        # Plot nodes
        ax.plot(self._nodes[:, 0], self._nodes[:, 1], 'ko', markersize=1, alpha=0.6)
        
        # Plot cell edges
        for face in self._faces:
            if len(face) >= 2:
                # For each face, plot the line connecting its nodes
                nodes = self._nodes[face]
                # Close the polygon if it has more than 2 nodes
                if len(face) > 2:
                    nodes = np.vstack([nodes, nodes[0]])
                ax.plot(nodes[:, 0], nodes[:, 1], 'k-', linewidth=0.5)
        
        if title:
            ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        
        return ax
    
    def savePlot(self, filename, title=None, dpi=150, format='pdf'):
        """
        Create and save a plot of the mesh directly to file.
        
        Parameters:
        -----------
        filename : str
            The name of the file to save the plot to
        title : str, optional
            Title for the plot
        dpi : int, optional
            The resolution in dots per inch
        format : str, optional
            The file format to save in ('pdf', 'png', etc.)
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot(ax, title)
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
        plt.close(fig) 