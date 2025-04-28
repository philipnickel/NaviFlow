"""
Structured mesh generation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from .mesh import Mesh

class StructuredMesh(Mesh):
    """
    Base structured mesh class for 2D CFD simulations.
    """
    def __init__(self, x_nodes, y_nodes):
        """
        Initialize a structured mesh with grid points.
        
        Parameters:
        -----------
        x_nodes : ndarray, shape (nx,)
            x-coordinates of grid points
        y_nodes : ndarray, shape (ny,)
            y-coordinates of grid points
        """
        self.x_nodes = np.asarray(x_nodes)
        self.y_nodes = np.asarray(y_nodes)
        
        self.nx = len(self.x_nodes)
        self.ny = len(self.y_nodes)
        
        # Create node coordinates
        self._generate_node_coordinates()
        
        # Create face indices
        self._generate_face_indices()
        
        # Create cell indices
        self._generate_cell_indices()
        
        # Compute geometric properties
        self._compute_geometry()
    
    def _generate_node_coordinates(self):
        """Generate coordinates for all nodes in the mesh."""
        # Create a 2D grid of nodes
        xx, yy = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        
        # Reshape to (nx*ny, 2) array
        self._node_coords = np.column_stack([xx.flatten(), yy.flatten()])
        
        # Create a mapping from (i,j) indices to flattened node index
        self._node_indices = np.arange(self.nx * self.ny).reshape(self.nx, self.ny)
    
    def _generate_face_indices(self):
        """Generate face indices for the structured mesh using vectorized operations."""
        # Calculate total number of faces
        n_x_faces = (self.nx-1) * self.ny  # Faces in x-direction
        n_y_faces = self.nx * (self.ny-1)  # Faces in y-direction
        total_faces = n_x_faces + n_y_faces
        
        # Pre-allocate arrays
        self._faces = []
        self._face_normals = np.zeros((total_faces, 2))
        self._face_areas = np.zeros(total_faces)
        self._face_centers = np.zeros((total_faces, 2))
        
        # Generate x-direction faces (vertical faces)
        face_idx = 0
        
        # Generate indices for all vertical faces at once
        i_indices, j_indices = np.meshgrid(
            np.arange(self.nx-1), 
            np.arange(self.ny), 
            indexing='ij'
        )
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        for idx in range(len(i_indices)):
            i, j = i_indices[idx], j_indices[idx]
            
            # Nodes for this face
            n1 = self._node_indices[i, j]
            n2 = self._node_indices[i+1, j]
            
            if j < self.ny-1:
                n3 = self._node_indices[i+1, j+1]
                n4 = self._node_indices[i, j+1]
                self._faces.append([n1, n2, n3, n4])
            else:
                self._faces.append([n1, n2])
            
            # Face normal is (1,0)
            self._face_normals[face_idx] = [1.0, 0.0]
            
            # Face length
            dy = self.y_nodes[j+1] - self.y_nodes[j] if j < self.ny-1 else 0
            self._face_areas[face_idx] = dy
            
            # Face center
            self._face_centers[face_idx] = [
                (self.x_nodes[i] + self.x_nodes[i+1]) / 2,
                self.y_nodes[j]
            ]
            
            face_idx += 1
        
        # Generate indices for all horizontal faces at once
        i_indices, j_indices = np.meshgrid(
            np.arange(self.nx), 
            np.arange(self.ny-1), 
            indexing='ij'
        )
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        for idx in range(len(i_indices)):
            i, j = i_indices[idx], j_indices[idx]
            
            # Nodes for this face
            n1 = self._node_indices[i, j]
            n2 = self._node_indices[i, j+1]
            
            if i < self.nx-1:
                n3 = self._node_indices[i+1, j+1]
                n4 = self._node_indices[i+1, j]
                self._faces.append([n1, n2, n3, n4])
            else:
                self._faces.append([n1, n2])
            
            # Face normal is (0,1)
            self._face_normals[face_idx] = [0.0, 1.0]
            
            # Face length
            dx = self.x_nodes[i+1] - self.x_nodes[i] if i < self.nx-1 else 0
            self._face_areas[face_idx] = dx
            
            # Face center
            self._face_centers[face_idx] = [
                self.x_nodes[i],
                (self.y_nodes[j] + self.y_nodes[j+1]) / 2
            ]
            
            face_idx += 1
    
    def _generate_cell_indices(self):
        """Generate cell indices for the structured mesh."""
        # Number of cells: (nx-1)*(ny-1)
        n_cells = (self.nx-1) * (self.ny-1)
        n_x_faces = (self.nx-1) * self.ny
        
        # Pre-allocate arrays
        self._cells = [[] for _ in range(n_cells)]
        self._cell_centers = np.zeros((n_cells, 2))
        self._cell_volumes = np.zeros(n_cells)
        
        # Generate indices for all cells at once
        i_indices, j_indices = np.meshgrid(
            np.arange(self.nx-1), 
            np.arange(self.ny-1), 
            indexing='ij'
        )
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        # Vectorized cell centers calculation
        i_midpoints = i_indices + 0.5
        j_midpoints = j_indices + 0.5
        
        # Interpolate x and y coordinates
        x_coords = np.interp(i_midpoints, np.arange(self.nx), self.x_nodes)
        y_coords = np.interp(j_midpoints, np.arange(self.ny), self.y_nodes)
        
        # Assign cell centers
        self._cell_centers = np.column_stack([x_coords, y_coords])
        
        # Calculate cell areas
        dx = np.diff(self.x_nodes)[i_indices]
        dy = np.diff(self.y_nodes)[j_indices]
        self._cell_volumes = dx * dy
        
        # Assign face indices to cells
        for cell_idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            # Face indices for this cell:
            # - West face: i + j*(nx-1)
            west_face = j * (self.nx-1) + i
            
            # - East face: (i+1) + j*(nx-1)
            east_face = j * (self.nx-1) + (i+1)
            
            # - South face: n_x_faces + i*(ny-1) + j
            south_face = n_x_faces + i * (self.ny-1) + j
            
            # - North face: n_x_faces + i*(ny-1) + (j+1)
            north_face = n_x_faces + i * (self.ny-1) + (j+1)
            
            self._cells[cell_idx] = [west_face, east_face, south_face, north_face]
    
    def _compute_geometry(self):
        """Compute geometric properties like owner and neighbor cells."""
        n_faces = len(self._faces)
        n_x_faces = (self.nx-1) * self.ny
        
        # Pre-allocate ownership arrays with -1 (boundary)
        self._owner_cells = np.full(n_faces, -1, dtype=int)
        self._neighbor_cells = np.full(n_faces, -1, dtype=int)
        
        # Compute ownership for x-direction faces
        # Generate indices for all vertical faces at once
        i_indices, j_indices = np.meshgrid(
            np.arange(self.nx-1), 
            np.arange(self.ny), 
            indexing='ij'
        )
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        for face_idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            # Skip faces outside cell domain
            if j >= self.ny-1:
                continue
                
            # The cell to the left (west) of this face is the owner
            if i > 0 or j > 0:
                cell_west = j * (self.nx-1) + i
                self._owner_cells[face_idx] = cell_west
            else:
                # For the first cell, set it as the owner
                self._owner_cells[face_idx] = 0
            
            # The cell to the right (east) of this face is the neighbor
            if i < self.nx-2 and j < self.ny-1:
                cell_east = j * (self.nx-1) + (i+1)
                self._neighbor_cells[face_idx] = cell_east
        
        # Compute ownership for y-direction faces
        i_indices, j_indices = np.meshgrid(
            np.arange(self.nx), 
            np.arange(self.ny-1), 
            indexing='ij'
        )
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()
        
        for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            face_idx = n_x_faces + idx
            
            # Skip faces outside cell domain
            if i >= self.nx-1:
                continue
                
            # The cell above (north) of this face is the owner
            if i < self.nx-1 and j < self.ny-1:
                cell_north = j * (self.nx-1) + i
                self._owner_cells[face_idx] = cell_north
            
            # The cell below (south) of this face is the neighbor
            if i < self.nx-1 and j > 0:
                cell_south = (j-1) * (self.nx-1) + i
                self._neighbor_cells[face_idx] = cell_south
    
    def get_node_positions(self):
        """Returns all node positions."""
        return self._node_coords
    
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
        """
        Returns the volumes (areas in 2D) of all cells in the mesh.
        
        Returns:
        --------
        ndarray : shape (C,)
            Areas of all cells
        """
        return self._cell_volumes
    
    def get_owner_neighbor(self):
        """Returns owner and neighbor cell indices for all faces."""
        return self._owner_cells, self._neighbor_cells
    
    @property
    def n_cells(self):
        """Returns the number of cells in the mesh."""
        return (self.nx - 1) * (self.ny - 1)
    
    @property
    def n_faces(self):
        """Returns the number of faces in the mesh."""
        return (self.nx-1)*self.ny + self.nx*(self.ny-1)
    
    @property
    def n_nodes(self):
        """Returns the number of nodes in the mesh."""
        return self.nx * self.ny
        
    def get_cell_sizes(self):
        """
        Returns the cell sizes.
        
        Returns:
        --------
        tuple : (dx, dy)
            The x and y cell sizes
        """
        if self.nx <= 1 or self.ny <= 1:
            return 0, 0
            
        # For non-uniform mesh, return average cell sizes
        dx_avg = (self.x_nodes[-1] - self.x_nodes[0]) / (self.nx - 1)
        dy_avg = (self.y_nodes[-1] - self.y_nodes[0]) / (self.ny - 1)
        return dx_avg, dy_avg
        
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
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        # Create meshgrid for efficient plotting
        xn, yn = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        
        # Plot nodes
        ax.plot(xn.flatten(), yn.flatten(), 'ko', markersize=1, alpha=0.6)
        
        # Plot grid lines (cell outlines) more efficiently
        # Vertical lines
        for i in range(self.nx):
            ax.plot([self.x_nodes[i], self.x_nodes[i]], 
                    [self.y_nodes[0], self.y_nodes[-1]], 
                    'k-', linewidth=0.5)
        
        # Horizontal lines
        for j in range(self.ny):
            ax.plot([self.x_nodes[0], self.x_nodes[-1]], 
                    [self.y_nodes[j], self.y_nodes[j]], 
                    'k-', linewidth=0.5)
        
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


class StructuredUniform(StructuredMesh):
    """
    Class for uniform structured meshes.
    
    This mesh has evenly spaced nodes in both x and y directions.
    """
    
    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
        """
        Initialize a uniform structured mesh.
        
        Parameters:
        -----------
        nx : int
            Number of nodes in x-direction
        ny : int
            Number of nodes in y-direction
        xmin : float, optional
            Minimum x-coordinate, defaults to 0.0
        xmax : float, optional
            Maximum x-coordinate, defaults to 1.0
        ymin : float, optional
            Minimum y-coordinate, defaults to 0.0
        ymax : float, optional
            Maximum y-coordinate, defaults to 1.0
        """
        # Create evenly spaced nodes
        x_nodes = np.linspace(xmin, xmax, nx)
        y_nodes = np.linspace(ymin, ymax, ny)
            
        # Initialize base class with the node coordinates
        super().__init__(x_nodes, y_nodes)


class StructuredNonUniform(StructuredMesh):
    """
    Class for non-uniform structured meshes.
    
    This mesh has unevenly spaced nodes, with clustering near boundaries.
    """
    
    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, 
                clustering_factor=0.2, beta_x=None, beta_y=None):
        """
        Initialize a non-uniform structured mesh specifically for lid-driven cavity problems.
        
        Parameters:
        -----------
        nx : int
            Number of nodes in x-direction
        ny : int
            Number of nodes in y-direction
        xmin : float, optional
            Minimum x-coordinate, defaults to 0.0
        xmax : float, optional
            Maximum x-coordinate, defaults to 1.0
        ymin : float, optional
            Minimum y-coordinate, defaults to 0.0
        ymax : float, optional
            Maximum y-coordinate, defaults to 1.0
        clustering_factor : float, optional
            Controls the degree of clustering near walls. Lower values (e.g., 0.1-0.3)
            create more clustering, higher values (e.g., 1.0+) create more uniform meshes.
            Defaults to 0.2
        beta_x : float, optional
            Clustering parameter in x-direction. Overrides clustering_factor for x.
        beta_y : float, optional
            Clustering parameter in y-direction. Overrides clustering_factor for y.
            Usually slightly higher than beta_x for more refinement near the lid.
        """
        # Set beta parameters based on clustering factor if not provided
        if beta_x is None:
            beta_x = clustering_factor
        if beta_y is None:
            beta_y = clustering_factor
        
        # Calculate domain dimensions
        length = xmax - xmin
        height = ymax - ymin
        
        # Create node distributions with clustering near walls using tanh function
        # This creates a symmetric grid with refinement near both ends
        xi = np.linspace(-1, 1, nx)
        eta = np.linspace(-1, 1, ny)
        
        # Apply tanh clustering function with potentially different 
        # clustering factors for x and y directions
        x_nodes = xmin + length * (0.5 + 0.5 * np.tanh(beta_x * xi) / np.tanh(beta_x))
        y_nodes = ymin + height * (0.5 + 0.5 * np.tanh(beta_y * eta) / np.tanh(beta_y))
        
        # Initialize base class with the node coordinates
        super().__init__(x_nodes, y_nodes) 