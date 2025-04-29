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
    
    def __init__(self):
        """
        Initialize the mesh with an empty boundary face mapping.
        """
        self.boundary_face_to_name = {}  # face index → boundary name
    
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
    
    def get_boundary_name(self, face_idx):
        """
        Return boundary name ('top', 'bottom', 'left', 'right') if face is a boundary face.
        
        Parameters:
        -----------
        face_idx : int
            Index of the face to check
            
        Returns:
        --------
        str or None
            Name of the boundary if face is a boundary face, None otherwise
        """
        return self.boundary_face_to_name.get(face_idx, None)
    
    def get_field_shapes(self):
        """
        Return the field shapes for collocated (u, v, p) fields.
        Base implementation that may be overridden by specific mesh types.

        Returns:
        --------
        tuple
            (u_shape, v_shape, p_shape)
        """
        # Default implementation for unstructured meshes
        n_cells = self.n_cells
        return n_cells, n_cells, n_cells
    
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
        super().__init__()
        
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
                
                # Normal is (dy, -dx) normalized for outward pointing convention
                # assuming counter-clockwise node ordering in faces relative to cells
                length = np.sqrt(dx*dx + dy*dy)
                if length > 1e-12: # Avoid division by zero for degenerate faces
                    self._face_normals[i] = np.array([dy/length, -dx/length])
                else:
                    self._face_normals[i] = np.array([0.0, 0.0]) # Or handle degenerate faces differently
                self._face_areas[i] = length
        
        # Store initial normals temporarily
        initial_normals = np.copy(self._face_normals)
        
        # Compute cell centers (using nodes) and volumes (using face centers - TODO revisit)
        self._cell_centers = np.zeros((n_cells, 2))
        self._cell_volumes = np.zeros(n_cells)
        
        for i, cell in enumerate(self._cells):
            # Get node indices for this cell (unique nodes forming the cell polygon)
            cell_node_indices = set()
            for face_idx in cell:
                cell_node_indices.update(self._faces[face_idx])
            cell_nodes = self._nodes[list(cell_node_indices)]

            # Cell center is the average of node coordinates (centroid of vertices)
            if cell_nodes.shape[0] > 0:
                 self._cell_centers[i] = np.mean(cell_nodes, axis=0)
            else:
                 self._cell_centers[i] = np.array([np.nan, np.nan]) # Handle empty cells if they occur

            # For a 2D cell, volume (area) can be computed using node coordinates (Shoelace formula)
            # Re-implement using nodes for potentially better accuracy
            x_nodes = cell_nodes[:, 0]
            y_nodes = cell_nodes[:, 1]
            # We need nodes in order. This requires more complex polygon reconstruction.
            # Sticking with face center average for volume for now, but center uses nodes.
            # TODO: Revisit volume calculation using ordered nodes if needed.
            if len(cell) > 0:
                cell_face_centers = self._face_centers[cell]
                x_face_centers = cell_face_centers[:, 0]
                y_face_centers = cell_face_centers[:, 1]
                self._cell_volumes[i] = 0.5 * np.abs(np.dot(x_face_centers, np.roll(y_face_centers, 1)) - np.dot(y_face_centers, np.roll(x_face_centers, 1)))
            else:
                self._cell_volumes[i] = 0.0
        
        # --- Pass 1: Assign Owner/Neighbor based on connectivity --- 
        self._owner_cells = np.full(n_faces, -1, dtype=int)
        self._neighbor_cells = np.full(n_faces, -1, dtype=int)

        for cell_idx, cell_faces in enumerate(self._cells):
            for face_idx in cell_faces:
                if self._owner_cells[face_idx] == -1:
                    # First cell encountering this face becomes the owner
                    self._owner_cells[face_idx] = cell_idx
                elif self._neighbor_cells[face_idx] == -1: 
                    # Second cell encountering this face becomes the neighbor
                    self._neighbor_cells[face_idx] = cell_idx
                # else: # Should not happen for triangular 2D meshes (max 2 cells per face)
                #    pass

        # --- Pass 2: Correct Normal Orientation based on Owner --- 
        # Ensure final stored normals point outward from the assigned owner cell
        for face_idx in range(n_faces):
            owner_idx = self._owner_cells[face_idx]
            initial_normal = initial_normals[face_idx]
            
            if owner_idx != -1:
                # Calculate vector from owner center to face center
                vec_owner_to_face = self._face_centers[face_idx] - self._cell_centers[owner_idx]
                
                # Check if owner center and face center are coincident
                if np.linalg.norm(vec_owner_to_face) > 1e-12:
                    dot_product = np.dot(initial_normal, vec_owner_to_face)
                    
                    if dot_product < -1e-9: # Initial normal points inward from owner
                        # Flip the normal for final storage
                        self._face_normals[face_idx] = -initial_normal
                    else:
                        # Initial normal points outward (or perpendicular), keep it
                        self._face_normals[face_idx] = initial_normal
                else:
                    # Coincident centers, keep initial normal orientation (arbitrary)
                    self._face_normals[face_idx] = initial_normal
            else:
                 # No owner assigned (e.g., orphan face - shouldn't happen in valid mesh)
                 # Keep the initial normal orientation
                 self._face_normals[face_idx] = initial_normal
                            
        # Compute geometric quantities for interpolation
        self._compute_interpolation_factors()
    
    def _compute_interpolation_factors(self):
        """Compute geometric factors needed for face value interpolation."""
        self._face_distances = np.zeros(self.n_faces)
        self._face_interpolation_factors = np.zeros((self.n_faces, 2)) # Stores [g_C, g_F]

        for face_idx in range(self.n_faces):
            owner_idx = self._owner_cells[face_idx]
            neighbor_idx = self._neighbor_cells[face_idx]

            Cf = self._face_centers[face_idx]
            Cc = self._cell_centers[owner_idx]

            if neighbor_idx != -1:
                # Internal face
                Cn = self._cell_centers[neighbor_idx]
                vec_CN = Cn - Cc
                dist_CN = np.linalg.norm(vec_CN)
                if dist_CN < 1e-12:
                    # Handle coincident cell centers (should ideally not happen)
                    g_C = 0.5
                    g_F = 0.5
                    dist_CfC = 0.0 # Or handle as error
                else:
                    vec_e = vec_CN / dist_CN
                    dist_CfC = np.dot(Cf - Cc, vec_e)
                    g_F = dist_CfC / dist_CN
                    g_C = 1.0 - g_F
                    # Clamp weights to avoid extrapolation issues, though ideally handled upstream
                    g_F = np.clip(g_F, 0.0, 1.0)
                    g_C = 1.0 - g_F
                self._face_distances[face_idx] = dist_CN
            else:
                # Boundary face
                # Distance is from cell center Cc to face center Cf
                # Interpolation weights are 1 for owner, 0 for neighbor
                vec_CfC = Cf - Cc
                dist_CfC = np.linalg.norm(vec_CfC)
                g_C = 1.0
                g_F = 0.0
                # Store distance from cell center to boundary face center
                # May need adjustment based on BC application
                self._face_distances[face_idx] = dist_CfC

            self._face_interpolation_factors[face_idx, 0] = g_C
            self._face_interpolation_factors[face_idx, 1] = g_F

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
    
    def get_face_interpolation_factors(self, face_idx):
        """
        Returns the interpolation factors (weights) for a given face.
        These factors are used to interpolate values from adjacent cell centers
        to the face center: phi_f = g_C * phi_C + g_F * phi_F
        where C is the owner cell and F is the neighbor cell.

        Parameters:
        -----------
        face_idx : int
            Index of the face

        Returns:
        --------
        tuple : (g_C, g_F)
            Interpolation weights for the owner (C) and neighbor (F) cells.
            For boundary faces, g_F is typically 0 and g_C is 1.
        """
        if face_idx < 0 or face_idx >= self.n_faces:
            raise IndexError(f"Face index {face_idx} out of bounds (0-{self.n_faces-1})")
        return self._face_interpolation_factors[face_idx, 0], self._face_interpolation_factors[face_idx, 1]

    def get_face_distances(self):
        """
        Returns the distances relevant for gradient calculations across faces.
        - For internal faces: Distance between owner and neighbor cell centers (d_CF).
        - For boundary faces: Distance between owner cell center and face center (d_Cf).

        Returns:
        --------
        ndarray : shape (F,)
            Distances for all faces.
        """
        return self._face_distances

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