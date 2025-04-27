"""
Base mesh class for CFD simulations.
Designed to support both structured and unstructured meshes.
"""

import numpy as np
from abc import ABC, abstractmethod

class Mesh(ABC):
    """
    Abstract base mesh class for CFD simulations.
    Serves as a common interface for all mesh types.
    """
    
    @abstractmethod
    def get_node_positions(self):
        """
        Returns the positions of all nodes in the mesh.
        
        Returns:
        --------
        ndarray : shape (N, 3)
            Coordinates of all mesh nodes
        """
        pass
    
    @abstractmethod
    def get_cell_centers(self):
        """
        Returns the center positions of all cells in the mesh.
        
        Returns:
        --------
        ndarray : shape (M, 3)
            Coordinates of all cell centers
        """
        pass
    
    @abstractmethod
    def get_face_centers(self):
        """
        Returns the center positions of all faces in the mesh.
        
        Returns:
        --------
        ndarray : shape (F, 3)
            Coordinates of all face centers
        """
        pass
    
    @abstractmethod
    def get_face_normals(self):
        """
        Returns the normal vectors for all faces in the mesh.
        
        Returns:
        --------
        ndarray : shape (F, 3)
            Normal vectors for all faces
        """
        pass
    
    @abstractmethod
    def get_face_areas(self):
        """
        Returns the areas of all faces in the mesh.
        
        Returns:
        --------
        ndarray : shape (F,)
            Areas of all faces
        """
        pass
    
    @abstractmethod
    def get_cell_volumes(self):
        """
        Returns the volumes of all cells in the mesh.
        
        Returns:
        --------
        ndarray : shape (C,)
            Volumes of all cells
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


class UnstructuredMesh(Mesh):
    """
    Unstructured mesh class for CFD simulations.
    Stores topology and geometry information for an unstructured mesh.
    """
    def __init__(self, nodes, faces, cells, boundary_markers=None):
        """
        Initialize an unstructured mesh with nodes, faces, and cells.
        
        Parameters:
        -----------
        nodes : ndarray, shape (N, 3)
            Coordinates of the mesh nodes (vertices)
        faces : list of lists
            Each list contains the indices of nodes that make up a face
        cells : list of lists
            Each list contains the indices of faces that make up a cell
        boundary_markers : dict, optional
            Dictionary mapping face indices to boundary condition types
        """
        self.nodes = np.asarray(nodes)
        self.faces = faces
        self.cells = cells
        self.boundary_markers = boundary_markers or {}
        
        # Geometric properties (to be computed)
        self._cell_centers = None
        self._cell_volumes = None
        self._face_centers = None
        self._face_normals = None
        self._face_areas = None
        
        # Topological properties (to be computed)
        self._owner_cells = None
        self._neighbor_cells = None
        
        # Compute geometric properties
        self.compute_geometry()
    
    def compute_geometry(self):
        """
        Compute all geometric properties of the mesh.
        """
        self.compute_face_centers_and_normals()
        self.compute_cell_centers_and_volumes()
        self.compute_ownership()
    
    def compute_face_centers_and_normals(self):
        """
        Compute centers, normals, and areas for all faces in the mesh.
        """
        num_faces = len(self.faces)
        self._face_centers = np.zeros((num_faces, 3))
        self._face_normals = np.zeros((num_faces, 3))
        self._face_areas = np.zeros(num_faces)
        
        for i, face in enumerate(self.faces):
            # Get coordinates of nodes for this face
            face_nodes = self.nodes[face]
            
            # Compute face center as average of nodes
            self._face_centers[i] = np.mean(face_nodes, axis=0)
            
            # For triangular faces, compute normal using cross product
            if len(face) == 3:
                v1 = face_nodes[1] - face_nodes[0]
                v2 = face_nodes[2] - face_nodes[0]
                normal = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(normal)
                if area > 0:
                    normal = normal / (2 * area)  # Normalize and adjust
            
            # For quadrilateral faces, split into triangles
            elif len(face) == 4:
                v1 = face_nodes[1] - face_nodes[0]
                v2 = face_nodes[2] - face_nodes[0]
                v3 = face_nodes[3] - face_nodes[0]
                
                normal1 = np.cross(v1, v2)
                normal2 = np.cross(v2, v3)
                
                area1 = 0.5 * np.linalg.norm(normal1)
                area2 = 0.5 * np.linalg.norm(normal2)
                
                normal = normal1 + normal2
                area = area1 + area2
                
                if area > 0:
                    normal = normal / (2 * area)  # Normalize and adjust
            
            # For more complex polygons, more sophisticated approach needed
            else:
                # Placeholder for more general polygon handling
                # For now, just compute a basic approximation
                # by triangulating from the first node
                normal = np.zeros(3)
                area = 0
                for j in range(1, len(face)-1):
                    v1 = face_nodes[j] - face_nodes[0]
                    v2 = face_nodes[j+1] - face_nodes[0]
                    tri_normal = np.cross(v1, v2)
                    tri_area = 0.5 * np.linalg.norm(tri_normal)
                    normal += tri_normal
                    area += tri_area
                
                if area > 0:
                    normal = normal / (2 * area)  # Normalize and adjust
            
            self._face_normals[i] = normal
            self._face_areas[i] = area
    
    def compute_cell_centers_and_volumes(self):
        """
        Compute centers and volumes for all cells in the mesh.
        """
        num_cells = len(self.cells)
        self._cell_centers = np.zeros((num_cells, 3))
        self._cell_volumes = np.zeros(num_cells)
        
        for i, cell in enumerate(self.cells):
            # For each cell, we need its faces
            cell_faces = [self.faces[face_idx] for face_idx in cell]
            
            # Get all unique nodes in the cell
            cell_node_indices = set()
            for face in cell_faces:
                cell_node_indices.update(face)
            
            cell_nodes = self.nodes[list(cell_node_indices)]
            
            # Simple approximation of cell center as average of nodes
            self._cell_centers[i] = np.mean(cell_nodes, axis=0)
            
            # Volume calculation requires more sophisticated approach
            # For now, we'll implement a simple approximation
            # For a more accurate implementation, we would decompose into tetrahedra
            # and sum their volumes
            
            # Placeholder for volume calculation
            self._cell_volumes[i] = 0  # To be implemented
    
    def compute_ownership(self):
        """
        Determine owner and neighbor cells for each face.
        Owner is the cell with lower index, neighbor is the higher index.
        Boundary faces have only an owner.
        """
        num_faces = len(self.faces)
        self._owner_cells = np.full(num_faces, -1, dtype=int)
        self._neighbor_cells = np.full(num_faces, -1, dtype=int)
        
        # Build a face-to-cell mapping
        face_to_cells = [[] for _ in range(num_faces)]
        
        for cell_idx, cell in enumerate(self.cells):
            for face_idx in cell:
                face_to_cells[face_idx].append(cell_idx)
        
        # Assign owner and neighbor
        for face_idx, cells in enumerate(face_to_cells):
            if len(cells) >= 1:
                self._owner_cells[face_idx] = cells[0]
            if len(cells) >= 2:
                self._neighbor_cells[face_idx] = cells[1]
            # If there are more than 2 cells sharing a face, that's an error
            # in the mesh definition, but we'll just ignore that for now
    
    def get_node_positions(self):
        """Returns all node positions."""
        return self.nodes
    
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
        Returns the volumes of all cells in the mesh.
        
        Returns:
        --------
        ndarray : shape (C,)
            Volumes of all cells
        """
        return self._cell_volumes
    
    def get_owner_neighbor(self):
        """Returns owner and neighbor cell indices for all faces."""
        return self._owner_cells, self._neighbor_cells
    
    @property
    def n_cells(self):
        """Returns the number of cells in the mesh."""
        return len(self.cells)
    
    @property
    def n_faces(self):
        """Returns the number of faces in the mesh."""
        return len(self.faces)
    
    @property
    def n_nodes(self):
        """Returns the number of nodes in the mesh."""
        return len(self.nodes)

    def plot(self, ax, title=None):
        """
        Plots the nodes and edges (faces in 2D) of the unstructured mesh.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        title : str, optional
            Title for the plot
        """
        nodes = self.get_node_positions()
        
        # Plot nodes
        ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=1, alpha=0.6)
        
        # Plot edges
        for face_nodes in self.faces:
            node_coords = nodes[face_nodes]
            ax.plot(node_coords[:, 0], node_coords[:, 1], 'k-', linewidth=0.5)
        
        if title:
            ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')


class StructuredMeshBase(Mesh):
    """
    Base class for structured meshes (both uniform and non-uniform).
    """
    def __init__(self, x_nodes, y_nodes, z_nodes=None):
        """
        Initialize a structured mesh with grid points.
        
        Parameters:
        -----------
        x_nodes : ndarray, shape (nx,)
            x-coordinates of grid points
        y_nodes : ndarray, shape (ny,)
            y-coordinates of grid points
        z_nodes : ndarray, shape (nz,), optional
            z-coordinates of grid points (for 3D meshes)
        """
        self.x_nodes = np.asarray(x_nodes)
        self.y_nodes = np.asarray(y_nodes)
        self.z_nodes = np.asarray(z_nodes) if z_nodes is not None else np.array([0.0])
        
        self.nx = len(self.x_nodes)
        self.ny = len(self.y_nodes)
        self.nz = len(self.z_nodes)
        
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
        # Create a 3D grid of nodes
        xx, yy, zz = np.meshgrid(self.x_nodes, self.y_nodes, self.z_nodes, indexing='ij')
        
        # Reshape to (nx*ny*nz, 3) array
        self._node_coords = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        
        # Create a mapping from (i,j,k) indices to flattened node index
        self._node_indices = np.arange(self.nx * self.ny * self.nz).reshape(self.nx, self.ny, self.nz)
    
    def _generate_face_indices(self):
        """Generate face indices for the structured mesh."""
        # Faces in x-direction (nx-1)*(ny)*(nz) faces
        # Faces in y-direction (nx)*(ny-1)*(nz) faces
        # Faces in z-direction (nx)*(ny)*(nz-1) faces
        
        # Calculate total number of faces
        total_faces = 0
        if self.nz > 1:  # 3D
            total_faces = (self.nx-1)*self.ny*self.nz + self.nx*(self.ny-1)*self.nz + self.nx*self.ny*(self.nz-1)
        else:  # 2D
            total_faces = (self.nx-1)*self.ny + self.nx*(self.ny-1)
        
        # Initialize arrays for face properties
        self._faces = []
        self._face_normals = np.zeros((total_faces, 3))
        self._face_areas = np.zeros(total_faces)
        self._face_centers = np.zeros((total_faces, 3))
        
        # Helper to get node index from (i,j,k)
        def node_idx(i, j, k):
            return self._node_indices[i, j, k]
        
        face_idx = 0
        
        # Create faces in x-direction (faces perpendicular to x-axis)
        for i in range(self.nx-1):
            for j in range(self.ny):
                for k in range(self.nz):
                    # Face between (i,j,k) and (i+1,j,k)
                    node1 = node_idx(i, j, k)
                    node2 = node_idx(i+1, j, k)
                    
                    if k < self.nz-1:
                        node3 = node_idx(i+1, j, k+1)
                        node4 = node_idx(i, j, k+1)
                        self._faces.append([node1, node2, node3, node4])
                    else:
                        node3 = node_idx(i+1, j, k)
                        node4 = node_idx(i, j, k)
                        self._faces.append([node1, node2, node3, node4])
                    
                    # Face normal is (1,0,0)
                    self._face_normals[face_idx] = [1.0, 0.0, 0.0]
                    
                    # Face area
                    if self.nz > 1:  # 3D
                        dy = self.y_nodes[j+1] - self.y_nodes[j] if j < self.ny-1 else 0
                        dz = self.z_nodes[k+1] - self.z_nodes[k] if k < self.nz-1 else 0
                        self._face_areas[face_idx] = dy * dz
                    else:  # 2D
                        dy = self.y_nodes[j+1] - self.y_nodes[j] if j < self.ny-1 else 0
                        self._face_areas[face_idx] = dy
                    
                    # Face center
                    if self.nz > 1:  # 3D
                        self._face_centers[face_idx] = [
                            (self.x_nodes[i] + self.x_nodes[i+1]) / 2,
                            self.y_nodes[j],
                            self.z_nodes[k]
                        ]
                    else:  # 2D
                        self._face_centers[face_idx] = [
                            (self.x_nodes[i] + self.x_nodes[i+1]) / 2,
                            self.y_nodes[j],
                            0.0
                        ]
                    
                    face_idx += 1
        
        # Create faces in y-direction (faces perpendicular to y-axis)
        for i in range(self.nx):
            for j in range(self.ny-1):
                for k in range(self.nz):
                    # Face between (i,j,k) and (i,j+1,k)
                    node1 = node_idx(i, j, k)
                    node2 = node_idx(i, j+1, k)
                    
                    if i < self.nx-1 and k < self.nz-1:
                        node3 = node_idx(i+1, j+1, k+1)
                        node4 = node_idx(i+1, j, k+1)
                        self._faces.append([node1, node2, node3, node4])
                    else:
                        node3 = node_idx(i, j+1, k)
                        node4 = node_idx(i, j, k)
                        self._faces.append([node1, node2, node3, node4])
                    
                    # Face normal is (0,1,0)
                    self._face_normals[face_idx] = [0.0, 1.0, 0.0]
                    
                    # Face area
                    if self.nz > 1:  # 3D
                        dx = self.x_nodes[i+1] - self.x_nodes[i] if i < self.nx-1 else 0
                        dz = self.z_nodes[k+1] - self.z_nodes[k] if k < self.nz-1 else 0
                        self._face_areas[face_idx] = dx * dz
                    else:  # 2D
                        dx = self.x_nodes[i+1] - self.x_nodes[i] if i < self.nx-1 else 0
                        self._face_areas[face_idx] = dx
                    
                    # Face center
                    if self.nz > 1:  # 3D
                        self._face_centers[face_idx] = [
                            self.x_nodes[i],
                            (self.y_nodes[j] + self.y_nodes[j+1]) / 2,
                            self.z_nodes[k]
                        ]
                    else:  # 2D
                        self._face_centers[face_idx] = [
                            self.x_nodes[i],
                            (self.y_nodes[j] + self.y_nodes[j+1]) / 2,
                            0.0
                        ]
                    
                    face_idx += 1
        
        # Create faces in z-direction (for 3D meshes)
        if self.nz > 1:
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz-1):
                        # Face between (i,j,k) and (i,j,k+1)
                        node1 = node_idx(i, j, k)
                        node2 = node_idx(i, j, k+1)
                        
                        if i < self.nx-1 and j < self.ny-1:
                            node3 = node_idx(i+1, j+1, k+1)
                            node4 = node_idx(i+1, j, k+1)
                            self._faces.append([node1, node2, node3, node4])
                        else:
                            node3 = node_idx(i, j, k+1)
                            node4 = node_idx(i, j, k)
                            self._faces.append([node1, node2, node3, node4])
                        
                        # Face normal is (0,0,1)
                        self._face_normals[face_idx] = [0.0, 0.0, 1.0]
                        
                        # Face area
                        dx = self.x_nodes[i+1] - self.x_nodes[i] if i < self.nx-1 else 0
                        dy = self.y_nodes[j+1] - self.y_nodes[j] if j < self.ny-1 else 0
                        self._face_areas[face_idx] = dx * dy
                        
                        # Face center
                        self._face_centers[face_idx] = [
                            (self.x_nodes[i] + self.x_nodes[i+1]) / 2 if i < self.nx-1 else self.x_nodes[i],
                            (self.y_nodes[j] + self.y_nodes[j+1]) / 2 if j < self.ny-1 else self.y_nodes[j],
                            (self.z_nodes[k] + self.z_nodes[k+1]) / 2
                        ]
                        
                        face_idx += 1
    
    def _generate_cell_indices(self):
        """Generate cell indices for the structured mesh."""
        # Number of cells: (nx-1)*(ny-1)*(nz-1)
        n_cells = (self.nx-1) * (self.ny-1) * (1 if self.nz <= 1 else self.nz-1)
        self._cells = [[] for _ in range(n_cells)]
        self._cell_centers = np.zeros((n_cells, 3))
        self._cell_volumes = np.zeros(n_cells)
        
        # Calculate face offsets for different directions
        # In 2D, faces in x-direction (vertical faces) are indexed first,
        # followed by faces in y-direction (horizontal faces)
        n_x_faces = (self.nx-1) * self.ny
        
        cell_idx = 0
        for i in range(self.nx-1):
            for j in range(self.ny-1):
                for k in range(self.nz-1) if self.nz > 1 else [0]:
                    # A cell is defined by its corner nodes
                    i0, j0, k0 = i, j, k
                    i1, j1, k1 = i+1, j+1, k+1 if self.nz > 1 else k
                    
                    # Corner nodes of the cell
                    node_indices = [
                        self._node_indices[i0, j0, k0],
                        self._node_indices[i1, j0, k0],
                        self._node_indices[i1, j1, k0],
                        self._node_indices[i0, j1, k0]
                    ]
                    
                    if self.nz > 1:  # 3D mesh
                        node_indices.extend([
                            self._node_indices[i0, j0, k1],
                            self._node_indices[i1, j0, k1],
                            self._node_indices[i1, j1, k1],
                            self._node_indices[i0, j1, k1]
                        ])
                    
                    # Cell center is the average of corner nodes
                    cell_nodes = np.array([self._node_coords[idx] for idx in node_indices])
                    self._cell_centers[cell_idx] = np.mean(cell_nodes, axis=0)
                    
                    # Cell volume (area in 2D)
                    dx = self.x_nodes[i1] - self.x_nodes[i0]
                    dy = self.y_nodes[j1] - self.y_nodes[j0]
                    
                    if self.nz > 1:  # 3D
                        dz = self.z_nodes[k1] - self.z_nodes[k0]
                        self._cell_volumes[cell_idx] = dx * dy * dz
                    else:  # 2D
                        self._cell_volumes[cell_idx] = dx * dy
                    
                    # For the mesh structure, identify the faces of this cell
                    if self.nz <= 1:  # 2D mesh
                        # Face indices for this cell:
                        # - West face: i + j*(nx-1)
                        # - East face: (i+1) + j*(nx-1)
                        # - South face: n_x_faces + i*(ny-1) + (j)
                        # - North face: n_x_faces + i*(ny-1) + (j+1)
                        
                        # West face
                        west_face = j * (self.nx-1) + i
                        self._cells[cell_idx].append(west_face)
                        
                        # East face
                        east_face = j * (self.nx-1) + (i+1)
                        self._cells[cell_idx].append(east_face)
                        
                        # South face
                        south_face = n_x_faces + i * (self.ny-1) + j
                        self._cells[cell_idx].append(south_face)
                        
                        # North face
                        north_face = n_x_faces + i * (self.ny-1) + (j+1)
                        self._cells[cell_idx].append(north_face)
                    else:
                        # 3D case - would need to add faces for all 6 sides of the cell
                        pass
                    
                    cell_idx += 1
    
    def _compute_geometry(self):
        """Compute geometric properties like owner and neighbor cells."""
        # Make sure faces have been generated
        if not hasattr(self, '_faces') or len(self._faces) == 0:
            self._generate_face_indices()
        
        # Make sure cells have been generated 
        if not hasattr(self, '_cells') or len(self._cells) == 0:
            self._generate_cell_indices()
        
        num_faces = len(self._faces)
        self._owner_cells = np.full(num_faces, -1, dtype=int)
        self._neighbor_cells = np.full(num_faces, -1, dtype=int)
        
        # For a structured grid, we can directly compute the ownership based on the face index
        # This is a 2D case implementation - would need to be extended for 3D
        
        # In 2D, faces in x-direction (vertical faces)
        # First (nx-1)*ny faces are in x-direction
        n_x_faces = (self.nx-1) * self.ny
        
        # x-direction faces (vertical faces)
        for j in range(self.ny):  # Loop over rows
            for i in range(self.nx-1):  # Loop over internal x-faces
                face_idx = j * (self.nx-1) + i
                
                # The cell to the left (west) of this face is the owner
                if i > 0 or j > 0:  # Skip the first cell if we're using it as a reference cell
                    cell_west = j * (self.nx-1) + i
                    self._owner_cells[face_idx] = cell_west
                else:
                    # For the first cell, set it as the owner anyway
                    self._owner_cells[face_idx] = 0
                
                # The cell to the right (east) of this face is the neighbor
                if i < self.nx-2 and j < self.ny-1:  # Only if there's a cell to the east
                    cell_east = j * (self.nx-1) + (i+1)
                    self._neighbor_cells[face_idx] = cell_east
        
        # y-direction faces (horizontal faces)
        for i in range(self.nx):
            for j in range(self.ny-1):  # Loop over internal y-faces
                face_idx = n_x_faces + i * (self.ny-1) + j
                
                # The cell above (north) of this face is the owner
                if i < self.nx-1 and j < self.ny-1:  # Only if there's a cell
                    cell_north = j * (self.nx-1) + i
                    self._owner_cells[face_idx] = cell_north
                
                # The cell below (south) of this face is the neighbor
                if i < self.nx-1 and j > 0:  # Only if there's a cell to the south and it's valid
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
        Returns the volumes of all cells in the mesh.
        
        Returns:
        --------
        ndarray : shape (C,)
            Volumes of all cells
        """
        return self._cell_volumes
    
    def get_owner_neighbor(self):
        """Returns owner and neighbor cell indices for all faces."""
        return self._owner_cells, self._neighbor_cells
    
    @property
    def n_cells(self):
        """Returns the number of cells in the mesh."""
        return (self.nx - 1) * (self.ny - 1) * (self.nz - 1 if self.nz > 1 else 1)
    
    @property
    def n_faces(self):
        """Returns the number of faces in the mesh."""
        if self.nz > 1:  # 3D
            return (self.nx-1)*self.ny*self.nz + self.nx*(self.ny-1)*self.nz + self.nx*self.ny*(self.nz-1)
        else:  # 2D
            return (self.nx-1)*self.ny + self.nx*(self.ny-1)
    
    @property
    def n_nodes(self):
        """Returns the number of nodes in the mesh."""
        return self.nx * self.ny * self.nz

    def plot(self, ax, title=None):
        """
        Plots the nodes and cell outlines of the structured mesh.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        title : str, optional
            Title for the plot
        """
        # Plot nodes
        xn, yn = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        ax.plot(xn, yn, 'ko', markersize=1, alpha=0.6)
        
        # Plot grid lines (cell outlines)
        for i in range(self.nx):
            ax.plot(np.full(self.ny, self.x_nodes[i]), self.y_nodes, 'k-', linewidth=0.5)
        for j in range(self.ny):
            ax.plot(self.x_nodes, np.full(self.nx, self.y_nodes[j]), 'k-', linewidth=0.5)
        
        if title:
            ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')


class UniformStructuredMesh(StructuredMeshBase):
    """
    Uniform structured mesh with constant spacing in each direction.
    """
    def __init__(self, xmin, xmax, ymin, ymax, nx, ny, zmin=0.0, zmax=0.0, nz=1):
        """
        Initialize a uniform structured mesh.
        
        Parameters:
        -----------
        xmin, xmax : float
            Domain limits in the x direction
        ymin, ymax : float
            Domain limits in the y direction
        nx, ny : int
            Number of nodes in x and y directions
        zmin, zmax : float, optional
            Domain limits in the z direction (for 3D meshes)
        nz : int, optional
            Number of nodes in z direction (for 3D meshes)
        """
        # Generate uniform grid points
        x_nodes = np.linspace(xmin, xmax, nx)
        y_nodes = np.linspace(ymin, ymax, ny)
        z_nodes = np.linspace(zmin, zmax, nz) if nz > 1 else np.array([0.0])
        
        super().__init__(x_nodes, y_nodes, z_nodes)


class NonUniformStructuredMesh(StructuredMeshBase):
    """
    Non-uniform structured mesh with variable spacing in each direction.
    """
    def __init__(self, x_nodes, y_nodes, z_nodes=None):
        """
        Initialize a non-uniform structured mesh.
        
        Parameters:
        -----------
        x_nodes : ndarray, shape (nx,)
            x-coordinates of grid points
        y_nodes : ndarray, shape (ny,)
            y-coordinates of grid points
        z_nodes : ndarray, shape (nz,), optional
            z-coordinates of grid points (for 3D meshes)
        """
        super().__init__(x_nodes, y_nodes, z_nodes) 