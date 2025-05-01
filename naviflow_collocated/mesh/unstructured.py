import numpy as np
import matplotlib.pyplot as plt
from .base import Mesh
import pygmsh
import sys


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
        """
        Compute all geometric properties of the mesh.
        This method delegates to specific subtasks and finalizes with shared base class logic.
        """
        # Calculate face and cell geometries
        self._compute_face_geometry()
        self._compute_cell_geometry()

        # Establish owner-neighbor relationships
        self._assign_owner_neighbor()

        # Store original normal directions for boundary identification
        # This is important because the _finalize_geometry may change them
        original_normals = np.copy(self._face_normals)

        # Identify boundary faces - must be done after _assign_owner_neighbor
        # But before _finalize_geometry, which may change the normals
        self._identify_boundary_faces(original_normals)

        # Finalize with shared logic from base class
        self._finalize_geometry()

    def _compute_face_geometry(self):
        """
        Compute face centers, normals, and areas.
        """
        n_faces = len(self._faces)

        # Initialize arrays
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
                length = np.sqrt(dx * dx + dy * dy)
                if length > 1e-12:  # Avoid division by zero for degenerate faces
                    self._face_normals[i] = np.array([dy / length, -dx / length])
                else:
                    self._face_normals[i] = np.array(
                        [0.0, 0.0]
                    )  # Or handle degenerate faces differently
                self._face_areas[i] = length

    def _compute_cell_geometry(self):
        """
        Compute cell centers and volumes.
        """
        n_cells = len(self._cells)

        # Initialize arrays
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
                self._cell_centers[i] = np.array(
                    [np.nan, np.nan]
                )  # Handle empty cells if they occur

            # For a 2D cell, calculate volume (area)
            if len(cell) > 0:
                cell_face_centers = self._face_centers[cell]
                x_face_centers = cell_face_centers[:, 0]
                y_face_centers = cell_face_centers[:, 1]
                self._cell_volumes[i] = 0.5 * np.abs(
                    np.dot(x_face_centers, np.roll(y_face_centers, 1))
                    - np.dot(y_face_centers, np.roll(x_face_centers, 1))
                )
            else:
                self._cell_volumes[i] = 0.0

    def _assign_owner_neighbor(self):
        """
        Assign owner and neighbor cells for each face.
        The owner cell is the first cell encountered for a face.
        The neighbor cell is the second cell encountered (if any).
        """
        n_faces = len(self._faces)

        # Initialize arrays
        self._owner_cells = np.full(n_faces, -1, dtype=int)
        self._neighbor_cells = np.full(n_faces, -1, dtype=int)

        # First pass: assign owners and neighbors based on connectivity
        for cell_idx, cell_faces in enumerate(self._cells):
            for face_idx in cell_faces:
                if self._owner_cells[face_idx] == -1:
                    # First cell encountering this face becomes the owner
                    self._owner_cells[face_idx] = cell_idx
                elif self._neighbor_cells[face_idx] == -1:
                    # Second cell encountering this face becomes the neighbor
                    self._neighbor_cells[face_idx] = cell_idx

    def _identify_boundary_faces(self, original_normals):
        """
        Identify boundary faces based on neighbor cell relationships.
        This is a base implementation that can be overridden by specific mesh types.

        Parameters:
        -----------
        original_normals : ndarray, shape (F, 2)
            The original face normals before any adjustments
        """
        # Base implementation just identifies boundaries, without naming them
        for face_idx in range(self.n_faces):
            if self._neighbor_cells[face_idx] == -1:
                # This is a boundary face
                owner_idx = self._owner_cells[face_idx]
                if owner_idx != -1:
                    # Calculate vector from owner center to face center
                    vec_owner_to_face = (
                        self._face_centers[face_idx] - self._cell_centers[owner_idx]
                    )

                    # Check if owner center and face center are coincident
                    if np.linalg.norm(vec_owner_to_face) > 1e-12:
                        dot_product = np.dot(
                            original_normals[face_idx], vec_owner_to_face
                        )

                        if (
                            dot_product < -1e-9
                        ):  # Initial normal points inward from owner
                            # Flip the normal for final storage
                            self._face_normals[face_idx] = -original_normals[face_idx]

    def get_boundary_cell_indices(self, name: str) -> np.ndarray:
        """
        Returns a unique list of owner cell indices whose faces lie on the named boundary.

        Parameters:
        -----------
        name : str
            Boundary name (e.g., 'top', 'left')

        Returns:
        --------
        np.ndarray
            1D array of unique cell indices adjacent to the named boundary.
        """
        face_indices = [
            f
            for f, n in self.boundary_face_to_name.items()
            if n.lower() == name.lower()
        ]
        owner_cells = self._owner_cells[face_indices]
        return np.unique(owner_cells)

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
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Plot nodes
        ax.plot(self._nodes[:, 0], self._nodes[:, 1], "ko", markersize=1, alpha=0.6)

        # Plot cell edges
        for face in self._faces:
            if len(face) >= 2:
                # For each face, plot the line connecting its nodes
                nodes = self._nodes[face]
                # Close the polygon if it has more than 2 nodes
                if len(face) > 2:
                    nodes = np.vstack([nodes, nodes[0]])
                ax.plot(nodes[:, 0], nodes[:, 1], "k-", linewidth=0.5)

        if title:
            ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

        return fig, ax

    def savePlot(self, filename, title=None, dpi=150, format="pdf"):
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
        plt.savefig(filename, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)


class UnstructuredUniform(UnstructuredMesh):
    """
    Class for uniform unstructured meshes.

    This mesh has approximately uniform element sizes across the domain.
    """

    def __init__(self, mesh_size, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
        """
        Initialize a uniform unstructured mesh.

        Parameters:
        -----------
        mesh_size : float
            Characteristic mesh element size
        xmin : float, optional
            Minimum x-coordinate, defaults to 0.0
        xmax : float, optional
            Maximum x-coordinate, defaults to 1.0
        ymin : float, optional
            Minimum y-coordinate, defaults to 0.0
        ymax : float, optional
            Maximum y-coordinate, defaults to 1.0
        """
        # Check dependencies
        if "pygmsh" not in sys.modules or "meshio" not in sys.modules:
            raise ImportError(
                "UnstructuredUniform mesh requires 'pygmsh' and 'meshio' packages."
            )

        # Store domain bounds
        self._domain_bounds = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

        # Calculate domain dimensions
        xmax - xmin
        ymax - ymin

        # Generate mesh using pygmsh
        with pygmsh.geo.Geometry() as geom:
            # Define rectangle corners
            p1 = geom.add_point([xmin, ymin, 0.0], mesh_size=mesh_size)
            p2 = geom.add_point([xmax, ymin, 0.0], mesh_size=mesh_size)
            p3 = geom.add_point([xmax, ymax, 0.0], mesh_size=mesh_size)
            p4 = geom.add_point([xmin, ymax, 0.0], mesh_size=mesh_size)

            # Create lines connecting points
            l1 = geom.add_line(p1, p2)
            l2 = geom.add_line(p2, p3)
            l3 = geom.add_line(p3, p4)
            l4 = geom.add_line(p4, p1)

            # Create boundary curve and surface
            boundary = geom.add_curve_loop([l1, l2, l3, l4])
            geom.add_plane_surface(boundary)

            # Generate the mesh
            mesh_obj = geom.generate_mesh()

        # Process the mesh data directly
        # Extract nodes (vertices)
        nodes = mesh_obj.points

        # Process triangular cells and extract faces
        triangles = mesh_obj.cells_dict["triangle"]

        # Create faces (edges) from triangles
        edges = set()
        for tri in triangles:
            for i in range(3):
                edge = (min(tri[i], tri[(i + 1) % 3]), max(tri[i], tri[(i + 1) % 3]))
                edges.add(edge)

        # Convert to list of faces
        faces = [list(edge) for edge in edges]

        # Create mapping from edge to face index
        edge_to_face = {
            (min(face[0], face[1]), max(face[0], face[1])): i
            for i, face in enumerate(faces)
        }

        # Create cells (each triangle connects to 3 faces)
        cells = []
        for tri in triangles:
            cell_faces = []
            for i in range(3):
                edge = (min(tri[i], tri[(i + 1) % 3]), max(tri[i], tri[(i + 1) % 3]))
                cell_faces.append(edge_to_face[edge])
            cells.append(cell_faces)

        # Initialize base class with the processed mesh data
        super().__init__(nodes, faces, cells)

    def _identify_boundary_faces(self, original_normals):
        """
        Identify and name boundary faces based on their position.

        Parameters:
        -----------
        original_normals : ndarray, shape (F, 2)
            The original face normals before any adjustments
        """
        # Get domain bounds
        xmin = self._domain_bounds["xmin"]
        xmax = self._domain_bounds["xmax"]
        ymin = self._domain_bounds["ymin"]
        ymax = self._domain_bounds["ymax"]

        # Small tolerance for floating point comparison
        tol = 1e-10

        # Identify boundary faces (where neighbor = -1)
        for face_idx in range(len(self._faces)):
            if self._neighbor_cells[face_idx] == -1:
                # This is a boundary face
                center = self._face_centers[face_idx]

                # Determine which boundary this face belongs to
                if abs(center[0] - xmin) < tol:
                    # Left boundary
                    self.boundary_face_to_name[face_idx] = "left"
                    self._face_normals[face_idx] = np.array([-1.0, 0.0])
                elif abs(center[0] - xmax) < tol:
                    # Right boundary
                    self.boundary_face_to_name[face_idx] = "right"
                    self._face_normals[face_idx] = np.array([1.0, 0.0])
                elif abs(center[1] - ymin) < tol:
                    # Bottom boundary
                    self.boundary_face_to_name[face_idx] = "bottom"
                    self._face_normals[face_idx] = np.array([0.0, -1.0])
                elif abs(center[1] - ymax) < tol:
                    # Top boundary
                    self.boundary_face_to_name[face_idx] = "top"
                    self._face_normals[face_idx] = np.array([0.0, 1.0])


class UnstructuredRefined(UnstructuredMesh):
    """
    Generator for refined unstructured meshes for the lid-driven cavity.
    Generates a rectangular domain with a mesh that is refined near walls,
    especially near the top moving lid.
    """

    def __init__(
        self,
        mesh_size_walls,
        mesh_size_lid,
        mesh_size_center,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
    ):
        """
        Initialize a refined unstructured mesh for lid-driven cavity flow.

        Parameters:
        -----------
        mesh_size_walls : float
            Characteristic mesh size near walls
        mesh_size_lid : float
            Characteristic mesh size near the lid (top)
        mesh_size_center : float
            Characteristic mesh size at the center of the domain
        xmin, xmax : float
            Domain limits in the x direction
        ymin, ymax : float
            Domain limits in the y direction
        """
        # Store domain bounds for later use
        self._domain_bounds = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

        # Get domain size
        length = xmax - xmin
        height = ymax - ymin

        # Ensure pygmsh is available
        try:
            import pygmsh
        except ImportError:
            raise ImportError(
                "Pygmsh is required for this mesh generator. Please install it."
            )

        # Try a simple approach with explicit points that ensures connectivity
        with pygmsh.geo.Geometry() as geom:
            # Define corner points first (these connect the walls)
            p1 = geom.add_point(
                [xmin, ymin, 0.0], mesh_size=mesh_size_walls
            )  # Bottom-left
            p2 = geom.add_point(
                [xmax, ymin, 0.0], mesh_size=mesh_size_walls
            )  # Bottom-right
            p3 = geom.add_point([xmax, ymax, 0.0], mesh_size=mesh_size_lid)  # Top-right
            p4 = geom.add_point([xmin, ymax, 0.0], mesh_size=mesh_size_lid)  # Top-left

            # Create lines for outer boundaries
            l1 = geom.add_line(p1, p2)  # Bottom wall
            l2 = geom.add_line(p2, p3)  # Right wall
            l3 = geom.add_line(p3, p4)  # Top wall (lid)
            l4 = geom.add_line(p4, p1)  # Left wall

            # Create boundary curve and plane surface
            curve_loop = geom.add_curve_loop([l1, l2, l3, l4])
            geom.add_plane_surface(curve_loop)

            # Add interior points for mesh size control
            # Create a field that controls mesh size based on distance from boundaries
            # We'll manually add points throughout the domain

            # Middle of each wall with fine mesh size
            geom.add_point(
                [xmin + length / 2, ymin, 0.0], mesh_size=mesh_size_walls * 1.2
            )  # Bottom wall
            geom.add_point(
                [xmax, ymin + height / 2, 0.0], mesh_size=mesh_size_walls * 1.2
            )  # Right wall
            geom.add_point(
                [xmin + length / 2, ymax, 0.0], mesh_size=mesh_size_lid * 1.2
            )  # Top wall (lid)
            geom.add_point(
                [xmin, ymin + height / 2, 0.0], mesh_size=mesh_size_walls * 1.2
            )  # Left wall

            # Quarter points on walls with intermediate mesh sizes
            geom.add_point(
                [xmin + length / 4, ymin, 0.0], mesh_size=mesh_size_walls
            )  # Bottom wall
            geom.add_point(
                [xmin + 3 * length / 4, ymin, 0.0], mesh_size=mesh_size_walls
            )  # Bottom wall
            geom.add_point(
                [xmax, ymin + height / 4, 0.0], mesh_size=mesh_size_walls
            )  # Right wall
            geom.add_point(
                [xmax, ymin + 3 * height / 4, 0.0], mesh_size=mesh_size_lid
            )  # Right wall
            geom.add_point(
                [xmin + length / 4, ymax, 0.0], mesh_size=mesh_size_lid
            )  # Top wall
            geom.add_point(
                [xmin + 3 * length / 4, ymax, 0.0], mesh_size=mesh_size_lid
            )  # Top wall
            geom.add_point(
                [xmin, ymin + height / 4, 0.0], mesh_size=mesh_size_walls
            )  # Left wall
            geom.add_point(
                [xmin, ymin + 3 * height / 4, 0.0], mesh_size=mesh_size_lid
            )  # Left wall

            # Interior points with coarser mesh
            # Center of domain (coarsest)
            geom.add_point(
                [xmin + length / 2, ymin + height / 2, 0.0], mesh_size=mesh_size_center
            )

            # Quarter points inside domain (intermediate sizes)
            h1 = (
                mesh_size_walls + mesh_size_center
            ) / 2  # Intermediate size near walls
            h2 = (mesh_size_lid + mesh_size_center) / 2  # Intermediate size near lid

            # Halfway between walls and center
            geom.add_point(
                [xmin + length / 4, ymin + height / 4, 0.0], mesh_size=h1
            )  # Bottom-left quadrant
            geom.add_point(
                [xmin + 3 * length / 4, ymin + height / 4, 0.0], mesh_size=h1
            )  # Bottom-right quadrant
            geom.add_point(
                [xmin + length / 4, ymin + 3 * height / 4, 0.0], mesh_size=h2
            )  # Top-left quadrant
            geom.add_point(
                [xmin + 3 * length / 4, ymin + 3 * height / 4, 0.0], mesh_size=h2
            )  # Top-right quadrant

            # Set mesh size field - customize element size field
            geom.set_mesh_size_callback(
                lambda dim, tag, x, y, z, lc: min(
                    mesh_size_center,  # Default size
                    # Near bottom/left/right walls - size based on distance
                    mesh_size_walls
                    + min(
                        1.0,
                        (
                            1.25
                            * min(
                                abs(x - xmin),  # Distance from left wall
                                abs(x - xmax),  # Distance from right wall
                                abs(y - ymin),  # Distance from bottom wall
                            )
                            / min(length, height)
                        ),
                    )
                    * (mesh_size_center - mesh_size_walls),
                    # Near lid - special size
                    mesh_size_lid
                    + min(1.0, (1.25 * abs(y - ymax) / height))
                    * (mesh_size_center - mesh_size_lid),
                )
            )

            # Generate the mesh
            mesh_obj = geom.generate_mesh()

        # Process the mesh data directly
        # Extract nodes (vertices)
        nodes = mesh_obj.points

        # Process triangular cells and extract faces
        triangles = mesh_obj.cells_dict["triangle"]

        # Create faces (edges) from triangles
        edges = set()
        for tri in triangles:
            for i in range(3):
                edge = (min(tri[i], tri[(i + 1) % 3]), max(tri[i], tri[(i + 1) % 3]))
                edges.add(edge)

        # Convert to list of faces
        faces = [list(edge) for edge in edges]

        # Create mapping from edge to face index
        edge_to_face = {
            (min(face[0], face[1]), max(face[0], face[1])): i
            for i, face in enumerate(faces)
        }

        # Create cells (each triangle connects to 3 faces)
        cells = []
        for tri in triangles:
            cell_faces = []
            for i in range(3):
                edge = (min(tri[i], tri[(i + 1) % 3]), max(tri[i], tri[(i + 1) % 3]))
                cell_faces.append(edge_to_face[edge])
            cells.append(cell_faces)

        # Initialize base class with the processed mesh data
        super().__init__(nodes, faces, cells)

    def _identify_boundary_faces(self, original_normals):
        """
        Identify and name boundary faces based on their position.

        Parameters:
        -----------
        original_normals : ndarray, shape (F, 2)
            The original face normals before any adjustments
        """
        # Get domain bounds
        xmin = self._domain_bounds["xmin"]
        xmax = self._domain_bounds["xmax"]
        ymin = self._domain_bounds["ymin"]
        ymax = self._domain_bounds["ymax"]

        # Small tolerance for floating point comparison
        tol = 1e-10

        # Identify boundary faces (where neighbor = -1)
        for face_idx in range(len(self._faces)):
            if self._neighbor_cells[face_idx] == -1:
                # This is a boundary face
                center = self._face_centers[face_idx]

                # Determine which boundary this face belongs to
                if abs(center[0] - xmin) < tol:
                    # Left boundary
                    self.boundary_face_to_name[face_idx] = "left"
                    self._face_normals[face_idx] = np.array([-1.0, 0.0])
                elif abs(center[0] - xmax) < tol:
                    # Right boundary
                    self.boundary_face_to_name[face_idx] = "right"
                    self._face_normals[face_idx] = np.array([1.0, 0.0])
                elif abs(center[1] - ymin) < tol:
                    # Bottom boundary
                    self.boundary_face_to_name[face_idx] = "bottom"
                    self._face_normals[face_idx] = np.array([0.0, -1.0])
                elif abs(center[1] - ymax) < tol:
                    # Top boundary
                    self.boundary_face_to_name[face_idx] = "top"
                    self._face_normals[face_idx] = np.array([0.0, 1.0])
