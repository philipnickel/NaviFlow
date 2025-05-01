"""
Base mesh class for CFD simulations.
Designed to support 2D meshes.
"""

import numpy as np
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
        self._face_distances = None
        self._face_interpolation_factors = None

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: {self.n_cells} cells, {self.n_faces} faces>"
        )

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

    def _finalize_geometry(self):
        """
        Finalize mesh geometry by adjusting face normals and computing interpolation factors.
        This method is called after mesh-specific geometry computation is complete.
        """
        self._adjust_face_normals_by_owner()
        self._compute_interpolation_factors()

    def _adjust_face_normals_by_owner(self):
        """
        Adjust face normals to ensure they point outward from the owner cell.
        For internal faces, the normal should point from owner to neighbor.
        For boundary faces, the normal should point outward from the domain.
        """
        face_centers = self.get_face_centers()
        cell_centers = self.get_cell_centers()
        face_normals = self.get_face_normals()
        owner_cells, neighbor_cells = self.get_owner_neighbor()

        for face_idx in range(self.n_faces):
            owner_idx = owner_cells[face_idx]
            if owner_idx == -1:
                continue  # skip invalid faces

            # Ensure normal points outward from owner
            # Vector from owner center to face center
            vec = face_centers[face_idx] - cell_centers[owner_idx]

            # Check if vector has non-zero length before dot product
            if np.linalg.norm(vec) > 1e-12:
                dot = np.dot(face_normals[face_idx], vec)
                if dot < -1e-9:  # Allow for small numerical inaccuracies
                    face_normals[face_idx] *= -1.0
            # else: Handle coincident owner and face centers? Maybe log a warning.
            #     print(f"Warning: Owner cell {owner_idx} center coincides with face {face_idx} center.")

    def _compute_interpolation_factors(self):
        """
        Compute geometric factors needed for face value interpolation.
        This function is called after the mesh geometry has been computed.
        """
        self._face_distances = np.zeros(self.n_faces)
        self._face_interpolation_factors = np.zeros(
            (self.n_faces, 2)
        )  # Stores [g_C, g_F]

        owner_cells, neighbor_cells = self.get_owner_neighbor()
        face_centers = self.get_face_centers()
        cell_centers = self.get_cell_centers()

        for face_idx in range(self.n_faces):
            owner_idx = owner_cells[face_idx]
            neighbor_idx = neighbor_cells[face_idx]

            Cf = face_centers[face_idx]
            Cc = cell_centers[owner_idx]

            if neighbor_idx != -1:
                # Internal face
                Cn = cell_centers[neighbor_idx]
                vec_CN = Cn - Cc
                dist_CN = np.linalg.norm(vec_CN)
                if dist_CN < 1e-12:
                    # Handle coincident cell centers (should ideally not happen)
                    g_C = 0.5
                    g_F = 0.5
                    dist_CfC = 0.0  # Or handle as error
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
            raise IndexError(
                f"Face index {face_idx} out of bounds (0-{self.n_faces - 1})"
            )
        return self._face_interpolation_factors[
            face_idx, 0
        ], self._face_interpolation_factors[face_idx, 1]

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
        pass
