import numpy as np
import matplotlib.pyplot as plt
import pygmsh
from .base import Mesh


class UnstructuredMesh(Mesh):
    def __init__(self, nodes, faces, cells):
        super().__init__()
        self._nodes = (
            np.asarray(nodes)[:, :2] if nodes.shape[1] > 2 else np.asarray(nodes)
        )
        self._faces = faces
        self._cells = cells
        self._compute_geometry()

    def _compute_geometry(self):
        self._compute_face_geometry()
        self._compute_cell_geometry()
        self._assign_owner_neighbor()
        original_normals = np.copy(self._face_normals)
        self._identify_boundary_faces(original_normals)
        self._finalize_geometry()

    def _compute_face_geometry(self):
        n_faces = len(self._faces)
        self._face_centers = np.zeros((n_faces, 2))
        self._face_normals = np.zeros((n_faces, 2))
        self._face_areas = np.zeros(n_faces)

        for i, face in enumerate(self._faces):
            face_nodes = self._nodes[face]
            self._face_centers[i] = np.mean(face_nodes, axis=0)
            if len(face) >= 2:
                dx, dy = (
                    face_nodes[-1, 0] - face_nodes[0, 0],
                    face_nodes[-1, 1] - face_nodes[0, 1],
                )
                length = np.hypot(dx, dy)
                if length > 1e-12:
                    self._face_normals[i] = np.array([dy / length, -dx / length])
                self._face_areas[i] = length

    def _compute_cell_geometry(self):
        n_cells = len(self._cells)
        self._cell_centers = np.zeros((n_cells, 2))
        self._cell_volumes = np.zeros(n_cells)

        for i, cell in enumerate(self._cells):
            cell_node_indices = set()
            for face_idx in cell:
                cell_node_indices.update(self._faces[face_idx])
            cell_nodes = self._nodes[list(cell_node_indices)]
            self._cell_centers[i] = (
                np.mean(cell_nodes, axis=0)
                if cell_nodes.size
                else np.array([np.nan, np.nan])
            )
            if len(cell) > 0:
                cell_face_centers = self._face_centers[cell]
                x, y = cell_face_centers[:, 0], cell_face_centers[:, 1]
                self._cell_volumes[i] = 0.5 * np.abs(
                    np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                )

    def _assign_owner_neighbor(self):
        n_faces = len(self._faces)
        self._owner_cells = np.full(n_faces, -1, dtype=int)
        self._neighbor_cells = np.full(n_faces, -1, dtype=int)
        cell_face_counts = np.zeros(self.n_cells, dtype=int)

        for cell_idx, cell_faces in enumerate(self._cells):
            if not cell_faces:
                raise ValueError(f"Cell {cell_idx} has no associated faces.")
            for face_idx in cell_faces:
                cell_face_counts[cell_idx] += 1
                if self._owner_cells[face_idx] == -1:
                    self._owner_cells[face_idx] = cell_idx
                elif self._neighbor_cells[face_idx] == -1:
                    self._neighbor_cells[face_idx] = cell_idx

        if np.any(cell_face_counts == 0):
            raise ValueError("One or more cells own no faces.")

    def _identify_boundary_faces(self, original_normals):
        for face_idx in range(self.n_faces):
            if self._neighbor_cells[face_idx] == -1:
                owner_idx = self._owner_cells[face_idx]
                if owner_idx != -1:
                    vec = self._face_centers[face_idx] - self._cell_centers[owner_idx]
                    if np.linalg.norm(vec) > 1e-12:
                        if np.dot(original_normals[face_idx], vec) < -1e-9:
                            self._face_normals[face_idx] = -original_normals[face_idx]

    def get_owner_neighbor(self):
        return self._owner_cells, self._neighbor_cells

    def get_cell_centers(self):
        return self._cell_centers

    def get_node_positions(self):
        return self._nodes

    def get_face_centers(self):
        return self._face_centers

    def get_face_normals(self):
        return self._face_normals

    def get_face_areas(self):
        return self._face_areas

    def get_cell_volumes(self):
        return self._cell_volumes

    @property
    def n_cells(self):
        return len(self._cells)

    @property
    def n_faces(self):
        return len(self._faces)

    @property
    def n_nodes(self):
        return len(self._nodes)

    def plot(self, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        ax.plot(self._nodes[:, 0], self._nodes[:, 1], "ko", markersize=1, alpha=0.6)
        for face in self._faces:
            if len(face) >= 2:
                nodes = self._nodes[face]
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
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot(ax, title)
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, format=format, bbox_inches="tight")
        plt.close(fig)

    def _extract_faces_and_cells(self, triangles):
        """
        Extract faces and cells from triangles. Each face is uniquely shared.
        Ensures that every cell (triangle) has exactly 3 valid face indices.
        """
        edge_to_face = {}
        faces = []
        cells = []

        for tri_idx, tri in enumerate(triangles):
            cell_faces = []
            for i in range(3):
                n1, n2 = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted((n1, n2)))
                if edge not in edge_to_face:
                    edge_to_face[edge] = len(faces)
                    faces.append([n1, n2])
                cell_faces.append(edge_to_face[edge])
            if len(cell_faces) != 3:
                raise ValueError(f"Cell {tri_idx} does not have 3 valid faces.")
            cells.append(cell_faces)

        return faces, cells

    def _identify_boundary_faces(self, original_normals):
        xmin, xmax = self._domain_bounds["xmin"], self._domain_bounds["xmax"]
        ymin, ymax = self._domain_bounds["ymin"], self._domain_bounds["ymax"]
        tol = 1e-10
        for i in range(len(self._faces)):
            if self._neighbor_cells[i] == -1:
                c = self._face_centers[i]
                if abs(c[0] - xmin) < tol:
                    self.boundary_face_to_name[i] = "left"
                    self._face_normals[i] = np.array([-1, 0])
                elif abs(c[0] - xmax) < tol:
                    self.boundary_face_to_name[i] = "right"
                    self._face_normals[i] = np.array([1, 0])
                elif abs(c[1] - ymin) < tol:
                    self.boundary_face_to_name[i] = "bottom"
                    self._face_normals[i] = np.array([0, -1])
                elif abs(c[1] - ymax) < tol:
                    self.boundary_face_to_name[i] = "top"
                    self._face_normals[i] = np.array([0, 1])


class UnstructuredRefined(UnstructuredMesh):
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
        self._domain_bounds = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

        with pygmsh.geo.Geometry() as geom:
            p1 = geom.add_point([xmin, ymin, 0.0], mesh_size=mesh_size_walls)
            p2 = geom.add_point([xmax, ymin, 0.0], mesh_size=mesh_size_walls)
            p3 = geom.add_point([xmax, ymax, 0.0], mesh_size=mesh_size_lid)
            p4 = geom.add_point([xmin, ymax, 0.0], mesh_size=mesh_size_lid)

            l1 = geom.add_line(p1, p2)
            l2 = geom.add_line(p2, p3)
            l3 = geom.add_line(p3, p4)
            l4 = geom.add_line(p4, p1)

            cloop = geom.add_curve_loop([l1, l2, l3, l4])
            geom.add_plane_surface(cloop)

            geom.set_mesh_size_callback(
                lambda dim, tag, x, y, z, lc: min(
                    mesh_size_center,
                    mesh_size_walls
                    + min(
                        1.0,
                        1.25
                        * min(abs(x - xmin), abs(x - xmax), abs(y - ymin))
                        / max(xmax - xmin, ymax - ymin),
                    )
                    * (mesh_size_center - mesh_size_walls),
                    mesh_size_lid
                    + min(1.0, 1.25 * abs(y - ymax) / (ymax - ymin))
                    * (mesh_size_center - mesh_size_lid),
                )
            )

            mesh = geom.generate_mesh()

        nodes = mesh.points
        triangles = mesh.cells_dict["triangle"]
        faces, cells = self._extract_faces_and_cells(triangles, nodes)

        super().__init__(nodes, faces, cells)

    def _extract_faces_and_cells(self, triangles, nodes):
        edge_to_face = {}
        faces = []
        cells = []

        for tri in triangles:
            if len(set(tri)) < 3 or np.any(np.array(tri) >= len(nodes)):
                continue  # Skip degenerate or out-of-bounds

            cell_faces = []
            for i in range(3):
                n1, n2 = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted((n1, n2)))
                if edge not in edge_to_face:
                    edge_to_face[edge] = len(faces)
                    faces.append([n1, n2])
                f_idx = edge_to_face[edge]
                cell_faces.append(f_idx)

            if len(cell_faces) == 3:
                cells.append(cell_faces)

        # Post-process: filter out cells that don't own any face
        face_to_owners = {i: [] for i in range(len(faces))}
        for c_idx, cell_faces in enumerate(cells):
            for f in cell_faces:
                face_to_owners[f].append(c_idx)

        face_owners = {f: owners[0] for f, owners in face_to_owners.items() if owners}

        surviving_cells = []
        surviving_indices = set(face_owners.values())
        for c_idx, cell_faces in enumerate(cells):
            if c_idx in surviving_indices:
                surviving_cells.append(cell_faces)

        return faces, surviving_cells

    def _identify_boundary_faces(self, original_normals):
        xmin, xmax = self._domain_bounds["xmin"], self._domain_bounds["xmax"]
        ymin, ymax = self._domain_bounds["ymin"], self._domain_bounds["ymax"]
        tol = 1e-6  # Slightly relaxed
        self.boundary_face_to_name = {}

        for i in range(len(self._faces)):
            if self._neighbor_cells[i] == -1:
                c = self._face_centers[i]
                if abs(c[0] - xmin) < tol:
                    self.boundary_face_to_name[i] = "left"
                    self._face_normals[i] = np.array([-1, 0])
                elif abs(c[0] - xmax) < tol:
                    self.boundary_face_to_name[i] = "right"
                    self._face_normals[i] = np.array([1, 0])
                elif abs(c[1] - ymin) < tol:
                    self.boundary_face_to_name[i] = "bottom"
                    self._face_normals[i] = np.array([0, -1])
                elif abs(c[1] - ymax) < tol:
                    self.boundary_face_to_name[i] = "top"
                    self._face_normals[i] = np.array([0, 1])
                else:
                    raise RuntimeError(
                        f"Boundary face {i} at {c} not classified: "
                        f"expected near edge of domain "
                        f"x=[{xmin},{xmax}], y=[{ymin},{ymax}]"
                        f"face_centers: {self._face_centers}"
                        f"face: {self._faces[i]}"
                    )
