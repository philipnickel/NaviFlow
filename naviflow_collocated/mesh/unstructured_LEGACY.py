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
        # Will be overridden in subclass using Gmsh physical tags
        pass

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


class UnstructuredRefined(Mesh):
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
        self._domain_bounds = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }

        with pygmsh.geo.Geometry() as geom:
            # Define corner points
            p1 = geom.add_point(
                [xmin, ymin, 0.0], mesh_size=mesh_size_walls
            )  # bottom-left
            p2 = geom.add_point(
                [xmax, ymin, 0.0], mesh_size=mesh_size_walls
            )  # bottom-right
            p3 = geom.add_point([xmax, ymax, 0.0], mesh_size=mesh_size_lid)  # top-right
            p4 = geom.add_point([xmin, ymax, 0.0], mesh_size=mesh_size_lid)  # top-left

            # Define boundary lines
            l_bottom = geom.add_line(p1, p2)
            l_right = geom.add_line(p2, p3)
            l_top = geom.add_line(p3, p4)
            l_left = geom.add_line(p4, p1)

            # Tag boundaries with physical names
            geom.add_physical([l_left], label="left")
            geom.add_physical([l_right], label="right")
            geom.add_physical([l_top], label="top")
            geom.add_physical([l_bottom], label="bottom")

            # Create surface and tag it
            boundary_loop = geom.add_curve_loop([l_bottom, l_right, l_top, l_left])
            surface = geom.add_plane_surface(boundary_loop)
            geom.add_physical(surface, label="fluid")

            # Custom refinement callback
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

        # Parse physical data
        self._line_edges = mesh.cells_dict.get("line", [])
        self._line_data = mesh.cell_data_dict["gmsh:physical"].get("line", [])
        self._line_tags = mesh.field_data

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
                continue

            cell_faces = []
            for i in range(3):
                n1, n2 = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted((n1, n2)))
                if edge not in edge_to_face:
                    edge_to_face[edge] = len(faces)
                    faces.append([n1, n2])
                cell_faces.append(edge_to_face[edge])

            if len(cell_faces) == 3:
                cells.append(cell_faces)

        face_to_owners = {i: [] for i in range(len(faces))}
        for c_idx, cell_faces in enumerate(cells):
            for f in cell_faces:
                face_to_owners[f].append(c_idx)

        face_owners = {f: owners[0] for f, owners in face_to_owners.items() if owners}
        surviving_cells = [
            cell_faces
            for c_idx, cell_faces in enumerate(cells)
            if c_idx in set(face_owners.values())
        ]

        return faces, surviving_cells

    def _identify_boundary_faces(self, original_normals):
        tag_map = {v[0]: k for k, v in self._line_tags.items() if v[1] == 1}
        self.boundary_face_to_name = {}

        for i, face in enumerate(self._faces):
            if self._neighbor_cells[i] != -1:
                continue
            a, b = sorted(face)
            for (v1, v2), tag in zip(self._line_edges, self._line_data):
                if {a, b} == {v1, v2}:
                    self.boundary_face_to_name[i] = tag_map.get(tag, "unknown")
                    break
            else:
                raise RuntimeError(
                    f"Could not classify face {i} with nodes {face} as a boundary."
                )
