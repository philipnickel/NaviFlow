import numpy as np
import matplotlib.pyplot as plt
from .mesh import Mesh

class StructuredMesh(Mesh):
    def __init__(self, n_cells_x, n_cells_y, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 is_uniform=True, clustering_factor=0.2, beta_x=None, beta_y=None,
                 boundary_names=None, boundary_conditions=None):
        super().__init__()

        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.nx = n_cells_x + 1
        self.ny = n_cells_y + 1

        if is_uniform:
            self.x_nodes, self.y_nodes = np.linspace(xmin, xmax, self.nx), np.linspace(ymin, ymax, self.ny)
        else:
            bx = beta_x if beta_x is not None else clustering_factor
            by = beta_y if beta_y is not None else clustering_factor
            self.x_nodes, self.y_nodes = self._generate_clustered_nodes(bx, by)

        self.boundary_names = boundary_names or {'left': 'left', 'right': 'right', 'bottom': 'bottom', 'top': 'top'}

        # Store boundary conditions, provide default for cavity if none given
        self._boundary_conditions = boundary_conditions or {
            'left': {'u': 0.0, 'v': 0.0},
            'right': {'u': 0.0, 'v': 0.0},
            'bottom': {'u': 0.0, 'v': 0.0},
            'top': {'u': 1.0, 'v': 0.0} # Default cavity BC
        }
        # Ensure all named boundaries have a BC entry
        for name in self.boundary_names.values():
            if name not in self._boundary_conditions:
                 # Add default zero velocity if a boundary name exists but has no BC specified
                 self._boundary_conditions[name] = {'u': 0.0, 'v': 0.0}

        self._generate_node_coordinates()
        self._generate_faces_and_cells()
        self._compute_geometry()
        self._cache_boundary_indices()

    def _generate_clustered_nodes(self, beta_x, beta_y):
        xi = np.linspace(-1, 1, self.nx)
        eta = np.linspace(-1, 1, self.ny)
        x = self.xmin + (self.xmax - self.xmin) * (0.5 + 0.5 * np.tanh(beta_x * xi) / np.tanh(beta_x))
        y = self.ymin + (self.ymax - self.ymin) * (0.5 + 0.5 * np.tanh(beta_y * eta) / np.tanh(beta_y))
        x[0], x[-1] = self.xmin, self.xmax
        y[0], y[-1] = self.ymin, self.ymax
        return x, y

    def _generate_node_coordinates(self):
        xx, yy = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
        self._node_coords = np.column_stack([xx.ravel(), yy.ravel()])
        self._node_indices = np.arange(self.nx * self.ny).reshape(self.nx, self.ny)

    def _generate_faces_and_cells(self):
        nx, ny = self.nx, self.ny
        self._faces = []
        self._face_centers = []
        self._face_areas = []
        self._cells = []
        self._cell_centers = []
        self._cell_volumes = []

        face_map = {}
        face_counter = 0
        self._cell_faces = []

        for i in range(nx - 1):
            for j in range(ny - 1):
                nodes = self._node_indices
                cell_nodes = [
                    nodes[i, j], nodes[i + 1, j],
                    nodes[i + 1, j + 1], nodes[i, j + 1]
                ]
                face_ids = []
                face_nodes = [
                    (cell_nodes[0], cell_nodes[3]),
                    (cell_nodes[1], cell_nodes[2]),
                    (cell_nodes[0], cell_nodes[1]),
                    (cell_nodes[3], cell_nodes[2])
                ]

                for fn in face_nodes:
                    fn = tuple(sorted(fn))
                    if fn in face_map:
                        face_id = face_map[fn]
                    else:
                        face_id = face_counter
                        face_map[fn] = face_id
                        self._faces.append(list(fn))
                        pts = self._node_coords[list(fn)]
                        center = np.mean(pts, axis=0)
                        area = np.linalg.norm(pts[1] - pts[0])
                        self._face_centers.append(center)
                        self._face_areas.append(area)
                        face_counter += 1
                    face_ids.append(face_id)

                cx = 0.25 * (self.x_nodes[i] + self.x_nodes[i+1] + self.x_nodes[i] + self.x_nodes[i+1])
                cy = 0.25 * (self.y_nodes[j] + self.y_nodes[j] + self.y_nodes[j+1] + self.y_nodes[j+1])
                vol = (self.x_nodes[i+1] - self.x_nodes[i]) * (self.y_nodes[j+1] - self.y_nodes[j])
                self._cells.append(face_ids)
                self._cell_centers.append([cx, cy])
                self._cell_volumes.append(vol)

        self._face_centers = np.array(self._face_centers)
        self._face_areas = np.array(self._face_areas)
        self._cell_centers = np.array(self._cell_centers)
        self._cell_volumes = np.array(self._cell_volumes)

    def _assign_owner_neighbor(self):
        n_faces = len(self._faces)
        self._owner_cells = np.full(n_faces, -1, dtype=int)
        self._neighbor_cells = np.full(n_faces, -1, dtype=int)

        for c_idx, f_list in enumerate(self._cells):
            for f_idx in f_list:
                if self._owner_cells[f_idx] == -1:
                    self._owner_cells[f_idx] = c_idx
                elif self._neighbor_cells[f_idx] == -1:
                    self._neighbor_cells[f_idx] = c_idx
                else:
                    raise RuntimeError(f"Face {f_idx} shared by more than 2 cells")

    def _compute_geometry(self):
        self._assign_owner_neighbor()
        original_normals = np.zeros((len(self._faces), 2))
        for i, face in enumerate(self._faces):
            nodes = self._node_coords[face]
            d = nodes[1] - nodes[0]
            n = np.array([d[1], -d[0]])
            l = np.linalg.norm(n)
            original_normals[i] = n / l if l > 1e-12 else np.array([0.0, 0.0])

        self._face_normals = original_normals.copy() # Store original normals before adjustment

        self._identify_boundary_faces() # Populates self.boundary_face_to_name

        # Ensure normals point outwards by checking against owner cell center
        self._face_normals_adjusted = np.zeros_like(self._face_normals)
        self._face_distances = np.zeros(self.n_faces) # d between owner and neighbor/boundary

        for f_idx in range(self.n_faces):
            owner = self._owner_cells[f_idx]
            neighbor = self._neighbor_cells[f_idx]
            face_center = self._face_centers[f_idx]

            if owner != -1:
                 owner_center = self._cell_centers[owner]
                 vec_oc_fc = face_center - owner_center
                 # Adjust normal direction if necessary
                 if np.dot(original_normals[f_idx], vec_oc_fc) < 0:
                     self._face_normals_adjusted[f_idx] = -original_normals[f_idx]
                 else:
                     self._face_normals_adjusted[f_idx] = original_normals[f_idx]

                 # Calculate distance d
                 if neighbor != -1:
                     neighbor_center = self._cell_centers[neighbor]
                     self._face_distances[f_idx] = np.linalg.norm(neighbor_center - owner_center)
                 else: # Boundary face
                     # Project owner center onto face normal to find distance to face plane
                     # d = |(owner_center - face_center) . normal| / |normal|^2 * normal -> simplify |(owner_center - face_center) . normal|
                     # Assuming adjusted normal points outwards
                     self._face_distances[f_idx] = np.abs(np.dot(owner_center - face_center, self._face_normals_adjusted[f_idx]))
                     # Alternative: distance from owner center to face center
                     # self._face_distances[f_idx] = np.linalg.norm(face_center - owner_center)
            else:
                 # Should not happen with current logic, but handle defensively
                 self._face_normals_adjusted[f_idx] = original_normals[f_idx]
                 self._face_distances[f_idx] = 1e-6 # Assign small default distance


        self._face_normals = self._face_normals_adjusted # Use adjusted normals from now on
        # Note: _finalize_geometry() was removed as its logic is integrated here

        # --- ADDED: Compute interpolation factors --- 
        self._face_interpolation_factors = np.zeros((self.n_faces, 2)) # Stores [g_C, g_F]
        owner_cells = self._owner_cells
        neighbor_cells = self._neighbor_cells
        face_centers = self._face_centers
        cell_centers = self._cell_centers

        for face_idx in range(self.n_faces):
            owner_idx = owner_cells[face_idx]
            neighbor_idx = neighbor_cells[face_idx]

            if owner_idx == -1: # Should not happen for valid mesh
                continue

            Cf = face_centers[face_idx]
            Cc = cell_centers[owner_idx]

            if neighbor_idx != -1:
                # Internal face
                Cn = cell_centers[neighbor_idx]
                vec_CN = Cn - Cc
                dist_CN = np.linalg.norm(vec_CN)
                if dist_CN < 1e-12:
                    g_C = 0.5
                    g_F = 0.5
                    dist_CfC = 0.0 
                else:
                    vec_e = vec_CN / dist_CN
                    dist_CfC = np.dot(Cf - Cc, vec_e)
                    g_F = dist_CfC / dist_CN
                    g_C = 1.0 - g_F
                    g_F = np.clip(g_F, 0.0, 1.0)
                    g_C = 1.0 - g_F
                # self._face_distances[face_idx] = dist_CN # Already computed earlier
            else:
                # Boundary face
                vec_CfC = Cf - Cc
                dist_CfC = np.linalg.norm(vec_CfC)
                g_C = 1.0
                g_F = 0.0
                # self._face_distances[face_idx] = dist_CfC # Already computed earlier

            self._face_interpolation_factors[face_idx, 0] = g_C
            self._face_interpolation_factors[face_idx, 1] = g_F
        # --- END ADDED --- 

    def _identify_boundary_faces(self):
        tol = 1e-10
        self.boundary_face_to_name = {}
        for f_idx, face in enumerate(self._faces):
            if self._neighbor_cells[f_idx] == -1:
                center = self._face_centers[f_idx]
                if abs(center[0] - self.xmin) < tol:
                    self.boundary_face_to_name[f_idx] = "left"
                elif abs(center[0] - self.xmax) < tol:
                    self.boundary_face_to_name[f_idx] = "right"
                elif abs(center[1] - self.ymin) < tol:
                    self.boundary_face_to_name[f_idx] = "bottom"
                elif abs(center[1] - self.ymax) < tol:
                    self.boundary_face_to_name[f_idx] = "top"

    def _cache_boundary_indices(self):
        nx, ny = self.n_cells_x, self.n_cells_y
        indices_2d = np.arange(nx * ny).reshape((nx, ny), order='F')
        self._boundary_indices = {
            'left': indices_2d[0, :],
            'right': indices_2d[-1, :],
            'bottom': indices_2d[:, 0],
            'top': indices_2d[:, -1],
        }
        for k, v in self.boundary_names.items():
            self._boundary_indices[v.lower()] = self._boundary_indices[k.lower()]

    @property
    def n_cells(self) -> int:
        return self.n_cells_x * self.n_cells_y

    @property
    def n_faces(self) -> int:
        return len(self._faces)

    @property
    def n_nodes(self) -> int:
        return self.nx * self.ny

    def get_node_positions(self) -> np.ndarray:
        return self._node_coords

    def get_face_centers(self) -> np.ndarray:
        return self._face_centers

    def get_cell_centers(self) -> np.ndarray:
        return self._cell_centers

    def get_face_normals(self) -> np.ndarray:
        # Return the normals guaranteed to point outward from the owner cell
        return self._face_normals # Now returns the adjusted normals

    def get_owner_neighbor(self) -> tuple[np.ndarray, np.ndarray]:
        return self._owner_cells, self._neighbor_cells

    def get_face_areas(self) -> np.ndarray:
        # For 2D structured grid, face 'area' is its length
        return self._face_areas

    def get_cell_volumes(self) -> np.ndarray:
        return self._cell_volumes

    def get_face_distances(self) -> np.ndarray:
        """Returns the distance 'd' used in diffusion calculations.
           For internal faces: distance between cell centers.
           For boundary faces: distance from owner cell center to the face plane.
        """
        return self._face_distances

    def get_field_shapes(self) -> tuple[tuple[int], tuple[int], tuple[int]]:
        """Returns the shapes for cell-centered fields (u, v, p)."""
        n_cells = self.n_cells
        shape = (n_cells,)
        return shape, shape, shape # u, v, p shapes

    def get_boundary_cell_indices(self, name: str) -> np.ndarray | None:
        """Returns the 1D indices of cells adjacent to the named boundary."""
        return self._boundary_indices.get(name.lower())

    def get_wall_flags(self) -> np.ndarray:
        """
        Returns a boolean array indicating which faces are boundary faces.
        In this structured mesh context, all boundary faces might potentially
        have velocity conditions applied via get_wall_velocities.
        """
        return self._neighbor_cells == -1

    def get_wall_velocities(self, is_u=True) -> np.ndarray:
        """
        Returns an array of prescribed velocity component values for boundary faces.
        Values are zero for non-boundary faces or if no BC is specified.
        """
        n_faces = self.n_faces
        wall_velocities = np.zeros(n_faces)
        component = 'u' if is_u else 'v'

        for f_idx in range(n_faces):
            if self._neighbor_cells[f_idx] == -1: # It's a boundary face
                boundary_name = self.boundary_face_to_name.get(f_idx)
                if boundary_name:
                    bc_dict = self._boundary_conditions.get(boundary_name)
                    if bc_dict and component in bc_dict:
                        wall_velocities[f_idx] = bc_dict[component]
                    # else: implicitly leaves velocity as 0.0 if name/component not found

        return wall_velocities

    def plot(self, ax=None, title: str = "Structured Mesh", show_nodes: bool = False, show_centers: bool = False, show_normals: bool = False, cell_ids: bool = False, face_ids: bool = False, node_ids: bool = False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        # Plot cell boundaries by iterating through faces
        for i, face_nodes in enumerate(self._faces):
            nodes = self.get_node_positions()[face_nodes]
            ax.plot(nodes[:, 0], nodes[:, 1], 'k-', lw=0.5)
            if face_ids:
                 center = self.get_face_centers()[i]
                 ax.text(center[0], center[1], f'{i}', color='blue', fontsize=6, ha='center', va='center')

        if show_nodes:
            nodes = self.get_node_positions()
            ax.plot(nodes[:, 0], nodes[:, 1], 'ko', ms=2, label="Nodes")
            if node_ids:
                for i, (x,y) in enumerate(nodes):
                   ax.text(x, y, f'{i}', color='red', fontsize=6, ha='right', va='bottom')


        if show_centers:
            cell_centers = self.get_cell_centers()
            ax.plot(cell_centers[:, 0], cell_centers[:, 1], 'go', ms=3, label="Cell Centers")
            if cell_ids:
                for i, (x,y) in enumerate(cell_centers):
                   ax.text(x, y, f'C{i}', color='green', fontsize=6, ha='center', va='center')

        if show_normals:
            face_centers = self.get_face_centers()
            normals = self.get_face_normals()
            ax.quiver(face_centers[:, 0], face_centers[:, 1], normals[:, 0], normals[:, 1], color='r', scale=20, width=0.005, label="Normals")

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.set_title(title)
        ax.set_xlim(self.xmin - 0.1*(self.xmax-self.xmin), self.xmax + 0.1*(self.xmax-self.xmin))
        ax.set_ylim(self.ymin - 0.1*(self.ymax-self.ymin), self.ymax + 0.1*(self.ymax-self.ymin))
        if show_nodes or show_centers or show_normals:
             ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        return fig, ax

    def savePlot(self, filename: str, **kwargs):
        fig, _ = self.plot(**kwargs)
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def get_dimensions(self):
        """Return number of cells in x and y directions."""
        return self.n_cells_x, self.n_cells_y

    # Implement all required Mesh ABC methods like get_node_positions etc. using standard patterns.
