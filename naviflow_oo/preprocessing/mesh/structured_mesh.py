"""
Structured mesh generation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from .mesh import Mesh

class StructuredMesh(Mesh):
    """
    Unified structured mesh class for 2D CFD simulations.
    Can generate uniform or non-uniform (clustered) meshes.
    """
    def __init__(self, n_cells_x, n_cells_y, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                 is_uniform=True, clustering_factor=0.2, beta_x=None, beta_y=None,
                 boundary_names=None):
        """
        Initialize a structured mesh.

        Parameters:
        -----------
        n_cells_x : int
            Number of cells in x-direction
        n_cells_y : int
            Number of cells in y-direction
        xmin : float, optional
            Minimum x-coordinate, defaults to 0.0
        xmax : float, optional
            Maximum x-coordinate, defaults to 1.0
        ymin : float, optional
            Minimum y-coordinate, defaults to 0.0
        ymax : float, optional
            Maximum y-coordinate, defaults to 1.0
        is_uniform : bool, optional
            If True, generate a uniform mesh. If False, generate a non-uniform
            mesh using clustering parameters. Defaults to True.
        clustering_factor : float, optional
            Default clustering factor if is_uniform is False and beta_x/beta_y
            are not provided. Defaults to 0.2.
        beta_x : float, optional
            Clustering parameter for x-direction (overrides clustering_factor).
        beta_y : float, optional
            Clustering parameter for y-direction (overrides clustering_factor).
        boundary_names : dict, optional
            Dictionary mapping boundary names ('left', 'right', 'top', 'bottom')
            to user-defined names. Defaults to standard names.
        """
        super().__init__()

        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # Calculate number of nodes
        self.nx = n_cells_x + 1
        self.ny = n_cells_y + 1

        # Generate node coordinates based on type
        if is_uniform:
            self.x_nodes, self.y_nodes = self._generate_uniform_nodes()
        else:
            # Use specific beta values if provided, otherwise use clustering_factor
            actual_beta_x = beta_x if beta_x is not None else clustering_factor
            actual_beta_y = beta_y if beta_y is not None else clustering_factor
            self.x_nodes, self.y_nodes = self._generate_clustered_nodes(actual_beta_x, actual_beta_y)

        # --- Common initialization steps ---
        # Store boundary info (moved from StructuredUniform)
        self.boundary_names = boundary_names or {'left': 'left', 'right': 'right', 'bottom': 'bottom', 'top': 'top'}
        self._boundary_indices = {}

        # Create node coordinates array and index map
        self._generate_node_coordinates()

        # Create face indices, normals, areas, centers
        self._generate_face_indices()

        # Create cell indices, centers, volumes
        self._generate_cell_indices()

        # Compute owner/neighbor connectivity and name boundaries
        self._compute_geometry()

        # Cache boundary cell indices (moved from StructuredUniform)
        self._cache_boundary_indices()

    def _generate_uniform_nodes(self):
        """Generate evenly spaced nodes."""
        x_nodes = np.linspace(self.xmin, self.xmax, self.nx)
        y_nodes = np.linspace(self.ymin, self.ymax, self.ny)
        return x_nodes, y_nodes

    def _generate_clustered_nodes(self, beta_x, beta_y):
        """Generate nodes clustered towards boundaries using tanh stretching."""
        length = self.xmax - self.xmin
        height = self.ymax - self.ymin

        # Create base distributions (-1 to 1)
        xi = np.linspace(-1, 1, self.nx)
        eta = np.linspace(-1, 1, self.ny)

        # Apply tanh clustering function
        # Avoid division by zero if beta is very small (effectively uniform)
        tanh_beta_x = np.tanh(beta_x) if abs(beta_x) > 1e-6 else 1.0
        tanh_beta_y = np.tanh(beta_y) if abs(beta_y) > 1e-6 else 1.0

        x_nodes = self.xmin + length * (0.5 + 0.5 * np.tanh(beta_x * xi) / tanh_beta_x)
        y_nodes = self.ymin + height * (0.5 + 0.5 * np.tanh(beta_y * eta) / tanh_beta_y)

        # Ensure endpoints match exactly due to potential floating point issues
        x_nodes[0], x_nodes[-1] = self.xmin, self.xmax
        y_nodes[0], y_nodes[-1] = self.ymin, self.ymax

        return x_nodes, y_nodes

    def _cache_boundary_indices(self):
        """Pre-calculate and store indices for named boundaries."""
        # Use number of cells (nx-1, ny-1) for reshaping
        nx_cells = self.n_cells_x # Use stored n_cells_x
        ny_cells = self.n_cells_y # Use stored n_cells_y

        if nx_cells <= 0 or ny_cells <= 0:
            self._boundary_indices = {}
            return # No cells to index

        all_indices = np.arange(self.n_cells)
        # Reshape indices to match grid structure (using cell counts)
        # Use Fortran order ('F') for reshaping if indexing='ij' was used in meshgrid for cells
        # Or ensure consistent indexing ('C' order with 'xy' indexing). Let's assume C order for now.
        try:
            indices_2d = all_indices.reshape((nx_cells, ny_cells)) # Assumes C-order flatten/reshape
        except ValueError as e:
             print(f"Error reshaping cell indices: {e}")
             print(f"Total cells: {self.n_cells}, nx_cells: {nx_cells}, ny_cells: {ny_cells}")
             self._boundary_indices = {}
             return


        self._boundary_indices = {} # Initialize dict

        # Use provided names, fall back to defaults
        names_to_process = self.boundary_names

        for name, location in names_to_process.items():
            loc = location.lower()
            # Adjust indexing based on how cells were flattened/ordered
            if loc == 'left':
                # Indices of cells in the first column (i=0)
                self._boundary_indices[name.lower()] = indices_2d[0, :].flatten()
            elif loc == 'right':
                # Indices of cells in the last column (i=nx_cells-1)
                self._boundary_indices[name.lower()] = indices_2d[nx_cells - 1, :].flatten()
            elif loc == 'bottom':
                # Indices of cells in the first row (j=0)
                self._boundary_indices[name.lower()] = indices_2d[:, 0].flatten()
            elif loc == 'top':
                # Indices of cells in the last row (j=ny_cells-1)
                self._boundary_indices[name.lower()] = indices_2d[:, ny_cells - 1].flatten()

    def get_boundary_cell_indices(self, boundary_name):
         """
         Get the indices of cells adjacent to a named boundary.

         Parameters:
         -----------
         boundary_name : str
             The name of the boundary (e.g., 'top', 'inlet').

         Returns:
         --------
         np.ndarray or None
             An array of cell indices for the boundary, or None if not found.
         """
         return self._boundary_indices.get(boundary_name.lower())

    def get_dimensions(self):
        """
        Get the number of cells in x and y directions.
        
        Returns:
        --------
        tuple : (nx, ny)
            Number of cells in each direction (nx-1, ny-1)
        """
        return self.nx - 1, self.ny - 1
    
    def get_field_shapes(self):
        """
        Return the field shapes for collocated (u, v, p) fields.

        Returns:
        --------
        tuple
            (u_shape, v_shape, p_shape)
        """
        nx, ny = self.get_dimensions()
        shape = (nx, ny)  # Same for all in collocated arrangement
        return shape, shape, shape
    
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
        self._faces = [[] for _ in range(total_faces)]
        self._face_normals = np.zeros((total_faces, 2))
        self._face_areas = np.zeros(total_faces)
        self._face_centers = np.zeros((total_faces, 2))
        
        # Generate x-direction faces (vertical faces)
        face_idx_counter = 0 # Simple counter for loop iterations

        # Pre-calculate node differences to avoid repeated calculations
        dx_nodes = np.diff(self.x_nodes) # Length nx-1
        dy_nodes = np.diff(self.y_nodes) # Length ny-1

        # Generate indices for iteration
        i_indices, j_indices = np.meshgrid(
            np.arange(self.nx - 1), # Corresponds to left cell index i: 0..nx-2
            np.arange(self.ny),     # Corresponds to y-node index j: 0..ny-1
            indexing='ij'
        )
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()

        for idx in range(len(i_indices)):
            i, j = i_indices[idx], j_indices[idx] # Grid indices from loop

            # Calculate the target linear face index
            target_face_idx = i + j * (self.nx - 1)

            # --- Normal Vector ---
            if target_face_idx < len(self._face_normals): self._face_normals[target_face_idx] = [1.0, 0.0]

            # --- Node Connectivity ---
            # Find nodes for this vertical face (at x=x_nodes[i+1], between y=y_nodes[j] and y=y_nodes[j+1])
            node_bl = self._node_indices[i + 1, j] # Bottom-left (of face)
            if j + 1 < self.ny:
                node_tl = self._node_indices[i + 1, j + 1] # Top-left (of face)
                # Ensure target index is valid before assignment
                if target_face_idx < total_faces:
                    self._faces[target_face_idx] = [node_bl, node_tl]
            else: # Handle face on boundary if index scheme requires it
                 if target_face_idx < total_faces:
                      self._faces[target_face_idx] = [node_bl] # Or define appropriately

            # --- Area Calculation ---
            # Face length dy requires y_nodes[j] and y_nodes[j+1]. Use pre-calculated dy_nodes[j].
            if j < self.ny - 1:
                dy = dy_nodes[j]
            else: # Handle face segment on top/bottom boundary line if needed by index scheme
                if self.ny > 1: dy = dy_nodes[self.ny - 2] # Use last valid segment length
                else: dy = 0.0
            if target_face_idx < len(self._face_areas): self._face_areas[target_face_idx] = dy

            # --- Center Calculation ---
            # Vertical face is at x = x_nodes[i+1] (between cell i and i+1)
            # Center y = (y_nodes[j] + y_nodes[j+1]) / 2
            center_x = self.x_nodes[i+1]
            if j < self.ny - 1:
                center_y = self.y_nodes[j] + dy / 2.0
            else: # Handle face segment on top boundary line
                center_y = self.y_nodes[j] # Center is just the node y-coord
            if target_face_idx < len(self._face_centers): self._face_centers[target_face_idx] = [center_x, center_y]

            # --- DEBUG ---
            # Keep debug prints if needed, using target_face_idx
            # if target_face_idx == 1: print(...)
            # --- END DEBUG ---

            face_idx_counter += 1

        n_x_faces = (self.nx - 1) * self.ny # Use theoretical count

        # Generate y-direction faces (horizontal faces)
        # Iterate using node indices i (0..nx-1) and j (0..ny-2)
        i_indices_h, j_indices_h = np.meshgrid(
            np.arange(self.nx),     # Node index i: 0..nx-1
            np.arange(self.ny - 1), # Node index j: 0..ny-2 (corresponds to bottom cell)
            indexing='ij'
        )
        i_indices_h = i_indices_h.flatten()
        j_indices_h = j_indices_h.flatten()

        for idx in range(len(i_indices_h)):
            i, j = i_indices_h[idx], j_indices_h[idx] # Node indices from loop

            # Calculate the target linear face index
            target_face_idx = n_x_faces + i * (self.ny - 1) + j

            # --- Normal Vector ---
            if target_face_idx < len(self._face_normals): self._face_normals[target_face_idx] = [0.0, 1.0]

            # --- Node Connectivity ---
            # Find nodes for this horizontal face (at y=y_nodes[j+1], between x=x_nodes[i] and x=x_nodes[i+1])
            node_bl = self._node_indices[i, j + 1] # Bottom-left (of face)
            if i + 1 < self.nx:
                node_br = self._node_indices[i + 1, j + 1] # Bottom-right (of face)
                if target_face_idx < total_faces:
                     self._faces[target_face_idx] = [node_bl, node_br]
            else: # Handle face on boundary
                 if target_face_idx < total_faces:
                      self._faces[target_face_idx] = [node_bl]

            # --- Area Calculation ---
            # Face length dx requires x_nodes[i] and x_nodes[i+1]. Use pre-calculated dx_nodes[i].
            if i < self.nx - 1:
                dx = dx_nodes[i]
            else: # Handle face segment on right/left boundary line
                if self.nx > 1: dx = dx_nodes[self.nx - 2] # Use last valid segment length
                else: dx = 0.0
            if target_face_idx < len(self._face_areas): self._face_areas[target_face_idx] = dx

            # --- Center Calculation ---
            # Horizontal face is at y = y_nodes[j] (between cell j-1 and j)
            # Center x = (x_nodes[i] + x_nodes[i+1]) / 2
            center_y = self.y_nodes[j] # Correct y-position for face between cell j-1 and j
            if i < self.nx - 1:
                center_x = self.x_nodes[i] + dx / 2.0
            else: # Handle face segment on right boundary line
                center_x = self.x_nodes[i] # Center is just the node x-coord
            if target_face_idx < len(self._face_centers): self._face_centers[target_face_idx] = [center_x, center_y]

             # --- DEBUG PRINT for horizontal faces ---
            # if i == self.nx - 1: print(...) # Use target_face_idx if debugging
            # --- END DEBUG ---

            face_idx_counter += 1 # Increment simple counter (optional)

        # Remove leftover debug prints from previous steps if they exist
        # Find and remove 'DEBUG Calc Face', 'DEBUG Store Face' prints

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
        
        # Assign cell centers - Ensure order matches cell indexing j*(nx-1)+i
        n_cells = (self.nx - 1) * (self.ny - 1)
        temp_centers = np.column_stack([x_coords, y_coords]) # Centers flattened based on meshgrid(ij) order
        self._cell_centers = np.zeros((n_cells, 2))
        idx_source = 0
        for i in range(self.nx - 1): # Cell column index
            for j in range(self.ny - 1): # Cell row index
                idx_target = j * (self.nx - 1) + i # Target index based on formula
                if idx_source < len(temp_centers) and idx_target < n_cells:
                     self._cell_centers[idx_target] = temp_centers[idx_source]
                idx_source += 1
        
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
        """Compute owner/neighbor cells and identify boundary faces."""
        n_faces = len(self._faces)
        n_cells = (self.nx - 1) * (self.ny - 1)
        n_x_faces = (self.nx-1) * self.ny

        # Initialize owners and neighbors
        self._owner_cells = np.full(n_faces, -1, dtype=int)
        self._neighbor_cells = np.full(n_faces, -1, dtype=int)
        self.boundary_face_to_name = {} # Initialize boundary mapping

        # --- Assign Owner/Neighbor ---
        # Vertical faces: West-East direction (j=0..ny-1)
        for j in range(self.ny):
            for i in range(self.nx - 1):
                face_idx = i + j * (self.nx - 1)

                # Cells on either side of this face
                # Check if owner cell (i,j) is valid
                is_owner_valid = (0 <= i < self.nx - 1) and (0 <= j < self.ny - 1)
                # Check if neighbor cell (i+1,j) is valid
                is_neighbor_valid = (0 <= i + 1 < self.nx - 1) and (0 <= j < self.ny - 1)

                if is_owner_valid:
                    self._owner_cells[face_idx] = j * (self.nx - 1) + i

                    # Boundary condition should be based on missing neighbor
                    # if i == 0: # OLD boundary logic was incorrect place
                    #     self.boundary_face_to_name[face_idx] = self.boundary_names.get("left", "left")

                # Check if neighbor cell (i+1, j) exists and assign it
                if is_neighbor_valid:
                    # Internal face - assign neighbor (cell i+1, j)
                    self._neighbor_cells[face_idx] = j * (self.nx - 1) + (i + 1) # Corrected neighbor index
                # else: # Neighbor is not valid, this is a boundary face
                    # The old logic was here:
                    # self._neighbor_cells[face_idx] = j * (self.nx - 1) + (i - 1) # WRONG: assigned cell i-1,j

                # Assign boundary names based on missing cells
                if is_owner_valid and not is_neighbor_valid: # Owner exists, neighbor doesn't -> Right boundary
                     self.boundary_face_to_name[face_idx] = self.boundary_names.get("right", "right")
                # Note: Left boundary faces (where owner is invalid) are handled in the final pass

        # Horizontal faces: South-North direction (i=0..nx-1, j=0..ny-2)
        for j in range(self.ny - 1):
            for i in range(self.nx): # Loop through all potential face x-indices
                face_idx = n_x_faces + i * (self.ny - 1) + j

                # Skip faces that would be outside our cell domain # REMOVED incorrect skip condition
                # if i >= self.nx - 1:
                #    continue

                # Cells on either side of this face
                # Check if owner cell (i,j) is valid
                is_owner_valid = (0 <= i < self.nx - 1) and (0 <= j < self.ny - 1)
                # Check if neighbor cell (i, j+1) is valid
                is_neighbor_valid = (0 <= i < self.nx - 1) and (0 <= j + 1 < self.ny - 1)

                # Assign ownership based on which cells exist
                if is_owner_valid:
                    self._owner_cells[face_idx] = j * (self.nx - 1) + i

                    # Boundary condition should be based on missing neighbor
                    # if j == 0: # OLD boundary logic was incorrect place
                    #    self.boundary_face_to_name[face_idx] = self.boundary_names.get("bottom", "bottom")

                # Check if neighbor cell (i, j+1) exists and assign it
                if is_neighbor_valid:
                    # Internal face - assign neighbor (cell i, j+1)
                    self._neighbor_cells[face_idx] = (j + 1) * (self.nx - 1) + i # Corrected neighbor index
                # else: # Neighbor is not valid, this is a boundary face
                    # The old logic was here:
                    # self._neighbor_cells[face_idx] = (j - 1) * (self.nx - 1) + i # WRONG: assigned cell i, j-1

                # Assign boundary names based on missing cells
                if is_owner_valid and not is_neighbor_valid: # Owner exists, neighbor doesn't -> Top boundary
                     self.boundary_face_to_name[face_idx] = self.boundary_names.get("top", "top")
                # Note: Bottom boundary faces (where owner is invalid) are handled in the final pass

        # Final pass to ensure no face has both owner and neighbor as -1
        # This pass also handles boundary faces where owner was initially assigned -1 (left, bottom)
        # or where the above logic didn't catch a boundary correctly.
        for face_idx in range(n_faces):
            owner = self._owner_cells[face_idx]
            neighbor = self._neighbor_cells[face_idx]

            if owner == -1 and neighbor == -1:
                # Find the closest valid cell and assign as owner
                # (Assuming this should only happen for faces outside the main cell domain, e.g., corners)
                face_center = self._face_centers[face_idx]
                min_dist = float('inf')
                closest_cell = -1
                for cell_idx in range(n_cells):
                    # Ensure cell_idx is valid before accessing center
                    if 0 <= cell_idx < len(self._cell_centers):
                        cell_center = self._cell_centers[cell_idx]
                        dist = np.linalg.norm(face_center - cell_center)
                        if dist < min_dist:
                            min_dist = dist
                            closest_cell = cell_idx
                    else:
                        # Handle cases where cell_idx might be out of bounds if n_cells calculation is off
                        print(f"Warning: Invalid cell_idx {cell_idx} encountered in final geometry pass.")


                if closest_cell != -1:
                    self._owner_cells[face_idx] = closest_cell
                    # Determine which boundary this face is on based on normal
                    normal = self._face_normals[face_idx]
                    if abs(normal[0]) > abs(normal[1]):  # Vertical face normal
                        if face_center[0] < self.xmin + 1e-6: # Check proximity to boundary
                             self.boundary_face_to_name[face_idx] = self.boundary_names.get("left", "left")
                        elif face_center[0] > self.xmax - 1e-6:
                             self.boundary_face_to_name[face_idx] = self.boundary_names.get("right", "right")
                        # Add robustness: Fallback based on normal if proximity fails
                        elif normal[0] < 0: # Points left
                           self.boundary_face_to_name[face_idx] = self.boundary_names.get("left", "left")
                        else: # Points right
                           self.boundary_face_to_name[face_idx] = self.boundary_names.get("right", "right")

                    else:  # Horizontal face normal
                         if face_center[1] < self.ymin + 1e-6: # Check proximity to boundary
                             self.boundary_face_to_name[face_idx] = self.boundary_names.get("bottom", "bottom")
                         elif face_center[1] > self.ymax - 1e-6:
                             self.boundary_face_to_name[face_idx] = self.boundary_names.get("top", "top")
                         # Add robustness: Fallback based on normal if proximity fails
                         elif normal[1] < 0: # Points down
                            self.boundary_face_to_name[face_idx] = self.boundary_names.get("bottom", "bottom")
                         else: # Points up
                            self.boundary_face_to_name[face_idx] = self.boundary_names.get("top", "top")

            # Assign boundary name if owner is valid but neighbor is not (covers left/bottom boundaries)
            elif owner != -1 and neighbor == -1 and face_idx not in self.boundary_face_to_name:
                 # Determine boundary based on normal direction relative to owner cell center
                 normal = self._face_normals[face_idx]
                 owner_center = self._cell_centers[owner]
                 face_center = self._face_centers[face_idx]
                 owner_to_face = face_center - owner_center

                 # Check dot product: positive -> normal points away from owner (expected for boundary)
            if np.dot(normal, owner_to_face) > 1e-9: # Check if face normal points "outward" from owner
                    if abs(normal[0]) > abs(normal[1]):  # Vertical face
                        if normal[0] < 0: # Points left
                              self.boundary_face_to_name[face_idx] = self.boundary_names.get("left", "left")
                        else: # Points right (Should have been caught earlier?)
                               self.boundary_face_to_name[face_idx] = self.boundary_names.get("right", "right")
                    else: # Horizontal face
                           if normal[1] < 0: # Points down
                               self.boundary_face_to_name[face_idx] = self.boundary_names.get("bottom", "bottom")
                           else: # Points up (Should have been caught earlier?)
                                self.boundary_face_to_name[face_idx] = self.boundary_names.get("top", "top")
                 #else:
                     # Normal points inward or is zero/tangential? This case might indicate other issues.
                     # print(f"Warning: Ambiguous boundary assignment for face {face_idx}")
                     # Fallback to proximity check if dot product is ambiguous
            if abs(normal[0]) > abs(normal[1]):  # Vertical face
                        if face_center[0] < (self.xmin + self.xmax) / 2:
                            self.boundary_face_to_name[face_idx] = self.boundary_names.get("left", "left")
                        else:
                            self.boundary_face_to_name[face_idx] = self.boundary_names.get("right", "right")
            else: # Horizontal face
                        if face_center[1] < (self.ymin + self.ymax) / 2:
                            self.boundary_face_to_name[face_idx] = self.boundary_names.get("bottom", "bottom")
                        else:
                            self.boundary_face_to_name[face_idx] = self.boundary_names.get("top", "top")

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
    
    def get_face_interpolation_factors(self, face_idx):
        """
        Calculate linear interpolation factors (weights) for a face based on 
        distances to adjacent cell centers (owner C, neighbor F).

        g_C = ||x_f - x_F|| / ||x_C - x_F||
        g_F = ||x_f - x_C|| / ||x_C - x_F||

        For boundary faces (neighbor F == -1), returns (g_C=1.0, g_F=0.0).

        Parameters:
        -----------
        face_idx : int
            The index of the face.

        Returns:
        --------
        tuple
            (g_C, g_F) interpolation factors.
        """
        owner_idx = self._owner_cells[face_idx]
        neighbor_idx = self._neighbor_cells[face_idx]
        
        # Handle invalid owner (should not happen for faces processed by solvers)
        if owner_idx < 0 or owner_idx >= self.n_cells:
            # Return default weights or raise error, depending on desired handling
            # For robustness, let's return default assuming owner dominates if neighbor invalid
             return (1.0, 0.0)
             
        # Handle boundary faces
        if neighbor_idx < 0 or neighbor_idx >= self.n_cells:
            return (1.0, 0.0)
            
        # Get coordinates
        face_center = self._face_centers[face_idx]
        owner_center = self._cell_centers[owner_idx]
        neighbor_center = self._cell_centers[neighbor_idx]
        
        # Calculate distances
        vec_CF = owner_center - neighbor_center
        dist_CF_sq = np.dot(vec_CF, vec_CF)
        
        # Avoid division by zero if cell centers coincide
        if dist_CF_sq < 1e-12:
            return (0.5, 0.5) # Or handle as error
        
        dist_CF = np.sqrt(dist_CF_sq)
        
        vec_fC = face_center - owner_center
        dist_fC = np.sqrt(np.dot(vec_fC, vec_fC))
        
        vec_fF = face_center - neighbor_center
        dist_fF = np.sqrt(np.dot(vec_fF, vec_fF))
        
        # Calculate factors
        g_C = dist_fF / dist_CF # || x_f - x_F || / || x_C - x_F ||
        g_F = dist_fC / dist_CF # || x_f - x_C || / || x_C - x_F ||
        
        # --- DEBUG --- 
        if face_idx == 1:
             print(f"DEBUG Face {face_idx}: xf={face_center}, xC={owner_center}, xF={neighbor_center}")
             print(f"DEBUG Face {face_idx}: dist_fF={dist_fF:.6f}, dist_fC={dist_fC:.6f}, dist_CF={dist_CF:.6f}")
             print(f"DEBUG Face {face_idx}: g_C={g_C:.6f}, g_F={g_F:.6f}, sum={g_C + g_F:.6f}")
        # --- END DEBUG ---
            
        return (g_C, g_F)

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

    def get_wall_flags(self):
        """Returns a boolean array where True indicates a wall (no-slip) boundary face."""
        flags = np.zeros(self.n_faces, dtype=bool)
        for face_idx, name in self.boundary_face_to_name.items():
            if name in ['left', 'right', 'top', 'bottom']:
                flags[face_idx] = True
        return flags

    def get_wall_velocities(self, is_u=True):
        """
        Returns an array of prescribed velocities on wall faces.
        For lid-driven cavity:
            - top boundary (u) = 1.0
            - all others = 0.0
        """
        velocities = np.zeros(self.n_faces)
        for face_idx, name in self.boundary_face_to_name.items():
            if name == 'top' and is_u:
                velocities[face_idx] = 1.0
            else:
                velocities[face_idx] = 0.0
        return velocities

    def get_face_distances(self):
        """Returns the normal distance from face center to adjacent cell center (1D array)."""
        distances = np.zeros(self.n_faces)
        face_centers = self.get_face_centers()
        cell_centers = self.get_cell_centers()
        owners, _ = self.get_owner_neighbor()

        for face_idx, owner_idx in enumerate(owners):
            if owner_idx == -1:
                continue
            distances[face_idx] = np.linalg.norm(face_centers[face_idx] - cell_centers[owner_idx])
        return distances

    def get_viscosity(self):
        """Returns the dynamic viscosity (assumed constant for now)."""
        return 1.0  # Set to match your fluid's viscosity (could be parameterized) 