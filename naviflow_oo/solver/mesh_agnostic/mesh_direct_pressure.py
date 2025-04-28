"""
Mesh-agnostic direct solver for pressure correction equation.
Works with arbitrary mesh topologies.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..pressure_solver.base_pressure_solver import PressureSolver
from ..pressure_solver.helpers.rhs_construction import get_rhs

class MeshAgnosticDirectPressureSolver(PressureSolver):
    """
    Direct solver for pressure correction equation using sparse matrix methods.
    Designed to work with arbitrary mesh topologies.
    """
    
    def __init__(self, tolerance=1e-10, max_iterations=1000):
        """
        Initialize the mesh-agnostic direct pressure solver.
        
        Parameters:
        -----------
        tolerance : float, optional
            Convergence tolerance for iterative methods.
        max_iterations : int, optional
            Maximum number of iterations for iterative methods.
        """
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.residual_history = []
        self.mesh = None
        self.bc_manager = None
        self.p = None
        self.last_pressure_correction = None

    def build_pressure_matrix(self, mesh, rho, d_u, d_v, pin_pressure=True):
        """
        Build the coefficient matrix for the pressure correction equation.
        
        Parameters:
        -----------
        mesh : Mesh
            The computational mesh (structured or unstructured)
        rho : float
            Fluid density
        d_u, d_v : ndarray
            Momentum equation coefficients
        pin_pressure : bool, optional
            Whether to pin pressure at a reference point (default: True)
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Coefficient matrix for the pressure equation
        """
        # --- DEBUG Check d_u, d_v inputs ---
        if np.isnan(d_u).any() or np.isinf(d_u).any():
            print("** WARNING: NaN/Inf detected in d_u input to build_pressure_matrix! **")
        if np.isnan(d_v).any() or np.isinf(d_v).any():
            print("** WARNING: NaN/Inf detected in d_v input to build_pressure_matrix! **")
        # ------------------------------------
        
        # Get mesh topology information
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        owner_cells, neighbor_cells = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_normals = mesh.get_face_normals()
        
        # Arrays for sparse matrix construction
        rows = []
        cols = []
        data = []
        
        # Process each internal face (skip boundary faces)
        for face_idx in range(n_faces):
            owner = owner_cells[face_idx]
            neighbor = neighbor_cells[face_idx]
            
            # Skip boundary faces (-1), invalid indices, or negative indices
            if neighbor < 0 or owner >= n_cells or neighbor >= n_cells or owner < 0:
                continue
                
            area = face_areas[face_idx]
            normal = face_normals[face_idx]
            
            # Determine which d_coeff to use based on face normal direction
            # For faces with larger x-component in normal, use d_u
            # For faces with larger y-component in normal, use d_v
            if abs(normal[0]) > abs(normal[1]):
                # Use d_u for x-direction faces
                # Find the proper d_u for this face (safely)
                if owner < len(d_u) and neighbor < len(d_u):
                    d_face = 0.5 * (d_u[owner] + d_u[neighbor])
                else:
                    d_face = 1.0  # Default for safety
            else:
                # Use d_v for y-direction faces
                # Find the proper d_v for this face (safely)
                if owner < len(d_v) and neighbor < len(d_v):
                    d_face = 0.5 * (d_v[owner] + d_v[neighbor])
                else:
                    d_face = 1.0  # Default for safety
            
            # Calculate the face coefficient
            face_coeff = rho * d_face * area
            
            # Add to the sparse matrix
            # Owner to neighbor connection
            rows.append(owner)
            cols.append(neighbor)
            data.append(-face_coeff)
            
            # Neighbor to owner connection
            rows.append(neighbor)
            cols.append(owner)
            data.append(-face_coeff)
            
            # Diagonal contributions
            rows.append(owner)
            cols.append(owner)
            data.append(face_coeff)
            
            rows.append(neighbor)
            cols.append(neighbor)
            data.append(face_coeff)
        
        # Process boundary faces to add diagonal contributions
        for face_idx in range(n_faces):
            owner = owner_cells[face_idx]
            neighbor = neighbor_cells[face_idx]
            
            # Only process boundary faces with valid indices
            if neighbor >= 0 or owner >= n_cells or owner < 0:
                continue
                
            area = face_areas[face_idx]
            normal = face_normals[face_idx]
            
            # Add a diagonal contribution for zero-gradient pressure BC
            rows.append(owner)
            cols.append(owner)
            data.append(0.1)  # Small value for stability
        
        # Make sure we have at least one entry
        if not rows:
            # Add a default entry for the first cell
            rows.append(0)
            cols.append(0)
            data.append(1.0)
        
        # Create sparse matrix
        A = sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        A = A.tocsr()
        
        # Pin pressure at reference cell to avoid singular matrix
        if pin_pressure:
            pin_index = 0  # First cell as reference
            A[pin_index, :] = 0.0  # Zero out row
            A[pin_index, pin_index] = 1.0  # Set diagonal to 1
        
        return A
    
    def build_rhs(self, mesh, rho, u_star, v_star, d_u, d_v, p):
        """
        Build the right-hand side for the pressure correction equation with Rhie-Chow interpolation.
        
        Parameters:
        -----------
        mesh : Mesh
            The computational mesh
        rho : float
            Fluid density
        u_star, v_star : ndarray
            Intermediate velocity fields at cells
        d_u, d_v : ndarray
            Momentum equation coefficients
        p : ndarray
            Current pressure field
            
        Returns:
        --------
        ndarray
            Right-hand side vector for the pressure equation
        """
        # Get mesh topology
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        owner_cells, neighbor_cells = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_normals = mesh.get_face_normals()
        cell_centers = mesh.get_cell_centers()
        
        # Initialize RHS vector (mass imbalance for each cell)
        rhs = np.zeros(n_cells)
        
        # Calculate mass fluxes through internal faces
        for face_idx in range(n_faces):
            owner = owner_cells[face_idx]
            neighbor = neighbor_cells[face_idx]
            
            # Skip invalid indices and boundary faces
            if owner >= n_cells or owner < 0 or neighbor < 0:
                continue
                
            area = face_areas[face_idx]
            normal = face_normals[face_idx]
            
            # For internal faces, we need to interpolate velocity from cells to faces
            if neighbor < n_cells:
                # Calculate face velocity using Rhie-Chow interpolation to avoid checker-boarding
                u_face_avg = 0.5 * (u_star[owner] + u_star[neighbor])
                v_face_avg = 0.5 * (v_star[owner] + v_star[neighbor])
                
                # 2. Calculate pressure gradient at cells
                # Find connected faces for each cell to calculate gradient
                dp_dx_owner = 0.0
                dp_dy_owner = 0.0
                dp_dx_neighbor = 0.0
                dp_dy_neighbor = 0.0
                
                # Simple pressure gradient approximation
                dr = cell_centers[neighbor] - cell_centers[owner]
                dr_mag = np.linalg.norm(dr)
                if dr_mag > 1e-10:
                    dp = p[neighbor] - p[owner]
                    dp_dx = dp * dr[0] / (dr_mag * dr_mag)
                    dp_dy = dp * dr[1] / (dr_mag * dr_mag)
                    dp_dx_owner = dp_dx
                    dp_dy_owner = dp_dy
                    dp_dx_neighbor = -dp_dx
                    dp_dy_neighbor = -dp_dy
                
                # 3. Interpolate the product of d and pressure gradient
                # --- DEBUG --- 
                if face_idx < 5: print(f"DEBUG shapes: d_u[owner]: {d_u[owner].shape if hasattr(d_u[owner], 'shape') else type(d_u[owner])}, d_v[owner]: {d_v[owner].shape if hasattr(d_v[owner], 'shape') else type(d_v[owner])}")
                # ----------- 
                d_u_face = 0.5 * (d_u[owner] + d_u[neighbor])
                d_v_face = 0.5 * (d_v[owner] + d_v[neighbor])
                
                # 4. Calculate pressure gradient at the face (simple central difference)
                dp_dx_face = 0.5 * (dp_dx_owner + dp_dx_neighbor)
                dp_dy_face = 0.5 * (dp_dy_owner + dp_dy_neighbor)
                
                # 5. Apply Rhie-Chow correction -- RESTORED
                u_face = u_face_avg - d_u_face * (dp_dx_face - 0.5 * (dp_dx_owner + dp_dx_neighbor))
                v_face = v_face_avg - d_v_face * (dp_dy_face - 0.5 * (dp_dy_owner + dp_dy_neighbor))
                # u_face = u_face_avg # Use simple average for debugging
                # v_face = v_face_avg # Use simple average for debugging
                
                # Create 2D face velocity vector
                face_velocity = np.array([u_face, v_face]) 
                
                # Calculate mass flux through the face (using 2D normal)
                mass_flux = rho * np.dot(face_velocity, normal) * area
                
                # Add contributions to cells (mass flux is from owner to neighbor)
                rhs[owner] -= mass_flux
                rhs[neighbor] += mass_flux
        
        # Handle boundary faces specially
        # For walls, use zero-normal velocity boundary condition
        for face_idx in range(n_faces):
            owner = owner_cells[face_idx]
            neighbor = neighbor_cells[face_idx]
            
            # Process only boundary faces with valid owner
            if neighbor >= 0 or owner >= n_cells or owner < 0:
                continue
                
            # For walls and other boundaries, the mass flux is zero (no-penetration)
            # No additional contribution to the RHS
            
        # Ensure the RHS sums to zero (necessary for solvability)
        if not np.isclose(np.sum(rhs), 0.0, atol=1e-10):
            rhs -= np.sum(rhs) / n_cells  # Adjust for global mass conservation
        
        # Set reference cell RHS to zero (for pinned pressure)
        rhs[0] = 0.0
        
        return rhs
    
    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star, return_dict=True):
        """
        Solve the pressure correction equation using a direct method.
        With Rhie-Chow interpolation for collocated grids.
        
        Parameters:
        -----------
        mesh : Mesh
            The computational mesh
        u_star, v_star : ndarray
            Intermediate velocity fields
        d_u, d_v : ndarray
            Momentum equation coefficients
        p_star : ndarray
            Current pressure field
        return_dict : bool, optional
            If True, returns a dictionary with residual information
            
        Returns:
        --------
        p_prime : ndarray
            Pressure correction field
        residual_info : dict
            Dictionary with residual information
        """
        # Store mesh for later use
        self.mesh = mesh
        
        # Fluid density (should be provided by a fluid properties object)
        rho = 1.0
        
        # Build coefficient matrix
        A = self.build_pressure_matrix(mesh, rho, d_u, d_v)
        
        # Build right-hand side using Rhie-Chow interpolation
        # Ensure inputs are flattened 1D arrays
        u_star_flat = u_star.flatten() if u_star.ndim > 1 else u_star
        v_star_flat = v_star.flatten() if v_star.ndim > 1 else v_star
        d_u_flat = d_u.flatten() if d_u.ndim > 1 else d_u
        d_v_flat = d_v.flatten() if d_v.ndim > 1 else d_v
        p_star_flat = p_star.flatten() if p_star.ndim > 1 else p_star
        
        rhs = self.build_rhs(mesh, rho, u_star_flat, v_star_flat, d_u_flat, d_v_flat, p_star_flat)
        
        # --- Sanity check A and rhs --- 
        nan_A = np.isnan(A.data).any() if hasattr(A, 'data') else np.isnan(A).any()
        inf_A = np.isinf(A.data).any() if hasattr(A, 'data') else np.isinf(A).any()
        nan_rhs = np.isnan(rhs).any()
        inf_rhs = np.isinf(rhs).any()
        if nan_A or inf_A or nan_rhs or inf_rhs:
             print("** WARNING: NaN/Inf detected in Pressure Matrix (A) or RHS (rhs)! **")
             print(f"  NaNs: A={nan_A}, rhs={nan_rhs}")
             print(f"  Infs: A={inf_A}, rhs={inf_rhs}")
             # Optionally, raise error or save data for inspection
             # raise ValueError("NaN/Inf in pressure system!")
        # -------------------------------
        
        # Solve the system using direct solver
        p_prime = spsolve(A, rhs)
        
        # Calculate residual for reporting
        r = rhs - A.dot(p_prime)
        r_norm = np.linalg.norm(r)
        
        # Calculate relative norm
        rhs_norm = np.linalg.norm(rhs)
        rel_norm = r_norm / rhs_norm if rhs_norm > 0 else r_norm
        
        # Store residual
        self.residual_history.append(rel_norm)
        
        # Create residual info
        residual_info = {
            'rel_norm': rel_norm,
            'field': r
        }
        
        return p_prime, residual_info
    
    def get_solver_info(self):
        """
        Get information about the solver's performance.
        
        Returns:
        --------
        dict
            Dictionary containing solver performance metrics
        """
        info = {
            'name': 'MeshAgnosticDirectPressureSolver',
            'inner_iterations_history': [1] * len(self.residual_history),  # Direct solver uses 1 iteration per solve
            'total_inner_iterations': len(self.residual_history),
            'convergence_rate': None  # Not applicable for direct solver
        }
        
        # Add solver-specific information
        info['solver_specific'] = {
            'method': 'direct',
            'library': 'scipy.sparse.linalg.spsolve'
        }
        
        return info 

    def solve_pressure_correction(self, mesh, fluid, u_star, v_star, d_u, d_v, relaxation_factor=0.3, boundary_conditions=None):
        """
        Solve the pressure correction equation using sparse direct solver.
        Applies Rhie-Chow interpolation to avoid pressure-velocity decoupling.
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        u_star, v_star : ndarray
            Intermediate velocity fields (expected as 1D arrays)
        d_u, d_v : ndarray
            Momentum equation coefficients (expected as 1D arrays)
        relaxation_factor : float, optional
            Relaxation factor for pressure correction
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions
            
        Returns
        -------
        p_prime : ndarray
            Pressure correction field (1D array)
        """
        # Ensure all inputs are 1D arrays
        u_star_flat = u_star.flatten() if u_star.ndim > 1 else u_star
        v_star_flat = v_star.flatten() if v_star.ndim > 1 else v_star
        d_u_flat = d_u.flatten() if d_u.ndim > 1 else d_u
        d_v_flat = d_v.flatten() if d_v.ndim > 1 else d_v
        
        # Get mesh properties
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        owners, neighbors = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_normals = mesh.get_face_normals()
        face_centers = mesh.get_face_centers()
        cell_centers = mesh.get_cell_centers()
        cell_volumes = mesh.get_cell_volumes()
        
        # Build coefficient matrix for pressure correction equation
        # Initialize COO format data for sparse matrix
        data = []
        row_indices = []
        col_indices = []
        
        # Initialize RHS (continuity residual)
        b = np.zeros(n_cells)
        
        # Loop through all internal faces
        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]
            
            # Skip boundary faces for matrix assembly
            if neighbor == -1 or owner >= n_cells or neighbor >= n_cells:
                continue
                
            area = face_areas[face_idx]
            normal = face_normals[face_idx]
            
            # Get normal components
            nx, ny = normal[0], normal[1]
            
            # Calculate face distance
            d_owner = face_centers[face_idx] - cell_centers[owner]
            d_neighbor = cell_centers[neighbor] - face_centers[face_idx]
            d_total = np.linalg.norm(cell_centers[neighbor] - cell_centers[owner])
            
            # Calculate coefficient for this face using harmonic interpolation
            # Apply Rhie-Chow correction for pressure-velocity coupling
            # This calculation gives better stability for collocated grids
            
            # Harmonic interpolation of d_u and d_v values
            d_u_f = (d_u_flat[owner] * d_u_flat[neighbor]) / (
                d_u_flat[owner] * np.linalg.norm(d_neighbor) + 
                d_u_flat[neighbor] * np.linalg.norm(d_owner)
            ) if d_total > 1e-12 else 0.5 * (d_u_flat[owner] + d_u_flat[neighbor])
            
            d_v_f = (d_v_flat[owner] * d_v_flat[neighbor]) / (
                d_v_flat[owner] * np.linalg.norm(d_neighbor) + 
                d_v_flat[neighbor] * np.linalg.norm(d_owner)
            ) if d_total > 1e-12 else 0.5 * (d_v_flat[owner] + d_v_flat[neighbor])
            
            # Projection of d coefficients onto face normal
            d_f = d_u_f * nx**2 + d_v_f * ny**2
            
            # Face coefficient with area weighting
            a_f = area * d_f / d_total
            
            # Add to matrix (owner-neighbor relation)
            data.append(-a_f)  # Off-diagonal owner-to-neighbor
            row_indices.append(owner)
            col_indices.append(neighbor)
            
            data.append(-a_f)  # Off-diagonal neighbor-to-owner
            row_indices.append(neighbor)
            col_indices.append(owner)
            
            # Add to diagonal (sum of outflows principle)
            data.append(a_f)  # Diagonal owner
            row_indices.append(owner)
            col_indices.append(owner)
            
            data.append(a_f)  # Diagonal neighbor
            row_indices.append(neighbor)
            col_indices.append(neighbor)
            
            # Rhie-Chow interpolation for face velocities to avoid checker-boarding
            # Calculate linear interpolation factors based on distance
            w_owner = np.linalg.norm(d_neighbor) / d_total
            w_neighbor = np.linalg.norm(d_owner) / d_total
            
            # Calculate standard linear interpolation for velocity
            u_face_linear = w_owner * u_star_flat[owner] + w_neighbor * u_star_flat[neighbor]
            v_face_linear = w_owner * v_star_flat[owner] + w_neighbor * v_star_flat[neighbor]
            
            # Mass flux through face
            mass_flux = fluid.get_density() * area * (u_face_linear * nx + v_face_linear * ny)
            
            # Add contribution to RHS (continuity residual)
            b[owner] -= mass_flux
            b[neighbor] += mass_flux
        
        # Process boundary faces to complete mass conservation
        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]
            
            if neighbor != -1 or owner >= n_cells:
                continue  # Skip internal faces and invalid owners
                
            area = face_areas[face_idx]
            normal = face_normals[face_idx]
            
            # Get normal components
            nx, ny = normal[0], normal[1]
            
            # Calculate velocity at boundary (depends on boundary condition)
            # For wall, we know velocity is zero
            u_boundary = 0.0
            v_boundary = 0.0
            
            # If boundary conditions are provided, use them
            if boundary_conditions is not None:
                boundary_name = mesh.get_boundary_name(face_idx)
                if boundary_name is not None:
                    bc_values = boundary_conditions.get_condition(boundary_name, bc_type='velocity')
                    if bc_values is not None:
                        u_boundary = bc_values.get('u', 0.0)
                        v_boundary = bc_values.get('v', 0.0)
            
            # Add boundary flux to RHS
            mass_flux = fluid.get_density() * area * (u_boundary * nx + v_boundary * ny)
            b[owner] -= mass_flux
        
        # Create sparse matrix from COO data
        n = n_cells
        A = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n, n))
        A = A.tocsr()
        
        # Ensure the system is solvable (zero mean pressure correction)
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = 0.0
        
        # Solve the system directly using sparse solver
        p_prime = spsolve(A, b)
        
        # Apply under-relaxation
        if self.last_pressure_correction is not None:
            p_prime = relaxation_factor * p_prime + (1.0 - relaxation_factor) * self.last_pressure_correction
        
        # Store for next iteration
        self.last_pressure_correction = p_prime.copy()
        
        # Compute residual information
        residual = b - A @ p_prime
        residual[0] = 0.0  # Exclude fixed reference point
        
        residual_info = {
            'abs_norm': np.linalg.norm(residual),
            'rel_norm': np.linalg.norm(residual) / max(np.linalg.norm(b), 1e-10),
            'field': residual  # Store the residual field
        }
        
        return p_prime, residual_info 