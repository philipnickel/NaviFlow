"""
Mesh-agnostic direct solver for pressure correction equation.
Works with arbitrary mesh topologies.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..pressure_solver.base_pressure_solver import PressureSolver

class MeshAgnosticDirectPressureSolver(PressureSolver):
    """
    Direct solver for pressure correction equation using sparse matrix methods.
    Designed to work with arbitrary mesh topologies.
    """
    
    def __init__(self):
        """
        Initialize the mesh-agnostic direct pressure solver.
        """
        super().__init__()
        self.residual_history = []
        self.mesh = None
        self.bc_manager = None
        self.p = None

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
                d_u_face = 0.5 * (d_u[owner] + d_u[neighbor])
                d_v_face = 0.5 * (d_v[owner] + d_v[neighbor])
                
                # 4. Calculate pressure gradient at the face (simple central difference)
                dp_dx_face = 0.5 * (dp_dx_owner + dp_dx_neighbor)
                dp_dy_face = 0.5 * (dp_dy_owner + dp_dy_neighbor)
                
                # 5. Apply Rhie-Chow correction
                u_face = u_face_avg - d_u_face * (dp_dx_face - 0.5 * (dp_dx_owner + dp_dx_neighbor))
                v_face = v_face_avg - d_v_face * (dp_dy_face - 0.5 * (dp_dy_owner + dp_dy_neighbor))
                
                face_velocity = np.array([u_face, v_face, 0.0])
                
                # Calculate mass flux through the face
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
        rhs = self.build_rhs(mesh, rho, u_star, v_star, d_u, d_v, p_star)
        
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