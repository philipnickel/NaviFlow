"""
Unstructured mesh classes for NaviFlow.
"""

import numpy as np
import os
import sys

try:
    import pygmsh
    import meshio
except ImportError:
    pass  # We'll check again when needed and raise appropriate error messages

from .mesh import UnstructuredMesh
from ..mesh_generators.unstructured_mesh import UnstructuredMeshGenerator


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
            raise ImportError("UnstructuredUniform mesh requires 'pygmsh' and 'meshio' packages.")
        
        # Calculate domain dimensions
        length = xmax - xmin
        height = ymax - ymin
        
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
            surface = geom.add_plane_surface(boundary)
            
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
                edge = (min(tri[i], tri[(i+1)%3]), max(tri[i], tri[(i+1)%3]))
                edges.add(edge)
        
        # Convert to list of faces
        faces = [list(edge) for edge in edges]
        
        # Create mapping from edge to face index
        edge_to_face = {(min(face[0], face[1]), max(face[0], face[1])): i for i, face in enumerate(faces)}
        
        # Create cells (each triangle connects to 3 faces)
        cells = []
        for tri in triangles:
            cell_faces = []
            for i in range(3):
                edge = (min(tri[i], tri[(i+1)%3]), max(tri[i], tri[(i+1)%3]))
                cell_faces.append(edge_to_face[edge])
            cells.append(cell_faces)
        
        # Initialize base class with the processed mesh data
        super().__init__(nodes, faces, cells)

    def _compute_geometry(self):
        """
        Compute geometric properties and identify boundary faces.
        Extends the parent class _compute_geometry method to identify boundary faces.
        """
        # Call the parent class method to compute basic geometric properties
        super()._compute_geometry()
        
        # Get domain bounds
        xmin = self._domain_bounds['xmin']
        xmax = self._domain_bounds['xmax']
        ymin = self._domain_bounds['ymin']
        ymax = self._domain_bounds['ymax']
        
        # Small tolerance for floating point comparison
        tol = 1e-10
        
        # Get face centers and normals
        face_centers = self._face_centers
        face_normals = self._face_normals
        neighbors = self._neighbor_cells
        
        # Identify boundary faces (where neighbor = -1)
        for face_idx in range(len(self._faces)):
            if neighbors[face_idx] == -1:
                # This is a boundary face
                center = face_centers[face_idx]
                normal = face_normals[face_idx]
                
                # Determine which boundary this face belongs to
                if abs(center[0] - xmin) < tol:
                    # Left boundary
                    self.boundary_face_to_name[face_idx] = "left"
                elif abs(center[0] - xmax) < tol:
                    # Right boundary
                    self.boundary_face_to_name[face_idx] = "right"
                elif abs(center[1] - ymin) < tol:
                    # Bottom boundary
                    self.boundary_face_to_name[face_idx] = "bottom"
                elif abs(center[1] - ymax) < tol:
                    # Top boundary
                    self.boundary_face_to_name[face_idx] = "top"


class UnstructuredRefined(UnstructuredMesh):
    """
    Generator for refined unstructured meshes for the lid-driven cavity.
    Generates a rectangular domain with a mesh that is refined near walls,
    especially near the top moving lid.
    """
    
    def __init__(self, mesh_size_walls, mesh_size_lid, mesh_size_center, 
                xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
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
        self._domain_bounds = {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax
        }
        
        # Get domain size
        length = xmax - xmin
        height = ymax - ymin
        
        # Ensure pygmsh is available
        try:
            import pygmsh
        except ImportError:
            raise ImportError("Pygmsh is required for this mesh generator. Please install it.")
        
        # Try a simple approach with explicit points that ensures connectivity
        with pygmsh.geo.Geometry() as geom:
            # Define corner points first (these connect the walls)
            p1 = geom.add_point([xmin, ymin, 0.0], mesh_size=mesh_size_walls)  # Bottom-left
            p2 = geom.add_point([xmax, ymin, 0.0], mesh_size=mesh_size_walls)  # Bottom-right
            p3 = geom.add_point([xmax, ymax, 0.0], mesh_size=mesh_size_lid)    # Top-right
            p4 = geom.add_point([xmin, ymax, 0.0], mesh_size=mesh_size_lid)    # Top-left
            
            # Create lines for outer boundaries
            l1 = geom.add_line(p1, p2)  # Bottom wall
            l2 = geom.add_line(p2, p3)  # Right wall
            l3 = geom.add_line(p3, p4)  # Top wall (lid)
            l4 = geom.add_line(p4, p1)  # Left wall
            
            # Create boundary curve and plane surface
            curve_loop = geom.add_curve_loop([l1, l2, l3, l4])
            surface = geom.add_plane_surface(curve_loop)
            
            # Add interior points for mesh size control
            # Create a field that controls mesh size based on distance from boundaries
            # We'll manually add points throughout the domain
            
            # Middle of each wall with fine mesh size
            geom.add_point([xmin + length/2, ymin, 0.0], mesh_size=mesh_size_walls*1.2)  # Bottom wall
            geom.add_point([xmax, ymin + height/2, 0.0], mesh_size=mesh_size_walls*1.2)  # Right wall 
            geom.add_point([xmin + length/2, ymax, 0.0], mesh_size=mesh_size_lid*1.2)    # Top wall (lid)
            geom.add_point([xmin, ymin + height/2, 0.0], mesh_size=mesh_size_walls*1.2)  # Left wall
            
            # Quarter points on walls with intermediate mesh sizes
            geom.add_point([xmin + length/4, ymin, 0.0], mesh_size=mesh_size_walls)       # Bottom wall
            geom.add_point([xmin + 3*length/4, ymin, 0.0], mesh_size=mesh_size_walls)     # Bottom wall
            geom.add_point([xmax, ymin + height/4, 0.0], mesh_size=mesh_size_walls)       # Right wall
            geom.add_point([xmax, ymin + 3*height/4, 0.0], mesh_size=mesh_size_lid)       # Right wall
            geom.add_point([xmin + length/4, ymax, 0.0], mesh_size=mesh_size_lid)         # Top wall
            geom.add_point([xmin + 3*length/4, ymax, 0.0], mesh_size=mesh_size_lid)       # Top wall
            geom.add_point([xmin, ymin + height/4, 0.0], mesh_size=mesh_size_walls)       # Left wall
            geom.add_point([xmin, ymin + 3*height/4, 0.0], mesh_size=mesh_size_lid)       # Left wall
            
            # Interior points with coarser mesh
            # Center of domain (coarsest)
            geom.add_point([xmin + length/2, ymin + height/2, 0.0], mesh_size=mesh_size_center)
            
            # Quarter points inside domain (intermediate sizes)
            h1 = (mesh_size_walls + mesh_size_center) / 2  # Intermediate size near walls
            h2 = (mesh_size_lid + mesh_size_center) / 2    # Intermediate size near lid
            
            # Halfway between walls and center
            geom.add_point([xmin + length/4, ymin + height/4, 0.0], mesh_size=h1)     # Bottom-left quadrant
            geom.add_point([xmin + 3*length/4, ymin + height/4, 0.0], mesh_size=h1)   # Bottom-right quadrant
            geom.add_point([xmin + length/4, ymin + 3*height/4, 0.0], mesh_size=h2)   # Top-left quadrant
            geom.add_point([xmin + 3*length/4, ymin + 3*height/4, 0.0], mesh_size=h2) # Top-right quadrant
            
            # Set mesh size field - customize element size field
            geom.set_mesh_size_callback(
                lambda dim, tag, x, y, z, lc: min(
                    mesh_size_center,  # Default size
                    # Near bottom/left/right walls - size based on distance
                    mesh_size_walls + min(1.0, (1.25 * min(
                        abs(x - xmin),   # Distance from left wall
                        abs(x - xmax),   # Distance from right wall
                        abs(y - ymin)    # Distance from bottom wall
                    ) / min(length, height))) * (mesh_size_center - mesh_size_walls),
                    # Near lid - special size
                    mesh_size_lid + min(1.0, (1.25 * abs(y - ymax) / height)) * (mesh_size_center - mesh_size_lid)
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
                edge = (min(tri[i], tri[(i+1)%3]), max(tri[i], tri[(i+1)%3]))
                edges.add(edge)
        
        # Convert to list of faces
        faces = [list(edge) for edge in edges]
        
        # Create mapping from edge to face index
        edge_to_face = {(min(face[0], face[1]), max(face[0], face[1])): i for i, face in enumerate(faces)}
        
        # Create cells (each triangle connects to 3 faces)
        cells = []
        for tri in triangles:
            cell_faces = []
            for i in range(3):
                edge = (min(tri[i], tri[(i+1)%3]), max(tri[i], tri[(i+1)%3]))
                cell_faces.append(edge_to_face[edge])
            cells.append(cell_faces)
        
        # Initialize base class with the processed mesh data
        super().__init__(nodes, faces, cells)

    def _compute_geometry(self):
        """
        Compute geometric properties and identify boundary faces.
        Extends the parent class _compute_geometry method to identify boundary faces.
        """
        # Call the parent class method to compute basic geometric properties
        super()._compute_geometry()
        
        # Get domain bounds
        xmin = self._domain_bounds['xmin']
        xmax = self._domain_bounds['xmax']
        ymin = self._domain_bounds['ymin']
        ymax = self._domain_bounds['ymax']
        
        # Small tolerance for floating point comparison
        tol = 1e-10
        
        # Get face centers and normals
        face_centers = self._face_centers
        face_normals = self._face_normals
        neighbors = self._neighbor_cells
        
        # Identify boundary faces (where neighbor = -1)
        for face_idx in range(len(self._faces)):
            if neighbors[face_idx] == -1:
                # This is a boundary face
                center = face_centers[face_idx]
                normal = face_normals[face_idx]
                
                # Determine which boundary this face belongs to
                if abs(center[0] - xmin) < tol:
                    # Left boundary
                    self.boundary_face_to_name[face_idx] = "left"
                elif abs(center[0] - xmax) < tol:
                    # Right boundary
                    self.boundary_face_to_name[face_idx] = "right"
                elif abs(center[1] - ymin) < tol:
                    # Bottom boundary
                    self.boundary_face_to_name[face_idx] = "bottom"
                elif abs(center[1] - ymax) < tol:
                    # Top boundary
                    self.boundary_face_to_name[face_idx] = "top"


class Unstructured:
    """
    Utility class for creating unstructured meshes.
    
    This class provides factory methods to create different types of unstructured meshes.
    It serves as a bridge between UnstructuredUniform and UnstructuredRefined.
    
    Note: This class is deprecated and maintained for backward compatibility.
    Please use UnstructuredUniform and UnstructuredRefined directly.
    """
    
    @staticmethod
    def create_uniform(mesh_size, length=1.0, height=1.0, origin=(0.0, 0.0), xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Create a uniform unstructured mesh. (Deprecated)
        
        Parameters:
        -----------
        mesh_size : float
            Characteristic mesh element size
        length : float, optional
            Length of the domain in x-direction, defaults to 1.0
        height : float, optional
            Height of the domain in y-direction, defaults to 1.0
        origin : tuple of float, optional
            Origin coordinates (x_origin, y_origin), defaults to (0.0, 0.0)
        xmin : float, optional
            Minimum x-coordinate, overrides origin[0] if provided
        xmax : float, optional
            Maximum x-coordinate, overrides length if provided with xmin
        ymin : float, optional
            Minimum y-coordinate, overrides origin[1] if provided
        ymax : float, optional
            Maximum y-coordinate, overrides height if provided with ymin
            
        Returns:
        --------
        UnstructuredUniform
            A uniform unstructured mesh with the specified parameters
        """
        # Handle alternative specification methods
        x_origin, y_origin = origin
        
        if xmin is None:
            xmin = x_origin
        if ymin is None:
            ymin = y_origin
            
        if xmax is None:
            xmax = xmin + length
        if ymax is None:
            ymax = ymin + height
            
        return UnstructuredUniform(
            mesh_size=mesh_size, 
            xmin=xmin, 
            xmax=xmax,
            ymin=ymin, 
            ymax=ymax
        )
    
    @staticmethod
    def create_refined_cavity(mesh_size_walls, mesh_size_lid, mesh_size_center, 
                              length=1.0, height=1.0, origin=(0.0, 0.0),
                              xmin=None, xmax=None, ymin=None, ymax=None,
                              dist_min_lid=0.05, dist_max_lid=0.25,
                              dist_min_walls=0.05, dist_max_walls=0.4):
        """
        Create a refined unstructured mesh for lid-driven cavity problems. (Deprecated)
        
        Parameters:
        -----------
        mesh_size_walls : float
            Mesh element size near the walls (bottom, left, right)
        mesh_size_lid : float
            Mesh element size near the lid (top boundary)
        mesh_size_center : float
            Mesh element size in the center of the domain
        length : float, optional
            Length of the domain in x-direction, defaults to 1.0
        height : float, optional
            Height of the domain in y-direction, defaults to 1.0
        origin : tuple of float, optional
            Origin coordinates (x_origin, y_origin), defaults to (0.0, 0.0)
        xmin : float, optional
            Minimum x-coordinate, overrides origin[0] if provided
        xmax : float, optional
            Maximum x-coordinate, overrides length if provided with xmin
        ymin : float, optional
            Minimum y-coordinate, overrides origin[1] if provided
        ymax : float, optional
            Maximum y-coordinate, overrides height if provided with ymin
            
        Returns:
        --------
        UnstructuredRefined
            A refined unstructured mesh suitable for lid-driven cavity simulations
        """
        # Handle alternative specification methods
        x_origin, y_origin = origin
        
        if xmin is None:
            xmin = x_origin
        if ymin is None:
            ymin = y_origin
            
        if xmax is None:
            xmax = xmin + length
        if ymax is None:
            ymax = ymin + height
            
        return UnstructuredRefined(
            mesh_size_walls=mesh_size_walls,
            mesh_size_lid=mesh_size_lid,
            mesh_size_center=mesh_size_center,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax
        ) 