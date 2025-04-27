"""
Unstructured mesh generation utilities.
"""

import numpy as np
import os
from ..mesh.mesh import UnstructuredMesh

class UnstructuredMeshGenerator:
    """
    Generator for unstructured meshes.
    """
    
    @staticmethod
    def from_gmsh(file_path):
        """
        Load an unstructured mesh from a Gmsh file.
        
        Parameters:
        -----------
        file_path : str
            Path to the Gmsh mesh file (.msh)
            
        Returns:
        --------
        mesh : UnstructuredMesh
            The loaded unstructured mesh
        """
        try:
            import gmsh
        except ImportError:
            raise ImportError("Gmsh Python API is required. Please install 'gmsh'.")
        
        # Initialize Gmsh
        gmsh.initialize()
        
        # Open the mesh file
        gmsh.open(file_path)
        
        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        
        # Reshape node coordinates to (N, 3)
        nodes = node_coords.reshape(-1, 3)
        
        # Get elements
        element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements()
        
        # Extract triangular and tetrahedral elements (2D and 3D)
        faces = []
        cells = []
        
        # Process triangular elements (2D mesh)
        tri_idx = None
        for i, etype in enumerate(element_types):
            if etype == 2:  # Triangle
                tri_idx = i
                break
        
        if tri_idx is not None:
            # For 2D mesh, triangles are cells
            triangles = node_tags_per_element[tri_idx].reshape(-1, 3) - 1  # Convert to 0-based indexing
            
            # Create faces (edges) from triangles
            edge_set = set()
            for tri in triangles:
                # Add the three edges of the triangle
                for i in range(3):
                    edge = (min(tri[i], tri[(i+1)%3]), max(tri[i], tri[(i+1)%3]))
                    edge_set.add(edge)
            
            # Convert edge set to list of faces
            faces = [list(edge) for edge in edge_set]
            
            # Create cells (each triangle is a cell with 3 faces)
            # This requires mapping triangle edges to face indices
            edge_to_face = {(min(face[0], face[1]), max(face[0], face[1])): i for i, face in enumerate(faces)}
            
            for tri in triangles:
                cell_faces = []
                for i in range(3):
                    edge = (min(tri[i], tri[(i+1)%3]), max(tri[i], tri[(i+1)%3]))
                    cell_faces.append(edge_to_face[edge])
                cells.append(cell_faces)
        
        # Process tetrahedral elements (3D mesh)
        tet_idx = None
        for i, etype in enumerate(element_types):
            if etype == 4:  # Tetrahedron
                tet_idx = i
                break
        
        if tet_idx is not None:
            # For 3D mesh, tetrahedra are cells
            tetrahedra = node_tags_per_element[tet_idx].reshape(-1, 4) - 1  # Convert to 0-based indexing
            
            # Create faces (triangles) from tetrahedra
            face_set = set()
            for tet in tetrahedra:
                # Add the four triangular faces of the tetrahedron
                for i in range(4):
                    # Skip the i-th vertex to get a triangular face
                    face = tuple(sorted([tet[j] for j in range(4) if j != i]))
                    face_set.add(face)
            
            # Convert face set to list of faces
            faces = [list(face) for face in face_set]
            
            # Create cells (each tetrahedron is a cell with 4 faces)
            # This requires mapping tetrahedron faces to face indices
            face_to_idx = {tuple(sorted(face)): i for i, face in enumerate(faces)}
            
            for tet in tetrahedra:
                cell_faces = []
                for i in range(4):
                    # Skip the i-th vertex to get a triangular face
                    face = tuple(sorted([tet[j] for j in range(4) if j != i]))
                    cell_faces.append(face_to_idx[face])
                cells.append(cell_faces)
        
        # Finalize Gmsh
        gmsh.finalize()
        
        # Create and return the mesh
        return UnstructuredMesh(nodes, faces, cells)
    
    @staticmethod
    def from_meshio(mesh):
        """
        Create an unstructured mesh from a meshio object.
        
        Parameters:
        -----------
        mesh : meshio.Mesh
            Meshio mesh object
            
        Returns:
        --------
        mesh : UnstructuredMesh
            The created unstructured mesh
        """
        try:
            import meshio
        except ImportError:
            raise ImportError("Meshio is required. Please install 'meshio'.")
        
        # Extract nodes
        nodes = mesh.points
        
        # Process by cell type
        faces = []
        cells = []
        
        # 2D case: triangular mesh
        if "triangle" in mesh.cells_dict:
            triangles = mesh.cells_dict["triangle"]
            
            # Create faces (edges) from triangles
            edge_set = set()
            for tri in triangles:
                # Add the three edges of the triangle
                for i in range(3):
                    edge = (min(tri[i], tri[(i+1)%3]), max(tri[i], tri[(i+1)%3]))
                    edge_set.add(edge)
            
            # Convert edge set to list of faces
            faces = [list(edge) for edge in edge_set]
            
            # Create cells (each triangle is a cell with 3 faces)
            # This requires mapping triangle edges to face indices
            edge_to_face = {(min(face[0], face[1]), max(face[0], face[1])): i for i, face in enumerate(faces)}
            
            for tri in triangles:
                cell_faces = []
                for i in range(3):
                    edge = (min(tri[i], tri[(i+1)%3]), max(tri[i], tri[(i+1)%3]))
                    cell_faces.append(edge_to_face[edge])
                cells.append(cell_faces)
        
        # 3D case: tetrahedral mesh
        elif "tetra" in mesh.cells_dict:
            tetrahedra = mesh.cells_dict["tetra"]
            
            # Create faces (triangles) from tetrahedra
            face_set = set()
            for tet in tetrahedra:
                # Add the four triangular faces of the tetrahedron
                for i in range(4):
                    # Skip the i-th vertex to get a triangular face
                    face = tuple(sorted([tet[j] for j in range(4) if j != i]))
                    face_set.add(face)
            
            # Convert face set to list of faces
            faces = [list(face) for face in face_set]
            
            # Create cells (each tetrahedron is a cell with 4 faces)
            # This requires mapping tetrahedron faces to face indices
            face_to_idx = {tuple(sorted(face)): i for i, face in enumerate(faces)}
            
            for tet in tetrahedra:
                cell_faces = []
                for i in range(4):
                    # Skip the i-th vertex to get a triangular face
                    face = tuple(sorted([tet[j] for j in range(4) if j != i]))
                    cell_faces.append(face_to_idx[face])
                cells.append(cell_faces)
        
        # Create and return the mesh
        return UnstructuredMesh(nodes, faces, cells)
    
    @staticmethod
    def from_pygmsh(geometry, mesh_size):
        """
        Generate an unstructured mesh from a pygmsh geometry.
        
        Parameters:
        -----------
        geometry : pygmsh.built_in.Geometry or pygmsh.geo.Geometry
            Pygmsh geometry object
        mesh_size : float
            Characteristic length of mesh elements
            
        Returns:
        --------
        mesh : UnstructuredMesh
            The generated unstructured mesh
        """
        try:
            import pygmsh
            import meshio
        except ImportError:
            raise ImportError("Pygmsh and meshio are required. Please install 'pygmsh' and 'meshio'.")
        
        # Generate mesh with pygmsh
        with pygmsh.geo.Geometry() as geom:
            # Define geometry based on input
            # This is a simplified example, actual implementation would need to handle
            # the geometry input more generally
            points = [
                geom.add_point([0.0, 0.0, 0.0], mesh_size=mesh_size),
                geom.add_point([1.0, 0.0, 0.0], mesh_size=mesh_size),
                geom.add_point([1.0, 1.0, 0.0], mesh_size=mesh_size),
                geom.add_point([0.0, 1.0, 0.0], mesh_size=mesh_size),
            ]
            
            lines = [
                geom.add_line(points[0], points[1]),
                geom.add_line(points[1], points[2]),
                geom.add_line(points[2], points[3]),
                geom.add_line(points[3], points[0]),
            ]
            
            curve_loop = geom.add_curve_loop(lines)
            surface = geom.add_plane_surface(curve_loop)
            
            # Generate mesh
            mesh = geom.generate_mesh()
        
        # Convert to our mesh format using from_meshio
        return UnstructuredMeshGenerator.from_meshio(mesh) 