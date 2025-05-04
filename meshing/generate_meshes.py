"""
Mesh Generation Script for NaviFlow-Collocated

Generates different mesh types for various CFD experiment scenarios.
Creates a structured directory hierarchy:
    meshing/
        experiments/
            experiment_type/
                mesh_type/
                    mesh files (.vtk)

Usage:
    python -m meshing.generate_meshes [experiment_name]
    
If no experiment name is provided, it will generate meshes for all experiments.
"""

import os
import sys
import gmsh
import argparse
import numpy as np

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import mesh generators
try:
    from naviflow_collocated.mesh.structured_uniform import generate as gen_uniform
    from naviflow_collocated.mesh.structured_refined import generate as gen_refined
    from naviflow_collocated.mesh.unstructured import generate as gen_unstructured
except ImportError as e:
    print(f"Error importing mesh generators: {e}")
    print("Ensure the script is run from the repository root or the path is correct.")
    sys.exit(1)

# Define experiment configurations
# Each experiment has specific parameters for the mesh types
EXPERIMENTS = {
    "lidDrivenCavity": {
        "description": "Lid-driven cavity flow in a square domain",
        "uniform": {
            "L": 1.0,
            "nx": 50,
            "ny": 50,
            "lc": 0.05,
            "description": "Standard uniform square mesh"
        },
        "refined": {
            "L": 1.0, 
            "nx": 40,
            "ny": 40,
            "refine_edge": "top",
            "ratio": 1.2,
            "description": "Refined near the top lid to capture boundary layer"
        },
        "unstructured": {
            "L": 1.0,
            "mesh_size_walls": 0.02,
            "mesh_size_lid": 0.01,
            "mesh_size_center": 0.05,
            "description": "Unstructured triangle mesh with refinement near walls and lid" 
        }
    },

    "flowAroundCylinder": {
        "description": "Flow around a circular cylinder",
     
        "unstructured": {
            "L": 2.2,
            "obstacle_radius": 0.1,
            "mesh_size_obstacle": 0.01,
            "mesh_size_walls": 0.04,
            "mesh_size_center": 0.1,
            "description": "Unstructured mesh with circular obstacle and refinement"
        }
    }
}

# Helper functions for mesh generation
def generate_structured_mesh(model_name, params, output_file):
    """Generate a structured mesh using given parameters."""
    if params.get("refine_edge", "none") != "none":
        # Structured refined mesh
        return gen_refined(
            L=params["L"],
            nx=params["nx"],
            ny=params["ny"],
            refine_edge=params["refine_edge"],
            ratio=params["ratio"],
            output_filename=output_file
        )
    else:
        # Structured uniform mesh
        return gen_uniform(
            L=params["L"],
            nx=params["nx"],
            ny=params["ny"],
            lc=params["lc"],
            output_filename=output_file
        )

def generate_unstructured_refined_mesh(model_name, params, output_file):
    """Generate an unstructured mesh with refinement using gmsh."""
    # Clear existing gmsh model
    gmsh.clear()
    gmsh.model.add(model_name)
    
    # Domain dimensions
    L = params.get("L", 1.0)
    
    # Get mesh size parameters
    mesh_size_walls = params.get("mesh_size_walls", 0.03)
    mesh_size_lid = params.get("mesh_size_lid", mesh_size_walls)  # Default to walls if not specified
    mesh_size_center = params.get("mesh_size_center", 0.08)
    
    # For experiments with obstacles
    obstacle_radius = params.get("obstacle_radius", 0)
    mesh_size_obstacle = params.get("mesh_size_obstacle", mesh_size_walls/2)
    
    # Create domain geometry
    if obstacle_radius > 0:
        # Domain with circular obstacle
        rect = gmsh.model.occ.addRectangle(0, 0, 0, L*2.2, L*0.41)
        disk = gmsh.model.occ.addDisk(0.2*L, 0.2*L, 0, obstacle_radius, obstacle_radius)
        # Perform the cut operation and get the result
        out, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
        domain = out  # out contains the resulting entities after cut
    else:
        # Simple rectangular domain
        rect = gmsh.model.occ.addRectangle(0, 0, 0, L, L)
        domain = [(2, rect)]
    
    # Synchronize to create geometry
    gmsh.model.occ.synchronize()
    
    # Add mesh size control points
    # Corners
    if obstacle_radius <= 0:  # Only for domains without obstacles
        # Let's get corners automatically based on the domain bounds
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, domain[0][1])
        
        # Get all the vertices of the domain
        boundary_vertices = []
        boundaries = gmsh.model.getBoundary(domain, recursive=True)
        for dim, tag in boundaries:
            if dim == 0:  # It's a point
                boundary_vertices.append((dim, tag))
                # Set default size for all boundary points
                gmsh.model.mesh.setSize([(dim, tag)], mesh_size_walls)
        
        # Add interior point for size control
        center_point = gmsh.model.geo.addPoint(xmin + (xmax-xmin)/2, ymin + (ymax-ymin)/2, 0, mesh_size_center)
        gmsh.model.geo.synchronize()
    
    # For domains with obstacles, refine near the obstacle
    if obstacle_radius > 0:
        # Get all edges on the domain boundary
        edges = []
        for dim, tag in domain:
            # Get the boundary of this face
            face_boundary = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False)
            edges.extend(face_boundary)
        
        # Add size control for all edges
        for dim, tag in edges:
            # Get the vertices of this edge
            edge_vertices = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False)
            for v_dim, v_tag in edge_vertices:
                gmsh.model.mesh.setSize([(v_dim, v_tag)], mesh_size_walls)
        
        # Add center point with larger mesh size
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, domain[0][1])
        center_point = gmsh.model.geo.addPoint(xmin + (xmax-xmin)/2, ymin + (ymax-ymin)/2, 0, mesh_size_center)
        gmsh.model.geo.synchronize()
    
    # Set up size fields for smooth size transitions
    # Define distance fields from boundaries
    field_walls = gmsh.model.mesh.field.add("Distance")
    
    # Get all edges on the boundary
    edges = []
    for dim, tag in domain:
        # Get the boundary of this face (edges)
        face_boundary = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False)
        edges.extend(face_boundary)
    
    # Set the edges for the distance field
    edge_tags = [tag for dim, tag in edges if dim == 1]
    gmsh.model.mesh.field.setNumbers(field_walls, "EdgesList", edge_tags)
    
    # Create threshold field to control size based on distance from boundaries
    field_size = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field_size, "IField", field_walls)
    gmsh.model.mesh.field.setNumber(field_size, "LcMin", mesh_size_walls)
    gmsh.model.mesh.field.setNumber(field_size, "LcMax", mesh_size_center)
    gmsh.model.mesh.field.setNumber(field_size, "DistMin", 0.01)
    gmsh.model.mesh.field.setNumber(field_size, "DistMax", 0.2)
    
    # Special field for lid (if applicable)
    if mesh_size_lid < mesh_size_walls and obstacle_radius <= 0:
        # Define points on the lid (top boundary)
        lid_points = []
        for dim, tag in boundaries:
            if dim == 0:  # It's a point
                x, y, z = gmsh.model.getValue(0, tag, [])
                if abs(y - ymax) < 1e-6:  # This is a point on the top boundary
                    lid_points.append((dim, tag))
        
        # Distance from lid points
        field_lid = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_lid, "NodesList", [tag for dim, tag in lid_points])
        
        # Threshold field for lid refinement
        field_lid_size = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_lid_size, "IField", field_lid)
        gmsh.model.mesh.field.setNumber(field_lid_size, "LcMin", mesh_size_lid)
        gmsh.model.mesh.field.setNumber(field_lid_size, "LcMax", mesh_size_walls)
        gmsh.model.mesh.field.setNumber(field_lid_size, "DistMin", 0.01)
        gmsh.model.mesh.field.setNumber(field_lid_size, "DistMax", 0.15)
        
        # Min field to combine both fields
        field_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field_size, field_lid_size])
        
        # Use this as the mesh size field
        gmsh.model.mesh.field.setAsBackgroundMesh(field_min)
    else:
        # Just use the wall distance field
        gmsh.model.mesh.field.setAsBackgroundMesh(field_size)
    
    # Generate the mesh
    gmsh.model.mesh.generate(2)
    
    # Save mesh file in VTK format
    gmsh.write(output_file)
    
    # Also save in native GMSH format (.msh)
    msh_file = os.path.splitext(output_file)[0] + ".msh"
    gmsh.write(msh_file)
    
    # Return mesh data (if needed)
    return None  # Simplified for now

def generate_experiment_meshes(exp_name, exp_config, base_dir):
    """Generate meshes for a specific experiment, saving only .vtk format for ParaView."""
    print(f"Generating meshes for experiment: {exp_name}")
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create a README for this experiment
    with open(os.path.join(exp_dir, "README.md"), "w") as f:
        f.write(f"# {exp_name} Experiment\n\n")
        f.write(f"{exp_config['description']}\n\n")
        f.write("## Mesh Types\n\n")
        f.write("- **structuredUniform**: " + exp_config["uniform"]["description"] + "\n")
        f.write("- **structuredRefined**: " + exp_config["refined"]["description"] + "\n")
        f.write("- **unstructured**: " + exp_config["unstructured"]["description"] + "\n")
    
    # ---------- Structured Uniform ----------
    mesh_dir = os.path.join(exp_dir, "structuredUniform")
    os.makedirs(mesh_dir, exist_ok=True)
    vtk_file = os.path.join(mesh_dir, f"{exp_name}_uniform.vtk")
    
    print(f"  Generating uniform mesh...")
    try:
        # Clear previous model and create a new one
        gmsh.clear()
        model_name = f"{exp_name}_uniform"
        gmsh.model.add(model_name)
        
        params = exp_config["uniform"]
        gen_uniform(
            L=params["L"],
            nx=params["nx"],
            ny=params["ny"],
            lc=params["lc"],
            output_filename=vtk_file,  # use vtk as target
            model_name=model_name  # Pass the model name
        )
        
        # Save in native GMSH format - make sure we're using the correct model
        msh_file = os.path.splitext(vtk_file)[0] + ".msh"
        gmsh.write(msh_file)
        print(f"  Saved to {vtk_file} and {msh_file}")
    except Exception as e:
        print(f"  Error generating uniform mesh: {e}")
    
    # ---------- Structured Refined ----------
    mesh_dir = os.path.join(exp_dir, "structuredRefined")
    os.makedirs(mesh_dir, exist_ok=True)
    vtk_file = os.path.join(mesh_dir, f"{exp_name}_refined.vtk")
    
    print(f"  Generating refined mesh...")
    try:
        # Clear previous model and create a new one
        gmsh.clear()
        model_name = f"{exp_name}_refined"
        gmsh.model.add(model_name)
        
        params = exp_config["refined"]
        gen_refined(
            L=params["L"],
            nx=params["nx"],
            ny=params["ny"],
            refine_edge=params["refine_edge"],
            ratio=params["ratio"],
            output_filename=vtk_file,
            model_name=model_name  # Pass the model name
        )
        
        # Save in native GMSH format - make sure we're using the correct model
        msh_file = os.path.splitext(vtk_file)[0] + ".msh"
        gmsh.write(msh_file)
        print(f"  Saved to {vtk_file} and {msh_file}")
    except Exception as e:
        print(f"  Error generating refined mesh: {e}")
    
    # ---------- Unstructured ----------
    mesh_dir = os.path.join(exp_dir, "unstructured")
    os.makedirs(mesh_dir, exist_ok=True)
    vtk_file = os.path.join(mesh_dir, f"{exp_name}_unstructured.vtk")
    
    print(f"  Generating unstructured mesh...")
    try:
        # Use the new unstructured refined mesh generator
        generate_unstructured_refined_mesh(
            f"{exp_name}_unstructured",
            exp_config["unstructured"],
            vtk_file
        )
        print(f"  Saved to {vtk_file} and {os.path.splitext(vtk_file)[0] + '.msh'}")
    except Exception as e:
        print(f"  Error generating unstructured mesh: {e}")

def generate_all_meshes(selected_experiment=None):
    """Generate meshes for all experiments or a specific one."""
    # Initialize gmsh
    gmsh.initialize()
    
    base_dir = os.path.join(script_dir, "experiments")
    os.makedirs(base_dir, exist_ok=True)
    
    # Create readme in base directory
    with open(os.path.join(base_dir, "README.md"), "w") as f:
        f.write("# CFD Experiment Meshes\n\n")
        f.write("This directory contains meshes for various CFD experiments.\n\n")
        f.write("## Experiments\n\n")
        for exp_name, exp_config in EXPERIMENTS.items():
            f.write(f"### {exp_name}\n")
            f.write(f"{exp_config['description']}\n\n")
    
    if selected_experiment:
        if selected_experiment in EXPERIMENTS:
            generate_experiment_meshes(selected_experiment, EXPERIMENTS[selected_experiment], base_dir)
        else:
            print(f"Error: Experiment '{selected_experiment}' not found")
            print("Available experiments:")
            for exp_name in EXPERIMENTS.keys():
                print(f"  - {exp_name}")
            sys.exit(1)
    else:
        # Generate meshes for all experiments
        for exp_name, exp_config in EXPERIMENTS.items():
            generate_experiment_meshes(exp_name, exp_config, base_dir)
    
    # Finalize gmsh
    gmsh.finalize()
    print("\nMesh generation complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate meshes for CFD experiments")
    parser.add_argument("experiment", nargs="?", help="Name of specific experiment to generate")
    args = parser.parse_args()
    
    generate_all_meshes(args.experiment) 