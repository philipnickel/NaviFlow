#!/usr/bin/env python3
"""
Mesh Generation Script for NaviFlow-Collocated

Generates different mesh types for various CFD experiment scenarios.
Creates a structured directory hierarchy:
    meshing/
        experiments/
            experiment_type/
                mesh_type/
                    mesh files (.msh)

Usage:
    python -m meshing.generate_meshes [experiment_name]
    
If no experiment name is provided, it will generate meshes for all experiments.
"""

import os
import sys
import gmsh
import argparse

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
            "obstacle_radius": 0.0,  # No obstacle for lid-driven cavity
            "mesh_size": 0.05,  # Control mesh element size
            "description": "Unstructured triangle mesh" 
        }
    },
    "channelFlow": {
        "description": "Flow in a rectangular channel",
        "uniform": {
            "L": 2.0,  # Longer domain for channel
            "nx": 80,
            "ny": 40,
            "lc": 0.05,
            "description": "Uniform rectangular mesh"
        },
        "refined": {
            "L": 2.0,
            "nx": 60,
            "ny": 30,
            "refine_edge": "left",
            "ratio": 1.2,
            "description": "Refined near inlet (left edge)"
        },
        "unstructured": {
            "L": 2.0,
            "obstacle_radius": 0.0,
            "mesh_size": 0.05,
            "description": "Unstructured triangle mesh"
        }
    },
    "flowAroundCylinder": {
        "description": "Flow around a circular cylinder",
        "uniform": {
            "L": 2.2,
            "nx": 60,
            "ny": 40,
            "lc": 0.05,
            "description": "Uniform mesh (not ideal for cylinder flow)"
        },
        "refined": {
            "L": 2.2,
            "nx": 50,
            "ny": 30,
            "refine_edge": "left",
            "ratio": 1.2,
            "description": "Refined near inlet, not cylinder-specific"
        },
        "unstructured": {
            "L": 2.2,
            "obstacle_radius": 0.1,
            "mesh_size": 0.03,
            "description": "Unstructured mesh with circular obstacle"
        }
    },
    "backwardFacingStep": {
        "description": "Flow over a backward-facing step",
        "uniform": {
            "L": 3.0,
            "nx": 90,
            "ny": 30,
            "lc": 0.05,
            "description": "Uniform rectangular mesh for step flow domain"
        },
        "refined": {
            "L": 3.0,
            "nx": 70,
            "ny": 25,
            "refine_edge": "bottom",
            "ratio": 1.3,
            "description": "Refined near bottom where step is located"
        },
        "unstructured": {
            "L": 3.0,
            "obstacle_radius": 0.0,
            "mesh_size": 0.04,
            "description": "Unstructured triangle mesh for step domain"
        }
    }
}

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
        gmsh.clear()
        gmsh.model.add(f"{exp_name}_uniform")
        params = exp_config["uniform"]
        gen_uniform(
            L=params["L"],
            nx=params["nx"],
            ny=params["ny"],
            lc=params["lc"],
            output_filename=vtk_file  # use vtk as target
        )
        gmsh.write(vtk_file)
        print(f"  Saved to {vtk_file}")
    except Exception as e:
        print(f"  Error generating uniform mesh: {e}")
    
    # ---------- Structured Refined ----------
    mesh_dir = os.path.join(exp_dir, "structuredRefined")
    os.makedirs(mesh_dir, exist_ok=True)
    vtk_file = os.path.join(mesh_dir, f"{exp_name}_refined.vtk")
    
    print(f"  Generating refined mesh...")
    try:
        gmsh.clear()
        gmsh.model.add(f"{exp_name}_refined")
        params = exp_config["refined"]
        gen_refined(
            L=params["L"],
            nx=params["nx"],
            ny=params["ny"],
            refine_edge=params["refine_edge"],
            ratio=params["ratio"],
            output_filename=vtk_file
        )
        gmsh.write(vtk_file)
        print(f"  Saved to {vtk_file}")
    except Exception as e:
        print(f"  Error generating refined mesh: {e}")
    
    # ---------- Unstructured ----------
    mesh_dir = os.path.join(exp_dir, "unstructured")
    os.makedirs(mesh_dir, exist_ok=True)
    vtk_file = os.path.join(mesh_dir, f"{exp_name}_unstructured.vtk")
    
    print(f"  Generating unstructured mesh...")
    try:
        gmsh.clear()
        gmsh.model.add(f"{exp_name}_unstructured")
        params = exp_config["unstructured"]
        if params["obstacle_radius"] <= 0:
            params["obstacle_radius"] = 0.001
        gen_unstructured(
            L=params["L"],
            obstacle_radius=params["obstacle_radius"],
            output_filename=vtk_file
        )
        gmsh.write(vtk_file)
        print(f"  Saved to {vtk_file}")
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