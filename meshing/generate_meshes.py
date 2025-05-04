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
EXPERIMENTS = {
    "lidDrivenCavity": {
        "description": "Classic lid-driven cavity problem with moving top wall",
        "uniform": {
            "L": 1.0,
            "nx": 50,
            "ny": 50,
            "lc": 0.02,
            "description": "Uniform mesh for lid-driven cavity"
        },
        "refined": {
            "L": 1.0,
            "nx": 50,
            "ny": 50,
            "refine_edge": "top",
            "ratio": 1.15,
            "description": "Mesh with refinement near the top (lid)"
        }
    },
    "flowAroundCylinder": {
        "description": "Flow around a circular cylinder",
        "unstructured": {
            "L": 2.2,
            "obstacle_radius": 0.2,
            "description": "Unstructured mesh with circular obstacle"
        }
    }
}

def generate_experiment_meshes(exp_name, exp_config, base_dir):
    print(f"Generating meshes for experiment: {exp_name}")
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "README.md"), "w") as f:
        f.write(f"# {exp_name} Experiment\n\n")
        f.write(f"{exp_config['description']}\n\n")
        f.write("## Mesh Types\n\n")
        if "uniform" in exp_config:
            f.write("- **structuredUniform**: " + exp_config["uniform"]["description"] + "\n")
        if "refined" in exp_config:
            f.write("- **structuredRefined**: " + exp_config["refined"]["description"] + "\n")
        if "unstructured" in exp_config:
            f.write("- **unstructured**: " + exp_config["unstructured"]["description"] + "\n")
        f.write("\n## File Formats\n\n")
        f.write("- **.msh**: Native GMSH format with all boundary information\n")
        f.write("  - Used for both simulation and visualization\n")
        f.write("  - For visualization in ParaView, use the meshio plugin\n")

    if "uniform" in exp_config:
        mesh_dir = os.path.join(exp_dir, "structuredUniform")
        os.makedirs(mesh_dir, exist_ok=True)
        msh_file = os.path.join(mesh_dir, f"{exp_name}_uniform.msh")

        print(f"  Generating uniform mesh...")
        try:
            gmsh.clear()
            model_name = f"{exp_name}_uniform"
            gen_uniform(
                L=exp_config["uniform"]["L"],
                nx=exp_config["uniform"]["nx"],
                ny=exp_config["uniform"]["ny"],
                lc=exp_config["uniform"]["lc"],
                output_filename=msh_file,
                model_name=model_name
            )
        except Exception as e:
            print(f"  Error generating uniform mesh: {e}")

    if "refined" in exp_config:
        mesh_dir = os.path.join(exp_dir, "structuredRefined")
        os.makedirs(mesh_dir, exist_ok=True)
        msh_file = os.path.join(mesh_dir, f"{exp_name}_refined.msh")

        print(f"  Generating refined mesh...")
        try:
            gmsh.clear()
            model_name = f"{exp_name}_refined"
            gen_refined(
                L=exp_config["refined"]["L"],
                nx=exp_config["refined"]["nx"],
                ny=exp_config["refined"]["ny"],
                refine_edge=exp_config["refined"]["refine_edge"],
                ratio=exp_config["refined"]["ratio"],
                output_filename=msh_file,
                model_name=model_name
            )
        except Exception as e:
            print(f"  Error generating refined mesh: {e}")

    if "unstructured" in exp_config:
        radius = exp_config["unstructured"].get("obstacle_radius", 0.0)
        if radius > 0.0:
            mesh_dir = os.path.join(exp_dir, "unstructured")
            os.makedirs(mesh_dir, exist_ok=True)
            msh_file = os.path.join(mesh_dir, f"{exp_name}_unstructured.msh")

            print(f"  Generating unstructured mesh...")
            try:
                gmsh.clear()
                model_name = f"{exp_name}_unstructured"
                gen_unstructured(
                    L=exp_config["unstructured"].get("L", 1.0),
                    obstacle_radius=radius,
                    output_filename=msh_file,
                    model_name=model_name
                )
            except Exception as e:
                print(f"  Error generating unstructured mesh: {e}")


def generate_all_meshes(selected_experiment=None):
    gmsh.initialize()

    base_dir = os.path.join(script_dir, "experiments")
    os.makedirs(base_dir, exist_ok=True)

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
        for exp_name, exp_config in EXPERIMENTS.items():
            generate_experiment_meshes(exp_name, exp_config, base_dir)

    gmsh.finalize()
    print("\nMesh generation complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate meshes for CFD experiments")
    parser.add_argument("experiment", nargs="?", help="Name of specific experiment to generate")
    args = parser.parse_args()

    generate_all_meshes(args.experiment)