"""
Mesh Generation Script for NaviFlow-Collocated

Generates different mesh types for various CFD experiment scenarios.
Creates .msh files with boundary and physical tagging.
Creates a structured directory hierarchy:
    meshing/
        experiments/
            experiment_type/
                mesh_type/
                    mesh files (.msh and .vtu)

Usage:
    python -m meshing.generate_meshes [experiment_name]

If no experiment name is provided, it will generate meshes for all experiments.
"""

import os
import sys
import gmsh
import meshio
import argparse

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import mesh generators
try:
    from naviflow_collocated.mesh.structured_uniform import generate as gen_uniform
    from naviflow_collocated.mesh.unstructured import generate as gen_unstructured
except ImportError as e:
    print(f"Error importing mesh generators: {e}")
    sys.exit(1)

def export_mesh(msh_file):
    """Export mesh to MSH format and convert to VTU."""
    try:
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(msh_file)
        print(f"  ✓ MSH (v2.2) written to: {msh_file}")

        # Convert to VTU
        mesh = meshio.read(msh_file)
        vtu_file = msh_file.replace(".msh", ".vtu")
        meshio.write(vtu_file, mesh)
        print(f"  ✓ VTU file written to: {vtu_file}")
    except Exception as e:
        print(f"  ❌ Mesh export failed: {e}")

EXPERIMENTS = {
    "lidDrivenCavity": {
        "description": "Classic lid-driven cavity problem with moving top wall",
        "uniform": {
            "L": 1.0, "nx": 30, "ny": 30, "lc": 0.02,
            "description": "Uniform mesh (30x30 Coarse) for lid-driven cavity"
        },
        "unstructured": {
            "Lx": 1.0, "Ly": 1.0, "n_cells": 3000, "ratio": 2.5,
            "description": "Unstructured mesh with boundary refinement using distance field"
        }
    },
    "channelFlow": {
        "description": "Channel flow with circular obstacle",
        "unstructured": {
            "Lx": 3.0, "Ly": 1.0, "n_cells": 4000, "ratio": 2.5,
            "obstacle": {
                "type": "circle",
                "center": (0.6, 0.5),
                "radius": 0.2
            },
            "description": "Unstructured mesh for channel flow with circular obstacle"
        }
    },
    "cavityWithObstacle": {
        "description": "Cavity flow with rectangular obstacle",
        "unstructured": {
            "Lx": 1.0, "Ly": 1.0, "n_cells": 3000, "ratio": 2.5,
            "obstacle": {
                "type": "rectangle",
                "start": (0.4, 0.2),
                "end": (0.6, 0.4)
            },
            "description": "Unstructured mesh for cavity with rectangular obstacle"
        }
    },
    "airfoilFlow": {
        "description": "External flow around a NACA airfoil",
        "unstructured": {
            "Lx": 5.0, "Ly": 3.0, "n_cells": 5000, "ratio": 3.0,
            "obstacle": {
                "type": "custom",
                "geometry": "naca",
                "params": {
                    "digits": "0012",     # NACA 0012 airfoil
                    "chord": 1.0,         # Chord length
                    "points": 100,        # Number of points to define airfoil
                    "angle": 5.0          # 5 degree angle of attack
                },
                "position": (1.5, 1.5),   # Position in domain
                "scale": 1.0              # Scale factor
            },
            "description": "Unstructured mesh for flow around NACA 0012 airfoil at 5° angle of attack"
        }
    }
}

def generate_experiment_meshes(exp_name, exp_config, base_dir):
    print(f"\n=== Generating meshes for experiment: {exp_name} ===")
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    readme_path = os.path.join(exp_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# {exp_name} Experiment\n\n{exp_config['description']}\n\n## Mesh Types\n\n")
        for key in ["uniform", "refined", "unstructured"]:
            if key in exp_config:
                f.write(f"- **{key}**: {exp_config[key]['description']}\n")
        f.write("\n## File Format\n- **.msh**: Gmsh format with boundary tags\n- **.vtu**: VTU format for ParaView\n")

    # ---------- Structured Uniform ----------
    if "uniform" in exp_config:
        print("→ Generating structured uniform mesh...")
        mesh_dir = os.path.join(exp_dir, "structuredUniform")
        os.makedirs(mesh_dir, exist_ok=True)
        msh_file = os.path.join(mesh_dir, f"{exp_name}_uniform.msh")

        try:
            gmsh.clear()
            gen_uniform(
                L=exp_config["uniform"]["L"],
                nx=exp_config["uniform"]["nx"],
                ny=exp_config["uniform"]["ny"],
                lc=exp_config["uniform"]["lc"],
                output_filename=None,  # Don't write yet
                model_name=f"{exp_name}_uniform"
            )
            export_mesh(msh_file)
        except Exception as e:
            print(f"  ❌ Error generating structured uniform mesh: {e}")

    # ---------- Unstructured ----------
    if "unstructured" in exp_config:
        print("→ Generating unstructured mesh...")
        mesh_dir = os.path.join(exp_dir, "unstructured")
        os.makedirs(mesh_dir, exist_ok=True)
        msh_file = os.path.join(mesh_dir, f"{exp_name}_unstructured.msh")

        try:
            gmsh.clear()
            # Handle different parameter sets
            if "Lx" in exp_config["unstructured"]:
                # Rectangular domain with possible obstacle
                gen_unstructured(
                    Lx=exp_config["unstructured"]["Lx"],
                    Ly=exp_config["unstructured"]["Ly"],
                    n_cells=exp_config["unstructured"]["n_cells"],
                    ratio=exp_config["unstructured"]["ratio"],
                    obstacle=exp_config["unstructured"].get("obstacle", None),
                    output_filename=None,  # Don't write yet
                )
            else:
                # Legacy square domain support
                gen_unstructured(
                    Lx=exp_config["unstructured"]["L"],
                    Ly=exp_config["unstructured"]["L"],
                    n_cells=exp_config["unstructured"]["n_cells"],
                    ratio=exp_config["unstructured"]["ratio"],
                    output_filename=None,  # Don't write yet
                )
            export_mesh(msh_file)
        except Exception as e:
            print(f"  ❌ Error generating unstructured mesh: {e}")


def generate_all_meshes(selected_experiment=None):
    # Initialize Gmsh ONCE at the beginning
    gmsh.initialize()
    # Ensure finalization even if errors occur
    try:
        base_dir = os.path.join(script_dir, "experiments")
        os.makedirs(base_dir, exist_ok=True)

        with open(os.path.join(base_dir, "README.md"), "w") as f:
            f.write("# CFD Experiment Meshes\n\n")
            for exp_name, exp_config in EXPERIMENTS.items():
                f.write(f"### {exp_name}\n{exp_config['description']}\n\n")

        if selected_experiment:
            if selected_experiment in EXPERIMENTS:
                generate_experiment_meshes(selected_experiment, EXPERIMENTS[selected_experiment], base_dir)
            else:
                print(f"Experiment '{selected_experiment}' not found.")
                # Don't call sys.exit here, let finalize run
                # sys.exit(1)
        else:
            for exp_name, exp_config in EXPERIMENTS.items():
                generate_experiment_meshes(exp_name, exp_config, base_dir)

        print("\n✅ Mesh generation complete.")

    finally:
        # Finalize Gmsh ONCE at the very end
        gmsh.finalize()
        # Removed the finalize call from here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate meshes for CFD experiments")
    parser.add_argument("experiment", nargs="?", help="Name of specific experiment to generate")
    args = parser.parse_args()
    generate_all_meshes(args.experiment)
