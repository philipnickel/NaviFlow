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
            "resolutions": {
                "coarse": {"L": 1.0, "nx": 5, "ny": 5, "lc": 0.05},
                "medium": {"L": 1.0, "nx": 10, "ny": 10, "lc": 0.025},
                "fine": {"L": 1.0, "nx": 20, "ny": 20, "lc": 0.0125},
            },
            "description": "Uniform structured mesh for lid-driven cavity at multiple resolutions",
        },
        "unstructured": {
            "resolutions": {
                "coarse": {"Lx": 1.0, "Ly": 1.0, "n_cells": 25, "ratio": 2.5},
                "medium": {"Lx": 1.0, "Ly": 1.0, "n_cells": 50, "ratio": 2.5},
                "fine": {"Lx": 1.0, "Ly": 1.0, "n_cells": 100, "ratio": 2.5},
            },
            "description": "Unstructured mesh with boundary refinement at multiple resolutions",
        },
    },
    "cylinderFlow": {
        "description": "Flow around a circular cylinder in a rectangular domain",
        "unstructured": {
            "resolutions": {
                "coarse": {
                    "Lx": 4.0, "Ly": 1.0, "n_cells": 200, "ratio": 2.5,
                    "obstacle": {
                        "type": "circle",
                        "center": (1.0, 0.5),
                        "radius": 0.1
                    },
                    "wake_refinement": True
                },
                "medium": {
                    "Lx": 4.0, "Ly": 1.0, "n_cells": 4000, "ratio": 2.5,
                    "obstacle": {
                        "type": "circle",
                        "center": (1.0, 0.5),
                        "radius": 0.1
                    },
                    "wake_refinement": True
                },
                "fine": {
                    "Lx": 4.0, "Ly": 1.0, "n_cells": 8000, "ratio": 2.5,
                    "obstacle": {
                        "type": "circle",
                        "center": (1.0, 0.5),
                        "radius": 0.1
                    },
                    "wake_refinement": True
                }
            },
            "description": "Unstructured mesh for flow around a cylinder at multiple resolutions"
        }
    }
}

def generate_experiment_meshes(exp_name, exp_config, base_dir, selected_resolution=None):
    print(f"\n=== Generating meshes for experiment: {exp_name} ===")
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    readme_path = os.path.join(exp_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# {exp_name} Experiment\n\n{exp_config['description']}\n\n## Mesh Types\n\n")
        for key in ["uniform", "refined", "unstructured"]:
            if key in exp_config:
                f.write(f"- **{key}**: {exp_config[key]['description']}\n")
        f.write("\n## Resolutions\n")
        f.write("- **coarse**: Lower resolution for quick tests\n")
        f.write("- **medium**: Balanced resolution for most simulations\n")
        f.write("- **fine**: Higher resolution for detailed flow features\n\n")
        f.write("\n## File Format\n- **.msh**: Gmsh format with boundary tags\n- **.vtu**: VTU format for ParaView\n")

    if "uniform" in exp_config:
        print("→ Generating structured uniform meshes...")
        mesh_dir = os.path.join(exp_dir, "structuredUniform")
        os.makedirs(mesh_dir, exist_ok=True)

        for res_name, res_config in exp_config["uniform"]["resolutions"].items():
            if selected_resolution and res_name != selected_resolution:
                continue

            print(f"  • Resolution: {res_name}")
            res_dir = os.path.join(mesh_dir, res_name)
            os.makedirs(res_dir, exist_ok=True)
            msh_file = os.path.join(res_dir, f"{exp_name}_uniform_{res_name}.msh")

            try:
                gmsh.clear()
                gen_uniform(
                    L=res_config["L"],
                    nx=res_config["nx"],
                    ny=res_config["ny"],
                    lc=res_config["lc"],
                    output_filename=msh_file,
                    model_name=f"{exp_name}_uniform_{res_name}"
                )
                export_mesh(msh_file)
            except Exception as e:
                print(f"  ❌ Error generating structured uniform mesh ({res_name}): {e}")

    if "unstructured" in exp_config:
        print("→ Generating unstructured meshes...")
        mesh_dir = os.path.join(exp_dir, "unstructured")
        os.makedirs(mesh_dir, exist_ok=True)

        for res_name, res_config in exp_config["unstructured"]["resolutions"].items():
            if selected_resolution and res_name != selected_resolution:
                continue

            print(f"  • Resolution: {res_name}")
            res_dir = os.path.join(mesh_dir, res_name)
            os.makedirs(res_dir, exist_ok=True)
            msh_file = os.path.join(res_dir, f"{exp_name}_unstructured_{res_name}.msh")

            try:
                gmsh.clear()
                gen_unstructured(
                    Lx=res_config["Lx"],
                    Ly=res_config["Ly"],
                    n_cells=res_config["n_cells"],
                    ratio=res_config["ratio"],
                    obstacle=res_config.get("obstacle", None),
                    wake_refinement=res_config.get("wake_refinement", False),
                    output_filename=msh_file,
                    model_name=f"{exp_name}_unstructured_{res_name}"
                )
                export_mesh(msh_file)
            except Exception as e:
                print(f"  ❌ Error generating unstructured mesh ({res_name}): {e}")

def generate_all_meshes(selected_experiment=None, selected_resolution=None):
    gmsh.initialize()
    try:
        base_dir = os.path.join(script_dir, "experiments")
        os.makedirs(base_dir, exist_ok=True)

        with open(os.path.join(base_dir, "README.md"), "w") as f:
            f.write("# CFD Experiment Meshes\n\n")
            for exp_name, exp_config in EXPERIMENTS.items():
                f.write(f"### {exp_name}\n{exp_config['description']}\n\n")

        if selected_experiment:
            if selected_experiment in EXPERIMENTS:
                generate_experiment_meshes(selected_experiment, EXPERIMENTS[selected_experiment], base_dir, selected_resolution)
            else:
                print(f"Experiment '{selected_experiment}' not found.")
        else:
            for exp_name, exp_config in EXPERIMENTS.items():
                generate_experiment_meshes(exp_name, exp_config, base_dir, selected_resolution)

        print("\n✅ Mesh generation complete.")

    finally:
        gmsh.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate meshes for CFD experiments")
    parser.add_argument("experiment", nargs="?", help="Name of specific experiment to generate")
    parser.add_argument("--resolution", "-r", choices=["coarse", "medium", "fine"], help="Generate only a specific resolution (coarse, medium, fine)")
    args = parser.parse_args()
    generate_all_meshes(args.experiment, args.resolution)
