import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from naviflow_collocated.mesh.mesh_loader import load_mesh, BC_WALL, BC_DIRICHLET, BC_NEUMANN, BC_ZEROGRADIENT


def validate_boundary_conditions(mesh, bc_config_file, plot=False):
    """
    Verifies that boundary types and values match those defined in the boundary config file.

    Parameters:
        mesh: MeshData2D
        bc_config_file: Path to the boundary conditions YAML file
        plot: If True, generates a visual verification of boundary values
    """
    # Load boundary configurations from file
    with open(bc_config_file, "r") as f:
        boundary_config = yaml.safe_load(f)
    
    boundary_conditions = boundary_config.get("boundaries", {})
    
    # Map BC type strings to internal codes
    bc_type_map = {
        "wall": BC_WALL,
        "dirichlet": BC_DIRICHLET,
        "neumann": BC_NEUMANN,
        "zerogradient": BC_ZEROGRADIENT
    }
    
    # Get mesh information
    face_ids = mesh.boundary_faces
    patch_ids = mesh.boundary_patches
    boundary_types = mesh.boundary_types
    boundary_values = mesh.boundary_values
    face_centers = mesh.face_centers
    
    # Create mapping from physical tag to boundary name
    tag_to_name = {}
    for face_id in face_ids:
        tag = patch_ids[face_id]
        if tag not in tag_to_name:
            # Find the name of this boundary from mesh data
            # We'll use the first instance of each tag
            # Get list of all boundary names from faces
            boundary_names = [name for name in boundary_conditions.keys()]
            for name in boundary_names:
                # Find if this face corresponds to this boundary name
                x_f = face_centers[face_id]
                expected_values = []
                
                # For each boundary patch, check if any faces match it by position
                # (This assumes boundaries are in distinct positions)
                # A more robust approach would be to include physical tags in the boundary config
                if name in ['left', 'right', 'top', 'bottom']:
                    # Simple geometric check for sides of square/rectangular domain
                    if name == 'left' and abs(x_f[0]) < 1e-6:
                        tag_to_name[tag] = name
                    elif name == 'right' and abs(x_f[0] - 1.0) < 1e-6:
                        tag_to_name[tag] = name
                    elif name == 'bottom' and abs(x_f[1]) < 1e-6:
                        tag_to_name[tag] = name
                    elif name == 'top' and abs(x_f[1] - 1.0) < 1e-6:
                        tag_to_name[tag] = name
        
    # Create visualization colors
    vel_colors = {
        BC_WALL: "blue",          # wall
        BC_DIRICHLET: "green",    # dirichlet
        BC_NEUMANN: "red",        # neumann
        BC_ZEROGRADIENT: "purple", # zerogradient
        -1: "gray",               # unknown
    }
    
    p_colors = {
        BC_WALL: "cyan",         # wall
        BC_DIRICHLET: "lightgreen", # dirichlet
        BC_NEUMANN: "salmon",    # neumann
        BC_ZEROGRADIENT: "lavender", # zerogradient
        -1: "lightgray",         # unknown
    }

    # Validate boundary conditions against config file
    for face in face_ids:
        patch_id = patch_ids[face]
        patch_name = tag_to_name.get(patch_id)
        
        if patch_name is None:
            continue  # Unknown patch
        
        # Get BC config for this boundary
        bc_config = boundary_conditions.get(patch_name, {})
        if not bc_config:
            continue
        
        # Get velocity BC info
        vel_bc = bc_config.get("velocity", {})
        vel_bc_type_str = vel_bc.get("bc", "zerogradient").lower()
        expected_vel_type = bc_type_map.get(vel_bc_type_str, -1)
        vel_value_raw = vel_bc.get("value", [0.0, 0.0])
        
        # Get pressure BC info
        p_bc = bc_config.get("pressure", {})
        p_bc_type_str = p_bc.get("bc", "zerogradient").lower()
        expected_p_type = bc_type_map.get(p_bc_type_str, -1)
        p_value_raw = p_bc.get("value", 0.0)
        
        # Process velocity values (handle expressions)
        x_f = face_centers[face]
        expected_vel_value = [0.0, 0.0]
        
        if isinstance(vel_value_raw, list):
            for i, item in enumerate(vel_value_raw[:2]):
                if isinstance(item, str):
                    try:
                        expected_vel_value[i] = eval(item, {"np": np, "x": x_f})
                    except Exception:
                        expected_vel_value[i] = 0.0
                else:
                    expected_vel_value[i] = item
        elif isinstance(vel_value_raw, (int, float)):
            expected_vel_value[0] = vel_value_raw
        
        # Process pressure value (handle expressions)
        expected_p_value = 0.0
        if isinstance(p_value_raw, str):
            try:
                expected_p_value = eval(p_value_raw, {"np": np, "x": x_f})
            except Exception:
                expected_p_value = 0.0
        else:
            expected_p_value = p_value_raw
            
        # Get actual values from mesh
        actual_vel_type = boundary_types[face, 0]
        actual_p_type = boundary_types[face, 1]
        actual_vel_value = boundary_values[face, 0:2]
        actual_p_value = boundary_values[face, 2]
        
        # Verify BC types
        assert actual_vel_type == expected_vel_type, (
            f"Face {face} ({patch_name}): expected velocity type {expected_vel_type} ('{vel_bc_type_str}'), got {actual_vel_type}"
        )
        
        assert actual_p_type == expected_p_type, (
            f"Face {face} ({patch_name}): expected pressure type {expected_p_type} ('{p_bc_type_str}'), got {actual_p_type}"
        )
        
        # Verify BC values
        assert np.allclose(actual_vel_value, expected_vel_value, atol=1e-12), (
            f"Face {face} ({patch_name}): expected velocity {expected_vel_value}, got {actual_vel_value.tolist()}"
        )
        
        assert np.isclose(actual_p_value, expected_p_value, atol=1e-12), (
            f"Face {face} ({patch_name}): expected pressure {expected_p_value}, got {actual_p_value}"
        )
    
    print("✅ All boundary conditions validated successfully.")
    
    if plot:
        # Plot face-centered velocity arrows colored by type
        plt.figure(figsize=(8, 6))
        for face in face_ids:
            x, y = face_centers[face]
            u, v = boundary_values[face, 0:2]
            p = boundary_values[face, 2]
            vel_type = boundary_types[face, 0]
            p_type = boundary_types[face, 1]
            patch_id = patch_ids[face]
            patch_name = tag_to_name.get(patch_id, "unknown")
            
            # Plot velocity vector
            vel_color = vel_colors.get(vel_type, "red")
            plt.quiver(x, y, u, v, angles="xy", scale_units="xy", scale=1, color=vel_color, alpha=0.8)
            
            # Add text label
            label = f"{patch_name}\nvel_type={vel_type}, p_type={p_type}\nu=({u:.2g},{v:.2g}), p={p:.2g}"
            plt.text(x, y, label, fontsize=6, color=vel_color)

        plt.scatter(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], color="black", s=5, alpha=0.3)
        plt.axis("equal")
        plt.title("Boundary Values and Types")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("tests/test_output/BCapplication", exist_ok=True)
        plt.savefig(f"tests/test_output/BCapplication/BC_application_{bc_config_file.split('/')[-1].split('.')[0]}.png")


# Test cases
def test_boundary_conditions_for_mms():
    mesh_path = "meshing/experiments/sanityCheck/structuredUniform/coarse/sanityCheck_uniform_coarse.msh"
    
    # Test diffusion case (sine function with Dirichlet BCs)
    diffusion_mesh = load_mesh(mesh_path, "shared_configs/domain/sanityCheckDiffusion.yaml")
    validate_boundary_conditions(diffusion_mesh, "shared_configs/domain/sanityCheckDiffusion.yaml", plot=True)
    
    # Test convection case (cosine function with mixed BCs)
    convection_mesh = load_mesh(mesh_path, "shared_configs/domain/sanityCheckConvection.yaml")
    validate_boundary_conditions(convection_mesh, "shared_configs/domain/sanityCheckConvection.yaml", plot=True)
    
    # Test combined case (sine function with mixed BCs)
    combined_mesh = load_mesh(mesh_path, "shared_configs/domain/sanityCheckCombined.yaml")
    validate_boundary_conditions(combined_mesh, "shared_configs/domain/sanityCheckCombined.yaml", plot=True)


if __name__ == "__main__":
    test_boundary_conditions_for_mms()
