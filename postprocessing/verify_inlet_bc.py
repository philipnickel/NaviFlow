import os
import argparse
import numpy as np
import yaml
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.mesh.helpers.mesh_loader_helpers import parse_physical_names

def main():
    parser = argparse.ArgumentParser(description="Verify inlet boundary condition application.")
    parser.add_argument("--config", required=True, help="Path to the simulation config file")
    args = parser.parse_args()

    # --- Load Config and Simulation Results ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    results_dir = os.path.join(os.path.dirname(args.config), "results")
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found at {results_dir}")
        return
        
    u_final_path = os.path.join(results_dir, "U_final.npy")
    if not os.path.exists(u_final_path):
        print(f"Error: U_final.npy not found at {u_final_path}")
        return
    U_final = np.load(u_final_path)

    # --- Load Mesh and BCs ---
    mesh_file = "meshing/experiments/cylinderFlow/unstructured/medium/cylinderFlow_unstructured_medium.msh"
    
    bc_file = config["domain"]["boundary_conditions"]
    mesh = load_mesh(mesh_file, bc_file)
    
    # Get the mapping from physical name to physical ID
    physical_names_map = parse_physical_names(mesh_file)
    name_to_id_map = {name: id for id, name in physical_names_map.items()}

    # --- Verification Logic ---
    print("\n--- Inlet Boundary Condition Verification ---")
    
    # Find the inlet patch from the boundary config
    with open(bc_file, 'r') as f:
        bc_config = yaml.safe_load(f)
    
    inlet_patch_name = None
    for patch_name, settings in bc_config['boundaries'].items():
        if settings.get('velocity', {}).get('bc') == 'inlet':
            inlet_patch_name = patch_name
            formula_u = settings['velocity']['value'][0]
            print(f"Found inlet patch: '{inlet_patch_name}' with U-velocity formula: '{formula_u}'")
            break
            
    if not inlet_patch_name:
        print("Error: Could not find an inlet patch in the boundary file.")
        return

    target_patch_id = name_to_id_map.get(inlet_patch_name)
    if target_patch_id is None:
        print(f"Error: Patch name '{inlet_patch_name}' not found in mesh physical groups: {list(name_to_id_map.keys())}")
        return

    # Find faces belonging to the inlet patch
    inlet_face_indices = []
    
    # mesh.boundary_patches is an array of size n_faces, where a non -1 value indicates a boundary face
    # and the value itself is the patch_id. We can find the face indices where the patch_id matches.
    all_face_indices = np.arange(len(mesh.boundary_patches))
    inlet_face_indices = all_face_indices[mesh.boundary_patches == target_patch_id]

    if len(inlet_face_indices) == 0:
        print("Error: No faces found for the inlet patch.")
        return

    print(f"Found {len(inlet_face_indices)} faces for patch '{inlet_patch_name}'.")
    print("\nVerifying velocities at face centers...")
    print("-" * 60)
    print(f"{'Face Index':<12} {'Y-Coord':<12} {'Expected U':<15} {'Actual U':<15}")
    print("-" * 60)

    discrepancies = 0
    for f_idx in inlet_face_indices:
        # Get face center y-coordinate
        y = mesh.face_centers[f_idx, 1]
        
        # Calculate expected U from formula
        # IMPORTANT: Use 'y' instead of 'x[1]' for eval, as we defined it
        # The formula from the yaml is: "4.0* 0.3 * x[1]*(0.41-x[1])/(0.41*0.41)"
        # We need to replace x[1] with y
        formula_u_safe = formula_u.replace("x[1]", "y")
        expected_u = eval(formula_u_safe, {"__builtins__": None}, {"y": y})
        
        # Get the actual U from the owner cell of that face
        owner_cell_idx = mesh.owner_cells[f_idx]
        actual_u = U_final[owner_cell_idx, 0]
        
        # Check for discrepancy
        if not np.isclose(expected_u, actual_u, atol=1e-5):
            discrepancies += 1
        
        print(f"{f_idx:<12} {y:<12.4f} {expected_u:<15.6f} {actual_u:<15.6f}")

    print("-" * 60)
    if discrepancies == 0:
        print("✅ Success: All actual inlet velocities match the expected formula.")
    else:
        print(f"❌ Found {discrepancies} discrepancies between expected and actual inlet velocities.")

if __name__ == "__main__":
    main() 