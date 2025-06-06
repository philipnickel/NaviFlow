import os
import argparse
import numpy as np
import yaml
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.mesh.helpers.mesh_loader_helpers import parse_physical_names

def main():
# ... existing code ...
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
# ... existing code ...
    if not inlet_patch_name:
        print("Error: Could not find an inlet patch in the boundary file.")
        return

    target_patch_id = name_to_id_map.get(inlet_patch_name)
    if target_patch_id is None:
        print(f"Error: Patch name '{inlet_patch_name}' not found in mesh physical groups: {list(name_to_id_map.keys())}")
        return

    # Find faces belonging to the inlet patch
    inlet_face_indices = []
    for f_idx in mesh.boundary_faces:
        # mesh.boundary_patches contains the physical ID for each face
        if mesh.boundary_patches[f_idx] == target_patch_id:
            inlet_face_indices.append(f_idx)

    if not inlet_face_indices:
# ... existing code ... 