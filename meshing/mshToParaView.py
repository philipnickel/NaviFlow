#!/usr/bin/env python3
"""
Recursively convert all .msh files under meshing/experiments/ to .vtu.
Only processes files that don't already have a matching .vtu.
"""

import os
import meshio

def convert_if_needed(msh_path):
    vtu_path = os.path.splitext(msh_path)[0] + ".vtu"
    if os.path.exists(vtu_path):
        print(f"✓ Skipping (already exists): {vtu_path}")
        return

    print(f"→ Converting: {msh_path}")
    mesh = meshio.read(msh_path)

    # Only keep 2D cells (triangles, quads)
    cell_types = ["triangle", "quad"]
    mesh.cells = [c for c in mesh.cells if c.type in cell_types]
    if not mesh.cells:
        print(f"  ⚠️ No 2D cells found in {msh_path}, skipping.")
        return

    # Filter corresponding cell data
    filtered_cell_data = {}
    for name, data in mesh.cell_data.items():
        filtered_cell_data[name] = [
            arr for cell_block, arr in zip(mesh.cells, data)
            if cell_block.type in cell_types
        ]
    mesh.cell_data = filtered_cell_data

    meshio.write(vtu_path, mesh)
    print(f"  ✓ Saved: {vtu_path}")

def convert_all_in_folder(root="meshing/experiments"):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".msh"):
                convert_if_needed(os.path.join(dirpath, fname))

if __name__ == "__main__":
    convert_all_in_folder()