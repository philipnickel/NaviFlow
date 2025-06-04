import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from naviflow_collocated.utils.postprocess.plot_style import plt  # This will apply the style

# Set the style for all plots
plt.style.use(['science', 'grid'])

def save_pdf(fig, path):
    """Save a figure as a PDF file."""
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    print(f"Saved: {path}")
    plt.close(fig)

def get_obstacle_mask_from_msh(x, y, experiment):
    """
    Use the original mesh tagging from the .msh file to assign obstacle tags to solution cells.
    For 'cylinderFlow', use a geometric mask based on known center and radius.
    Returns a boolean mask where True means obstacle cell (physical tag 5 or inside obstacle geometry).
    """
    if experiment == "cylinderFlow" or "cylinderFlow" in experiment:
        # Cylinder center and radius from mesh generation
        center = np.array([0.2, 0.2])  # Updated cylinder center
        radius = 0.05
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask = dist < radius
        return mask
    # Fallback to original .msh-based logic for other experiments
    msh_file = os.path.join("meshing", "experiments", experiment, "unstructured", "medium", f"{experiment}_unstructured_medium.msh")
    try:
        with open(msh_file, 'r') as f:
            lines = f.readlines()
        # Parse $Nodes section
        node_section = lines.index('$Nodes\n')
        n_nodes = int(lines[node_section+1])
        node_lines = lines[node_section+2:node_section+2+n_nodes]
        node_coords = {}
        for line in node_lines:
            parts = line.strip().split()
            idx = int(parts[0])
            coord = tuple(map(float, parts[1:4]))
            node_coords[idx] = coord
        # Parse $Elements section
        elem_section = lines.index('$Elements\n')
        n_elems = int(lines[elem_section+1])
        elem_lines = lines[elem_section+2:elem_section+2+n_elems]
        centroids = []
        tags = []
        for line in elem_lines:
            parts = line.strip().split()
            elem_type = int(parts[1])
            if elem_type == 2:  # triangle (2D cell)
                num_tags = int(parts[2])
                physical_tag = int(parts[3])
                # Node indices are at the end
                node_ids = list(map(int, parts[3+num_tags:]))
                coords = [node_coords[nid] for nid in node_ids]
                centroid = tuple(np.mean(coords, axis=0))
                centroids.append(centroid)
                tags.append(physical_tag)
        centroids = np.array(centroids)
        tags = np.array(tags)
        # Only use x, y for centroid matching
        centroids_2d = centroids[:, :2]
        # Build KDTree for triangle centroids
        tree = cKDTree(centroids_2d)
        sol_xy = np.column_stack((x, y))
        _, idx = tree.query(sol_xy)
        tags_for_solution = tags[idx]
        return tags_for_solution == 5
    except Exception as e:
        print(f"Warning: Could not robustly detect obstacle cells from .msh: {e}")
        return np.zeros_like(x, dtype=bool)

def flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary into a single level dictionary with dot notation keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items) 