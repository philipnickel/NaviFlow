import platform
import subprocess
import time
import uuid
from datetime import datetime
import os
import numpy as np
import re
import hashlib



def format_floats(data):
    if isinstance(data, dict):
        return {k: format_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [format_floats(v) for v in data]
    elif isinstance(data, float):
        # Check if we need scientific notation
        abs_val = abs(data)
        if abs_val < 0.01 or abs_val >= 100000:
            # Use scientific notation for very small or large numbers
            mantissa, exponent = f"{data:.2e}".split('e')
            mantissa = mantissa.rstrip('0').rstrip('.')
            return f"{mantissa}e{int(exponent)}"
        else:
            # Use regular decimal notation for numbers between 0.01 and 100
            return f"{data:.2f}".rstrip('0').rstrip('.')
    else:
        return data

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def get_experiment_from_config_path(config_path):
    """Extract experiment name from config file path."""
    # Split path and get the experiment name (e.g., 'lidDrivenCavity' from 'experiments/lidDrivenCavity/debugging/config.yaml')
    parts = config_path.split('/')
    if len(parts) >= 2:
        return parts[1]  # Return the experiment name
    return "unknown"

def collect_metadata(
    args,
    config,
    mesh,
    mesh_file,
    bc_file,
    results_dir,
    Re,
    rho,
    mu,
    u_l2norm,
    v_l2norm,
    continuity_l2norm,
    start_time,
    end_time,
    logger_status=None,
    cd=None,
    cl=None,
):
    # Generate a unique run ID with date and hash
    random_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
    run_id = f"{random_hash}"

    # Get git info
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        git_commit = "unknown"

    num_cells = mesh.cell_centers.shape[0]
    num_faces = mesh.face_centers.shape[0]
    print(f"Mesh file: {mesh_file}")

    # Get experiment name from config path
    experiment = get_experiment_from_config_path(args.config)

    metadata = {
        # Simulation identification and tracking
        "Simulation id": run_id,
        "Experiment": experiment,
        "Git commit": git_commit,
        "Wall time (s)": round(end_time - start_time, 3),
        
        # Physical parameters
        "Reynolds number": Re,
        
        # Mesh and geometry
        "Mesh type": "Structured Uniform" if "structuredUniform" in mesh_file else "Unstructured",
        "Number of control volumes": num_cells,
        "Boundary conditions": bc_file,
        
        # Algorithm settings
        "Algorithm": config["algorithm"]["type"],
        "Convection scheme": config["algorithm"]["convection_discretization"],
        "Limiter": config["algorithm"].get("limiter", "No limiter used"),
        "Convergence tolerance": config["algorithm"]["convergence_criteria"]["residual"],
        
        # Relaxation factors
        "Momentum relaxation": config["algorithm"]["relaxation_factors"]["velocity"],
        "Pressure relaxation": config["algorithm"]["relaxation_factors"]["pressure"],
        
        # Momentum solver settings
        "Momentum solver": config["linear_solvers"]["momentum"]["type"],
        "Momentum solver preconditioner": config["linear_solvers"]["momentum"]["preconditioner"],
        "Momentum solver tolerance": config["linear_solvers"]["momentum"]["tolerance"],
        
        # Pressure solver settings
        "Pressure solver": config["linear_solvers"]["pressure"]["type"],
        "Pressure solver preconditioner": config["linear_solvers"]["pressure"]["preconditioner"],
        "Pressure solver tolerance": config["linear_solvers"]["pressure"]["tolerance"],
        "Non-orthogonal corrections": config["algorithm"]["non_orthogonal_corrections"],
        
        # Results and convergence
        "Number of iterations": len(u_l2norm),
        "Final u-residual": float(u_l2norm[-1]),
        "Final v-residual": float(v_l2norm[-1]),
        "Final continuity-residual": float(continuity_l2norm[-1]),
    }
    
     # Convert numpy types and format floats
    metadata = convert_numpy_types(metadata)
    metadata = format_floats(metadata)
    
    return metadata
