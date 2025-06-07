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
            if mantissa == "":
                mantissa = "0"
            return f"{mantissa}e{int(exponent)}"
        else:
            # Use regular decimal notation for numbers between 0.01 and 100
            return f"{data:.2f}".rstrip('0').rstrip('.')
    elif isinstance(data, str):
        try:
            # Try to convert string to float and format it
            return format_floats(float(data))
        except (ValueError, TypeError):
            return data
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

    # Read the last residuals from residuals.log
    residuals_log = os.path.join(results_dir, "residuals.log")
    u_res = v_res = cont_res = 0.0
    
    if os.path.exists(residuals_log):
        try:
            with open(residuals_log, 'r') as f:
                # Skip header
                next(f)
                # Read all lines and get the last one
                last_line = None
                for line in f:
                    if line.strip():  # Only consider non-empty lines
                        last_line = line
                if last_line:
                    # Parse the last line
                    _, u_res_str, v_res_str, cont_res_str = last_line.strip().split(',')
                    u_res = float(u_res_str)
                    v_res = float(v_res_str)
                    cont_res = float(cont_res_str)
                    print(f"Read residuals from log: u={u_res}, v={v_res}, cont={cont_res}")
        except Exception as e:
            print(f"Warning: Failed to read residuals from log: {e}")
            # Fallback to solver residuals
            u_res = float(u_l2norm[-1]) if len(u_l2norm) > 0 else 0.0
            v_res = float(v_l2norm[-1]) if len(v_l2norm) > 0 else 0.0
            cont_res = float(continuity_l2norm[-1]) if len(continuity_l2norm) > 0 else 0.0
    else:
        print("Warning: residuals.log not found")
        # Fallback to solver residuals
        u_res = float(u_l2norm[-1]) if len(u_l2norm) > 0 else 0.0
        v_res = float(v_l2norm[-1]) if len(v_l2norm) > 0 else 0.0
        cont_res = float(continuity_l2norm[-1]) if len(continuity_l2norm) > 0 else 0.0

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
        "Final u-residual": u_res,
        "Final v-residual": v_res,
        "Final continuity-residual": cont_res,
        
        # Add the full config for complete reproducibility
        "Config": config,
    }
    
    # Add force coefficients if available
    if cd is not None and cl is not None:
        metadata["results"] = {
            "drag_coefficient": cd,
            "lift_coefficient": cl
        }
    
    # Convert numpy types and format floats
    metadata = convert_numpy_types(metadata)
    metadata = format_floats(metadata)
    
    return metadata
