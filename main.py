import os
import argparse
import yaml
import numpy as np
from numba import config as numba_config
import time
from datetime import datetime
import signal
from naviflow_collocated.mesh.mesh_loader import load_mesh  
from naviflow_collocated.core.simple_algorithm import simple_algorithm, calculate_cylinder_forces  
from naviflow_collocated.utils.logger import ResidualLogger
from naviflow_collocated.utils.metadata import collect_metadata

interrupted = False

def handle_sigterm(signum, frame):
    global interrupted
    print(f"\nReceived termination signal ({signum}). Preparing graceful shutdown...")
    interrupted = True

# Register the handler
signal.signal(signal.SIGTERM, handle_sigterm)  # for `bkill`
signal.signal(signal.SIGINT, handle_sigterm)   # for Ctrl+C, dev use

start_time = time.time()
start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------
# CLI argument parsing
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True)
parser.add_argument("--max_iterations", type=int)
parser.add_argument("--reynolds_number", type=float)
parser.add_argument("--velocity_relaxation", type=float)
parser.add_argument("--pressure_relaxation", type=float)
parser.add_argument("--tolerance_exponent", type=int, help="Convergence tolerance as 10^-x (e.g., 4 for 1e-4)")
parser.add_argument("--print_interval", type=int, default=100, help="Print residuals every N iterations")
args = parser.parse_args()

# ----------------------------
# Resolve experiment path and config
# ----------------------------
experiment_path = os.path.join("experiments", args.experiment)
config_path = os.path.join(experiment_path, "config.yaml")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# ----------------------------
# Apply CLI overrides to config
# ----------------------------
if args.max_iterations is not None:
    config["algorithm"]["max_iterations"] = args.max_iterations
if args.reynolds_number is not None:
    config["physical_properties"]["reynolds_number"] = args.reynolds_number
if args.velocity_relaxation is not None:
    config["algorithm"]["relaxation_factors"]["velocity"] = args.velocity_relaxation
if args.pressure_relaxation is not None:
    config["algorithm"]["relaxation_factors"]["pressure"] = args.pressure_relaxation
if args.tolerance_exponent is not None:
    config["algorithm"]["convergence_criteria"]["residual"] = 10 ** -args.tolerance_exponent

# ----------------------------
# Load mesh and BCs
# ----------------------------
experiment_id = config["tags"][0]
mesh_type, resolution = config["domain"]["mesh"]

mesh_file = os.path.join(
    "meshing", "experiments", experiment_id,
    "structuredUniform" if "uniform" in mesh_type else "unstructured",
    resolution,
    f"{experiment_id}_{mesh_type}_{resolution}.msh"
)

bc_file = config["domain"]["boundary_conditions"]

print(f"Loading mesh: {mesh_file}")
mesh = load_mesh(mesh_file, bc_file)

# ----------------------------
# Set up result output directory
# ----------------------------
results_dir = os.path.join(experiment_path, "results")
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# ----------------------------
# Physical properties
# ----------------------------
rho = config["physical_properties"]["rho"]
U = config["physical_properties"].get("characteristic_velocity", 1.0)
D = config["physical_properties"].get("characteristic_length", 1.0)
Re = config["physical_properties"]["reynolds_number"]
mu = (U * D) / Re

# ----------------------------
# Solver config
# ----------------------------
alpha_uv = config["algorithm"]["relaxation_factors"]["velocity"]
alpha_p  = config["algorithm"]["relaxation_factors"]["pressure"]
max_iter = config["algorithm"]["max_iterations"]
tolerance = config["algorithm"]["convergence_criteria"]["residual"]
scheme = config["algorithm"]["convection_discretization"]
limiter = config["algorithm"].get("limiter", None)
algorithm = config["algorithm"]["type"]
n_nonortho_corrections = config["algorithm"].get("non_orthogonal_corrections", 0)  # Default to 2 if not specified

# Get linear solver settings from config
linear_solver_settings = {
    'momentum': {
        'type': config.get('linear_solvers', {}).get('momentum', {}).get('type', 'bcgs'),
        'preconditioner': config.get('linear_solvers', {}).get('momentum', {}).get('preconditioner', 'hypre'),
        'tolerance': config.get('linear_solvers', {}).get('momentum', {}).get('tolerance', 1e-6),
        'max_iterations': config.get('linear_solvers', {}).get('momentum', {}).get('max_iterations', 1000)
    },
    'pressure': {
        'type': config.get('linear_solvers', {}).get('pressure', {}).get('type', 'bcgs'),
        'preconditioner': config.get('linear_solvers', {}).get('pressure', {}).get('preconditioner', 'hypre'),
        'tolerance': config.get('linear_solvers', {}).get('pressure', {}).get('tolerance', 1e-6),
        'max_iterations': config.get('linear_solvers', {}).get('pressure', {}).get('max_iterations', 1000)
    }
}

# ----------------------------
# Set Numba thread count
# ----------------------------
numba_cores = config["numba_cores"]
if numba_cores != "default":
    os.environ["NUMBA_NUM_THREADS"] = str(numba_cores)

print(f"Using {numba_cores} threads")

# ----------------------------
# Run SIMPLE
# ----------------------------
logger = ResidualLogger(
    results_dir,
    divergence_factor=10000.0,
    allow_unsteady=False,
    convergence_tolerance=tolerance,
    print_every=args.print_interval
)

if algorithm == "PISO":
    print("Running PISO solver...")
    PISO = True
    PISO_corrections = config.get("algorithm", {}).get("PISO_corrections", 3)
else:
    PISO = False

# Initialize arrays to store force coefficients history
cd_history = np.zeros(max_iter)
cl_history = np.zeros(max_iter)

print("Running SIMPLE solver...")
U, p, continuity_field, u_l2norm, v_l2norm, continuity_l2norm, u_residual, v_residual, mdot_star, cd_history, cl_history = simple_algorithm(
    mesh,
    alpha_uv, alpha_p,
    rho, mu,
    max_iter, tolerance,
    scheme, limiter,
    PISO=PISO,
    progress_callback=logger.update,
    interruption_flag=lambda: interrupted,
    linear_solver_settings=linear_solver_settings,
    n_nonortho_corrections=n_nonortho_corrections
)
logger.close()
status = logger.status()

if status["diverging"]:
    print("TERMINATING run: residual divergence detected.")

if status["stalled"]:
    print("Residuals stalled — notify or flag post-analysis.")

# Calculate final lift and drag coefficients
U_inf = config["physical_properties"].get("characteristic_velocity", 1.0)
cd, cl = calculate_cylinder_forces(mesh, p, U, mu, rho, U_inf, D)
print(f"DEBUG: cd type: {type(cd)}, shape: {getattr(cd, 'shape', 'scalar')}")
print(f"DEBUG: cl type: {type(cl)}, shape: {getattr(cl, 'shape', 'scalar')}")
cd = cd.item() if hasattr(cd, 'item') else float(cd)
cl = cl.item() if hasattr(cl, 'item') else float(cl)
print(f"Drag coefficient (Cd): {cd:.6f}")
print(f"Lift coefficient (Cl): {cl:.6f}")

end_time = time.time()
wall_time_sec = end_time - start_time
print(f"SIMPLE solver completed in {wall_time_sec:.2f} seconds.")

print("Saving final state...")

# metadata
metadata = collect_metadata(args, config, mesh, mesh_file, bc_file, results_dir, Re, rho, mu, 
                          u_l2norm=u_l2norm, v_l2norm=v_l2norm, continuity_l2norm=continuity_l2norm,
                          start_time=start_time, end_time=end_time,
                          cd=cd, cl=cl)

np.save(os.path.join(results_dir, "U_final.npy"), U)
np.save(os.path.join(results_dir, "p_final.npy"), p)
np.savez(os.path.join(results_dir, "residuals.npz"),
         u=u_l2norm, v=v_l2norm, cont=continuity_l2norm)
np.savez(os.path.join(results_dir, "cell_centers.npz"),
         x=mesh.cell_centers[:, 0],
         y=mesh.cell_centers[:, 1])
np.savez(os.path.join(results_dir, "force_coefficients.npz"),
         cd=cd_history, cl=cl_history)
with open(os.path.join(results_dir, "metadata.yaml"), "w") as f:
    yaml.dump(metadata, f, sort_keys=False)

np.save(os.path.join(results_dir, "u_residual.npy"), u_residual)
np.save(os.path.join(results_dir, "v_residual.npy"), v_residual)
np.save(os.path.join(results_dir, "continuity_field.npy"), continuity_field)

print("State saved. Exiting.")
