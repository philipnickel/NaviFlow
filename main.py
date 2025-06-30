import os
import argparse
import yaml
import numpy as np
from numba import config as numba_config
import time
from datetime import datetime
import signal
from naviflow_collocated.mesh.mesh_loader import load_mesh  
from naviflow_collocated.core.simple_algorithm import simple_algorithm
from naviflow_collocated.utils.logger import ResidualLogger
from naviflow_collocated.utils.metadata import collect_metadata
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients



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
parser.add_argument("--config", required=True, help="Path to the config file")
parser.add_argument("--max_iterations", type=int)
parser.add_argument("--reynolds_number", type=float)
parser.add_argument("--velocity_relaxation", type=float)
parser.add_argument("--pressure_relaxation", type=float)
parser.add_argument("--tolerance_exponent", type=int, help="Convergence tolerance as 10^-x (e.g., 4 for 1e-4)")
parser.add_argument("--print_interval", type=int, default=10, help="Print residuals every N iterations")
args = parser.parse_args()

# ----------------------------
# Load config
# ----------------------------
config_path = args.config
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
experiment_id = config.get("experiment", "default_experiment")
mesh_type, resolution = config["domain"]["mesh"]

# Construct a more flexible mesh path
mesh_base_path = os.path.join("meshing", "experiments")
potential_mesh_path = os.path.join(
    mesh_base_path, experiment_id,
    "structuredUniform" if "uniform" in mesh_type else "unstructured",
    resolution,
    f"{experiment_id}_{mesh_type}_{resolution}.msh"
)

# A fallback for cases like our transient one, where the mesh from the original experiment is used.
fallback_mesh_id = experiment_id.replace("transient_", "")
fallback_mesh_path = os.path.join(
    mesh_base_path, fallback_mesh_id,
    "structuredUniform" if "uniform" in mesh_type else "unstructured",
    resolution,
    f"{fallback_mesh_id}_{mesh_type}_{resolution}.msh"
)

if os.path.exists(potential_mesh_path):
    mesh_file = potential_mesh_path
elif os.path.exists(fallback_mesh_path):
    print(f"Warning: Mesh for '{experiment_id}' not found. Using mesh from '{fallback_mesh_id}'.")
    mesh_file = fallback_mesh_path
else:
    raise FileNotFoundError(f"Could not find mesh at '{potential_mesh_path}' or '{fallback_mesh_path}'")


bc_file = config["domain"]["boundary_conditions"]

print(f"Loading mesh: {mesh_file}")
mesh = load_mesh(mesh_file, bc_file)

# ----------------------------
# Set up result output directory
# ----------------------------
results_dir = os.path.join(os.path.dirname(config_path), "results")
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
print(f"Viscosity: {mu}")

# ----------------------------
# Solver config
# ----------------------------
alpha_uv = config["algorithm"]["relaxation_factors"]["velocity"]
alpha_p  = config["algorithm"]["relaxation_factors"]["pressure"]
max_iter = config["algorithm"]["max_iterations"]
tolerance = config["algorithm"]["convergence_criteria"]["residual"]
scheme = config["algorithm"]["convection_discretization"]
limiter = config["algorithm"].get("limiter", "MUSCL")
algorithm = config["algorithm"]["type"]
n_nonortho_corrections = config["algorithm"].get("non_orthogonal_corrections", 0)  # Default to 2 if not specified

# Get linear solver settings from config
linear_solver_settings = {
    'momentum': {
        'solver_type': config.get('linear_solvers', {}).get('momentum', {}).get('type', 'bcgs'),
        'preconditioner': config.get('linear_solvers', {}).get('momentum', {}).get('preconditioner', 'hypre'),
        'tolerance': config.get('linear_solvers', {}).get('momentum', {}).get('tolerance', 1e-6),
        'max_iterations': config.get('linear_solvers', {}).get('momentum', {}).get('max_iterations', 1000)
    },
    'pressure': {
        'solver_type': config.get('linear_solvers', {}).get('pressure', {}).get('type', 'bcgs'),
        'preconditioner': config.get('linear_solvers', {}).get('pressure', {}).get('preconditioner', 'hypre'),
        'tolerance': config.get('linear_solvers', {}).get('pressure', {}).get('tolerance', 1e-6),
        'max_iterations': config.get('linear_solvers', {}).get('pressure', {}).get('max_iterations', 1000)
    }
}

# Get output settings from config
output_settings = config.get("output_settings", {})
save_interval = output_settings.get("save_interval", 0)  # Default to 0 (only final state)

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

# Get PISO settings
PISO = config["algorithm"].get("PISO", False)
PISO_corrections = config["algorithm"].get("PISO_corrections", 1)

# Get time scheme settings
time_scheme = config["algorithm"].get("time_scheme", "Euler")  # Default to Euler scheme

transient_params = {
    'transient': config.get("algorithm", {}).get("transient", False),
    'dt': config.get("algorithm", {}).get("dt", 0.0),
    'end_time': config.get("algorithm", {}).get("end_time", 0.0)
}

print("Running SIMPLE solver...")
p, U, mdot, residuals, final_iter_count, is_converged, u_residual, v_residual, continuity_field = simple_algorithm(
    mesh=mesh,
    alpha_uv=alpha_uv,
    alpha_p=alpha_p,
    rho=rho,
    mu=mu,
    max_iter=max_iter,
    tol=tolerance,
    convection_scheme=scheme,
    limiter=limiter,
    PISO=PISO,
    PISO_corrections=PISO_corrections,
    progress_callback=logger,
    interruption_flag=lambda: interrupted,
    linear_solver_settings=linear_solver_settings,
    n_nonortho_corrections=n_nonortho_corrections,
    transient=transient_params['transient'],
    dt=transient_params['dt'],
    end_time=transient_params['end_time'],
    time_scheme=time_scheme,
    results_dir=results_dir,
    save_interval=save_interval
)
logger.close()
status = logger.status()

if status["diverging"]:
    print("TERMINATING run: residual divergence detected.")

if status["stalled"]:
    print("Residuals stalled — notify or flag post-analysis.")

end_time = time.time()
wall_time_sec = end_time - start_time
print(f"SIMPLE solver completed in {wall_time_sec:.2f} seconds.")

print("Saving final state metadata and residuals...")

(u_l2norm, v_l2norm, continuity_l2norm) = residuals

# metadata
metadata = collect_metadata(args, config, mesh, mesh_file, bc_file, results_dir, Re, rho, mu, 
                          u_l2norm=u_l2norm, v_l2norm=v_l2norm, continuity_l2norm=continuity_l2norm,
                          start_time=start_time, end_time=end_time)

# The primary fields (p, U) are now saved inside the solver loop.
# We no longer save them here to avoid redundancy.

# Save residuals from logger's history
np.savez(os.path.join(results_dir, "residuals.npz"),
         u=np.array(logger.history["u"]),
         v=np.array(logger.history["v"]),
         cont=np.array(logger.history["cont"]))

np.savez(os.path.join(results_dir, "cell_centers.npz"),
         x=mesh.cell_centers[:, 0],
         y=mesh.cell_centers[:, 1])
with open(os.path.join(results_dir, "metadata.yaml"), "w") as f:
    yaml.dump(metadata, f, sort_keys=False)

# Save residual fields
np.save(os.path.join(results_dir, "u_residual.npy"), u_residual)
np.save(os.path.join(results_dir, "v_residual.npy"), v_residual)
np.save(os.path.join(results_dir, "continuity_field.npy"), continuity_field)

print("State saved. Exiting.")
