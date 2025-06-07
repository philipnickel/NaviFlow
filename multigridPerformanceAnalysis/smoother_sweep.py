import sys
import os
import numpy as np
import pandas as pd
import time
from itertools import product

# Add the project root to the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import solvers and helpers
from naviflow_staggered.preprocessing.mesh.structured import StructuredMesh
from naviflow_staggered.solver.pressure_solver.multigrid import MultiGridSolver
from naviflow_staggered.solver.pressure_solver.jacobi import JacobiSolver
from naviflow_staggered.solver.pressure_solver.gauss_seidel import GaussSeidelSolver
from naviflow_staggered.solver.pressure_solver.helpers.matrix_free import compute_Ap_product

# --- MMS Setup ---
def p_manufactured(x, y):
    """Manufactured solution for pressure with zero-gradient boundaries."""
    return (1 - np.cos(2 * np.pi * x)) * (1 - np.cos(2 * np.pi * y))

def setup_mms_problem(grid_size):
    """Creates the mesh, manufactured solution, and RHS for a given grid size."""
    mesh = StructuredMesh(nx=grid_size, ny=grid_size, length=1.0, height=1.0)
    nx, ny = mesh.get_dimensions()
    dx, dy = mesh.get_cell_sizes()
    
    p_mms = p_manufactured(mesh.X, mesh.Y)
    p_mms_flat = p_mms.flatten('F')

    rho = 1.0
    d_u = np.ones((nx + 1, ny))
    d_v = np.ones((nx, ny + 1))

    b_manufactured = compute_Ap_product(p_mms_flat, nx, ny, dx, dy, rho, d_u, d_v, pin_pressure=True)
    
    problem_data = {
        "mesh": mesh,
        "p_mms_flat": p_mms_flat,
        "b_manufactured": b_manufactured,
        "rho": rho,
        "d_u": d_u,
        "d_v": d_v
    }
    return problem_data

# --- Main Sweep Logic ---
def run_single_case(params, problem_data, num_cycles=1):
    """Runs a single simulation case and returns performance metrics."""
    cycle_type, smoother_name, nu1, nu2 = params
    
    # Unpack problem data
    mesh = problem_data["mesh"]
    p_mms_flat = problem_data["p_mms_flat"]
    b_manufactured = problem_data["b_manufactured"]
    rho = problem_data["rho"]
    d_u = problem_data["d_u"]
    d_v = problem_data["d_v"]
    
    nx, ny = mesh.get_dimensions()
    dx, dy = mesh.get_cell_sizes()

    # Initialize the correct smoother
    if smoother_name == 'Jacobi':
        smoother = JacobiSolver(omega=2./3., max_iterations=1)
    elif smoother_name == 'Gauss-Seidel':
        smoother = GaussSeidelSolver(omega=1.0, max_iterations=1, method_type='standard')
    elif smoother_name == 'Red-Black GS':
        smoother = GaussSeidelSolver(omega=1.0, max_iterations=1, method_type='red_black')
    else:
        raise ValueError(f"Unknown smoother: {smoother_name}")

    # Initialize MultiGridSolver
    mg_solver = MultiGridSolver(
        smoother=smoother,
        max_iterations=num_cycles,
        tolerance=1e-12, 
        pre_smoothing=nu1,
        post_smoothing=nu2,
        cycle_type=cycle_type,
        coarsest_grid_size=7
    )

    # Run the simulation
    p_solution = np.zeros_like(b_manufactured)
    start_time = time.time()
    
    cycle_method = getattr(mg_solver, f"_{cycle_type}_cycle")
    
    # Arguments differ between v_cycle (p, rhs) and w_cycle (u, f)
    cycle_args = {
        'mesh': mesh, 'rho': rho, 'd_u': d_u, 'd_v': d_v,
        'omega': smoother.omega, 'pre_smoothing': nu1, 'post_smoothing': nu2
    }
    if cycle_type == 'v':
        cycle_args['p'] = p_solution
        cycle_args['rhs'] = b_manufactured
    else: # w_cycle
        cycle_args['u'] = p_solution
        cycle_args['f'] = b_manufactured

    for _ in range(num_cycles):
        if cycle_type == 'v':
            cycle_args['p'] = cycle_method(**cycle_args)
        else: # w_cycle
            cycle_args['u'] = cycle_method(**cycle_args)
    
    p_solution = cycle_args.get('p', cycle_args.get('u'))
    
    end_time = time.time()
    wall_time = end_time - start_time

    # Calculate metrics
    Ap = compute_Ap_product(p_solution, nx, ny, dx, dy, rho, d_u, d_v, pin_pressure=True)
    residual_norm = np.linalg.norm(b_manufactured - Ap)
    initial_residual_norm = np.linalg.norm(b_manufactured)
    conv_rate = (residual_norm / initial_residual_norm) ** (1.0 / num_cycles) if initial_residual_norm > 0 else 0

    work_factor = (4./3.) if cycle_type == 'v' else 2.0
    work_units = num_cycles * (nu1 + nu2) * work_factor

    return {
        'CycleType': cycle_type,
        'Smoother': smoother_name,
        'nu1': nu1,
        'nu2': nu2,
        'Time': wall_time,
        'WorkUnits': work_units,
        'ConvRate': conv_rate
    }

def run_and_print_summary_table():
    """Performs the parameter sweep and prints the summary table."""
    grid_size = 63
    print(f"Setting up MMS problem for grid size: {grid_size}x{grid_size}")
    problem_data = setup_mms_problem(grid_size)
    
    # Define parameter space
    cycle_types = ['v', 'w']
    smoother_types = ['Jacobi', 'Gauss-Seidel', 'Red-Black GS']
    nu_values = [3, 5, 7]
    
    param_combinations = list(product(cycle_types, smoother_types, nu_values))
    results = []
    
    print(f"Starting smoother sweep for {len(param_combinations)} combinations...")
    for i, (cycle_type, smoother_name, nu) in enumerate(param_combinations):
        params = (cycle_type, smoother_name, nu, nu)
        # Use carriage return to show progress on a single line
        print(f"  Running case {i+1}/{len(param_combinations)}: {params}", end='\r')
        result = run_single_case(params, problem_data, num_cycles=1)
        results.append(result)
    print("\nSweep complete.                                          ")
        
    df = pd.DataFrame(results)
    
    # --- Process and Print Table ---
    df['TotalSmooths'] = df['nu1'] + df['nu2']
    
    summary_table = df[['CycleType', 'Smoother', 'TotalSmooths', 'ConvRate', 'WorkUnits', 'Time']].copy()
    summary_table.sort_values(by=['CycleType', 'Smoother', 'TotalSmooths'], inplace=True)
    summary_table.rename(columns={
        'CycleType': 'Cycle',
        'TotalSmooths': 'Total Smooths (ν₁+ν₂)',
        'WorkUnits': 'Work',
        'ConvRate': 'Conv. Rate'
    }, inplace=True)
    
    print("\n--- Single-Cycle Convergence Rate Summary ---")
    # Use a unicode character for nu in the printout for better readability
    print(summary_table.to_string(index=False, float_format="%.3f").replace('ν', 'ν'))

if __name__ == "__main__":
    run_and_print_summary_table() 