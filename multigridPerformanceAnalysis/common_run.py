import os
import subprocess
import re
import sys
import tracemalloc
import time

def run_simulation(script_path, params):
    """
    Runs a simulation with specified parameters, tracking performance of the solve step.
    Disables plotting and post-processing data generation.
    """
    results_dir_name = f'gs{params["grid_size"]}_re{params["reynolds"]}_it{params["max_iterations"]}_tol{params["tolerance"]}_ptol{params["pressure_tolerance"]}'
    results_dir = os.path.join(
        'multigridPerformanceAnalysis',
        f'results_{params["name"]}',
        results_dir_name
    )
    os.makedirs(results_dir, exist_ok=True)

    with open(script_path, 'r') as f:
        lines = f.readlines()

    # Find the full algorithm.solve() call
    solve_call_start_index = -1
    solve_call_end_index = -1
    open_parentheses = 0
    for i, line in enumerate(lines):
        if "result = algorithm.solve(" in line:
            solve_call_start_index = i
            open_parentheses += line.count('(')
            open_parentheses -= line.count(')')
            if open_parentheses == 0:
                solve_call_end_index = i
                break
        elif solve_call_start_index != -1:
            open_parentheses += line.count('(')
            open_parentheses -= line.count(')')
            if open_parentheses == 0:
                solve_call_end_index = i
                break
    
    if solve_call_start_index == -1 or solve_call_end_index == -1:
        raise ValueError(f"Could not find full 'result = algorithm.solve(...)' call in the script {script_path}.")

    indentation = lines[solve_call_start_index][:len(lines[solve_call_start_index]) - len(lines[solve_call_start_index].lstrip())]

    # Insert performance tracking code
    lines.insert(solve_call_start_index, f"{indentation}tracemalloc.start()\n")
    lines.insert(solve_call_start_index + 1, f"{indentation}start_wall_time = time.perf_counter()\n")
    lines.insert(solve_call_start_index + 2, f"{indentation}start_cpu_time = time.process_time()\n")
    
    solve_call_end_index += 3 # Adjust for inserted lines
    
    lines.insert(solve_call_end_index + 1, f"\n{indentation}end_wall_time = time.perf_counter()\n")
    lines.insert(solve_call_end_index + 2, f"{indentation}end_cpu_time = time.process_time()\n")
    lines.insert(solve_call_end_index + 3, f"{indentation}current, peak = tracemalloc.get_traced_memory()\n")
    lines.insert(solve_call_end_index + 4, f"{indentation}tracemalloc.stop()\n")
    lines.insert(solve_call_end_index + 5, f"{indentation}print(f'--- Performance for algorithm.solve() ---')\n")
    lines.insert(solve_call_end_index + 6, f"{indentation}print(f'Wall time: {{end_wall_time - start_wall_time:.4f}} seconds')\n")
    lines.insert(solve_call_end_index + 7, f"{indentation}print(f'CPU time: {{end_cpu_time - start_cpu_time:.4f}} seconds')\n")
    lines.insert(solve_call_end_index + 8, f"{indentation}print(f'Peak memory usage: {{peak / 10**6:.4f}} MB')\n")
    lines.insert(solve_call_end_index + 9, f"{indentation}print(f'------------------------------------')\n")

    new_lines = []
    new_lines.append("import tracemalloc\n")
    new_lines.append("import time\n")
    new_lines.append("try:\n")
    new_lines.append("    import petsc4py\n")
    new_lines.append("    petsc4py.PETSc.Options().setValue('-log_view', None)\n")
    new_lines.append("except (ImportError, AttributeError):\n")
    new_lines.append("    pass\n")

    # Comment out all lines after the solve call block
    lines_after_solve = lines[solve_call_end_index + 10:]
    for i in range(len(lines_after_solve)):
        lines[solve_call_end_index + 10 + i] = "#" + lines[solve_call_end_index + 10 + i]
    
    lines_to_comment_out = [
        "plot_combined_results", "plot_final_residuals", "metadata.yaml",
        "U_final.npy", "p_final.npy", "cell_centers.npz", "residuals.npz",
        "u_residual.npy", "v_residual.npy", "continuity_field.npy",
        "pseudo_config.yaml", 'print("Saving data in collocated format...")',
        'print(f"Data saved in collocated format to {results_dir}")',
        'print(f"You can now run post-processing with:")',
        'print("Created pseudo_config.yaml for post-processing compatibility")'
    ]
    
    in_plotting_block = False
    for line in lines:
        new_line = line
        # This is a bit of a hack, but it's the most reliable way to inject the verbose flag
        if "verbose_mg" in params and params["verbose_mg"]:
            if "smoother=smoother," in new_line:
                new_line = new_line.replace("smoother=smoother,", "smoother=smoother, verbose=True,")
        
        if any(trigger in new_line for trigger in ["result.plot_combined_results", "plot_final_residuals"]):
            in_plotting_block = True
        
        should_comment = in_plotting_block or any(trigger in new_line for trigger in lines_to_comment_out)

        # Parameter replacement
        if not should_comment:
            processed_line = new_line
            for key, value in params.items():
                if key not in ["name", "grid_size", "reynolds", "max_iterations", "tolerance", "pressure_tolerance", "verbose_mg"]:
                    processed_line = processed_line.replace(f"{key}={params[key]}", f"{key}={value}", 1)
            
            processed_line = processed_line.replace(f"nx, ny = 2**7-1, 2**7-1", f"nx, ny = {params['grid_size']}, {params['grid_size']}")
            processed_line = processed_line.replace(f"nx, ny = 2**6-1, 2**6-1", f"nx, ny = {params['grid_size']}, {params['grid_size']}")
            processed_line = processed_line.replace(f"reynolds = 100", f"reynolds = {params['reynolds']}")
            processed_line = processed_line.replace(f"max_iterations = 35000", f"max_iterations = {params['max_iterations']}")
            processed_line = processed_line.replace(f"max_iterations = 3000", f"max_iterations = {params['max_iterations']}")
            processed_line = processed_line.replace(f"tolerance = 1e-8", f"tolerance = {params['tolerance']}")
            processed_line = processed_line.replace("tolerance = 1e-3", f"tolerance = {params['tolerance']}")
            processed_line = processed_line.replace(f"pressure_tolerance = 1e-3", f"pressure_tolerance = {params['pressure_tolerance']}")
            processed_line = processed_line.replace(f"pressure_tolerance = 1e-7", f"pressure_tolerance = {params['pressure_tolerance']}")
            processed_line = processed_line.replace("pressure_tolerance = expected_disc_error", f"pressure_tolerance = {params['pressure_tolerance']}")

            if "results_dir = os.path.join(os.path.dirname(__file__), 'results')" in processed_line:
                processed_line = f"results_dir = r'{results_dir}'\n"
            elif "os.makedirs(results_dir, exist_ok=True)" in processed_line:
                processed_line = ""

            if "import tracemalloc" not in processed_line and "import time" not in processed_line:
                new_lines.append(processed_line)
        else:
            new_lines.append("#" + new_line)

        if in_plotting_block and ")" in new_line:
            in_plotting_block = False
    
    temp_script_path = f'temp_{params["name"]}_script.py'
    with open(temp_script_path, 'w') as f:
        f.writelines(new_lines)

    subprocess.run([sys.executable, temp_script_path], check=True)
    os.remove(temp_script_path) 