"""
Profiling utilities for naviflow_oo.

This module provides classes and functions for profiling the performance of
algorithms and solvers in the NaviFlow framework.
"""

import time
import os
import psutil
import platform
import subprocess
from datetime import datetime
import numpy as np

class Profiler:
    """
    Profiler class for tracking performance metrics.
    
    This class provides methods for tracking execution time, memory usage,
    and other performance metrics for CFD algorithms and solvers.
    """
    
    def __init__(self, algorithm_name, mesh, fluid):
        """
        Initialize the profiler.
        
        Parameters:
        -----------
        algorithm_name : str
            Name of the algorithm being profiled
        mesh : StructuredMesh
            The computational mesh
        fluid : FluidProperties
            Fluid properties
        """
        self.algorithm_name = algorithm_name
        self.mesh = mesh
        self.fluid = fluid
        self.initialize()
    
    def initialize(self):
        """Initialize profiling data structures."""
        # Get detailed processor information
        processor_info = self._get_detailed_processor_info()
        
        self.profiling_data = {
            'total_time': 0.0,
            'cpu_time': 0.0,
            'momentum_solve_time': 0.0,
            'pressure_solve_time': 0.0,
            'velocity_update_time': 0.0,
            'boundary_condition_time': 0.0,
            'other_time': 0.0,
            'iterations': 0,
            'memory_usage': [],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'platform': platform.platform(),
                'processor': processor_info,
                'python_version': platform.python_version(),
                'memory_total': psutil.virtual_memory().total / (1024 ** 3)  # GB
            },
            'convergence_info': {
                'tolerance': None,
                'final_residual': None,
                'converged': False,
                'residual_history': []
            },
            'detailed_residuals': {
                'iterations': [],
                'wall_times': [],
                'cpu_times': [],
                'total_residuals': [],
                'momentum_residuals': [],
                'pressure_residuals': [],
                'rss_values': [],
                'vms_values': [],
                'infinity_norm_errors': []  # Added for tracking infinity norm errors
            },
            'pressure_solver_info': {
                'name': None,
                'total_inner_iterations': 0,
                'avg_inner_iterations_per_outer': 0.0,
                'max_inner_iterations': 0,
                'min_inner_iterations': float('inf'),
                'inner_iterations_history': [],
                'convergence_rate': None,
                'solver_specific': {}
            }
        }
        
        # For tracking time spent in each part of the algorithm
        self._start_time = None
        self._start_cpu_time = None
        self._section_start_time = None
        self._section_start_cpu_time = None
    
    def _get_detailed_processor_info(self):
        """Get detailed processor information based on the platform."""
        basic_info = platform.processor()
        
        # For macOS (Darwin), try to get more detailed info
        if platform.system() == 'Darwin':
            try:
                # Use sysctl to get processor info on macOS
                output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
                if output:
                    return output
                
                # Try system_profiler for Apple Silicon
                output = subprocess.check_output(['system_profiler', 'SPHardwareDataType']).decode('utf-8')
                for line in output.split('\n'):
                    if 'Chip' in line or 'Processor' in line:
                        return line.split(':')[1].strip()
            except (subprocess.SubprocessError, IndexError, FileNotFoundError):
                pass
        
        # For Linux, try to get more detailed info from /proc/cpuinfo
        elif platform.system() == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':')[1].strip()
            except (IOError, IndexError):
                pass
        
        # For Windows, try to use WMI
        elif platform.system() == 'Windows':
            try:
                import wmi
                c = wmi.WMI()
                for processor in c.Win32_Processor():
                    return processor.Name
            except (ImportError, AttributeError):
                pass
        
        return basic_info
    
    def start(self):
        """Start profiling."""
        self._start_time = time.time()
        self._start_cpu_time = time.process_time()
        self.update_memory_usage()
    
    def end(self):
        """End profiling and calculate total time."""
        if self._start_time is not None:
            self.profiling_data['total_time'] = time.time() - self._start_time
            self.profiling_data['cpu_time'] = time.process_time() - self._start_cpu_time
            self.update_memory_usage()
    
    def start_section(self):
        """Start timing a section."""
        self._section_start_time = time.time()
        self._section_start_cpu_time = time.process_time()
        return self._section_start_time
    
    def end_section(self, section_name):
        """
        End timing a section and add to the appropriate counter.
        
        Parameters:
        -----------
        section_name : str
            Name of the section being timed
        """
        if self._section_start_time is not None:
            elapsed_wall = time.time() - self._section_start_time
            elapsed_cpu = time.process_time() - self._section_start_cpu_time
            
            # Store wall time
            if section_name in self.profiling_data:
                self.profiling_data[section_name] += elapsed_wall
            else:
                self.profiling_data[section_name] = elapsed_wall
                
            # Store CPU time
            cpu_section_name = f"{section_name}_cpu"
            if cpu_section_name in self.profiling_data:
                self.profiling_data[cpu_section_name] += elapsed_cpu
            else:
                self.profiling_data[cpu_section_name] = elapsed_cpu
                
            self._section_start_time = None
            self._section_start_cpu_time = None
    
    def update_memory_usage(self):
        """Update memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.profiling_data['memory_usage'].append({
            'timestamp': time.time() - self._start_time if self._start_time else 0,
            'rss': memory_info.rss / (1024 ** 2),  # MB
            'vms': memory_info.vms / (1024 ** 2)   # MB
        })
    
    def set_iterations(self, iterations):
        """Set the number of iterations performed."""
        self.profiling_data['iterations'] = iterations
    
    def set_convergence_info(self, tolerance, final_residual, residual_history, converged=None):
        """
        Set convergence information.
        
        Parameters:
        -----------
        tolerance : float
            Convergence tolerance used
        final_residual : float
            Final residual value
        residual_history : list
            History of residuals during iterations
        converged : bool, optional
            Whether the simulation converged (if None, determined from final_residual and tolerance)
        """
        if converged is None:
            converged = final_residual <= tolerance
            
        self.profiling_data['convergence_info']['tolerance'] = tolerance
        self.profiling_data['convergence_info']['final_residual'] = final_residual
        self.profiling_data['convergence_info']['residual_history'] = residual_history
        self.profiling_data['convergence_info']['converged'] = converged
    
    def add_residual_data(self, iteration, total_residual, momentum_residual, pressure_residual, infinity_norm_error=None):
        """
        Add detailed residual data for the current iteration.
        
        Parameters:
        -----------
        iteration : int
            Current iteration number
        total_residual : float
            Total residual from the outer loop
        momentum_residual : float
            Residual from the momentum equations
        pressure_residual : float
            Residual from the pressure equation
        infinity_norm_error : float, optional
            Infinity norm error against benchmark data
        """
        # Get current times
        if self._start_time is not None:
            wall_time = time.time() - self._start_time
            cpu_time = time.process_time() - self._start_cpu_time
        else:
            wall_time = 0.0
            cpu_time = 0.0
            
        # Get current memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        rss = memory_info.rss / (1024 ** 2)  # MB
        vms = memory_info.vms / (1024 ** 2)  # MB
        
        # Store the data
        self.profiling_data['detailed_residuals']['iterations'].append(iteration)
        self.profiling_data['detailed_residuals']['wall_times'].append(wall_time)
        self.profiling_data['detailed_residuals']['cpu_times'].append(cpu_time)
        self.profiling_data['detailed_residuals']['total_residuals'].append(total_residual)
        self.profiling_data['detailed_residuals']['momentum_residuals'].append(momentum_residual)
        self.profiling_data['detailed_residuals']['pressure_residuals'].append(pressure_residual)
        self.profiling_data['detailed_residuals']['rss_values'].append(rss)
        self.profiling_data['detailed_residuals']['vms_values'].append(vms)
        
        # Always append infinity norm error (None if not provided)
        # This ensures alignment with iteration count
        self.profiling_data['detailed_residuals']['infinity_norm_errors'].append(infinity_norm_error)
    
    def set_pressure_solver_info(self, solver_name, inner_iterations=None, convergence_rate=None, solver_specific=None):
        """
        Set information about the pressure solver performance.
        
        Parameters:
        -----------
        solver_name : str
            Name of the pressure solver used
        inner_iterations : list, optional
            List of inner iterations performed in each outer iteration
        convergence_rate : float, optional
            Average convergence rate of the pressure solver
        solver_specific : dict, optional
            Solver-specific metrics (e.g., grid levels for multigrid)
        """
        # Initialize pressure solver info if it doesn't exist
        if 'pressure_solver_info' not in self.profiling_data:
            self.profiling_data['pressure_solver_info'] = {
                'name': None,
                'total_inner_iterations': 0,
                'avg_inner_iterations_per_outer': 0.0,
                'max_inner_iterations': 0,
                'min_inner_iterations': float('inf'),
                'inner_iterations_history': [],
                'convergence_rate': None,
                'solver_specific': {}
            }
        
        self.profiling_data['pressure_solver_info']['name'] = solver_name
        
        if inner_iterations is not None:
            self.profiling_data['pressure_solver_info']['inner_iterations_history'] = inner_iterations
            self.profiling_data['pressure_solver_info']['total_inner_iterations'] = sum(inner_iterations)
            
            if len(inner_iterations) > 0:
                self.profiling_data['pressure_solver_info']['avg_inner_iterations_per_outer'] = sum(inner_iterations) / len(inner_iterations)
                self.profiling_data['pressure_solver_info']['max_inner_iterations'] = max(inner_iterations)
                self.profiling_data['pressure_solver_info']['min_inner_iterations'] = min(inner_iterations)
        
        if convergence_rate is not None:
            self.profiling_data['pressure_solver_info']['convergence_rate'] = convergence_rate
            
        if solver_specific is not None:
            self.profiling_data['pressure_solver_info']['solver_specific'] = solver_specific
    
    def save(self, filename=None, profile_dir='results/profiles'):
        """
        Save profiling data to a file.
        
        Parameters:
        -----------
        filename : str, optional
            Name of the file to save the data to. If None, a default name is generated.
        profile_dir : str, optional
            Directory to save profiling data
        
        Returns:
        --------
        str
            Path to the saved file
        """
        if filename is None:
            # Generate a default filename based on the algorithm name and timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            nx, ny = self.mesh.get_dimensions()
            reynolds = int(self.fluid.get_reynolds_number())
            filename = os.path.join(
                profile_dir, 
                f"{self.algorithm_name}_Re{reynolds}_mesh{nx}x{ny}_profile.txt"
            )
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Calculate derived metrics
        if self.profiling_data['iterations'] > 0:
            avg_time_per_iteration = self.profiling_data['total_time'] / self.profiling_data['iterations']
        else:
            avg_time_per_iteration = 0
            
        # Calculate percentages
        total_time = self.profiling_data['total_time']
        if total_time > 0:
            momentum_percent = (self.profiling_data['momentum_solve_time'] / total_time) * 100
            pressure_percent = (self.profiling_data['pressure_solve_time'] / total_time) * 100
            velocity_percent = (self.profiling_data['velocity_update_time'] / total_time) * 100
            boundary_percent = (self.profiling_data['boundary_condition_time'] / total_time) * 100
            other_percent = (self.profiling_data['other_time'] / total_time) * 100
        else:
            momentum_percent = pressure_percent = velocity_percent = boundary_percent = other_percent = 0
        
        # Get peak memory usage
        if self.profiling_data['memory_usage']:
            peak_memory = max(item['rss'] for item in self.profiling_data['memory_usage'])
        else:
            peak_memory = 0
        
        # Write the profiling data to the file
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"NAVIFLOW PROFILING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Algorithm: {self.algorithm_name}\n")
            f.write(f"Timestamp: {self.profiling_data['timestamp']}\n")
            f.write(f"Mesh Size: {self.mesh.get_dimensions()[0]}x{self.mesh.get_dimensions()[1]}\n")
            f.write(f"Reynolds Number: {self.fluid.get_reynolds_number()}\n\n")
            
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Platform: {self.profiling_data['system_info']['platform']}\n")
            f.write(f"Processor: {self.profiling_data['system_info']['processor']}\n")
            f.write(f"Python Version: {self.profiling_data['system_info']['python_version']}\n")
            f.write(f"Total System Memory: {self.profiling_data['system_info']['memory_total']:.2f} GB\n\n")
            
            f.write("CONVERGENCE INFORMATION\n")
            f.write("-" * 80 + "\n")
            tolerance = self.profiling_data['convergence_info']['tolerance']
            final_residual = self.profiling_data['convergence_info']['final_residual']
            converged = self.profiling_data['convergence_info']['converged']
            
            f.write(f"Convergence Tolerance: {tolerance:.6e}\n")
            f.write(f"Final Residual: {final_residual:.6e}\n")
            
            if converged:
                f.write(f"Status: Converged after {self.profiling_data['iterations']} iterations\n\n")
            else:
                f.write(f"Status: Did not converge (reached maximum iterations)\n\n")
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Wall Time: {self.profiling_data['total_time']:.4f} seconds\n")
            f.write(f"Total CPU Time: {self.profiling_data['cpu_time']:.4f} seconds\n")
            f.write(f"Total Iterations: {self.profiling_data['iterations']}\n")
            f.write(f"Average Wall Time per Iteration: {avg_time_per_iteration:.4f} seconds\n")
            f.write(f"Peak Memory Usage: {peak_memory:.2f} MB\n\n")
            
            f.write("TIME BREAKDOWN (WALL TIME)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Momentum Equations: {self.profiling_data['momentum_solve_time']:.4f} seconds ({momentum_percent:.2f}%)\n")
            f.write(f"Pressure Equation: {self.profiling_data['pressure_solve_time']:.4f} seconds ({pressure_percent:.2f}%)\n")
            f.write(f"Velocity Update: {self.profiling_data['velocity_update_time']:.4f} seconds ({velocity_percent:.2f}%)\n")
            f.write(f"Boundary Conditions: {self.profiling_data['boundary_condition_time']:.4f} seconds ({boundary_percent:.2f}%)\n")
            f.write(f"Other Operations: {self.profiling_data['other_time']:.4f} seconds ({other_percent:.2f}%)\n\n")
            
            # Add CPU time breakdown if available
            if 'momentum_solve_time_cpu' in self.profiling_data:
                f.write("TIME BREAKDOWN (CPU TIME)\n")
                f.write("-" * 80 + "\n")
                
                # Calculate CPU time percentages
                cpu_total = self.profiling_data['cpu_time']
                if cpu_total > 0:
                    momentum_cpu = self.profiling_data.get('momentum_solve_time_cpu', 0.0)
                    pressure_cpu = self.profiling_data.get('pressure_solve_time_cpu', 0.0)
                    velocity_cpu = self.profiling_data.get('velocity_update_time_cpu', 0.0)
                    boundary_cpu = self.profiling_data.get('boundary_condition_time_cpu', 0.0)
                    other_cpu = self.profiling_data.get('other_time_cpu', 0.0)
                    
                    momentum_cpu_percent = (momentum_cpu / cpu_total) * 100
                    pressure_cpu_percent = (pressure_cpu / cpu_total) * 100
                    velocity_cpu_percent = (velocity_cpu / cpu_total) * 100
                    boundary_cpu_percent = (boundary_cpu / cpu_total) * 100
                    other_cpu_percent = (other_cpu / cpu_total) * 100
                    
                    f.write(f"Momentum Equations: {momentum_cpu:.4f} seconds ({momentum_cpu_percent:.2f}%)\n")
                    f.write(f"Pressure Equation: {pressure_cpu:.4f} seconds ({pressure_cpu_percent:.2f}%)\n")
                    f.write(f"Velocity Update: {velocity_cpu:.4f} seconds ({velocity_cpu_percent:.2f}%)\n")
                    f.write(f"Boundary Conditions: {boundary_cpu:.4f} seconds ({boundary_cpu_percent:.2f}%)\n")
                    f.write(f"Other Operations: {other_cpu:.4f} seconds ({other_cpu_percent:.2f}%)\n\n")
            
            # Add pressure solver information if available
            if 'pressure_solver_info' in self.profiling_data:
                pressure_info = self.profiling_data['pressure_solver_info']
                if pressure_info.get('name') is not None:
                    f.write("PRESSURE SOLVER INFORMATION\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Solver: {pressure_info['name']}\n")
                    
                    if pressure_info.get('total_inner_iterations', 0) > 0:
                        f.write(f"Total Inner Iterations: {pressure_info['total_inner_iterations']}\n")
                        f.write(f"Average Inner Iterations per Outer: {pressure_info['avg_inner_iterations_per_outer']:.2f}\n")
                        f.write(f"Maximum Inner Iterations: {pressure_info['max_inner_iterations']}\n")
                        f.write(f"Minimum Inner Iterations: {pressure_info['min_inner_iterations']}\n")
                    
                    if pressure_info.get('convergence_rate') is not None:
                        f.write(f"Average Convergence Rate: {pressure_info['convergence_rate']:.6f}\n")
                    
                    # Add solver-specific information
                    if pressure_info.get('solver_specific'):
                        f.write("\nSolver-Specific Information:\n")
                        for key, value in pressure_info['solver_specific'].items():
                            if isinstance(value, float):
                                f.write(f"  {key}: {value:.6f}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                    f.write("\n")
            
            # Add detailed residual history if available
            detailed = self.profiling_data['detailed_residuals']
            if detailed['iterations'] and len(detailed['iterations']) > 0:
                f.write("DETAILED RESIDUAL HISTORY\n")
                f.write("-" * 80 + "\n")
                
                # Check if infinity norm errors are available
                has_inf_norm = 'infinity_norm_errors' in detailed and len(detailed['infinity_norm_errors']) > 0
                
                if has_inf_norm:
                    f.write(f"{'Iter':>5} {'Wall Time':>12} {'CPU Time':>12} {'Total Res':>12} {'Momentum Res':>12} {'Pressure Res':>12} {'Inf Norm Err':>12} {'RSS (MB)':>12} {'VMS (MB)':>12}\n")
                else:
                    f.write(f"{'Iter':>5} {'Wall Time':>12} {'CPU Time':>12} {'Total Res':>12} {'Momentum Res':>12} {'Pressure Res':>12} {'RSS (MB)':>12} {'VMS (MB)':>12}\n")
                
                for i in range(len(detailed['iterations'])):
                    # Prepare the common part of the line
                    line = f"{detailed['iterations'][i]:5d} " \
                           f"{detailed['wall_times'][i]:12.4f} " \
                           f"{detailed['cpu_times'][i]:12.4f} " \
                           f"{detailed['total_residuals'][i]:12.6e} " \
                           f"{detailed['momentum_residuals'][i]:12.6e} " \
                           f"{detailed['pressure_residuals'][i]:12.6e} "
                    
                    # Add infinity norm error if available
                    if has_inf_norm:
                        inf_norm_value = detailed['infinity_norm_errors'][i] if i < len(detailed['infinity_norm_errors']) else None
                        if inf_norm_value is not None:
                            line += f"{inf_norm_value:12.6e} "
                        else:
                            line += f"{'N/A':>12} "
                    
                    # Add memory usage
                    line += f"{detailed['rss_values'][i]:12.2f} " \
                            f"{detailed['vms_values'][i]:12.2f}"
                    
                    f.write(line + "\n")
                f.write("\n")
            
            # Add simple residual history if available but no detailed history
            elif residual_history and len(residual_history) > 0:
                f.write("RESIDUAL HISTORY\n")
                f.write("-" * 80 + "\n")
                f.write("Iteration    Residual\n")
        
        # Also save detailed residual history as CSV for data processing
        if detailed['iterations'] and len(detailed['iterations']) > 0:
            # Create CSV filename by replacing .txt with .csv
            csv_filename = os.path.splitext(filename)[0] + '_residuals.csv'
            
            with open(csv_filename, 'w') as f:
                # Write header
                header = "Iteration,Wall_Time,CPU_Time,Total_Residual,Momentum_Residual,Pressure_Residual"
                if has_inf_norm:
                    header += ",Infinity_Norm_Error"
                header += ",RSS_MB,VMS_MB"
                f.write(header + "\n")
                
                # Write data rows
                for i in range(len(detailed['iterations'])):
                    row = f"{detailed['iterations'][i]},{detailed['wall_times'][i]},{detailed['cpu_times'][i]}"
                    row += f",{detailed['total_residuals'][i]},{detailed['momentum_residuals'][i]},{detailed['pressure_residuals'][i]}"
                    
                    if has_inf_norm:
                        inf_norm_value = detailed['infinity_norm_errors'][i] if i < len(detailed['infinity_norm_errors']) else None
                        if inf_norm_value is not None:
                            row += f",{inf_norm_value}"
                        else:
                            row += ","  # Empty value for CSV
                    
                    row += f",{detailed['rss_values'][i]},{detailed['vms_values'][i]}"
                    f.write(row + "\n")
            
            print(f"Detailed residual history also saved as CSV: {csv_filename}")
        
        return os.path.abspath(filename) 