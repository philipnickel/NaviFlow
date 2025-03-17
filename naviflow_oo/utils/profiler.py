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
    
    def __init__(self, algorithm_name, mesh, fluid, algorithm=None):
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
        algorithm : BaseAlgorithm, optional
            The algorithm instance being profiled
        """
        self.algorithm_name = algorithm_name
        self.mesh = mesh
        self.fluid = fluid
        self.algorithm = algorithm
        self.initialize()
    
    def initialize(self):
        """Initialize profiling data structures."""
        # Get detailed processor information
        processor_info = self._get_detailed_processor_info()
        
        self.profiling_data = {
            'total_time': 0.0,
            'cpu_time': 0.0,
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
    
    def end(self):
        """End profiling and calculate total time."""
        if self._start_time is not None:
            self.profiling_data['total_time'] = time.time() - self._start_time
            self.profiling_data['cpu_time'] = time.process_time() - self._start_cpu_time
    
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
            
       
        # Store the data
        self.profiling_data['detailed_residuals']['iterations'].append(iteration)
        self.profiling_data['detailed_residuals']['wall_times'].append(wall_time)
        self.profiling_data['detailed_residuals']['cpu_times'].append(cpu_time)
        self.profiling_data['detailed_residuals']['total_residuals'].append(total_residual)
        self.profiling_data['detailed_residuals']['momentum_residuals'].append(momentum_residual)
        self.profiling_data['detailed_residuals']['pressure_residuals'].append(pressure_residual)
        
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
            
        
        # Write the profiling data to the file
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"NAVIFLOW PROFILING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Algorithm: {self.algorithm_name}\n")
            f.write(f"Timestamp: {self.profiling_data['timestamp']}\n")
            f.write(f"Mesh Size: {self.mesh.get_dimensions()[0]}x{self.mesh.get_dimensions()[1]}\n")
            f.write(f"Reynolds Number: {self.fluid.get_reynolds_number()}\n\n")
            
            # Add algorithm and solver parameters section
            f.write("ALGORITHM AND SOLVER PARAMETERS\n")
            f.write("-" * 80 + "\n")
            
            # Add algorithm-specific parameters
            if hasattr(self, 'algorithm') and self.algorithm is not None:
                if hasattr(self.algorithm, 'alpha_p'):
                    f.write(f"Pressure Relaxation Factor (alpha_p): {self.algorithm.alpha_p:.3f}\n")
                if hasattr(self.algorithm, 'alpha_u'):
                    f.write(f"Velocity Relaxation Factor (alpha_u): {self.algorithm.alpha_u:.3f}\n")
            
            # Add pressure solver parameters
            if hasattr(self, 'algorithm') and self.algorithm is not None and self.algorithm.pressure_solver is not None:
                pressure_solver = self.algorithm.pressure_solver
                f.write(f"Pressure Solver: {pressure_solver.__class__.__name__}\n")
                
                # Add solver-specific parameters
                if hasattr(pressure_solver, 'tolerance'):
                    f.write(f"Pressure Solver Tolerance: {pressure_solver.tolerance:.6e}\n")
                if hasattr(pressure_solver, 'max_iterations'):
                    f.write(f"Pressure Solver Max Iterations: {pressure_solver.max_iterations}\n")
                if hasattr(pressure_solver, 'matrix_free'):
                    f.write(f"Matrix-Free Mode: {pressure_solver.matrix_free}\n")
                if hasattr(pressure_solver, 'smoother'):
                    f.write(f"Smoother Type: {pressure_solver.smoother}\n")
                if hasattr(pressure_solver, 'cycle_type'):
                    f.write(f"Multigrid Cycle Type: {pressure_solver.cycle_type}\n")
            
            # Add momentum solver parameters
            if hasattr(self, 'algorithm') and self.algorithm is not None and self.algorithm.momentum_solver is not None:
                momentum_solver = self.algorithm.momentum_solver
                f.write(f"Momentum Solver: {momentum_solver.__class__.__name__}\n")
            
            f.write("\n")
            
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Platform: {self.profiling_data['system_info']['platform']}\n")
            f.write(f"Processor: {self.profiling_data['system_info']['processor']}\n")
            f.write(f"Python Version: {self.profiling_data['system_info']['python_version']}\n")
            
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
                    f.write(f"{'Iter':>5} {'Wall Time':>12} {'CPU Time':>12} {'Total Res':>12} {'Momentum Res':>12} {'Pressure Res':>12} {'Inf Norm Err':>12}\n")
                else:
                    f.write(f"{'Iter':>5} {'Wall Time':>12} {'CPU Time':>12} {'Total Res':>12} {'Momentum Res':>12} {'Pressure Res':>12}\n")
                
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
                   
                    f.write(line + "\n")
                f.write("\n")
            
        
        
        return os.path.abspath(filename) 