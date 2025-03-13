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
            }
        }
        
        # For tracking time spent in each part of the algorithm
        self._start_time = None
        self._section_start_time = None
    
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
        self.update_memory_usage()
    
    def end(self):
        """End profiling and calculate total time."""
        if self._start_time is not None:
            self.profiling_data['total_time'] = time.time() - self._start_time
            self.update_memory_usage()
    
    def start_section(self):
        """Start timing a section."""
        self._section_start_time = time.time()
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
            elapsed = time.time() - self._section_start_time
            if section_name in self.profiling_data:
                self.profiling_data[section_name] += elapsed
            else:
                self.profiling_data[section_name] = elapsed
            self._section_start_time = None
    
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
            f.write(f"Total Runtime: {self.profiling_data['total_time']:.4f} seconds\n")
            f.write(f"Total Iterations: {self.profiling_data['iterations']}\n")
            f.write(f"Average Time per Iteration: {avg_time_per_iteration:.4f} seconds\n")
            f.write(f"Peak Memory Usage: {peak_memory:.2f} MB\n\n")
            
            f.write("TIME BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            f.write(f"Momentum Equations: {self.profiling_data['momentum_solve_time']:.4f} seconds ({momentum_percent:.2f}%)\n")
            f.write(f"Pressure Equation: {self.profiling_data['pressure_solve_time']:.4f} seconds ({pressure_percent:.2f}%)\n")
            f.write(f"Velocity Update: {self.profiling_data['velocity_update_time']:.4f} seconds ({velocity_percent:.2f}%)\n")
            f.write(f"Boundary Conditions: {self.profiling_data['boundary_condition_time']:.4f} seconds ({boundary_percent:.2f}%)\n")
            f.write(f"Other Operations: {self.profiling_data['other_time']:.4f} seconds ({other_percent:.2f}%)\n\n")
            
            # Add residual history if available
            residual_history = self.profiling_data['convergence_info']['residual_history']
            if residual_history and len(residual_history) > 0:
                f.write("RESIDUAL HISTORY\n")
                f.write("-" * 80 + "\n")
                f.write("Iteration    Residual\n")
                
                # If there are too many iterations, sample the history
                if len(residual_history) > 100:
                    indices = np.linspace(0, len(residual_history)-1, 100, dtype=int)
                    for i, idx in enumerate(indices):
                        f.write(f"{idx+1:<12} {residual_history[idx]:.6e}\n")
                else:
                    for i, res in enumerate(residual_history):
                        f.write(f"{i+1:<12} {res:.6e}\n")
                f.write("\n")
            
            f.write("MEMORY USAGE OVER TIME\n")
            f.write("-" * 80 + "\n")
            f.write("Time (s)    RSS (MB)    VMS (MB)\n")
            for item in self.profiling_data['memory_usage']:
                f.write(f"{item['timestamp']:.2f}        {item['rss']:.2f}        {item['vms']:.2f}\n")
        
        return os.path.abspath(filename) 