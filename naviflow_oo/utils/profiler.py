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
import pandas as pd

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
        Save profiling data to an HDF5 file with a more natural structure.
        
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
                f"{self.algorithm_name}_Re{reynolds}_mesh{nx}x{ny}_profile.h5"
            )
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Create a nested dictionary for metadata with logical grouping
        metadata = {
            'simulation': {
                'algorithm': self.algorithm_name,
                'timestamp': self.profiling_data['timestamp'],
                'mesh_size': {
                    'x': self.mesh.get_dimensions()[0],
                    'y': self.mesh.get_dimensions()[1]
                },
                'reynolds_number': self.fluid.get_reynolds_number()
            },
            'performance': {
                'total_time': self.profiling_data['total_time'],
                'cpu_time': self.profiling_data['cpu_time'],
                'iterations': self.profiling_data['iterations'],
                'avg_time_per_iteration': self.profiling_data['total_time'] / self.profiling_data['iterations'] if self.profiling_data['iterations'] > 0 else 0
            },
            'convergence': {
                'tolerance': self.profiling_data['convergence_info']['tolerance'],
                'final_residual': self.profiling_data['convergence_info']['final_residual'],
                'converged': self.profiling_data['convergence_info']['converged']
            },
            'system': {
                'platform': self.profiling_data['system_info']['platform'],
                'processor': self.profiling_data['system_info']['processor'],
                'python_version': self.profiling_data['system_info']['python_version']
            }
        }
        
        # Add algorithm parameters if available
        if self.algorithm:
            algorithm_params = {}
            
            # Get relaxation factors
            if hasattr(self.algorithm, 'alpha_p'):
                algorithm_params['alpha_p'] = self.algorithm.alpha_p
            if hasattr(self.algorithm, 'alpha_u'):
                algorithm_params['alpha_u'] = self.algorithm.alpha_u
            
            if algorithm_params:
                metadata['algorithm'] = algorithm_params
            
            # Get pressure solver info
            if hasattr(self.algorithm, 'pressure_solver') and self.algorithm.pressure_solver:
                pressure_solver = self.algorithm.pressure_solver
                pressure_solver_params = {
                    'type': pressure_solver.__class__.__name__
                }
                
                # Get solver parameters
                if hasattr(pressure_solver, 'tolerance'):
                    pressure_solver_params['tolerance'] = pressure_solver.tolerance
                if hasattr(pressure_solver, 'max_iterations'):
                    pressure_solver_params['max_iterations'] = pressure_solver.max_iterations
                if hasattr(pressure_solver, 'matrix_free'):
                    pressure_solver_params['matrix_free'] = pressure_solver.matrix_free
                    
                # Get multigrid-specific parameters
                multigrid_params = {}
                if hasattr(pressure_solver, 'cycle_type'):
                    multigrid_params['cycle_type'] = pressure_solver.cycle_type
                if hasattr(pressure_solver, 'pre_smoothing'):
                    multigrid_params['pre_smoothing'] = pressure_solver.pre_smoothing
                if hasattr(pressure_solver, 'post_smoothing'):
                    multigrid_params['post_smoothing'] = pressure_solver.post_smoothing
                
                # Get smoother info
                smoother_params = {}
                if hasattr(pressure_solver, 'smoother'):
                    if hasattr(pressure_solver.smoother, '__class__'):
                        smoother_params['type'] = pressure_solver.smoother.__class__.__name__
                    else:
                        smoother_params['type'] = str(pressure_solver.smoother)
                if hasattr(pressure_solver, 'smoother_iterations'):
                    smoother_params['iterations'] = pressure_solver.smoother_iterations
                if hasattr(pressure_solver, 'smoother_omega'):
                    smoother_params['omega'] = pressure_solver.smoother_omega
                    
                if smoother_params:
                    pressure_solver_params['smoother'] = smoother_params
                if multigrid_params:
                    pressure_solver_params['multigrid'] = multigrid_params
                    
                metadata['pressure_solver'] = pressure_solver_params
            
            # Get momentum solver info
            if hasattr(self.algorithm, 'momentum_solver') and self.algorithm.momentum_solver:
                momentum_solver = self.algorithm.momentum_solver
                metadata['momentum_solver'] = {
                    'type': momentum_solver.__class__.__name__
                }
        
        # Create residual history DataFrame
        detailed = self.profiling_data['detailed_residuals']
        if detailed['iterations'] and len(detailed['iterations']) > 0:
            residual_data = {
                'iteration': detailed['iterations'],
                'wall_time': detailed['wall_times'],
                'cpu_time': detailed['cpu_times'],
                'total_residual': detailed['total_residuals'],
                'momentum_residual': detailed['momentum_residuals'],
                'pressure_residual': detailed['pressure_residuals']
            }
            if 'infinity_norm_errors' in detailed and len(detailed['infinity_norm_errors']) > 0:
                residual_data['infinity_norm_error'] = detailed['infinity_norm_errors']
            residual_df = pd.DataFrame(residual_data)
        else:
            residual_df = pd.DataFrame()

        # Save to HDF5 file
        import h5py
        
        with h5py.File(filename, 'w') as f:
            # Store metadata as attributes in groups
            for group_name, group_data in metadata.items():
                group = f.create_group(group_name)
                self._store_dict_to_h5_group(group, group_data)
            
            # Store residual history as a dataset
            if not residual_df.empty:
                residual_group = f.create_group('residual_history')
                for col in residual_df.columns:
                    residual_group.create_dataset(col, data=residual_df[col].values)
        
        return os.path.abspath(filename)

    def _store_dict_to_h5_group(self, group, data):
        """Helper method to recursively store a dictionary in an HDF5 group"""
        for key, value in data.items():
            if isinstance(value, dict):
                # Create a new group for nested dictionaries
                subgroup = group.create_group(key)
                self._store_dict_to_h5_group(subgroup, value)
            else:
                # Store simple values as attributes
                group.attrs[key] = value 