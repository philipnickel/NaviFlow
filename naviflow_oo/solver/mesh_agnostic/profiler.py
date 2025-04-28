"""
Performance profiling utilities for mesh-agnostic solvers.
"""

import time
import os
import h5py
import numpy as np

class SimpleProfiler:
    """
    A simplified profiler for mesh-agnostic solvers.
    Tracks execution time, convergence info, and solver statistics.
    """
    
    def __init__(self):
        """Initialize the profiler."""
        self.start_time = None
        self.end_time = None
        self.total_time = None
        self.sections = {}
        self.current_section = None
        self.current_section_start = None
        
        # Algorithm statistics
        self.iterations = None
        self.tolerance = None
        self.final_residual = None
        self.residual_history = None
        self.converged = None
        
        # Pressure solver statistics
        self.pressure_solver_name = None
        self.pressure_solver_inner_iterations = None
        self.pressure_solver_convergence_rate = None
        self.pressure_solver_specific = None
    
    def start(self):
        """Start the profiler timer."""
        self.start_time = time.time()
    
    def end(self):
        """End the profiler timer and calculate total time."""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
    
    def start_section(self, name=None):
        """
        Start timing a specific section.
        
        Parameters:
        -----------
        name : str, optional
            Name of the section. If None, a name will be auto-generated.
        """
        self.current_section = name or f"Section_{len(self.sections) + 1}"
        self.current_section_start = time.time()
    
    def end_section(self, name=None):
        """
        End timing for the current section.
        
        Parameters:
        -----------
        name : str, optional
            Name of the section to end. If provided, it must match the current section.
        """
        if self.current_section is None:
            return
        
        if name is not None and name != self.current_section:
            print(f"Warning: Attempted to end section {name} but current section is {self.current_section}")
            return
        
        section_time = time.time() - self.current_section_start
        self.sections[self.current_section] = section_time
        
        self.current_section = None
        self.current_section_start = None
    
    def set_iterations(self, iterations):
        """
        Set the number of iterations.
        
        Parameters:
        -----------
        iterations : int
            Number of iterations performed
        """
        self.iterations = iterations
    
    def set_convergence_info(self, tolerance, final_residual, residual_history, converged):
        """
        Set convergence information.
        
        Parameters:
        -----------
        tolerance : float
            Convergence tolerance
        final_residual : float
            Final residual value
        residual_history : list
            History of residual values
        converged : bool
            Whether the solver converged
        """
        self.tolerance = tolerance
        self.final_residual = final_residual
        self.residual_history = residual_history
        self.converged = converged
    
    def set_pressure_solver_info(self, solver_name, inner_iterations, convergence_rate, solver_specific):
        """
        Set pressure solver information.
        
        Parameters:
        -----------
        solver_name : str
            Name of the pressure solver
        inner_iterations : list
            History of inner iterations
        convergence_rate : float or None
            Convergence rate if available
        solver_specific : dict
            Additional solver-specific information
        """
        self.pressure_solver_name = solver_name
        self.pressure_solver_inner_iterations = inner_iterations
        self.pressure_solver_convergence_rate = convergence_rate
        self.pressure_solver_specific = solver_specific
    
    def save_to_file(self, filename):
        """
        Save profiling data to an HDF5 file.
        
        Parameters:
        -----------
        filename : str
            Path to the output file
            
        Returns:
        --------
        str
            Path to the saved file
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with h5py.File(filename, 'w') as f:
            # Save basic timing info
            timing = f.create_group('timing')
            timing.attrs['total_time'] = self.total_time
            
            # Save section timings
            for section_name, section_time in self.sections.items():
                timing.attrs[section_name] = section_time
            
            # Save algorithm info
            algorithm = f.create_group('algorithm')
            algorithm.attrs['iterations'] = self.iterations
            algorithm.attrs['tolerance'] = self.tolerance
            algorithm.attrs['final_residual'] = self.final_residual
            algorithm.attrs['converged'] = self.converged
            
            # Save residual history
            if self.residual_history is not None:
                algorithm.create_dataset('residual_history', data=np.array(self.residual_history))
            
            # Save pressure solver info
            if self.pressure_solver_name is not None:
                pressure = f.create_group('pressure_solver')
                pressure.attrs['name'] = self.pressure_solver_name
                
                if self.pressure_solver_inner_iterations is not None:
                    pressure.create_dataset('inner_iterations', data=np.array(self.pressure_solver_inner_iterations))
                
                if self.pressure_solver_convergence_rate is not None:
                    pressure.attrs['convergence_rate'] = self.pressure_solver_convergence_rate
                
                if self.pressure_solver_specific is not None:
                    specific = pressure.create_group('specific')
                    for key, value in self.pressure_solver_specific.items():
                        specific.attrs[key] = str(value)
        
        return filename 