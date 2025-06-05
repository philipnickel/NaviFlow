"""
Class to store and process simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from .visualization import plot_velocity_field, plot_combined_results_matrix
from .validation import BenchmarkData
from .validation.cavity_flow import calculate_infinity_norm_error, calculate_l2_norm_error

class SimulationResult:
    """
    Store and analyze simulation results.
    """
    def __init__(self, u, v, p, mesh, iterations=0, residuals=None, divergence=None, reynolds=None,
                 momentum_residuals=None, pressure_residuals=None,
                 u_residual_field=None, v_residual_field=None, p_residual_field=None):
        """
        Initialize the simulation result.
        
        Parameters:
        -----------
        u, v : ndarray
            Velocity fields
        p : ndarray
            Pressure field
        mesh : StructuredMesh
            The computational mesh
        iterations : int
            Number of iterations performed
        residuals : list, optional
            Convergence history
        divergence : ndarray, optional
            Divergence field
        reynolds : float, optional
            Reynolds number
        momentum_residuals : list, optional
            Momentum convergence history
        pressure_residuals : list, optional
            Pressure convergence history
        u_residual_field : ndarray, optional
            Final algebraic u-momentum residual field
        v_residual_field : ndarray, optional
            Final algebraic v-momentum residual field
        p_residual_field : ndarray, optional
            Final algebraic pressure residual field
        """
        self.u = u
        self.v = v
        self.p = p
        self.mesh = mesh
        self.iterations = iterations
        self.residuals = residuals or []
        self.momentum_residuals = momentum_residuals or []
        self.pressure_residuals = pressure_residuals or []
        self.divergence = divergence
        self.reynolds = reynolds
        self.infinity_norm_error = None
        self.l2_norm_error = None
        # Store the final residual fields
        self.u_residual_field = u_residual_field
        self.v_residual_field = v_residual_field
        self.p_residual_field = p_residual_field
        # Dictionary to store custom history data
        self._custom_histories = {}
    
    def add_history(self, name, data):
        """
        Add a custom history data series to this simulation result.
        
        Parameters:
        -----------
        name : str
            Name of the history data (e.g., 'u_momentum_relaxed')
        data : list or ndarray
            The history data to store
        """
        self._custom_histories[name] = data
    
    def get_history(self, name):
        """
        Get a custom history data series by name.
        
        Parameters:
        -----------
        name : str
            Name of the history data to retrieve
            
        Returns:
        --------
        list or ndarray
            The requested history data, or None if not found
        """
        return self._custom_histories.get(name, None)
    
    def plot_velocity_field(self, title=None, filename=None, show=True):
        """
        Plot the velocity field.
        
        Parameters:
        -----------
        title : str, optional
            Plot title
        filename : str, optional
            If provided, saves the figure to this filename
        show : bool, optional
            Whether to display the plot
        """
        plot_velocity_field(
            self.u, self.v, self.mesh.x, self.mesh.y, 
            title=title, filename=filename, show=show
        )
    
    def plot_combined_results(self, title=None, filename=None, show=True):
        """
        Plot combined results (velocity, pressure, streamlines).
        
        Parameters:
        -----------
        title : str, optional
            Plot title
        filename : str, optional
            If provided, saves the figure to this filename
        show : bool, optional
            Whether to display the plot
        """
        # Get mesh dimensions and create coordinates
        nx, ny = self.mesh.get_dimensions()
        dx, dy = self.mesh.get_cell_sizes()
        
        # Create x and y coordinates
        x = np.linspace(dx/2, 1-dx/2, nx)
        y = np.linspace(dy/2, 1-dy/2, ny)
        
        # Use the stored Reynolds number
        Re = self.reynolds
        
        # Extract Reynolds number from title if not available
        if Re is None and title is not None:
            import re
            match = re.search(r'Re=(\d+)', title)
            if match:
                Re = int(match.group(1))
                print(f"Using Reynolds number extracted from title: {Re}")
        
        # Call the visualization function with all necessary parameters
        plot_combined_results_matrix(
            self.u, self.v, self.p, x, y, 
            title=title, filename=filename, show=show, Re=Re
        )
    
    def get_max_divergence(self):
        """
        Return the maximum absolute divergence in the interior of the domain.
        
        This method calculates the divergence of the velocity field and returns
        the maximum absolute value, excluding boundary cells where divergence
        calculations may be affected by boundary conditions.
        
        Returns:
        --------
        float
            Maximum absolute divergence
        """
        if self.divergence is None:
            from .validation.cavity_flow import calculate_divergence
            dx, dy = self.mesh.get_cell_sizes()
            self.divergence = calculate_divergence(self.u, self.v, dx, dy)
        
        # Get dimensions
        nx, ny = self.mesh.get_dimensions()
        
        # Create a mask to exclude boundary cells (one cell in from each boundary)
        mask = np.ones_like(self.divergence, dtype=bool)
        mask[0, :] = False  # Left boundary
        mask[-1, :] = False  # Right boundary
        mask[:, 0] = False  # Bottom boundary
        mask[:, -1] = False  # Top boundary
        
        # Calculate maximum divergence in the interior
        interior_divergence = self.divergence[mask]
        max_div = np.max(np.abs(interior_divergence))
        
        return max_div
    
    def validate_against_benchmark(self, Re=100):
        """
        Validate results against benchmark data.
        
        Parameters:
        -----------
        Re : int
            Reynolds number
            
        Returns:
        --------
        bool
            True if validation is successful
        """
        # Get benchmark data
        benchmark = BenchmarkData(case_type='cavity', reynolds=Re)
        
        # Get mesh dimensions
        nx, ny = self.mesh.get_dimensions()
        dx, dy = self.mesh.get_cell_sizes()
        
        # Create x and y coordinates
        x = np.linspace(dx/2, 1-dx/2, nx)
        y = np.linspace(dy/2, 1-dy/2, ny)
        
        # Extract centerline data from simulation
        u_centerline = self.u[nx//2, :]  # u along vertical centerline
        v_centerline = self.v[:, ny//2]  # v along horizontal centerline
        
        # Get benchmark data
        y_benchmark, u_benchmark = benchmark.get_centerline_data('u')
        x_benchmark, v_benchmark = benchmark.get_centerline_data('v')
        
        # Interpolate simulation data to benchmark coordinates
        from scipy.interpolate import interp1d
        u_interp = interp1d(y, u_centerline, kind='cubic', fill_value="extrapolate")
        v_interp = interp1d(x, v_centerline, kind='cubic', fill_value="extrapolate")
        
        u_sim_at_benchmark = u_interp(y_benchmark)
        v_sim_at_benchmark = v_interp(x_benchmark)
        
        # Calculate error
        u_error = np.abs(u_sim_at_benchmark - u_benchmark)
        v_error = np.abs(v_sim_at_benchmark - v_benchmark)
        
        max_u_error = np.max(u_error)
        max_v_error = np.max(v_error)
        
        print(f"Maximum u-velocity error: {max_u_error:.6f}")
        print(f"Maximum v-velocity error: {max_v_error:.6f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(u_benchmark, y_benchmark, 'ro', label='Benchmark')
        plt.plot(u_sim_at_benchmark, y_benchmark, 'b-', label='Simulation')
        plt.xlabel('u-velocity')
        plt.ylabel('y-coordinate')
        plt.title('u-velocity along vertical centerline')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(x_benchmark, v_benchmark, 'ro', label='Benchmark')
        plt.plot(x_benchmark, v_sim_at_benchmark, 'b-', label='Simulation')
        plt.xlabel('x-coordinate')
        plt.ylabel('v-velocity')
        plt.title('v-velocity along horizontal centerline')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'validation_Re{Re}.pdf')
        plt.show()
        
        # Consider validation successful if error is below threshold
        threshold = 0.1  # 10% error threshold
        return max_u_error < threshold and max_v_error < threshold 
    
    def calculate_infinity_norm_error(self):
        """
        Calculate the infinity norm error against Ghia data.
        
        Returns:
        --------
        float
            Infinity norm error
        """
        if self.reynolds is None:
            raise ValueError("Reynolds number must be set to calculate infinity norm error")
        
        self.infinity_norm_error = calculate_infinity_norm_error(self.u, self.v, self.mesh, self.reynolds)
        return self.infinity_norm_error 

    def calculate_l2_norm_error(self):
        """
        Calculate the L2 norm error against Ghia data.
        
        Returns:
        --------
        float
            L2 norm error
        """
        if self.reynolds is None:
            raise ValueError("Reynolds number must be set to calculate L2 norm error")
        
        self.l2_norm_error = calculate_l2_norm_error(self.u, self.v, self.mesh, self.reynolds)
        return self.l2_norm_error

    def save_solution(self, filename):
        """
        Save the solution fields to a NumPy .npz file.
        
        Parameters:
        -----------
        filename : str
            Path to save the solution file
        """
        np.savez(
            filename,
            u=self.u,
            v=self.v,
            p=self.p,
            x=self.mesh.x,
            y=self.mesh.y,
            reynolds=self.reynolds
        )
        return filename 
        
    def plot_residuals(self, title=None, filename=None, show=True):
        """
        Plot the residual history.
        
        Parameters:
        -----------
        title : str, optional
            Plot title
        filename : str, optional
            If provided, saves the figure to this filename
        show : bool, optional
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if not self.residuals:
            raise ValueError("No residuals available to plot")
        
        # Apply stall detection and trimming
        residual_dict = {
            'total': self.residuals,
            'momentum': self.momentum_residuals if self.momentum_residuals else [],
            'pressure': self.pressure_residuals if self.pressure_residuals else []
        }
        
        # Remove empty arrays
        residual_dict = {k: v for k, v in residual_dict.items() if v}
        
        trimmed_residuals, stall_info = self._trim_stalled_residuals(residual_dict, keep_stalled_iterations=500)
        
        plt.figure(figsize=(10, 6))
        
        # Plot total residuals
        total_residuals = trimmed_residuals['total']
        iterations = range(1, len(total_residuals) + 1)
        plt.semilogy(iterations, total_residuals, 'b-', linewidth=2, label='Total Residual')
        
        # Plot component residuals if available
        if 'momentum' in trimmed_residuals:
            momentum_residuals = trimmed_residuals['momentum']
            if len(momentum_residuals) == len(total_residuals):
                plt.semilogy(iterations, momentum_residuals, 'r--', linewidth=1.5, label='Momentum Residual')
            
        if 'pressure' in trimmed_residuals:
            pressure_residuals = trimmed_residuals['pressure']
            if len(pressure_residuals) == len(total_residuals):
                plt.semilogy(iterations, pressure_residuals, 'g-.', linewidth=1.5, label='Pressure Residual')
        
        # Add stall indicator if residuals were trimmed
        if stall_info['stalled']:
            stall_line_pos = stall_info['earliest_stall']
            if 0 <= stall_line_pos < len(total_residuals):
                plt.axvline(x=stall_line_pos, color='red', linestyle='--', alpha=0.7, 
                           label=f'Stall detected (iter {stall_info["earliest_stall"]})')
        
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        
        # Update title to show trimming information
        if title:
            plot_title = title
        else:
            plot_title = f'Residual History (Re={self.reynolds})' if self.reynolds else 'Residual History'
        
        if stall_info['stalled']:
            plot_title += f" (Trimmed: {stall_info['trimmed_length']}/{stall_info['original_length']} iterations)"
        
        plt.title(plot_title)
        plt.legend()
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Final residuals plot saved to {filename}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return plt.gcf()
    
    def _detect_stalled_residuals(self, residual_array, stall_window=500, stall_threshold=1e-2):
        """
        Detect if residuals have stalled (stopped decreasing significantly).
        
        Parameters:
        -----------
        residual_array : array-like
            Array of residual values
        stall_window : int
            Number of iterations to look back for stall detection
        stall_threshold : float
            Relative change threshold below which residuals are considered stalled
            
        Returns:
        --------
        stall_start : int or None
            Index where stalling begins, or None if no stalling detected
        """
        if len(residual_array) < stall_window * 2:
            return None
        
        # Look for stalling in the last part of the simulation
        for i in range(stall_window, len(residual_array) - stall_window):
            # Check if the relative change over stall_window iterations is small
            window_start = residual_array[i]
            window_end = residual_array[i + stall_window]
            
            # Avoid division by zero
            if abs(window_start) < 1e-20:
                continue
                
            relative_change = abs(window_end - window_start) / abs(window_start)
            
            # If change is very small, consider it stalled
            if relative_change < stall_threshold:
                return i
        
        return None

    def _trim_stalled_residuals(self, residual_arrays, keep_stalled_iterations=500):
        """
        Trim stalled residuals from multiple residual arrays while keeping some stalled data.
        
        Parameters:
        -----------
        residual_arrays : dict
            Dictionary of residual arrays (e.g., {'total': [...], 'momentum': [...], 'pressure': [...]})
        keep_stalled_iterations : int
            Number of stalled iterations to keep to show stalling behavior
            
        Returns:
        --------
        trimmed_arrays : dict
            Dictionary of trimmed residual arrays
        stall_info : dict
            Information about detected stalling
        """
        stall_starts = {}
        
        # Detect stalling for each residual type
        for key, residuals in residual_arrays.items():
            if residuals:  # Only check non-empty arrays
                stall_start = self._detect_stalled_residuals(residuals)
                if stall_start is not None:
                    stall_starts[key] = stall_start
        
        # If any residuals are stalled, find the earliest stall point
        if stall_starts:
            earliest_stall = min(stall_starts.values())
            trim_point = earliest_stall + keep_stalled_iterations
            
            # Ensure we don't go beyond the array length
            min_length = min(len(arr) for arr in residual_arrays.values() if arr)
            trim_point = min(trim_point, min_length)
            
            # Trim all arrays to the same length
            trimmed_arrays = {key: arr[:trim_point] if arr else [] for key, arr in residual_arrays.items()}
            
            stall_info = {
                'stalled': True,
                'stall_starts': stall_starts,
                'earliest_stall': earliest_stall,
                'trim_point': trim_point,
                'original_length': min_length,
                'trimmed_length': trim_point
            }
        else:
            trimmed_arrays = residual_arrays.copy()
            stall_info = {'stalled': False}
        
        return trimmed_arrays, stall_info 