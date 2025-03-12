"""
Class to store and process simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from .visualization import plot_velocity_field, plot_combined_results_matrix
from .validation import BenchmarkData

class SimulationResult:
    """
    Store and analyze simulation results.
    """
    def __init__(self, u, v, p, mesh, iterations=0, residuals=None, divergence=None, reynolds=None):
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
        """
        self.u = u
        self.v = v
        self.p = p
        self.mesh = mesh
        self.iterations = iterations
        self.residuals = residuals or []
        self.divergence = divergence
        self.reynolds = reynolds
    
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