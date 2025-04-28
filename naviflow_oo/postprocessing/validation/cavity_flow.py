"""
Lid-driven cavity validation utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import inspect
from numba import njit, prange


def check_divergence_free(imax, jmax, dx, dy, u, v):
    """Check if velocity field is divergence free."""
    # Fully vectorized implementation
    i_indices = np.arange(imax)
    j_indices = np.arange(jmax)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    divergence = (u[i_grid+1, j_grid] - u[i_grid, j_grid])/dx + \
                 (v[i_grid, j_grid+1] - v[i_grid, j_grid])/dy
    
    return divergence


class BenchmarkData:
    """Class to store benchmark data for comparison."""
    
    # Ghia et al. (1982) data for Re = 100
    GHIA_RE_100 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                      0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, 
                      -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 
                      0.10091, 0.09233, 0.00000]),
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                      0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
        'u': np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, 
                      -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 
                      0.78871, 1.00000])
    }
    
    # Ghia et al. (1982) data for Re = 400
    GHIA_RE_400 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                      0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, 
                      -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 
                      0.19713, 0.18360, 0.00000]),
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                      0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
        'u': np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, 
                      -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 
                      0.68439, 1.00000])
    }
    
    # Ghia et al. (1982) data for Re = 1000
    GHIA_RE_1000 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                       0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, 
                       -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 
                       0.29012, 0.27485, 0.00000]),
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                      0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
        'u': np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, 
                      -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 
                      0.57492, 1.00000])
    }
    
    # Ghia et al. (1982) data for Re = 3200
    GHIA_RE_3200 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                       0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.39017, -0.47425, -0.52357, -0.54053, -0.44307, -0.37401, 
                       -0.31184, 0.00999, 0.28188, 0.29030, 0.37119, 0.42768, 0.41906, 
                       0.40917, 0.39560, 0.00000]),
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                      0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
        'u': np.array([0.00000, -0.32407, -0.35344, -0.37827, -0.41933, -0.34323, -0.24427, 
                      -0.86636, -0.04272, 0.07156, 0.19791, 0.34682, 0.46101, 0.46547, 
                      0.48296, 1.00000])
    }
    
    # Ghia et al. (1982) data for Re = 5000
    GHIA_RE_5000 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                       0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.41165, -0.52876, -0.55408, -0.55069, -0.41442, -0.36214, 
                       -0.30018, 0.00945, 0.27280, 0.28066, 0.35368, 0.41824, 0.43564, 
                       0.43154, 0.42735, 0.00000]),
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                      0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
        'u': np.array([0.00000, -0.41165, -0.42901, -0.43643, -0.40435, -0.33050, -0.22855, 
                      -0.07404, -0.03039, 0.08183, 0.20087, 0.33556, 0.46036, 0.45992, 
                      0.46120, 1.00000])
    }
    
    # Ghia et al. (1982) data for Re = 7500
    GHIA_RE_7500 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                       0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.43154, -0.55216, -0.59756, -0.55460, -0.41824, -0.36435, 
                       -0.30448, 0.00824, 0.29598, 0.30448, 0.36089, 0.41349, 0.43453, 
                       0.43759, 0.43736, 0.00000]),
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                      0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
        'u': np.array([0.00000, -0.43154, -0.43590, -0.43025, -0.38324, -0.32393, -0.23176, 
                      -0.07503, -0.03800, 0.08342, 0.20591, 0.34228, 0.47167, 0.47323, 
                      0.47048, 1.00000])
    }
    
    # Ghia et al. (1982) data for Re = 10000
    GHIA_RE_10000 = {
        'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                       0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
        'v': np.array([0.00000, -0.42735, -0.57492, -0.65928, -0.68439, -0.43025, -0.37582, 
                       -0.31966, 0.00831, 0.30719, 0.31586, 0.37401, 0.42160, 0.44265, 
                       0.44407, 0.43979, 0.00000]),
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                      0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 1.0000]),
        'u': np.array([0.00000, -0.42735, -0.42537, -0.41657, -0.38000, -0.32709, -0.23186, 
                      -0.07540, -0.03111, 0.08344, 0.20673, 0.34635, 0.47804, 0.48070, 
                      0.47783, 1.00000])
    }
    
    @classmethod
    def get_ghia_data(cls, Re):
        """Get Ghia et al. data for given Reynolds number."""
        if Re == 100:
            return cls.GHIA_RE_100
        elif Re == 400:
            return cls.GHIA_RE_400
        elif Re == 1000:
            return cls.GHIA_RE_1000
        elif Re == 3200:
            return cls.GHIA_RE_3200
        elif Re == 5000:
            return cls.GHIA_RE_5000
        elif Re == 7500:
            return cls.GHIA_RE_7500
        elif Re == 10000:
            return cls.GHIA_RE_10000
        else:
            return None


def calculate_divergence(u, v, dx, dy, use_numba=False):
    """
    Calculate the divergence of the velocity field.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity fields
    dx, dy : float
        Grid spacing
    use_numba : bool, optional
        Whether to use numba for acceleration
        
    Returns:
    --------
    ndarray
        Divergence field
    """
    # Basic calculation of divergence from velocity components
    imax, jmax = u.shape[0] - 1, u.shape[1]
    
    # Create grid for vectorized calculation
    i_indices = np.arange(imax)
    j_indices = np.arange(jmax)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    # Ensure we don't go out of bounds
    valid_j = j_grid < min(jmax, v.shape[1] - 1)
    
    # Initialize divergence field
    divergence = np.zeros((imax, jmax))
    
    # Calculate du/dx component everywhere
    divergence = (u[i_grid+1, j_grid] - u[i_grid, j_grid]) / dx
    
    # Add dv/dy component where valid
    if np.any(valid_j):
        divergence[valid_j] += (v[i_grid[valid_j], j_grid[valid_j]+1] - v[i_grid[valid_j], j_grid[valid_j]]) / dy
    
    return divergence


def calculate_divergence_unstructured(u, v, mesh):
    """
    Calculate the divergence of the velocity field for unstructured meshes.
    
    For unstructured meshes, the divergence is calculated using the Gauss divergence theorem.
    For each cell, we compute the outward flux through all faces and normalize by the cell volume.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components at cell centers (1D arrays)
    mesh : UnstructuredMesh or MeshAdapter containing an UnstructuredMesh
        The computational mesh
        
    Returns:
    --------
    ndarray
        Divergence at each cell center (1D array)
    """
    # If it's a MeshAdapter, get the underlying mesh
    unstructured_mesh = mesh.mesh if hasattr(mesh, 'mesh') else mesh
    
    # Get mesh information
    n_cells = unstructured_mesh.n_cells
    owner_cells, neighbor_cells = unstructured_mesh.get_owner_neighbor()
    face_areas = unstructured_mesh.get_face_areas()
    face_normals = unstructured_mesh.get_face_normals()
    cell_volumes = unstructured_mesh.get_cell_volumes()
    
    # Initialize divergence array
    divergence = np.zeros(n_cells)
    
    # For each face, compute the contribution to divergence
    for face_idx in range(len(owner_cells)):
        owner = owner_cells[face_idx]
        neighbor = neighbor_cells[face_idx]
        
        # Face normal and area
        normal = face_normals[face_idx]
        area = face_areas[face_idx]
        
        # Velocity at face (simple interpolation)
        if neighbor >= 0:  # Internal face
            u_face = 0.5 * (u[owner] + u[neighbor])
            v_face = 0.5 * (v[owner] + v[neighbor])
        else:  # Boundary face - use cell center value for now
            u_face = u[owner]
            v_face = v[owner]
        
        # Compute face flux (dot product of velocity and area-weighted normal)
        flux = (u_face * normal[0] + v_face * normal[1]) * area
        
        # Add contribution to owner cell (outgoing flux is positive divergence)
        divergence[owner] += flux
        
        # Add contribution to neighbor cell (incoming flux is negative divergence)
        if neighbor >= 0:
            divergence[neighbor] -= flux
    
    # Normalize by cell volume to get divergence
    # Handle possible zero volumes to avoid division by zero
    for i in range(n_cells):
        if cell_volumes[i] > 1e-15:  # Avoid division by very small values
            divergence[i] /= cell_volumes[i]
        else:
            # For cells with zero or very small volume, set divergence to 0
            divergence[i] = 0.0
    
    return divergence


def calculate_infinity_norm_error(u, v, mesh, reynolds):
    """
    Calculate the infinity norm error against Ghia data.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity fields
    mesh : StructuredMesh
        The computational mesh
    reynolds : int
        Reynolds number
        
    Returns:
    --------
    float
        Infinity norm error
    """
    # Get mesh dimensions and coordinates
    nx, ny = mesh.get_dimensions()
    dx, dy = mesh.get_cell_sizes()
    
    # Create x and y coordinates
    x = np.linspace(dx/2, 1-dx/2, nx)
    y = np.linspace(dy/2, 1-dy/2, ny)
    
    # Get benchmark data for the given Reynolds number
    ghia_data = BenchmarkData.get_ghia_data(reynolds)
    if ghia_data is None:
        # If no exact match, use the closest available Reynolds number
        available_re = [100, 400, 1000, 3200, 5000, 7500, 10000]
        closest_re = min(available_re, key=lambda re: abs(re - reynolds))
        ghia_data = BenchmarkData.get_ghia_data(closest_re)
        print(f"Warning: No Ghia data for Re={reynolds}, using closest available Re={closest_re}")
    
    # Extract benchmark data
    x_benchmark = ghia_data['x']
    v_benchmark = ghia_data['v']
    y_benchmark = ghia_data['y']
    u_benchmark = ghia_data['u']
    
    # Interpolate simulation data to benchmark coordinates
    u_centerline = u[nx//2, :]  # u along vertical centerline
    v_centerline = v[:, ny//2]  # v along horizontal centerline
    
    # Create interpolation functions
    u_interp = interp1d(y, u_centerline, kind='cubic', bounds_error=False, fill_value="extrapolate")
    v_interp = interp1d(x, v_centerline, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    # Evaluate at benchmark coordinates
    u_sim_at_benchmark = u_interp(y_benchmark)
    v_sim_at_benchmark = v_interp(x_benchmark)
    
    # Calculate errors
    u_error = np.abs(u_sim_at_benchmark - u_benchmark)
    v_error = np.abs(v_sim_at_benchmark - v_benchmark)
    
    # Infinity norm error is the maximum of all errors
    inf_norm_error = max(np.max(u_error), np.max(v_error))
    
    return inf_norm_error


def calculate_l2_norm_error(u, v, mesh, reynolds):
    """
    Calculate the L2 norm error against Ghia data.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity fields
    mesh : StructuredMesh
        The computational mesh
    reynolds : int
        Reynolds number
        
    Returns:
    --------
    float
        L2 norm error
    """
    # Get mesh dimensions and coordinates
    nx, ny = mesh.get_dimensions()
    dx, dy = mesh.get_cell_sizes()
    
    # Create x and y coordinates
    x = np.linspace(dx/2, 1-dx/2, nx)
    y = np.linspace(dy/2, 1-dy/2, ny)
    
    # Get benchmark data for the given Reynolds number
    ghia_data = BenchmarkData.get_ghia_data(reynolds)
    if ghia_data is None:
        # If no exact match, use the closest available Reynolds number
        available_re = [100, 400, 1000, 3200, 5000, 7500, 10000]
        closest_re = min(available_re, key=lambda re: abs(re - reynolds))
        ghia_data = BenchmarkData.get_ghia_data(closest_re)
        print(f"Warning: No Ghia data for Re={reynolds}, using closest available Re={closest_re}")
    
    # Extract benchmark data
    x_benchmark = ghia_data['x']
    v_benchmark = ghia_data['v']
    y_benchmark = ghia_data['y']
    u_benchmark = ghia_data['u']
    
    # Interpolate simulation data to benchmark coordinates
    u_centerline = u[nx//2, :]  # u along vertical centerline
    v_centerline = v[:, ny//2]  # v along horizontal centerline
    
    # Create interpolation functions
    u_interp = interp1d(y, u_centerline, kind='cubic', bounds_error=False, fill_value="extrapolate")
    v_interp = interp1d(x, v_centerline, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    # Evaluate at benchmark coordinates
    u_sim_at_benchmark = u_interp(y_benchmark)
    v_sim_at_benchmark = v_interp(x_benchmark)
    
    # Calculate squared errors
    u_error_squared = (u_sim_at_benchmark - u_benchmark) ** 2
    v_error_squared = (v_sim_at_benchmark - v_benchmark) ** 2
    
    # L2 norm error is the square root of mean squared error
    l2_norm_error = np.sqrt((np.sum(u_error_squared) + np.sum(v_error_squared)) / (len(u_benchmark) + len(v_benchmark)))
    
    return l2_norm_error

 