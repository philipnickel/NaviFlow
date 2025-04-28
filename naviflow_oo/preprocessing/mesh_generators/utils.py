"""
Utility functions for mesh generation.
"""

import numpy as np

def create_tanh_clustered_nodes(min_val, max_val, n_points, alpha=3.0):
    """
    Creates non-uniform spacing clustered towards both ends using tanh.
    
    Parameters:
    -----------
    min_val : float
        Minimum coordinate value
    max_val : float
        Maximum coordinate value
    n_points : int
        Number of points to generate
    alpha : float, optional
        Controls the strength of clustering (higher values = stronger clustering)
        
    Returns:
    --------
    nodes : ndarray, shape (n_points,)
        Clustered node coordinates
    """
    if n_points <= 1:
        return np.array([min_val]) if n_points == 1 else np.array([])
    
    # Create uniform points in [0, 1]
    x_uniform = np.linspace(0.0, 1.0, n_points)
    
    # Apply tanh stretching function
    # y = 0.5 * (1 + tanh(alpha * (2*x - 1)) / tanh(alpha))
    tanh_alpha = np.tanh(alpha)
    if tanh_alpha == 0:  # Avoid division by zero if alpha is extremely small
        y_stretched = x_uniform
    else:
        y_stretched = 0.5 * (1.0 + np.tanh(alpha * (2.0 * x_uniform - 1.0)) / tanh_alpha)
        
    # Scale and shift to the desired range [min_val, max_val]
    nodes = min_val + (max_val - min_val) * y_stretched
    return nodes

def create_geometric_progression(min_val, max_val, n_points, growth_factor=1.1):
    """
    Creates non-uniform spacing based on geometric progression.
    
    Parameters:
    -----------
    min_val : float
        Minimum coordinate value
    max_val : float
        Maximum coordinate value
    n_points : int
        Number of points to generate
    growth_factor : float, optional
        Growth factor between successive intervals (>1 for expanding, <1 for contracting)
        
    Returns:
    --------
    nodes : ndarray, shape (n_points,)
        Node coordinates with geometric progression
    """
    if n_points <= 1:
        return np.array([min_val]) if n_points == 1 else np.array([])

    # For growth_factor = 1, use uniform spacing
    if abs(growth_factor - 1.0) < 1e-10:
        return np.linspace(min_val, max_val, n_points)
    
    # Calculate geometric sum for normalization
    if growth_factor != 1.0:
        total = (1 - growth_factor**(n_points-1)) / (1 - growth_factor)
    else:
        total = n_points - 1
    
    # Generate the geometric sequence (starting with first interval = 1)
    intervals = np.ones(n_points - 1)
    for i in range(1, n_points - 1):
        intervals[i] = intervals[i-1] * growth_factor
    
    # Normalize to the desired total length
    interval_sum = np.sum(intervals)
    intervals = intervals * (max_val - min_val) / interval_sum
    
    # Construct node positions
    nodes = np.zeros(n_points)
    nodes[0] = min_val
    for i in range(1, n_points):
        nodes[i] = nodes[i-1] + intervals[i-1]
    
    return nodes

def create_sine_clustered_nodes(min_val, max_val, n_points, cluster_factor=0.5):
    """
    Creates non-uniform spacing with sine-based clustering.
    
    Parameters:
    -----------
    min_val : float
        Minimum coordinate value
    max_val : float
        Maximum coordinate value
    n_points : int
        Number of points to generate
    cluster_factor : float, optional
        Controls the clustering (0.0 = uniform, 1.0 = max clustering)
        
    Returns:
    --------
    nodes : ndarray, shape (n_points,)
        Node coordinates with sine-based clustering
    """
    if n_points <= 1:
        return np.array([min_val]) if n_points == 1 else np.array([])
    
    # Clamp cluster_factor to [0, 1]
    cluster_factor = max(0, min(1, cluster_factor))
    
    # Create uniform points in [0, 1]
    x_uniform = np.linspace(0.0, 1.0, n_points)
    
    # Apply sine-based mapping (when cluster_factor=0, this reduces to uniform)
    # When cluster_factor=1, clustering is maximized at the ends
    blend = cluster_factor * np.sin(np.pi * x_uniform) + (1 - cluster_factor) * x_uniform
    
    # Scale and shift to the desired range [min_val, max_val]
    nodes = min_val + (max_val - min_val) * blend
    return nodes 