"""
Validation module for CFD simulations.
"""

from .cavity_flow import BenchmarkData, calculate_divergence

# Define a function to get the closest Ghia data
def get_closest_ghia_data(Re):
    """
    Get the closest Ghia et al. data for the given Reynolds number.
    
    Parameters:
    -----------
    Re : float
        Reynolds number
        
    Returns:
    --------
    dict
        Dictionary containing Ghia et al. data
    """
    # Available Reynolds numbers in Ghia et al. data
    re_values = [100, 400, 1000, 3200, 5000, 7500, 10000]
    
    # Find closest Reynolds number
    closest_re = min(re_values, key=lambda x: abs(x - Re))
    
    # Get data for closest Reynolds number
    return BenchmarkData.get_ghia_data(closest_re)
