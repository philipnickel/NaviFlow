"""
NaviFlow - A Python package for computational fluid dynamics simulations.
"""

"""
# Main algorithms
from .algorithms.simple import simple_algorithm

# Core solvers
from .solvers.momentum.standard import u_momentum, v_momentum, solve_momentum
from .solvers.pressure.direct import get_rhs, get_coeff_mat, penta_diag_solve, solve_pressure_correction
from .solvers.velocity.standard import update_velocity

# Utility functions
"""
# main algorithms
from .algorithms.simple import *

# solvers
from .solvers.momentum.standard import *
from .solvers.pressure.direct import *
from .solvers.pressure.matrix_free import *
from .solvers.velocity.standard import *

# utils
from .utils.validation import *
from .utils.plotting import *



"""
Here is old stuff
from .basic_cavity_vectorized import * 
from .utils.validation import check_divergence_free, calculate_divergence, compare_with_ghia
from .utils.plotting import plot_velocity_field, plot_streamlines
"""


# Version information
__version__ = '0.1.0'


