"""
Utilities for testing pressure solvers in naviflow_oo.

This module provides functions to test the convergence properties of
pressure solvers.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from ..preprocessing.mesh.structured import StructuredMesh
from ..solver.pressure_solver.helpers.rhs_construction import get_rhs
from ..solver.pressure_solver.helpers.coeff_matrix import get_coeff_mat

def generate_test_problem(nx=65, ny=65, problem_type='poisson'):
    """
    Generate a test problem for pressure solvers.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions
    problem_type : str
        Type of test problem to generate:
        - 'poisson': Standard Poisson equation with known analytical solution
        - 'cavity': Lid-driven cavity flow pressure equation
        
    Returns:
    --------
    dict
        Dictionary containing the test problem data
    """
    mesh = StructuredMesh(nx=nx, ny=ny, length=1.0, height=1.0)
    dx, dy = mesh.get_cell_sizes()
    
    if problem_type == 'poisson':
        # Create a Poisson problem with known analytical solution
        # u(x,y) = sin(pi*x) * sin(pi*y)
        # -∇²u = 2*pi²*sin(pi*x)*sin(pi*y)
        
        # Create grid points
        x = np.linspace(dx/2, 1-dx/2, nx)
        y = np.linspace(dy/2, 1-dy/2, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Exact solution
        exact_solution = np.sin(np.pi*X) * np.sin(np.pi*Y)
        
        # Create dummy d_u and d_v (uniform)
        d_u = np.ones((nx+1, ny)) * dy / dx
        d_v = np.ones((nx, ny+1)) * dx / dy
        
        # Create dummy velocity fields
        u_star = np.zeros((nx+1, ny))
        v_star = np.zeros((nx, ny+1))
        
        # Initial guess
        p_star = np.zeros((nx, ny))
        
        # Create right-hand side directly from the exact solution
        # For a Poisson equation: -∇²u = f
        # f = 2*pi²*sin(pi*x)*sin(pi*y)
        rhs_values = 2 * np.pi**2 * exact_solution
        
        # Modify u_star and v_star to create the desired RHS
        # This is a bit of a hack, but it allows us to use the existing RHS construction
        for i in range(nx):
            for j in range(ny):
                if i < nx-1:
                    u_star[i+1, j] = rhs_values[i, j] * dx / 2
                if j < ny-1:
                    v_star[i, j+1] = rhs_values[i, j] * dy / 2
        
        return {
            'mesh': mesh,
            'u_star': u_star,
            'v_star': v_star,
            'd_u': d_u,
            'd_v': d_v,
            'p_star': p_star,
            'exact_solution': exact_solution
        }
        
    elif problem_type == 'cavity':
        # Create a lid-driven cavity flow problem
        # This doesn't have an exact analytical solution
        
        # Create dummy velocity fields for a lid-driven cavity
        u_star = np.zeros((nx+1, ny))
        u_star[:, -1] = 1.0  # Top lid velocity
        
        v_star = np.zeros((nx, ny+1))
        
        # Create d_u and d_v (uniform for simplicity)
        d_u = np.ones((nx+1, ny)) * dy / dx
        d_v = np.ones((nx, ny+1)) * dx / dy
        
        # Initial guess
        p_star = np.zeros((nx, ny))
        
        return {
            'mesh': mesh,
            'u_star': u_star,
            'v_star': v_star,
            'd_u': d_u,
            'd_v': d_v,
            'p_star': p_star
        }
    
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

def test_solver_convergence(solver, problem_size=65, problem_type='poisson', 
                           max_iterations=1000, tolerance=1e-6, 
                           title=None, save_plot=False, filename=None):
    """
    Test the convergence of a pressure solver on a specified problem.
    
    Parameters:
    -----------
    solver : PressureSolver
        The solver to test
    problem_size : int or tuple
        Size of the problem (n or (nx, ny))
    problem_type : str
        Type of problem to test ('poisson' or 'cavity')
    max_iterations : int
        Maximum number of iterations for the solver
    tolerance : float
        Convergence tolerance for the solver
    title : str, optional
        Title for the plot
    save_plot : bool
        Whether to save the plot to a file
    filename : str, optional
        Filename for the saved plot
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    # Set up problem size
    if isinstance(problem_size, int):
        nx = ny = problem_size
    else:
        nx, ny = problem_size
    
    # Generate test problem
    problem_data = generate_test_problem(nx=nx, ny=ny, problem_type=problem_type)
    
    # Set solver parameters
    solver.tolerance = tolerance
    solver.max_iterations = max_iterations
    
    # Start timing
    start_time = time.time()
    
    # Solve the problem
    p_prime = solver.solve(
        mesh=problem_data['mesh'],
        u_star=problem_data['u_star'],
        v_star=problem_data['v_star'],
        d_u=problem_data['d_u'],
        d_v=problem_data['d_v'],
        p_star=problem_data['p_star']
    )
    
    # End timing
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Get convergence history if available
    convergence_history = []
    if hasattr(solver, 'residual_history'):
        convergence_history = solver.residual_history
    
    # Calculate error if exact solution is available
    error = None
    if 'exact_solution' in problem_data:
        exact = problem_data['exact_solution']
        error = np.max(np.abs(p_prime - exact))
    
    # Print results
    solver_name = solver.__class__.__name__
    print(f"Solver: {solver_name}")
    print(f"Problem: {problem_type} ({nx}x{ny})")
    print(f"Time taken: {time_taken:.4f} seconds")
    if error is not None:
        print(f"Maximum absolute error: {error:.6e}")
    print(f"Iterations: {len(convergence_history)}")
    if convergence_history:
        print(f"Final residual: {convergence_history[-1]:.6e}")
    
    # Plot convergence history
    if convergence_history:
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(1, len(convergence_history) + 1), convergence_history)
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Residual (log scale)')
        
        if title:
            plt.title(f'{title} - Convergence History')
        else:
            plt.title(f'{solver_name} - Convergence History')
        
        plt.tight_layout()
        
        if save_plot and filename:
            plt.savefig(f"{filename}_convergence.png")
    
    # Return results
    return {
        'solution': p_prime,
        'error': error,
        'iterations': len(convergence_history),
        'convergence_history': convergence_history,
        'time_taken': time_taken
    } 