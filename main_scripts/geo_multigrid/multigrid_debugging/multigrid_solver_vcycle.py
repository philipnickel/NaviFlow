#%% imports
import numpy as np
import matplotlib.pyplot as plt
from E03 import smooth, calc_b, u_exact, laplacian, calc_Gu, calc_Au
from time import time



def coarsen(fine_grid: np.ndarray) -> np.ndarray:
    """
    Reduces a fine grid to a coarse grid by taking every other point.
    
    Parameters:
        fine_grid (np.ndarray): The input fine grid to be coarsened
        
    Returns:
        np.ndarray: The coarsened grid
    """
    if fine_grid.ndim == 1:
        return fine_grid
    else:
        return fine_grid[1::2, 1::2]

def interpolate(coarse_grid: np.ndarray, m: int) -> np.ndarray:
    """
    Interpolates a coarse grid to a fine grid using linear interpolation.
    
    Parameters:
        coarse_grid (np.ndarray): The input coarse grid to be interpolated
        m (int): Size of the target fine grid
        
    Returns:
        np.ndarray: The interpolated fine grid
    """
    fine_grid = np.zeros((m, m))

    fine_grid[1::2, 1::2] = coarse_grid.copy() 
    fine_grid[1::2, 2::2] += fine_grid[1::2, 1::2] / 2 
    fine_grid[1::2, :-1:2] += fine_grid[1::2, 1::2] / 2 

    fine_grid[2::2, :] += fine_grid[1::2, :] / 2 
    fine_grid[:-1:2, :] += fine_grid[1::2, :] / 2 
        
    return fine_grid


# %% v cycle
import numpy as np
from scipy.linalg import norm

# Exact solution and RHS
def u(x, y):
    return u_exact(x, y)

def f(x, y):
    return laplacian(x, y)

def form_rhs(m):
    return calc_b(m)

def Amult(U):
    # Flatten U before passing it to calc_Gu
    return calc_Au(U)

def Vcycle(U: np.ndarray, 
           omega: float, 
           nsmooth: int, 
           m: int, 
           F: np.ndarray,
           grid_calculations: list) -> np.ndarray:
    """Performs one V-cycle of the multigrid method to solve the Poisson equation."""
    # approximate solve A*U = F
    h = 1.0 / (m + 1)
    current_level = len(grid_calculations)
    grid_calculations.append(m)
    
    if m == 1: 
        # Store data for the coarsest level
        vcycle_data.append({
            'level': current_level,
            'step': 'initial_solution',
            'data': U.reshape((m, m)).copy()
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'after_presmooth',
            'data': U.reshape((m, m)).copy()
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'residual',
            'data': F.reshape((m, m)).copy()
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'restricted_residual',
            'data': F.reshape((m, m)).copy()
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'coarse_correction',
            'data': (-1/16*F).reshape((m, m)).copy()
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'interpolated_correction',
            'data': np.zeros((m*2+1, m*2+1)).copy()
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'before_correction',
            'data': U.reshape((m, m)).copy()
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'after_correction',
            'data': U.reshape((m, m)).copy()
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'after_postsmooth',
            'data': U.reshape((m, m)).copy()
        })
        return -1/16*F
    else:
        # Store initial solution
        vcycle_data.append({
            'level': current_level,
            'step': 'initial_solution',
            'data': U.reshape((m, m)).copy()
        })
        
        # 1. pre-smooth the error
        U = smooth(U.flatten(), omega, F, nsmooth)
        vcycle_data.append({
            'level': current_level,
            'step': 'after_presmooth',
            'data': U.reshape((m, m)).copy()
        })
        
        # 2. calculate the residual
        R = F + Amult(U.flatten())
        vcycle_data.append({
            'level': current_level,
            'step': 'residual',
            'data': R.reshape((m, m)).copy()
        })
        
        # 3. coarsen the residual
        R = R.reshape((m, m))
        Rcoarse = coarsen(R)
        vcycle_data.append({
            'level': current_level,
            'step': 'restricted_residual',
            'data': Rcoarse.copy()
        })

        # 4. recurse to Vcycle on a coarser grid
        mc = (m - 1) // 2
        Ecoarse = Vcycle(np.zeros((mc, mc)), omega, nsmooth, mc, Rcoarse.flatten(), grid_calculations)
        vcycle_data.append({
            'level': current_level,
            'step': 'coarse_correction',
            'data': Ecoarse.reshape((mc, mc)).copy()
        })
        
        # 5. interpolate the error
        E = interpolate(Ecoarse.reshape((mc, mc)), m)
        vcycle_data.append({
            'level': current_level,
            'step': 'interpolated_correction',
            'data': E.copy()
        })
        
        # Store solution before correction
        vcycle_data.append({
            'level': current_level,
            'step': 'before_correction',
            'data': U.reshape((m, m)).copy()
        })
        
        # 6. update the solution given the interpolated error
        U += E.flatten()
        vcycle_data.append({
            'level': current_level,
            'step': 'after_correction',
            'data': U.reshape((m, m)).copy()
        })
        
        # 7. post-smooth the error
        U = smooth(U, omega, F, nsmooth)
        vcycle_data.append({
            'level': current_level,
            'step': 'after_postsmooth',
            'data': U.reshape((m, m)).copy()
        })

    return U

def plot_vcycle_results():
    """Plot the V-cycle results from stored data."""
    # Define the order of steps we want to show
    step_order = [
        'initial_solution',
        'after_presmooth',
        'residual',
        'restricted_residual',
        'coarse_correction',
        'interpolated_correction',
        'before_correction',
        'after_correction',
        'after_postsmooth'
    ]
    
    # Get unique levels
    levels = sorted(set(d['level'] for d in vcycle_data))
    
    # Create subplots for each level
    n_levels = len(levels)
    n_steps = len(step_order)
    
    # Adjust figure size based on number of levels and steps
    fig_width = 3 * n_steps  # 3 inches per step
    fig_height = 3 * n_levels  # 3 inches per level
    fig, axes = plt.subplots(n_levels, n_steps, figsize=(fig_width, fig_height))
    
    if n_levels == 1:
        axes = axes.reshape(1, -1)
    
    # Add a main title to the figure
    fig.suptitle('V-cycle Analysis - Steps in Chronological Order', fontsize=16, y=1.02)
    
    # Define scale ranges for different types of data
    scale_ranges = {
        'solution': {'vmin': -1, 'vmax': 1},  # For solution fields
        'residual': {'vmin': -0.1, 'vmax': 0.1},  # For residuals
        'correction': {'vmin': -0.1, 'vmax': 0.1},  # For corrections
        'default': {'vmin': -1, 'vmax': 1}  # Default range
    }
    
    # Map steps to scale types
    step_scales = {
        'initial_solution': 'solution',
        'after_presmooth': 'solution',
        'residual': 'residual',
        'restricted_residual': 'residual',
        'coarse_correction': 'correction',
        'interpolated_correction': 'correction',
        'before_correction': 'solution',
        'after_correction': 'solution',
        'after_postsmooth': 'solution'
    }
    
    for i, level in enumerate(levels):
        for j, step in enumerate(step_order):
            ax = axes[i, j]
            
            # Try to get data for this level and step
            data = next((d['data'] for d in vcycle_data if d['level'] == level and d['step'] == step), None)
            if data is not None:
                # Get appropriate scale range for this step
                scale_type = step_scales.get(step, 'default')
                scale_range = scale_ranges[scale_type]
                
                # Plot the data with appropriate scaling
                im = ax.imshow(data, cmap='viridis', 
                             vmin=scale_range['vmin'], 
                             vmax=scale_range['vmax'])
                ax.set_title(f'Level {level}\n{step}\nShape: {data.shape}', fontsize=10)
                plt.colorbar(im, ax=ax)
                
                # Add statistics
                stats_text = f'min: {np.min(data):.2e}\nmax: {np.max(data):.2e}\nmean: {np.mean(data):.2e}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data', 
                       horizontalalignment='center', 
                       verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title(f'Level {level}\n{step}')
            
            # Remove axis ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    # Save to PDF instead of showing
    plt.savefig('vcycle_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory

def main():
    # Set up the problem
    m = 63  # Grid size (2^6 - 1)
    dx = 1.0 / (m + 1)  # Grid spacing
    x = np.linspace(0, 1, m+2)[1:-1]  # Interior points
    y = np.linspace(0, 1, m+2)[1:-1]
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Right-hand side (source term)
    f = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Exact solution
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Initial guess
    u = np.zeros((m, m))
    
    # Initialize V-cycle data storage
    vcycle_data = []
    
    # Run one V-cycle for analysis
    print("\nStarting V-cycle analysis...")
    u, residual_history = Vcycle(u, f, m, dx, vcycle_data)
    
    # Plot V-cycle analysis
def main() -> None:
    global vcycle_data  # Make vcycle_data accessible to Vcycle function
    vcycle_data = []  # Initialize vcycle_data list
    
    epsilon = 1e-12
    max_iterations = 1  # Only run one iteration to analyze V-cycle
    omega = 0.5
    nsmooth = 3

    # Study dependence on grid size
    results = []
    convergence_data = {}  # To store convergence data for each grid size
    grid_calculations = []
    
    # Use 63x63 grid (2^6 - 1)
    m = 63
    U_vcycle = np.zeros((m, m))
    F = form_rhs(m)
    relative_residuals = []  # To store relative residuals for this grid size
    print(f'\nSolving problem for grid size m = {m}...')
    t1 = time()
    
    for i in range(1, max_iterations + 1):
        U_vcycle = Vcycle(U_vcycle, omega, nsmooth, m, F, grid_calculations)
        R = F + Amult(U_vcycle.flatten())
        relative_residual = norm(R, 2) / norm(F, 2)
        relative_residuals.append(relative_residual)
        
        if i > 1:
            convergence_rate = relative_residual / relative_residuals[-2]
            print(f'iter: {i}, residual: {relative_residual:.6e}, convergence rate: {convergence_rate:.6e}')
        else:
            print(f'iter: {i}, residual: {relative_residual:.6e}')
        
        if relative_residual < epsilon:
            results.append((m, i))
            print(f'MG_solver: Converged after {i} iterations')
            print(f'\tFinal residual: {relative_residual:.14f}')
            break
    
    t2 = time()
    print(f'\tTime elapsed: {t2 - t1} seconds\n')
    
    # Plot V-cycle analysis
    plot_vcycle_results()
    print("V-cycle analysis saved to 'vcycle_analysis.pdf'")

if __name__ == "__main__":
    main()

# %%
