#%% imports
import numpy as np
import matplotlib.pyplot as plt
from E03 import smooth, calc_b, u_exact, laplacian, calc_Gu, calc_Au
from time import time
import os
from scipy.linalg import norm
import pandas as pd

# Global variable to store V-cycle data for visualization
vcycle_data = []

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
            'data': U.reshape((m, m)).copy(),
            'shape': (m, m),
            'min': float(np.min(U)),
            'max': float(np.max(U)),
            'mean': float(np.mean(U))
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'after_presmooth',
            'data': U.reshape((m, m)).copy(),
            'shape': (m, m),
            'min': float(np.min(U)),
            'max': float(np.max(U)),
            'mean': float(np.mean(U))
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'residual',
            'data': F.reshape((m, m)).copy(),
            'shape': (m, m),
            'min': float(np.min(F)),
            'max': float(np.max(F)),
            'mean': float(np.mean(F))
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'restricted_residual',
            'data': F.reshape((m, m)).copy(),
            'shape': (m, m),
            'min': float(np.min(F)),
            'max': float(np.max(F)),
            'mean': float(np.mean(F))
        })
        
        # Add coarse_solution step
        coarse_sol = (-1/16*F).reshape((m, m)).copy()
        vcycle_data.append({
            'level': current_level,
            'step': 'coarse_solution',
            'data': coarse_sol.copy(),
            'shape': (m, m),
            'min': float(np.min(coarse_sol)),
            'max': float(np.max(coarse_sol)),
            'mean': float(np.mean(coarse_sol))
        })
        
        vcycle_data.append({
            'level': current_level,
            'step': 'coarse_correction',
            'data': coarse_sol.copy(),
            'shape': (m, m),
            'min': float(np.min(coarse_sol)),
            'max': float(np.max(coarse_sol)),
            'mean': float(np.mean(coarse_sol))
        })
        
        # Empty interpolated correction for the coarsest level
        empty_corr = np.zeros((m*2+1, m*2+1)).copy()
        vcycle_data.append({
            'level': current_level,
            'step': 'interpolated_correction',
            'data': empty_corr.copy(),
            'shape': empty_corr.shape,
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'before_correction',
            'data': U.reshape((m, m)).copy(),
            'shape': (m, m),
            'min': float(np.min(U)),
            'max': float(np.max(U)),
            'mean': float(np.mean(U))
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'after_correction',
            'data': U.reshape((m, m)).copy(),
            'shape': (m, m),
            'min': float(np.min(U)),
            'max': float(np.max(U)),
            'mean': float(np.mean(U))
        })
        vcycle_data.append({
            'level': current_level,
            'step': 'after_postsmooth',
            'data': U.reshape((m, m)).copy(),
            'shape': (m, m),
            'min': float(np.min(U)),
            'max': float(np.max(U)),
            'mean': float(np.mean(U))
        })
        return -1/16*F
    else:
        # Store initial solution
        vcycle_data.append({
            'level': current_level,
            'step': 'initial_solution',
            'data': U.reshape((m, m)).copy(),
            'shape': (m, m),
            'min': float(np.min(U)),
            'max': float(np.max(U)),
            'mean': float(np.mean(U))
        })
        
        # 1. pre-smooth the error
        U = smooth(U.flatten(), omega, F, nsmooth)
        U_reshaped = U.reshape((m, m))
        vcycle_data.append({
            'level': current_level,
            'step': 'after_presmooth',
            'data': U_reshaped.copy(),
            'shape': (m, m),
            'min': float(np.min(U_reshaped)),
            'max': float(np.max(U_reshaped)),
            'mean': float(np.mean(U_reshaped))
        })
        
        # 2. calculate the residual
        R = F + Amult(U.flatten())
        R_reshaped = R.reshape((m, m))
        vcycle_data.append({
            'level': current_level,
            'step': 'residual',
            'data': R_reshaped.copy(),
            'shape': (m, m),
            'min': float(np.min(R_reshaped)),
            'max': float(np.max(R_reshaped)),
            'mean': float(np.mean(R_reshaped))
        })
        
        # 3. coarsen the residual
        R = R.reshape((m, m))
        Rcoarse = coarsen(R)
        vcycle_data.append({
            'level': current_level,
            'step': 'restricted_residual',
            'data': Rcoarse.copy(),
            'shape': Rcoarse.shape,
            'min': float(np.min(Rcoarse)),
            'max': float(np.max(Rcoarse)),
            'mean': float(np.mean(Rcoarse))
        })

        # 4. recurse to Vcycle on a coarser grid
        mc = (m - 1) // 2
        Ecoarse = Vcycle(np.zeros((mc, mc)), omega, nsmooth, mc, Rcoarse.flatten(), grid_calculations)
        Ecoarse_reshaped = Ecoarse.reshape((mc, mc))
        
        # Add coarse_solution step
        vcycle_data.append({
            'level': current_level,
            'step': 'coarse_solution',
            'data': Ecoarse_reshaped.copy(),
            'shape': Ecoarse_reshaped.shape,
            'min': float(np.min(Ecoarse_reshaped)),
            'max': float(np.max(Ecoarse_reshaped)),
            'mean': float(np.mean(Ecoarse_reshaped))
        })
        
        vcycle_data.append({
            'level': current_level,
            'step': 'coarse_correction',
            'data': Ecoarse_reshaped.copy(),
            'shape': Ecoarse_reshaped.shape,
            'min': float(np.min(Ecoarse_reshaped)),
            'max': float(np.max(Ecoarse_reshaped)),
            'mean': float(np.mean(Ecoarse_reshaped))
        })
        
        # 5. interpolate the error
        E = interpolate(Ecoarse.reshape((mc, mc)), m)
        vcycle_data.append({
            'level': current_level,
            'step': 'interpolated_correction',
            'data': E.copy(),
            'shape': E.shape,
            'min': float(np.min(E)),
            'max': float(np.max(E)),
            'mean': float(np.mean(E))
        })
        
        # Store solution before correction
        U_before = U.reshape((m, m))
        vcycle_data.append({
            'level': current_level,
            'step': 'before_correction',
            'data': U_before.copy(),
            'shape': (m, m),
            'min': float(np.min(U_before)),
            'max': float(np.max(U_before)),
            'mean': float(np.mean(U_before))
        })
        
        # 6. update the solution given the interpolated error
        U += E.flatten()
        U_after = U.reshape((m, m))
        vcycle_data.append({
            'level': current_level,
            'step': 'after_correction',
            'data': U_after.copy(),
            'shape': (m, m),
            'min': float(np.min(U_after)),
            'max': float(np.max(U_after)),
            'mean': float(np.mean(U_after))
        })
        
        # 7. post-smooth the error
        U = smooth(U, omega, F, nsmooth)
        U_final = U.reshape((m, m))
        vcycle_data.append({
            'level': current_level,
            'step': 'after_postsmooth',
            'data': U_final.copy(),
            'shape': (m, m),
            'min': float(np.min(U_final)),
            'max': float(np.max(U_final)),
            'mean': float(np.mean(U_final))
        })

    return U

def plot_vcycle_results(output_path='vcycle_analysis.pdf'):
    """Plot the V-cycle results from stored data."""
    # Convert data to DataFrame for easier manipulation
    df = pd.DataFrame(vcycle_data)
    
    # Define the order of steps we want to show
    step_order = [
        'initial_solution',
        'after_presmooth',
        'residual',
        'restricted_residual',
        'coarse_solution',
        'coarse_correction',
        'interpolated_correction',
        'before_correction',
        'after_correction',
        'after_postsmooth'
    ]
    
    # Get unique levels
    levels = sorted(df['level'].unique())
    
    # Create subplots for each level
    n_levels = len(levels)
    n_steps = len(step_order)
    fig, axes = plt.subplots(n_levels, n_steps, figsize=(5*n_steps, 5*n_levels))
    
    if n_levels == 1:
        axes = axes.reshape(1, -1)
    
    # Add a main title to the figure
    fig.suptitle('V-cycle Analysis - Steps in Chronological Order', fontsize=16, y=1.02)
    
    for i, level in enumerate(levels):
        for j, step in enumerate(step_order):
            ax = axes[i, j]
            
            # Try to get data for this level and step
            data_filter = df[(df['level'] == level) & (df['step'] == step)]
            if not data_filter.empty:
                data = data_filter.iloc[0]
                
                # Plot the data
                im = ax.imshow(data['data'], cmap='viridis')
                ax.set_title(f'Level {level}\n{step}\nShape: {data["shape"]}', fontsize=10)
                plt.colorbar(im, ax=ax)
                
                # Add statistics
                stats_text = f'min: {data["min"]:.2e}\nmax: {data["max"]:.2e}\nmean: {data["mean"]:.2e}'
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
    # Save to PDF
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory

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
    m = 31
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
    
    # Plot V-cycle analysis with a specific output path
    debug_dir = 'debug_output'
    os.makedirs(debug_dir, exist_ok=True)
    output_path = os.path.join(debug_dir, 'vcycle_analysis.pdf')
    plot_vcycle_results(output_path)
    print(f"V-cycle analysis saved to '{output_path}'")

if __name__ == "__main__":
    main()

# %%
