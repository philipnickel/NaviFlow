"""
This script demonstrates how to fix the visualization in vcycle_analysis.pdf
by using origin='lower' in matplotlib's imshow function to display grids
in the mathematical convention (y-axis increasing upward).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# Create the debug output directory if it doesn't exist
debug_dir = 'debug_output'
os.makedirs(debug_dir, exist_ok=True)

def create_test_grids():
    """
    Create test grids to simulate the multigrid V-cycle data.
    """
    # Create a linear gradient test grid
    size = 31
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    grid = X + Y

    # Create special corner markers
    grid[0, 0] = 0.0      # Bottom-left (in mathematical coordinates)
    grid[0, -1] = 0.5     # Bottom-right
    grid[-1, -1] = 1.0    # Top-right
    grid[-1, 0] = 0.7     # Top-left

    # Create restricted grid using our fixed restrict function
    restricted = grid[1::2, 1::2]
    
    # Create a dataframe to simulate multigrid data
    vcycle_data = [
        {
            'level': 0,
            'step': 'original',
            'shape': grid.shape,
            'min': np.min(grid),
            'max': np.max(grid),
            'mean': np.mean(grid),
            'data': grid
        },
        {
            'level': 0,
            'step': 'restricted',
            'shape': restricted.shape,
            'min': np.min(restricted),
            'max': np.max(restricted),
            'mean': np.mean(restricted),
            'data': restricted
        }
    ]
    
    return pd.DataFrame(vcycle_data)

def plot_with_default_origin(df, output_path):
    """
    Plot the grids using the default origin='upper' in imshow,
    which causes the y-axis to appear flipped in the visualization.
    """
    # Create a PDF to save the plots
    with PdfPages(output_path) as pdf:
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot the original grid with default origin (upper)
        original_data = df[df['step'] == 'original'].iloc[0]
        im1 = axes[0].imshow(original_data['data'], cmap='viridis')
        axes[0].set_title(f"Original Grid\nShape: {original_data['shape']}\nDefault origin='upper'")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot the restricted grid with default origin (upper)
        restricted_data = df[df['step'] == 'restricted'].iloc[0]
        im2 = axes[1].imshow(restricted_data['data'], cmap='viridis')
        axes[1].set_title(f"Restricted Grid\nShape: {restricted_data['shape']}\nDefault origin='upper'")
        plt.colorbar(im2, ax=axes[1])
        
        # Remove axis ticks for cleaner look
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add annotations to show corner values in original grid
        original_grid = original_data['data']
        corners = [
            (0, 0, "BL: {:.1f}".format(original_grid[0, 0])),
            (0, original_grid.shape[1]-1, "BR: {:.1f}".format(original_grid[0, -1])),
            (original_grid.shape[0]-1, original_grid.shape[1]-1, "TR: {:.1f}".format(original_grid[-1, -1])),
            (original_grid.shape[0]-1, 0, "TL: {:.1f}".format(original_grid[-1, 0]))
        ]
        
        for y, x, text in corners:
            # In default origin='upper', the y-coordinate appears flipped
            axes[0].text(x, y, text, color='white', fontweight='bold', 
                      ha='center', va='center', backgroundcolor='black')
        
        # Add annotations to show corner values in restricted grid
        restricted_grid = restricted_data['data']
        corners = [
            (0, 0, "BL: {:.1f}".format(restricted_grid[0, 0])),
            (0, restricted_grid.shape[1]-1, "BR: {:.1f}".format(restricted_grid[0, -1])),
            (restricted_grid.shape[0]-1, restricted_grid.shape[1]-1, "TR: {:.1f}".format(restricted_grid[-1, -1])),
            (restricted_grid.shape[0]-1, 0, "TL: {:.1f}".format(restricted_grid[-1, 0]))
        ]
        
        for y, x, text in corners:
            # In default origin='upper', the y-coordinate appears flipped
            axes[1].text(x, y, text, color='white', fontweight='bold',
                      ha='center', va='center', backgroundcolor='black')
        
        plt.suptitle("Default Visualization (origin='upper')", fontsize=16)
        plt.tight_layout()
        
        # Save the figure to the PDF
        pdf.savefig(fig)
        plt.close()

def plot_with_lower_origin(df, output_path):
    """
    Plot the grids using origin='lower' in imshow,
    which displays the grids in the mathematical convention (y-axis increasing upward).
    """
    # Create a PDF to save the plots
    with PdfPages(output_path) as pdf:
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot the original grid with origin='lower'
        original_data = df[df['step'] == 'original'].iloc[0]
        im1 = axes[0].imshow(original_data['data'], cmap='viridis', origin='lower')
        axes[0].set_title(f"Original Grid\nShape: {original_data['shape']}\nFixed with origin='lower'")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot the restricted grid with origin='lower'
        restricted_data = df[df['step'] == 'restricted'].iloc[0]
        im2 = axes[1].imshow(restricted_data['data'], cmap='viridis', origin='lower')
        axes[1].set_title(f"Restricted Grid\nShape: {restricted_data['shape']}\nFixed with origin='lower'")
        plt.colorbar(im2, ax=axes[1])
        
        # Remove axis ticks for cleaner look
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add annotations to show corner values in original grid
        original_grid = original_data['data']
        corners = [
            (0, 0, "BL: {:.1f}".format(original_grid[0, 0])),
            (0, original_grid.shape[1]-1, "BR: {:.1f}".format(original_grid[0, -1])),
            (original_grid.shape[0]-1, original_grid.shape[1]-1, "TR: {:.1f}".format(original_grid[-1, -1])),
            (original_grid.shape[0]-1, 0, "TL: {:.1f}".format(original_grid[-1, 0]))
        ]
        
        for y, x, text in corners:
            # With origin='lower', coordinates match mathematical convention
            axes[0].text(x, y, text, color='white', fontweight='bold',
                      ha='center', va='center', backgroundcolor='black')
        
        # Add annotations to show corner values in restricted grid
        restricted_grid = restricted_data['data']
        corners = [
            (0, 0, "BL: {:.1f}".format(restricted_grid[0, 0])),
            (0, restricted_grid.shape[1]-1, "BR: {:.1f}".format(restricted_grid[0, -1])),
            (restricted_grid.shape[0]-1, restricted_grid.shape[1]-1, "TR: {:.1f}".format(restricted_grid[-1, -1])),
            (restricted_grid.shape[0]-1, 0, "TL: {:.1f}".format(restricted_grid[-1, 0]))
        ]
        
        for y, x, text in corners:
            # With origin='lower', coordinates match mathematical convention
            axes[1].text(x, y, text, color='white', fontweight='bold',
                      ha='center', va='center', backgroundcolor='black')
        
        plt.suptitle("Fixed Visualization (origin='lower')", fontsize=16)
        plt.tight_layout()
        
        # Save the figure to the PDF
        pdf.savefig(fig)
        plt.close()

def plot_comparison(df, output_path):
    """
    Plot a comparison between the default and fixed visualizations.
    """
    # Create a PDF to save the plots
    with PdfPages(output_path) as pdf:
        # Create a figure with four subplots (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Upper row: Default origin='upper'
        # Plot the original grid
        original_data = df[df['step'] == 'original'].iloc[0]
        im1 = axes[0, 0].imshow(original_data['data'], cmap='viridis')
        axes[0, 0].set_title(f"Original Grid\nDefault origin='upper'")
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot the restricted grid
        restricted_data = df[df['step'] == 'restricted'].iloc[0]
        im2 = axes[0, 1].imshow(restricted_data['data'], cmap='viridis')
        axes[0, 1].set_title(f"Restricted Grid\nDefault origin='upper'")
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Lower row: Fixed origin='lower'
        # Plot the original grid
        im3 = axes[1, 0].imshow(original_data['data'], cmap='viridis', origin='lower')
        axes[1, 0].set_title(f"Original Grid\nFixed origin='lower'")
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Plot the restricted grid
        im4 = axes[1, 1].imshow(restricted_data['data'], cmap='viridis', origin='lower')
        axes[1, 1].set_title(f"Restricted Grid\nFixed origin='lower'")
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Remove axis ticks for cleaner look
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle("Comparison of Visualization Methods", fontsize=16)
        plt.tight_layout()
        
        # Save the figure to the PDF
        pdf.savefig(fig)
        plt.close()

def main():
    """
    Main function to demonstrate the visualization fix.
    """
    # Create test data
    df = create_test_grids()
    
    # Plot with default origin
    plot_with_default_origin(df, os.path.join(debug_dir, 'default_visualization.pdf'))
    
    # Plot with fixed origin
    plot_with_lower_origin(df, os.path.join(debug_dir, 'fixed_visualization.pdf'))
    
    # Plot comparison
    plot_comparison(df, os.path.join(debug_dir, 'visualization_comparison.pdf'))
    
    print("Visualization demonstrations created in debug_output directory:")
    print(f"  - {os.path.join(debug_dir, 'default_visualization.pdf')}")
    print(f"  - {os.path.join(debug_dir, 'fixed_visualization.pdf')}")
    print(f"  - {os.path.join(debug_dir, 'visualization_comparison.pdf')}")
    
    print("\nTo fix the visualization issue in vcycle_analysis.pdf, modify the plot_vcycle_results method in:")
    print("  - naviflow_oo/solver/pressure_solver/multigrid.py")
    print("Add origin='lower' to the imshow call around line 94:\n")
    print("    im = ax.imshow(data['data'], cmap='viridis', origin='lower')")

if __name__ == "__main__":
    main() 