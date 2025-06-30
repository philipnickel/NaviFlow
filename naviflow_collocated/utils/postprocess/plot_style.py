import matplotlib.pyplot as plt
import scienceplots

# Set the style for all plots
plt.style.use(['science'])  # Removed 'grid' from default style

# Additional style customizations
plt.rcParams.update({
    'figure.figsize': (8, 6),  # Default figure size
    'figure.dpi': 300,         # High resolution
    'savefig.dpi': 300,        # High resolution for saved figures
    'savefig.format': 'pdf',   # Default save format
    'savefig.bbox': 'tight',   # Tight bounding box
    'savefig.pad_inches': 0.1, # Padding around the figure
    'font.size': 10,           # Default font size
    'axes.labelsize': 12,      # Axis label size
    'axes.titlesize': 14,      # Title size
    'xtick.labelsize': 10,     # Tick label size
    'ytick.labelsize': 10,
    'legend.fontsize': 10,     # Legend font size
    'legend.frameon': True,    # Frame around legend
    'legend.framealpha': 0.8,  # Legend frame transparency
    'legend.edgecolor': 'gray',# Legend frame color
    'lines.linewidth': 1.5,    # Line width
    'lines.markersize': 4,     # Marker size
    'image.cmap': 'coolwarm',  # Default colormap
}) 