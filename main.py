import numpy as np
import os
from naviflow_collocated.mesh.mesh_loader import load_mesh  
from naviflow_collocated.core.simple_algorithm import simple_algorithm  
from matplotlib.backends.backend_pdf import PdfPages
from utils.plot_style import plt

# Configure mesh and SIMPLE parameters
mesh_file = "meshing/experiments/lidDrivenCavity/unstructured/coarse/lidDrivenCavity_unstructured_coarse.msh" 
#mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh" 
bc_file = "shared_configs/domain/boundaries_lid_driven_cavity.yaml" 
mesh = load_mesh(mesh_file, bc_file)

# Determine mesh type from file path
mesh_type = "structured" if "structured" in mesh_file else "unstructured"

alpha_uv = 0.6
alpha_p = 0.4 
reynolds_number = 100
max_iter =10000
tolerance = 1e-4
scheme = "TVD"
limiter = "MUSCL"

# Run SIMPLE
print("Running SIMPLE solver...")
u_field, p , continuity_field, res_v, res_u, u_residuals, v_residuals, continuity_residuals = simple_algorithm(mesh, alpha_uv, alpha_p, reynolds_number, max_iter, tolerance, scheme, limiter)
print("SIMPLE solver completed.")

# Plotting
x = mesh.cell_centers[:, 0]
y = mesh.cell_centers[:, 1]
# Compute velocity magnitude
velocity_magnitude = np.sqrt(u_field[:, 0]**2 + u_field[:, 1]**2)

# Get number of cells
n_cells = len(mesh.cell_centers)

# Prepare PDF file name
pdf_filename = f"plots/LDC_Re{reynolds_number}_ncells{n_cells}_{scheme}_{mesh_type}.pdf"
os.makedirs("plots", exist_ok=True)

with PdfPages(pdf_filename) as pdf:
    # --- Page 1: Flow Field Visualization ---
    fig1 = plt.figure(figsize=(15, 10))
    fig1.suptitle(f"Lid-Driven Cavity Flow Analysis\nRe = {reynolds_number}, Number of Cells = {n_cells}, Scheme = {scheme}, {mesh_type.capitalize()} Mesh", fontsize=16, y=0.98)
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[0, 1])
    ax3 = fig1.add_subplot(gs[1, 0])
    ax4 = fig1.add_subplot(gs[1, 1])
    # U velocity
    cf1 = ax1.tricontourf(x, y, u_field[:, 0], levels=50, cmap="coolwarm")
    fig1.colorbar(cf1, ax=ax1)
    ax1.set_title("U Velocity")
    ax1.set_aspect('equal', 'box')
    # V velocity
    cf2 = ax2.tricontourf(x, y, u_field[:, 1], levels=50, cmap="coolwarm")
    fig1.colorbar(cf2, ax=ax2)
    ax2.set_title("V Velocity")
    ax2.set_aspect('equal', 'box')
    # Velocity magnitude
    cf3 = ax3.tricontourf(x, y, velocity_magnitude, levels=50, cmap="coolwarm")
    fig1.colorbar(cf3, ax=ax3)
    ax3.set_title("Velocity Magnitude")
    ax3.set_aspect('equal', 'box')
    # Pressure (no streamlines)
    cf4 = ax4.tricontourf(x, y, p, levels=50, cmap="coolwarm")
    fig1.colorbar(cf4, ax=ax4)
    ax4.set_title("Pressure")
    ax4.set_aspect('equal', 'box')
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig1)
    plt.close(fig1)

    # --- Page 2: Residual History ---
    fig2 = plt.figure(figsize=(10, 6))
    fig2.suptitle(f"Residual History\nRe = {reynolds_number}, Number of Cells = {n_cells}, Scheme = {scheme}, {mesh_type.capitalize()} Mesh", fontsize=14)
    ax_hist = fig2.add_subplot(1,1,1)
    iterations = range(len(u_residuals))
    ax_hist.semilogy(iterations, u_residuals, 'b-', label='$u$-velocity')
    ax_hist.semilogy(iterations, v_residuals, 'r-', label='$v$-velocity') 
    ax_hist.semilogy(iterations, continuity_residuals, 'g-', label='Continuity')
    ax_hist.grid(True)
    ax_hist.set_xlabel('Iteration')
    ax_hist.set_ylabel('Residual')
    ax_hist.set_title('Residual History')
    ax_hist.legend()
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig2)
    plt.close(fig2)

    # --- Page 3: Residual Fields ---
    fig3 = plt.figure(figsize=(15, 5))
    fig3.suptitle(f"Residual Fields\nRe = {reynolds_number}, Number of Cells = {n_cells}, Scheme = {scheme}, {mesh_type.capitalize()} Mesh", fontsize=14)
    gs2 = plt.GridSpec(1, 3)
    ax5 = fig3.add_subplot(gs2[0, 0])
    ax6 = fig3.add_subplot(gs2[0, 1])
    ax7 = fig3.add_subplot(gs2[0, 2])
    # U-velocity residual field
    cf5 = ax5.tricontourf(x, y, np.abs(res_u), levels=50, cmap="viridis")
    fig3.colorbar(cf5, ax=ax5)
    ax5.set_title("U-Velocity Residual")
    ax5.set_aspect('equal', 'box')
    # V-velocity residual field
    cf6 = ax6.tricontourf(x, y, np.abs(res_v), levels=50, cmap="viridis")
    fig3.colorbar(cf6, ax=ax6)
    ax6.set_title("V-Velocity Residual")
    ax6.set_aspect('equal', 'box')
    # Continuity residual field
    cf7 = ax7.tricontourf(x, y, np.abs(continuity_field), levels=50, cmap="viridis")
    fig3.colorbar(cf7, ax=ax7)
    ax7.set_title("Mass Flux Imbalance")
    ax7.set_aspect('equal', 'box')
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig3)
    plt.close(fig3)

