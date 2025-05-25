import numpy as np
import os
os.environ["NUMBA_NUM_THREADS"] = "11"
from naviflow_collocated.mesh.mesh_loader import load_mesh  
from naviflow_collocated.core.simple_algorithm import simple_algorithm  
from matplotlib.backends.backend_pdf import PdfPages
from utils.plot_style import plt
from numba import config

print(f"Using {config.NUMBA_NUM_THREADS} threads for momentum relaxation")


# Configure mesh and SIMPLE parameters
#mesh_file = "meshing/experiments/lidDrivenCavity/unstructured/coarse/lidDrivenCavity_unstructured_coarse.msh" 
#mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh" 
mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/medium/lidDrivenCavity_uniform_medium.msh" 
#mesh_file = "meshing/experiments/cylinderFlow/unstructured/fine/cylinderFlow_unstructured_fine.msh"
#mesh_file = "meshing/experiments/cylinderFlow/unstructured/coarse/cylinderFlow_unstructured_coarse.msh"
#mesh_file = "meshing/experiments/sanityCheck/unstructured/coarse/sanityCheck_unstructured_coarse.msh"
#mesh_file ="meshing/experiments/nacaFlow/unstructured/coarse/nacaFlow_unstructured_coarse.msh"
bc_file = "shared_configs/domain/boundaries_lid_driven_cavity.yaml" 
#bc_file = "shared_configs/domain/boundaries_object_flow.yaml" 
mesh = load_mesh(mesh_file, bc_file)

# Determine mesh type from file path
mesh_type = "structured" if "structured" in mesh_file else "unstructured"

alpha_uv = 0.04
alpha_p = 1.0 
max_iter =1000
tolerance = 1e-4
scheme = "QUICK"
limiter = "MUSCL"
PISO = True
PISO_corrections = 1
rho = 1.00
reynolds_number = 5
U = 0.3
#U = 1.0 
#D = 1.0
D = 0.1
#mu = 2*D*U/(3*reynolds_number)
mu = (rho * U * D)/ reynolds_number
#mu = 1.0
# Run SIMPLE
print("Running SIMPLE solver...")
U, p , continuity_field, u_l2norm, v_l2norm, continuity_l2norm, u_residual, v_residual = simple_algorithm(mesh, alpha_uv, alpha_p, rho, mu, max_iter, tolerance, scheme, limiter, PISO, PISO_corrections)
print("SIMPLE solver completed.")

# Plotting
x = mesh.cell_centers[:, 0]
y = mesh.cell_centers[:, 1]
# Compute velocity magnitude
velocity_magnitude = np.sqrt(U[:, 0]**2 + U[:, 1]**2)

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
    cf1 = ax1.tricontourf(x, y, U[:, 0], levels=50, cmap="coolwarm")
    fig1.colorbar(cf1, ax=ax1)
    ax1.set_title("U Velocity")
    ax1.set_aspect('equal', 'box')
    # V velocity
    cf2 = ax2.tricontourf(x, y, U[:, 1], levels=50, cmap="coolwarm")
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
    iterations = range(len(u_l2norm))
    ax_hist.semilogy(iterations, u_l2norm, 'b-', label='$u$-velocity')
    ax_hist.semilogy(iterations, v_l2norm, 'r-', label='$v$-velocity') 
    ax_hist.semilogy(iterations, continuity_l2norm, 'g-', label='Continuity')
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
    cf5 = ax5.tricontourf(x, y, np.abs(u_residual), levels=50, cmap="viridis")
    fig3.colorbar(cf5, ax=ax5)
    ax5.set_title("U-Velocity Residual")
    ax5.set_aspect('equal', 'box')
    # V-velocity residual field
    cf6 = ax6.tricontourf(x, y, np.abs(v_residual), levels=50, cmap="viridis")
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

    # --- Page 4: Comparison with Ghia et al. (1982) ---
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

    fig4 = plt.figure(figsize=(12, 5))
    fig4.suptitle(f"Comparison with Ghia et al. (1982)\nRe = {reynolds_number}, Number of Cells = {n_cells}, Scheme = {scheme}, {mesh_type.capitalize()} Mesh", fontsize=14)
    
    # Plot u-velocity along vertical centerline (x = 0.5)
    ax8 = fig4.add_subplot(121)
    
    # Get points along vertical centerline (x = 0.5)
    centerline_mask = np.abs(x - 0.5) < 0.01
    vertical_y = y[centerline_mask]
    vertical_u = U[centerline_mask, 0]
    
    # Sort by y-coordinate
    sort_idx = np.argsort(vertical_y)
    vertical_y = vertical_y[sort_idx]
    vertical_u = vertical_u[sort_idx]
    
    
    ax8.plot(vertical_u, vertical_y, 'b-', label='Numerical')
    ax8.plot(GHIA_RE_100['u'], GHIA_RE_100['y'], 'ro', label='Ghia et al.')
    ax8.set_xlabel('u-velocity')
    ax8.set_ylabel('y')
    ax8.set_title('u-velocity along x = 0.5')
    ax8.grid(True)
    ax8.legend()

    # Plot v-velocity along horizontal centerline (y = 0.5)
    ax9 = fig4.add_subplot(122)
    
    # Get points along horizontal centerline (y = 0.5)
    centerline_mask = np.abs(y - 0.5) < 0.01
    horizontal_x = x[centerline_mask]
    horizontal_v = U[centerline_mask, 1]
    
    # Sort by x-coordinate
    sort_idx = np.argsort(horizontal_x)
    horizontal_x = horizontal_x[sort_idx]
    horizontal_v = horizontal_v[sort_idx]
    
    
    ax9.plot(horizontal_x, horizontal_v, 'b-', label='Numerical')
    ax9.plot(GHIA_RE_100['x'], GHIA_RE_100['v'], 'ro', label='Ghia et al.')
    ax9.set_xlabel('x')
    ax9.set_ylabel('v-velocity')
    ax9.set_title('v-velocity along y = 0.5')
    ax9.grid(True)
    ax9.legend()



    fig4.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig4)
    plt.close(fig4)