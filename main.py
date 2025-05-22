import numpy as np
import matplotlib.pyplot as plt
import os
from naviflow_collocated.mesh.mesh_loader import load_mesh  
from naviflow_collocated.core.simple_algorithm import simple_algorithm  

# Configure mesh and SIMPLE parameters
#mesh_file = "meshing/experiments/lidDrivenCavity/unstructured/coarse/lidDrivenCavity_unstructured_coarse.msh" 
mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh" 
bc_file = "shared_configs/domain/boundaries_lid_driven_cavity.yaml" 
mesh = load_mesh(mesh_file, bc_file)

alpha_uv = 0.6
alpha_p = 0.1 
reynolds_number = 10
max_iter =1000
tolerance = 1e-10
scheme = "TVD"
limiter = "MUSCL"

# Run SIMPLE
print("Running SIMPLE solver...")
u_field, p = simple_algorithm(mesh, alpha_uv, alpha_p, reynolds_number, max_iter, tolerance, scheme, limiter)
print("SIMPLE solver completed.")

# Plotting
x = mesh.cell_centers[:, 0]
y = mesh.cell_centers[:, 1]
# Compute velocity magnitude
velocity_magnitude = np.sqrt(u_field[:, 0]**2 + u_field[:, 1]**2)

# Create 2x2 subplot figure
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("SIMPLE Solver Results", fontsize=16)

# U velocity
cf1 = axs[0, 0].tricontourf(x, y, u_field[:, 0], levels=50, cmap="coolwarm")
fig.colorbar(cf1, ax=axs[0, 0])
axs[0, 0].set_title("U Velocity")

# V velocity
cf2 = axs[0, 1].tricontourf(x, y, u_field[:, 1], levels=50, cmap="coolwarm")
fig.colorbar(cf2, ax=axs[0, 1])
axs[0, 1].set_title("V Velocity")

# Velocity magnitude
cf3 = axs[1, 0].tricontourf(x, y, velocity_magnitude, levels=50, cmap="viridis")
fig.colorbar(cf3, ax=axs[1, 0])
axs[1, 0].set_title("Velocity Magnitude")

# Pressure
cf4 = axs[1, 1].tricontourf(x, y, p, levels=50, cmap="coolwarm")
fig.colorbar(cf4, ax=axs[1, 1])
axs[1, 1].set_title("Pressure")

# Improve layout
for ax in axs.flat:
    ax.set_aspect('equal', 'box')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/simple_solver_results.png", dpi=300)
