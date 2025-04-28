import numpy as np
import matplotlib.pyplot as plt

# Import your code
from naviflow_oo.preprocessing.mesh.unstructured import UnstructuredRefined
from naviflow_oo.constructor.properties.fluid import FluidProperties
from naviflow_oo.solver.momentum_solver.AMG_solver import AMGMomentumSolver
from naviflow_oo.constructor.boundary_conditions import BoundaryConditionManager

# 1. Create unstructured refined mesh
print("Creating unstructured refined mesh...")
length_x, length_y = 1.0, 1.0
mesh = UnstructuredRefined(
    mesh_size_walls=0.05, 
    mesh_size_lid=0.02, 
    mesh_size_center=0.1,
    xmin=0.0, xmax=length_x, ymin=0.0, ymax=length_y
)
n_cells = mesh.n_cells
print(f"Mesh created with {n_cells} cells.")

# 2. Create fluid properties
reynolds = 100
fluid = FluidProperties(reynolds_number=reynolds, characteristic_velocity=1.0)

# 3. Initialize collocated fields (1D arrays for mesh-agnostic)
u = np.zeros(n_cells)
v = np.zeros(n_cells)
p = np.zeros(n_cells)

# 4. Boundary conditions (collocated version)
bc_manager = BoundaryConditionManager()
bc_manager.set_condition('top', 'velocity', {'u': 1.0, 'v': 0.0})
bc_manager.set_condition('bottom', 'wall')
bc_manager.set_condition('left', 'wall')
bc_manager.set_condition('right', 'wall')

# 5. Create momentum solver (collocated)
momentum_solver = AMGMomentumSolver(discretization_scheme='power_law')

# 6. Solve u-momentum
print("Solving u-momentum...")
u_star, d_u, residual_info_u = momentum_solver.solve_u_momentum(
    mesh, fluid, u, v, p, relaxation_factor=0.7,
    boundary_conditions=bc_manager, return_dict=True
)

# 7. Solve v-momentum
print("Solving v-momentum...")
v_star, d_v, residual_info_v = momentum_solver.solve_v_momentum(
    mesh, fluid, u, v, p, relaxation_factor=0.7,
    boundary_conditions=bc_manager, return_dict=True
)

# 8. Print residuals
print(f"U-momentum residual (relative): {residual_info_u['rel_norm']:.2e}")
print(f"V-momentum residual (relative): {residual_info_v['rel_norm']:.2e}")

# 9. Plot results (using tricontourf for unstructured mesh)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Get cell centers
cell_centers = mesh.get_cell_centers()
x_centers = cell_centers[:, 0]
y_centers = cell_centers[:, 1]

# --- Plot u_star using tricontourf ---
levels_u = np.linspace(u_star.min(), u_star.max(), 15)
cf1 = axs[0].tricontourf(x_centers, y_centers, u_star, levels=levels_u, cmap='viridis', extend='both')
# Overlay mesh edges (optional, can be slow for large meshes)
# mesh.plot(ax=axs[0]) # Use the mesh plot method if available
axs[0].set_title('Intermediate u velocity (u*)')
axs[0].set_aspect('equal', adjustable='box')
fig.colorbar(cf1, ax=axs[0])

# --- Plot v_star using tricontourf ---
# Add small epsilon to min/max if they are equal to avoid contour error
min_v, max_v = v_star.min(), v_star.max()
if np.isclose(min_v, max_v):
    levels_v = np.linspace(min_v - 1e-9, max_v + 1e-9, 15)
else:
    levels_v = np.linspace(min_v, max_v, 15)
cf2 = axs[1].tricontourf(x_centers, y_centers, v_star, levels=levels_v, cmap='viridis', extend='both')
# mesh.plot(ax=axs[1]) # Overlay mesh edges
axs[1].set_title('Intermediate v velocity (v*)')
axs[1].set_aspect('equal', adjustable='box')
fig.colorbar(cf2, ax=axs[1])


plt.tight_layout()
plt.show()
