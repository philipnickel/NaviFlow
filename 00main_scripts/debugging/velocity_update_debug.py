import numpy as np
from naviflow_staggered.preprocessing.mesh.structured import StructuredUniform
from naviflow_staggered.constructor.boundary_conditions import BoundaryConditionManager
from naviflow_staggered.solver.velocity_solver.standard import StandardVelocityUpdater

# --- Setup Mesh and Fields ---
nx, ny = 31, 31
mesh = StructuredUniform(nx, ny)
n_cells = mesh.n_cells

# --- Boundary Conditions ---
bc_manager = BoundaryConditionManager()
bc_manager.set_condition("top", "velocity", {"u": 1.0, "v": 0.0})
bc_manager.set_condition("bottom", "wall", {})
bc_manager.set_condition("left", "wall", {})
bc_manager.set_condition("right", "wall", {})

# --- Inputs: artificial u*, v*, p', and d ---
u_star = 0.1 * np.ones(n_cells)
v_star = 0.05 * np.ones(n_cells)
p_prime = np.random.rand(n_cells) * 0.01  # small pressure correction

# Diagonal coefficients (assume realistic range)
d_u = np.full(n_cells, 0.03)
d_v = np.full(n_cells, 0.04)

# --- Velocity Updater ---
updater = StandardVelocityUpdater()
u_corr, v_corr = updater.update_velocity(mesh, u_star, v_star, p_prime, d_u, d_v, bc_manager)

# --- Output Check ---
print("u_corr stats:", np.min(u_corr), np.max(u_corr), np.mean(u_corr))
print("v_corr stats:", np.min(v_corr), np.max(v_corr), np.mean(v_corr))

# Check for NaNs/Infs
assert not np.isnan(u_corr).any() and not np.isinf(u_corr).any(), "NaNs/Infs in u_corr!"
assert not np.isnan(v_corr).any() and not np.isinf(v_corr).any(), "NaNs/Infs in v_corr!"
