import numpy as np
from naviflow_oo.solver.pressure_solver.helpers.coeff_matrix import get_coeff_mat
from naviflow_oo.solver.pressure_solver.helpers.matrix_free import compute_Ap_product

nx = 10
ny = 10
dx = 1.0
dy = 1.0
rho = 1.0
# Use random values instead of ones
d_u = np.random.rand(nx+1, ny) + 0.1  # Adding 0.1 to ensure positive values
d_v = np.random.rand(nx, ny+1) + 0.1  # Adding 0.1 to ensure positive values
A = get_coeff_mat(nx, ny, dx, dy, rho, d_u, d_v)
p = np.random.rand(nx * ny)
b1 = compute_Ap_product(p, nx, ny, dx, dy, rho, d_u, d_v)
b2 = A @ p
print(np.allclose(b1, b2))  # Should be True within tolerance
# Check symmetry for both matrix-free and coefficient matrix
x = np.random.rand(nx * ny)
y = np.random.rand(nx * ny)

# Matrix-free symmetry check
Ax_mf = compute_Ap_product(x, nx, ny, dx, dy, rho, d_u, d_v)
Ay_mf = compute_Ap_product(y, nx, ny, dx, dy, rho, d_u, d_v)
sym_diff_mf = np.dot(x, Ay_mf) - np.dot(y, Ax_mf)
print(f"Matrix-free symmetry check: xᵀAy - yᵀAx = {sym_diff_mf:.3e}")

# Coefficient matrix symmetry check
Ax_coeff = A @ x
Ay_coeff = A @ y
sym_diff_coeff = np.dot(x, Ay_coeff) - np.dot(y, Ax_coeff)
print(f"Coefficient matrix symmetry check: xᵀAy - yᵀAx = {sym_diff_coeff:.3e}")
