import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from pathlib import Path
from sympy import symbols, sin, cos, pi, lambdify
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix
import matplotlib.tri as tri

def pressure_mms_test(mesh_file: str, bc_file: str):
    print(f"Running MMS pressure test on mesh: {mesh_file}")
    mesh = load_mesh(mesh_file, bc_file)
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    n = len(x)

    # === Define analytical pressure field p(x, y) ===
    x_sym, y_sym = symbols("x y")
    p_expr = sin(4*pi*(x_sym + y_sym)) + cos(4*pi*x_sym*y_sym)

    # === Laplacian of p ===
    from sympy import diff
    laplacian_expr = diff(p_expr, x_sym, 2) + diff(p_expr, y_sym, 2)

    # === Lambdify both expressions
    p_func = lambdify((x_sym, y_sym), p_expr, modules='numpy')
    laplacian_func = lambdify((x_sym, y_sym), laplacian_expr, modules='numpy')

    # === Evaluate exact pressure and its Laplacian at cell centers
    p_exact = p_func(x, y)
    rhs_p = -laplacian_func(x, y)  # sign convention: ∇²p = -rhs

    # === Pin pressure node
    rhs_p = np.ascontiguousarray(rhs_p) * mesh.cell_volumes
    rhs_p[0] = 1.0

    # === Dummy Ap (uniform unit coefficients)
    Ap_u = np.ones(n)
    Ap_v = np.ones(n)

    # === Assemble pressure matrix
    row, col, data = assemble_pressure_correction_matrix(mesh, Ap_u, Ap_v)
    A_p = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    # === Solve for numeric pressure
    p_numeric = spsolve(A_p, rhs_p)

    # === Compute error
    abs_error = np.abs(p_numeric - p_exact)
    l2_error = np.sqrt(np.mean(abs_error ** 2))
    max_error = np.max(abs_error)
    print(f"[Numeric p] L2 Error = {l2_error:.3e}, Max Error = {max_error:.3e}")

    # === Residual check
    residual = A_p @ p_numeric - rhs_p
    residual_l2 = np.linalg.norm(residual)
    print(f"[Residual] L2 Norm = {residual_l2:.3e}")

    # === Output
    outdir = Path("tests/test_output/pressure_mms_analytic_rhs")
    outdir.mkdir(parents=True, exist_ok=True)

    # === Plot all in subplots ===
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    triang = tri.Triangulation(x, y)

    fields = [
        (p_exact, r"$p_{\mathrm{exact}}$"),
        (p_numeric, r"$p_{\mathrm{numeric}}$"),
        (abs_error, r"$|p_{\mathrm{numeric}} - p_{\mathrm{exact}}|$"),
        (residual, r"$\mathbf{A}_p p - \mathbf{rhs}$")
    ]

    for ax, (field, title) in zip(axs.flat, fields):
        cs = ax.tricontourf(triang, field, levels=50, cmap='coolwarm')
        fig.colorbar(cs, ax=ax)
        ax.set_title(title)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(outdir / "pressure_mms_subplots.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    mesh_file = "meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh"
    bc_file = "shared_configs/domain/boundaries_lid_driven_cavity.yaml"
    pressure_mms_test(mesh_file, bc_file)