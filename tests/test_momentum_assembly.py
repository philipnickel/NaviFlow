import os
os.environ["NUMBA_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "4"





import sympy as sp
from sympy.utilities.lambdify import lambdify
import numpy as np
from utils.plot_style import plt
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.assembly.momentumMatrix import assemble_diffusion_convection_matrix
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients

def plot_field(mesh, field, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    try:
        import matplotlib.tri as tri
        triang = tri.Triangulation(x, y)
        cs = ax.tricontourf(triang, field, levels=30, cmap="viridis")
        plt.colorbar(cs, ax=ax, shrink=0.75)
        if title:
            ax.set_title(title)
    except Exception as e:
        print(f"Failed to plot tricontourf: {e}")
        sc = ax.scatter(x, y, c=field, cmap="viridis", s=30, edgecolor="k", linewidth=0.3)
        plt.colorbar(sc, ax=ax, shrink=0.75)
        if title:
            ax.set_title(title)
    ax.set_aspect("equal")

def run_mms_test(mesh_file, bc_file, u_exact_fn, u_field_fn, rhs_fn, grad_fn, mu, rho, tag_prefix, component_idx=0, beta=0.0):
    mesh = load_mesh(mesh_file, bc_file)

    # Get the exact solution and ensure it has the right shape
    phi_exact = u_exact_fn[component_idx](mesh.cell_centers)
    if np.isscalar(phi_exact) or phi_exact.size == 1:
        # If it's a scalar value, broadcast to all cells
        phi_exact = np.full(mesh.cell_centers.shape[0], float(phi_exact))
    phi_exact = np.atleast_1d(np.asarray(phi_exact).ravel())
    phi_exact = np.ascontiguousarray(phi_exact)
    
    u_field = u_field_fn(mesh.cell_centers)
    u_field = np.ascontiguousarray(u_field)
    #grad_phi_num = compute_cell_gradients(mesh, phi_exact)
    grad_phi = np.ascontiguousarray(grad_fn(mesh.cell_centers), dtype=np.float64)


    row, col, data, b_correction = assemble_diffusion_convection_matrix(
        mesh, grad_phi, u_field, 
        rho, mu, component_idx, phi=phi_exact, beta=beta
    )

    A = coo_matrix((data, (row, col)), shape=(mesh.cell_centers.shape[0],) * 2).tocsr()
    rhs = rhs_fn(mesh.cell_centers) * mesh.cell_volumes + b_correction

    phi_numeric = spsolve(A, rhs)

    # Compute the residual A·phi_exact - rhs_fn(mesh.cell_centers)*vol - b_correction
    assert np.all(np.isfinite(phi_exact))
    assert np.all(np.isfinite(u_field))
    assert np.all(np.isfinite(grad_phi))
    assert np.all(np.isfinite(A.data))
    assert np.all(np.isfinite(rhs))
    
    # Make sure phi_exact matches the shape expected by A (same as the RHS)
    if phi_exact.shape != rhs.shape:
        phi_exact = phi_exact.reshape(rhs.shape)
        
    Aphi_exact = A.dot(phi_exact)
    residual = Aphi_exact - rhs_fn(mesh.cell_centers) * mesh.cell_volumes - b_correction
    flux_imbalance = np.sum(residual)

    err = np.abs(phi_numeric - phi_exact)
    max_err, l2_err = np.max(err), np.sqrt(np.mean(err ** 2))
    print(f"[{tag_prefix}] Max error: {max_err:.2e}, L2 error: {l2_err:.2e}, Flux inbalance: {flux_imbalance:.2e}")

    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(f"MMS Solution Fields - {tag_prefix}", fontsize=20)
    plot_field(mesh, phi_numeric, ax=axs[0, 0], title=r"$\phi_{\mathrm{num}}$")
    plot_field(mesh, phi_exact, ax=axs[0, 1], title=r"$\phi_{\mathrm{exact}}$") 
    plot_field(mesh, err, ax=axs[1, 0], title=r"$|\phi_{\mathrm{num}} - \phi_{\mathrm{exact}}|$")
    plot_field(mesh, residual, ax=axs[1, 1], title=r"$\mathbf{A}\phi_{\mathrm{exact}} - \mathbf{rhs_{\mathrm{MMS}}} - \mathbf{b}_{\mathrm{corr}}$")
    plt.tight_layout()
    outdir = Path("tests/test_output/MMS_solutions")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"mms_{tag_prefix}.png", dpi=300)
    plt.close()


# The following test follows the Method of Manufactured Solutions (MMS) approach
# to verify spatial convergence of the numerical scheme by comparing numerical
# and exact solutions on a sequence of refined meshes.
def run_convergence_study(mesh_files, bc_file, u_exact_fn, u_field_fn, rhs_fn, grad_fn, mu, rho, tag_prefix, component_idx=0, beta=0.0, ax=None, marker=None):
    hs = []
    errors = []

    for mesh_file in mesh_files:
        mesh = load_mesh(mesh_file, bc_file)
        h = np.sqrt(np.mean(mesh.cell_volumes))
        hs.append(h)

        phi_exact = u_exact_fn[component_idx](mesh.cell_centers)
        if np.isscalar(phi_exact) or phi_exact.size == 1:
            # If it's a scalar value, broadcast to all cells
            phi_exact = np.full(mesh.cell_centers.shape[0], float(phi_exact))
        phi_exact = np.atleast_1d(np.asarray(phi_exact).ravel())
        phi_exact = np.ascontiguousarray(phi_exact)
        
        u_field = u_field_fn(mesh.cell_centers)
        u_field = np.ascontiguousarray(u_field)
        grad_phi = compute_cell_gradients(mesh, phi_exact)



        row, col, data, b_correction = assemble_diffusion_convection_matrix(
            mesh, grad_phi, u_field,
            rho, mu, component_idx, phi=phi_exact, beta=beta
        )


        A = coo_matrix((data, (row, col)), shape=(mesh.cell_centers.shape[0],) * 2).tocsr()
        diag = A.diagonal()
        rhs = rhs_fn(mesh.cell_centers) * mesh.cell_volumes + b_correction

        phi_numeric = spsolve(A, rhs)
        
        # Make sure phi_exact matches the shape expected by A (same as the RHS)
        if phi_exact.shape != rhs.shape:
            phi_exact = phi_exact.reshape(rhs.shape)
            
        residual = A.dot(phi_exact) - rhs_fn(mesh.cell_centers) * mesh.cell_volumes - b_correction
        flux_imbalance = np.sum(residual)

        err = np.abs(phi_numeric - phi_exact)
        l2_err = np.sqrt(np.mean(err**2))
        errors.append(l2_err)

        print(f"[{tag_prefix}] h = {h:.4f}, L2 error = {l2_err:.3e}, Flux imbalance = {flux_imbalance:.3e}")

    hs = np.array(hs)
    errors = np.array(errors)
    from numpy.linalg import lstsq
    X = np.log(hs).reshape(-1, 1)
    X = np.hstack([X, np.ones_like(X)])
    y = np.log(errors)
    (p, _), *_ = lstsq(X, y, rcond=None)

    print(f"\nObserved convergence rate (global fit): {tag_prefix} --> p ≈ {p:.2f}")

    if ax is not None:
        ax.loglog(hs, errors, label=rf"{tag_prefix} (p $\approx$ {p:.2f})", marker=marker)
    return errors


# === MMS Functions ===
# The exact solution u_sin is chosen to be smooth and not exactly representable by the discrete scheme,
# to validate the convergence behavior of the numerical method.
def generate_mms_functions(expr_str, mu, rho):
    x, y = sp.symbols("x y")
    u_x_expr = sp.sympify(expr_str[0])
    u_y_expr = sp.sympify(expr_str[1])
    phi_expr = u_x_expr  # or u_y_expr depending on which component you're testing

    grad_exprs = [sp.diff(phi_expr, var) for var in (x, y)]
    laplacian_expr = sum(sp.diff(phi_expr, var, 2) for var in (x, y))
    conv_term_expr = rho * (u_x_expr * grad_exprs[0] + u_y_expr * grad_exprs[1])
    diff_term_expr = mu * laplacian_expr
    rhs_expr = conv_term_expr - diff_term_expr

    # Vectorized lambdify with explicit numpy module
    u_x_func = lambdify((x, y), u_x_expr, modules="numpy")
    u_y_func = lambdify((x, y), u_y_expr, modules="numpy")
    grad_func = lambdify((x, y), grad_exprs, modules="numpy")
    rhs_func = lambdify((x, y), rhs_expr, modules="numpy")

    def eval_at(f):
        return lambda xy: np.column_stack([
            np.broadcast_to(g, (xy.shape[0],)) if np.isscalar(g) or np.ndim(g) == 0 else g
            for g in f(xy[:, 0], xy[:, 1])
        ])

    def u_field(xy):
        ux = u_x_func(xy[:, 0], xy[:, 1])
        uy = u_y_func(xy[:, 0], xy[:, 1])
        ux = np.broadcast_to(ux, (xy.shape[0],)) if np.isscalar(ux) or np.ndim(ux) == 0 else ux
        uy = np.broadcast_to(uy, (xy.shape[0],)) if np.isscalar(uy) or np.ndim(uy) == 0 else uy
        return np.column_stack([ux, uy])
    u_x = lambda xy: u_x_func(xy[:, 0], xy[:, 1])
    u_y = lambda xy: u_y_func(xy[:, 0], xy[:, 1])
    def grad(xy):
        vals = grad_func(xy[:, 0], xy[:, 1])
        vals = [np.full((xy.shape[0],), v) if np.isscalar(v) or np.ndim(v) == 0 else np.asarray(v).ravel() for v in vals]
        return np.column_stack(vals)
    rhs = lambda xy: rhs_func(xy[:, 0], xy[:, 1])

    return [u_x, u_y], u_field, grad, rhs




# === Run Tests ===
if __name__ == "__main__":
    structured_uniform = {
        "coarse": "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh",
        "medium": "meshing/experiments/lidDrivenCavity/structuredUniform/medium/lidDrivenCavity_uniform_medium.msh",
        "fine": "meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh",
    }
    unstructured = {
        "coarse": "meshing/experiments/lidDrivenCavity/unstructured/coarse/lidDrivenCavity_unstructured_coarse.msh",
        "medium": "meshing/experiments/lidDrivenCavity/unstructured/medium/lidDrivenCavity_unstructured_medium.msh",
        "fine": "meshing/experiments/lidDrivenCavity/unstructured/fine/lidDrivenCavity_unstructured_fine.msh",
    }

    # === Additional MMS Cases ===
    mms_cases = {
        "Sinusoidal": ("-cos(pi*x)*sin(pi*y)", "sin(pi*x)*cos(pi*y)"),
        "Quadratic": ("x**2 + y**2", "x*y"),
        #"Backwards": ("-1.0 + x*0.0", "0.0 + y*0.0"),
        #"Uniform": ("1.0 + x*0.0", "0.0 + y*0.0"),
        #"Linear": ("x", "y")
    }
    BC_files = {
        "Sinusoidal": "shared_configs/domain/sanityChecks/sanityCheckSIN.yaml",
        "Quadratic": "shared_configs/domain/sanityChecks/sanityCheckQUAD.yaml",
        #"Backwards": "shared_configs/domain/sanityChecks/sanityCheckBackwards.yaml",
        #"Uniform": "shared_configs/domain/sanityChecks/sanityCheckUniformFlow.yaml",
        #"Linear": "shared_configs/domain/sanityChecks/sanityCheckLinear.yaml"
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    marker_cycle = iter(['o', 's', '^', 'D', 'v', 'p', '*', 'x'])
    time_start = time.time()

    for tag, expr in mms_cases.items():
        mu = 0.01
        rho = 1.0
        beta = 1.0
        u_fn, u_field_fn, grad_fn, rhs_fn = generate_mms_functions(expr, mu=mu, rho=rho)
        bc_file = BC_files[tag]
        """

        run_mms_test(
            structured_uniform["fine"],
            bc_file,
            u_fn, u_field_fn, rhs_fn, grad_fn, mu, rho,
            tag_prefix=f"{tag} structured",
            beta=beta,
        )
        run_mms_test(
            unstructured["fine"],
            bc_file,
            u_fn, u_field_fn, rhs_fn, grad_fn, mu, rho,
            tag_prefix=f"{tag} unstructured",
            beta=beta,
        )

        """ 
        # Uncomment to run convergence studies
        errors = run_convergence_study(
            [structured_uniform["coarse"], structured_uniform["medium"], structured_uniform["fine"]],
            bc_file,
            u_fn, u_field_fn, rhs_fn, grad_fn, mu, rho,
            tag_prefix=f"{tag}_structured",
            beta=beta,
            ax=ax,
            marker=next(marker_cycle)
        )
        errors = run_convergence_study(
            [unstructured["coarse"], unstructured["medium"], unstructured["fine"]],
            bc_file,
            u_fn, u_field_fn, rhs_fn, grad_fn, mu, rho,
            tag_prefix=f"{tag}_unstructured",
            beta=beta,
            ax=ax,
            marker=next(marker_cycle)
        )

    hs = np.array([np.sqrt(np.mean(load_mesh(f, next(iter(BC_files.values()))).cell_volumes)) for f in [
        structured_uniform["coarse"],
        structured_uniform["medium"],
        structured_uniform["fine"]
    ]])
    ref_slope = np.min(errors)*15 * (hs / hs[0])#**2  # Normalize ref slope to first error value

    ax.loglog(hs, ref_slope, 'k--', label='First-order (ref)')

    ax.grid(True, which="both")
    ax.set_xlabel(r"Grid size $h$")
    ax.set_ylabel(r"L2 Error")
    ax.set_title("Order of Accuracy", fontsize=14)
    ax.legend(loc="lower right")
    Path("tests/test_output/MMS_convergence").mkdir(parents=True, exist_ok=True)
    plt.savefig("tests/test_output/MMS_convergence/convergence_plot_combined.pdf", dpi=300)
    plt.close()
    print(f"Time taken: {time.time() - time_start:.2f} seconds")