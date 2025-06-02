import numpy as np
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients
from scipy.sparse import coo_matrix
from naviflow_collocated.linear_solvers.petsc_solver import petsc_solver
from pathlib import Path
from sympy import symbols, sin, cos, pi, lambdify, diff
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix, pressure_correction_loop_term
from naviflow_collocated.core.helpers import interpolate_to_face
import matplotlib.tri as tri
from utils.plot_style import plt

def plot_field(mesh, field, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    try:
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

def run_mms_test(mesh_file: str, bc_file: str, mesh_type="structured"):
    print(f"Running MMS pressure test on {mesh_type} mesh: {mesh_file}")
    mesh = load_mesh(mesh_file, bc_file)
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    n = len(x)

    # === Define analytical pressure field p(x, y) ===
    x_sym, y_sym = symbols("x y")
    p_expr = cos(2*pi*(x_sym)) * cos(2*pi*(y_sym))

    # === Laplacian of p ===
    laplacian_expr = diff(p_expr, x_sym, 2) + diff(p_expr, y_sym, 2)

    # === Lambdify both expressions
    p_func = lambdify((x_sym, y_sym), p_expr, modules='numpy')
    laplacian_func = lambdify((x_sym, y_sym), laplacian_expr, modules='numpy')

    # === Evaluate exact pressure and its Laplacian at cell centers
    p_exact = p_func(x, y)
    rhs_p = -laplacian_func(x, y)  # sign convention: ∇²p = -rhs

    # === Pin pressure node
    rhs_p = np.ascontiguousarray(rhs_p) * mesh.cell_volumes
    #cell_centers = mesh.cell_centers
    #pinned_cell_coords = [0.5, 0.5]
    #pinned_cell = np.argmin(np.linalg.norm(cell_centers - pinned_cell_coords, axis=1))

    # === Assemble pressure matrix
    row, col, data = assemble_pressure_correction_matrix(mesh, rho=1.0)
    A_p = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    p_numeric, _, _ = petsc_solver(A_p, rhs_p, remove_nullspace=True, tolerance=1e-20)

    # === Compute error
    error = p_numeric - p_exact
    l2_error = np.linalg.norm(error)
    max_error = np.max(error)
    print(f"[Numeric p] L2 Error = {l2_error:.3e}, Max Error = {max_error:.3e}")

    # === Residual check
    residual = A_p @ p_exact - rhs_p
    residual_l2 = np.linalg.norm(residual)
    print(f"[Residual] L2 Norm = {residual_l2:.3e}")

    # === Output
    outdir = Path("tests/test_output/MMS_solutions")
    outdir.mkdir(parents=True, exist_ok=True)

    # === Plot all in subplots ===
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(f"Pressure MMS Solution Fields - {mesh_type.capitalize()} Mesh", fontsize=20)
    
    fields = [
        (p_numeric, r"$p_{\mathrm{num}}$"),
        (p_exact, r"$p_{\mathrm{exact}}$"),
        (error, r"$|p_{\mathrm{num}} - p_{\mathrm{exact}}|$"),
        (residual, r"$\mathbf{A}_p p - \mathbf{rhs}$")
    ]

    for ax, (field, title) in zip(axs.flat, fields):
        plot_field(mesh, field, ax=ax, title=title)

    plt.tight_layout()
    plt.savefig(outdir / f"pressure_mms_solution_{mesh_type}.pdf", dpi=300)
    plt.close()

def run_convergence_study(mesh_files: list[str], bc_file: str, ax=None, marker=None, mesh_type="structured"):
    """
    Run convergence study for pressure MMS test on multiple mesh resolutions.
    
    Args:
        mesh_files: List of mesh file paths in order of increasing resolution
        bc_file: Path to boundary conditions file
        ax: Optional matplotlib axis for plotting
        marker: Optional marker style for the plot
        mesh_type: Type of mesh ("structured" or "unstructured")
    """
    print(f"\nRunning Pressure MMS Convergence Study - {mesh_type.capitalize()} Mesh")
    print("======================================")
    
    # Store results for each mesh
    h_values = []  # Characteristic mesh size
    l2_errors = []  # L2 error norms
    residual_l2s = []  # L2 residual norms
    
    for mesh_file in mesh_files:
        print(f"\nProcessing {mesh_type} mesh: {mesh_file}")
        mesh = load_mesh(mesh_file, bc_file)
        
        # Calculate characteristic mesh size (average cell size)
        h = np.sqrt(np.mean(mesh.cell_volumes))
        h_values.append(h)
        
        # Run MMS test and get errors
        x = mesh.cell_centers[:, 0]
        y = mesh.cell_centers[:, 1]
        n = len(x)
        
        # Define analytical pressure field
        x_sym, y_sym = symbols("x y")
        p_expr = cos(2*pi*(x_sym)) * cos(2*pi*(y_sym))
        laplacian_expr = diff(p_expr, x_sym, 2) + diff(p_expr, y_sym, 2)
        
        p_func = lambdify((x_sym, y_sym), p_expr, modules='numpy')
        laplacian_func = lambdify((x_sym, y_sym), laplacian_expr, modules='numpy')
        
        p_exact = p_func(x, y)
        rhs_p = -laplacian_func(x, y)
        rhs_p = np.ascontiguousarray(rhs_p) * mesh.cell_volumes
        
        # Assemble and solve
        row, col, data = assemble_pressure_correction_matrix(mesh, rho=1.0)
        A_p = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
        
        p_numeric, _, _ = petsc_solver(A_p, rhs_p, remove_nullspace=True, tolerance=1e-20)
        
        # Compute errors
        error = np.abs(p_numeric - p_exact)
        l2_error = np.sqrt(np.sum(error**2) / n)
        max_error = np.max(error)
        print(f"[Numeric p] L2 Error = {l2_error:.3e}, Max Error = {max_error:.3e}")
        
        # === Residual check
        residual = A_p @ p_exact- rhs_p
        residual_l2 = np.linalg.norm(residual)
        
        l2_errors.append(l2_error)
        residual_l2s.append(residual_l2)
        
        print(f"Mesh size (h) = {h:.3e}")
        print(f"L2 Error = {l2_error:.3e}")
        print(f"Residual L2 = {residual_l2:.3e}")
    
    h_values = np.array(h_values)
    l2_errors = np.array(l2_errors)
    
    # Calculate convergence rate using least squares fit
    from numpy.linalg import lstsq
    X = np.log(h_values).reshape(-1, 1)
    X = np.hstack([X, np.ones_like(X)])
    y = np.log(l2_errors)
    (p, _), *_ = lstsq(X, y, rcond=None)
    
    print(f"\nObserved convergence rate for {mesh_type} mesh (global fit): p ≈ {p:.2f}")
    
    if ax is not None:
        ax.loglog(h_values, l2_errors, 
                 label=f"MMS - observed order: {p:.2f}", 
                 marker=marker)
    
    return l2_errors, residual_l2s

if __name__ == "__main__":
    structured_uniform = {
        "coarse": "meshing/experiments/lidDrivenCavity/structuredUniform/coarse/lidDrivenCavity_uniform_coarse.msh",
        "medium": "meshing/experiments/lidDrivenCavity/structuredUniform/medium/lidDrivenCavity_uniform_medium.msh",
        "fine": "meshing/experiments/lidDrivenCavity/structuredUniform/fine/lidDrivenCavity_uniform_fine.msh",
    }

    
    bc_file = "shared_configs/domain/boundaries_lidDrivenCavity.yaml"
    
    # Run individual MMS tests with different numbers of non-orthogonal corrections
    run_mms_test(structured_uniform["coarse"], bc_file, mesh_type="structured")
    
    
    # Run convergence study
    
    
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Run convergence study for structured mesh with different numbers of non-orthogonal corrections
    
    
    errors_structured = run_convergence_study(
        [structured_uniform["coarse"], structured_uniform["medium"], structured_uniform["fine"]],
        bc_file,
        ax=ax,
        marker='o',
        mesh_type="structured",
    )


    
    # Add reference line for second order
    hs = np.array([np.sqrt(np.mean(load_mesh(f, bc_file).cell_volumes)) for f in [
        structured_uniform["coarse"],
        structured_uniform["medium"],
        structured_uniform["fine"]
    ]])
    ref_slope = np.min(errors_structured)*2 * (hs / hs[0])**2  # Normalize ref slope to first error value
    
    ax.loglog(hs, ref_slope, 'k--', label=r'$\mathcal{O}(h^2)$')
    
    ax.grid(True, which="both")
    ax.set_xlabel(r"Grid size $h$")
    ax.set_ylabel(r"L2 Error")
    ax.set_title("Order of Accuracy Poisson like equation", fontsize=14)
    ax.legend(loc="lower right")
    
    Path("tests/test_output/MMS_convergence").mkdir(parents=True, exist_ok=True)
    plt.savefig("tests/test_output/MMS_convergence/pressure_convergence_plot.pdf", dpi=300)
    plt.close()    
    
    