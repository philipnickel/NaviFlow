import numpy as np
#from naviflow_collocated.discretization.gradient.gauss import compute_cell_gradients
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients
from scipy.sparse import coo_matrix
from naviflow_collocated.linear_solvers.petsc_solver import petsc_solver
from pathlib import Path
from sympy import symbols, sin, cos, pi, lambdify, diff
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix, pressure_correction_loop_term
from naviflow_collocated.core.helpers import interpolate_to_face, get_unique_cells_from_faces
import matplotlib.tri as tri
from utils.plot_style import plt

# Set number of non-orthogonal corrections globally
#N_NONORTHO_CORRECTIONS = 0

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


def compute_mesh_quality_metrics(mesh):
    """
    Compute mesh quality metrics for diagnostic purposes.
    """
    n_internal = mesh.internal_faces.shape[0]
    orthogonality_errors = []
    skewness_magnitudes = []
    
    # Internal faces only
    for i in range(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]
        
        # Orthogonality: angle between face normal and cell-center connection
        S_f = mesh.vector_S_f[f]
        d_CF = mesh.vector_d_CE[f]
        
        S_f_norm = np.linalg.norm(S_f)
        d_CF_norm = np.linalg.norm(d_CF)
        
        if S_f_norm > 1e-12 and d_CF_norm > 1e-12:
            cos_angle = np.dot(S_f, d_CF) / (S_f_norm * d_CF_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(np.abs(cos_angle))
            angle_deg = np.degrees(angle_rad)
            orthogonality_errors.append(angle_deg)
        
        # Skewness: magnitude of skewness vector
        skew_mag = np.linalg.norm(mesh.vector_skewness[f])
        skewness_magnitudes.append(skew_mag)
    
    orthogonality_errors = np.array(orthogonality_errors)
    skewness_magnitudes = np.array(skewness_magnitudes)
    
    return orthogonality_errors, skewness_magnitudes

def run_mms_test_with_nonortho(mesh_file: str, bc_file: str, mesh_type="structured", n_nonortho_corrections=0):
    mesh = load_mesh(mesh_file, bc_file)
    
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    n = len(x)
    x_sym, y_sym = symbols("x y")
    
    # Original trigonometric MMS problem
    p_expr = cos(2*pi*x_sym) * cos(2*pi*y_sym)
    laplacian_expr = diff(p_expr, x_sym, 2) + diff(p_expr, y_sym, 2)
    
    p_func = lambdify((x_sym, y_sym), p_expr, modules='numpy')
    laplacian_func = lambdify((x_sym, y_sym), laplacian_expr, modules='numpy')
    p_exact = p_func(x, y)
    # Fix: Make exact solution have zero mean (consistent with nullspace removal)
    p_exact = p_exact - np.mean(p_exact)
    rhs_p = -laplacian_func(x, y) * mesh.cell_volumes
    p_numeric = pressure_mms_with_nonortho(mesh, -rhs_p, p_exact, n_nonortho_corrections=n_nonortho_corrections, rho=1.0)
    
    error = p_numeric - p_exact
    # Only use interior cells for L2 error
    interior_cells = get_unique_cells_from_faces(mesh, mesh.internal_faces)
    vol = mesh.cell_volumes[interior_cells]
    err_interior = error[interior_cells]
    l2_error = np.sqrt(np.sum(vol * err_interior**2) / np.sum(vol))
    max_error = np.max(np.abs(err_interior))
    row, col, data = assemble_pressure_correction_matrix(mesh, 1.0)
    from scipy.sparse import coo_matrix
    A_p = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    residual = A_p @ p_exact - rhs_p
    residual_l2 = np.linalg.norm(residual)
    
    return mesh, p_numeric, p_exact, error, residual, l2_error, max_error, residual_l2

def pressure_mms_with_nonortho(mesh, rhs_p, p_exact, n_nonortho_corrections=0, rho=1.0):
    """
    Solve pressure Poisson equation with non-orthogonal + skewness correction.
    Returns: p_numeric
    """
    n = mesh.cell_centers.shape[0]
    row, col, data = assemble_pressure_correction_matrix(mesh, rho)
    from scipy.sparse import coo_matrix
    A_p = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    from naviflow_collocated.linear_solvers.petsc_solver import petsc_solver
    # First solve (use negative sign as in SIMPLE)
    p_prime, _, ksp_1 = petsc_solver(A_p, -rhs_p, remove_nullspace=True, tolerance=1e-20)
    if n_nonortho_corrections > 0:
        accumulated_correction = np.zeros_like(rhs_p)
        grad_p_prime = compute_cell_gradients(mesh, np.ascontiguousarray(p_prime), weighted=True, weight_exponent=0.5, use_limiter=False)
        grad_p_prime_face = interpolate_to_face(mesh, grad_p_prime)
        beta_nonortho = 0.6
        for _ in range(n_nonortho_corrections):
            correction_term = pressure_correction_loop_term(mesh, rho, grad_p_prime_face)
            accumulated_correction = correction_term
            rhs_corrected = rhs_p + beta_nonortho * accumulated_correction
            # Correction solve (use negative sign as in SIMPLE)
            p_prime_new, _, _ = petsc_solver(A_p, -rhs_corrected, ksp=ksp_1, remove_nullspace=True, tolerance=1e-20)
            p_prime = p_prime_new
            grad_p_prime = compute_cell_gradients(mesh, p_prime, weighted=True, weight_exponent=0.5, use_limiter=False)
            grad_p_prime_face = interpolate_to_face(mesh, grad_p_prime)
    return p_prime

def run_convergence_study_with_nonortho(mesh_files: list[str], bc_file: str, ax=None, marker=None, mesh_type="structured", n_corrections=0):
    h_values = []
    l2_errors = []
    residual_l2s = []
    for mesh_file in mesh_files:
        mesh, p_numeric, p_exact, error, residual, l2_error, max_error, residual_l2 = run_mms_test_with_nonortho(
            mesh_file, bc_file, mesh_type=mesh_type, n_nonortho_corrections=n_corrections)
        h = np.sqrt(np.mean(mesh.cell_volumes))
        h_values.append(h)
        l2_errors.append(l2_error)
        residual_l2s.append(residual_l2)
    h_values = np.array(h_values)
    l2_errors = np.array(l2_errors)
    from numpy.linalg import lstsq
    X = np.log(h_values).reshape(-1, 1)
    X = np.hstack([X, np.ones_like(X)])
    y = np.log(l2_errors)
    (p, _), *_ = lstsq(X, y, rcond=None)
    
    print(f"Pressure {mesh_type} mesh ({n_corrections} corrections): observed order = {p:.2f}")
    
    if ax is not None:
        label_suffix = f" ({n_corrections} corrections)" if mesh_type == "unstructured" else ""
        ax.loglog(h_values, l2_errors, label=f"{mesh_type.capitalize()}{label_suffix} (order: {p:.2f})", marker=marker, linewidth=2.0, markersize=8)
    return l2_errors, residual_l2s, p

if __name__ == "__main__":
    print("Running pressure convergence study...")
    
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
    bc_file = "shared_configs/domain/boundaries_lidDrivenCavity.yaml"
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Structured grid reference
    errors_structured, _, order_structured = run_convergence_study_with_nonortho(
        [structured_uniform["coarse"], structured_uniform["medium"], structured_uniform["fine"]],
        bc_file,
        ax=ax,
        marker='o',
        mesh_type="structured",
        n_corrections=0  # No corrections needed for structured
    )
    
    # Unstructured grids with different correction levels
    markers = ['s', '^', 'D', 'o', 'x']
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:blue', 'tab:purple']
    correction_levels = [0, 2]
    unstructured_results = {}
    
    for i, n_corr in enumerate(correction_levels):
        errors_unstr, _, order_unstr = run_convergence_study_with_nonortho(
            [unstructured["coarse"], unstructured["medium"], unstructured["fine"]],
            bc_file,
            ax=ax,
            marker=markers[i],
            mesh_type="unstructured",
            n_corrections=n_corr
        )
        unstructured_results[n_corr] = (errors_unstr, order_unstr)
        
        # Set color for this line
        ax.get_lines()[-1].set_color(colors[i])
    
    # Add reference line for second order using professional pattern (following postprocessing standards)
    hs = np.array([np.sqrt(np.mean(load_mesh(f, bc_file).cell_volumes)) for f in [
        structured_uniform["coarse"],
        structured_uniform["medium"], 
        structured_uniform["fine"]
    ]])
    
    # Use minimum error as reference (following postprocessing pattern)
    error_ref = np.min(errors_structured)
    
    # Create reference grid size range
    h_ref = np.array([hs.min() * 0.8, hs.max() * 1.2])
    
    # Second order reference (O(h^2)) - only one needed for pressure
    ref_2nd = error_ref * 2 * (h_ref / h_ref[0])**2
    ax.loglog(h_ref, ref_2nd, color='black', linestyle='--', alpha=0.6, linewidth=1.2, label=r'$\mathcal{O}(h^2)$')
    
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(r"Grid size $h$", fontsize=12)
    ax.set_ylabel(r"L2 Error", fontsize=12)
    ax.set_title(f"Order of Accuracy: Pressure Poisson Equation\nEffect of Non-Orthogonal Corrections on Unstructured Grids", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    
    Path("tests/test_output/MMS_convergence").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig("tests/test_output/MMS_convergence/pressure_convergence_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Pressure convergence study completed. Results saved to tests/test_output/MMS_convergence/")
    
    