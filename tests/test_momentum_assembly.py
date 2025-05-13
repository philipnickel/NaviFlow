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
    sc = ax.scatter(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1],
                    c=field, cmap="viridis", s=30, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.75)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")

def run_mms_test(mesh_file, bc_file, u_exact_fn, rhs_fn, mu, rho, tag_prefix, component_idx=0):
    mesh = load_mesh(mesh_file, bc_file)

    phi_exact = u_exact_fn(mesh.cell_centers)
    u_field = np.zeros((mesh.cell_centers.shape[0], 2)) # both components 
    phi_field = np.zeros((mesh.cell_centers.shape[0])) # scalar field
    scalar_u_field = u_field[:, component_idx]
    grad_component = compute_cell_gradients(mesh,scalar_u_field)
    grad_phi = compute_cell_gradients(mesh, phi_exact)


    row, col, data, b_correction = assemble_diffusion_convection_matrix(
        mesh, phi_field, grad_phi, u_field, grad_component,
        rho, mu, component_idx
    )

    A = coo_matrix((data, (row, col)), shape=(mesh.cell_centers.shape[0],) * 2).tocsr()
    print(f"MMS rhs = {np.sum(rhs_fn(mesh.cell_centers) * mesh.cell_volumes)}")
    rhs = rhs_fn(mesh.cell_centers) * mesh.cell_volumes + b_correction
    
    print(f" sum(b_correction) = {np.sum(b_correction)}")

    phi_numeric = spsolve(A, rhs)

    err = np.abs(phi_numeric - phi_exact)
    max_err, l2_err = np.max(err), np.sqrt(np.mean(err ** 2))
    print(f"[{tag_prefix}] Max error: {max_err:.2e}, L2 error: {l2_err:.2e}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plot_field(mesh, phi_numeric, ax=axs[0], title="Numerical")
    plot_field(mesh, phi_exact, ax=axs[1], title="Exact")
    plot_field(mesh, err, ax=axs[2], title="Error")
    plt.tight_layout()
    outdir = Path("tests/test_output/MMS_solutions")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"mms_{tag_prefix}.png", dpi=300)
    plt.close()


# The following test follows the Method of Manufactured Solutions (MMS) approach
# to verify spatial convergence of the numerical scheme by comparing numerical
# and exact solutions on a sequence of refined meshes.
def run_convergence_study(mesh_files, bc_file, u_exact_fn, rhs_fn, mu, rho, tag_prefix, component_idx=0):
    hs = []
    errors = []

    for mesh_file in mesh_files:
        mesh = load_mesh(mesh_file, bc_file)
        h = np.sqrt(np.mean(mesh.cell_volumes))
        hs.append(h)


        phi_exact = u_exact_fn(mesh.cell_centers)
        u_field = np.zeros((mesh.cell_centers.shape[0], 2)) # both components 
        phi_field = np.zeros((mesh.cell_centers.shape[0])) # scalar field
        scalar_u_field = u_field[:, component_idx]
        grad_component = compute_cell_gradients(mesh,scalar_u_field)
        grad_phi = compute_cell_gradients(mesh, phi_exact)


        row, col, data, b_correction = assemble_diffusion_convection_matrix(
            mesh, phi_field, grad_phi, u_field, grad_component,
            rho, mu, component_idx
        )


        A = coo_matrix((data, (row, col)), shape=(mesh.cell_centers.shape[0],) * 2).tocsr()
        rhs = rhs_fn(mesh.cell_centers) * mesh.cell_volumes + b_correction

        phi_numeric = spsolve(A, rhs)

        err = np.abs(phi_numeric - phi_exact)
        l2_err = np.sqrt(np.mean(err**2))
        errors.append(l2_err)

        print(f"[{tag_prefix}] h = {h:.4f}, L2 error = {l2_err:.3e}")

    hs = np.array(hs)
    errors = np.array(errors)
    rate = np.log(errors[:-1] / errors[1:]) / np.log(hs[:-1] / hs[1:])
    print("\nObserved convergence rates:")
    # The slope of the error vs. grid size in log-log scale approximates the observed order of convergence.
    for i, r in enumerate(rate):
        print(f"{tag_prefix}: from h={hs[i]:.4f} to h={hs[i+1]:.4f} --> rate ≈ {r:.2f}")

    plt.figure()
    plt.loglog(hs, errors, label=tag_prefix)
    # Add reference line for second-order convergence
    ref_slope = errors[0] * (hs / hs[0])**2 *0.9
    plt.loglog(hs, ref_slope, 'k--', label='Second-order')
    plt.grid(True, which="both")
    plt.xlabel("Grid size h")
    plt.ylabel("L2 Error")
    plt.title("Grid Convergence Study")
    plt.legend()
    Path("tests/test_output/MMS_convergence").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"tests/test_output/MMS_convergence/convergence_plot_{tag_prefix}.png", dpi=300)
    plt.close()



# === MMS Functions ===
# The exact solution u_sin is chosen to be smooth and not exactly representable by the discrete scheme,
# to validate the convergence behavior of the numerical method.

def generate_mms_functions(expr_str, velocity=("1.0", "0.0"), mu=1.0, rho=1.0):
    x, y = sp.symbols("x y")
    expr = sp.sympify(expr_str)
    u_x_expr = sp.sympify(velocity[0])
    u_y_expr = sp.sympify(velocity[1])

    grad = [sp.diff(expr, var) for var in (x, y)]
    laplacian = sum(sp.diff(expr, var, 2) for var in (x, y))

    conv_term = rho * (u_x_expr * grad[0] + u_y_expr * grad[1])
    diff_term = mu * laplacian
    total_rhs = conv_term - diff_term

    u_func = lambdify((x, y), expr, modules="numpy")
    grad_func = lambdify((x, y), grad, modules="numpy")
    rhs_func = lambdify((x, y), total_rhs, modules="numpy")

    def u(xy): return u_func(xy[:, 0], xy[:, 1])
    def rhs(xy): return rhs_func(xy[:, 0], xy[:, 1])
    def grad(xy):
        grad_vals = grad_func(xy[:, 0], xy[:, 1])
        return np.column_stack(grad_vals)

    return u, grad, rhs

def u_zero(mesh):
    x, y = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1]
    u_x = np.sin(np.pi * x) * np.sin(np.pi * y)
    u_y = np.zeros_like(x)
    return np.column_stack([u_x, u_y])

def u_mms(mesh):
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    u_x = -np.cos(np.pi * x) * np.sin(np.pi * y)
    u_y =  np.sin(np.pi * x) * np.cos(np.pi * y)
    return np.column_stack([u_x, u_y])



def u_constant(mesh):
    # Constant velocity field, e.g., [1, 0] everywhere
    return np.ones((mesh.cell_centers.shape[0], 2))
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
        "sinusoidal": "sin(pi*x)*sin(pi*y)",
        "quadratic": "x**2 + y**2 + x*y",
        "anisotropic": "sin(2*pi*x) + 0.5*cos(3*pi*y)"
    }
    BC_files = {
        "sinusoidal": "shared_configs/domain/sanityChecks/sanityCheckDiffusionSIN.yaml",
        "quadratic": "shared_configs/domain/sanityChecks/sanityCheckDiffusionQUADD.yaml",
        "anisotropic": "shared_configs/domain/sanityChecks/sanityCheckDiffusionANS.yaml"
    }

    for tag, expr in mms_cases.items():
        u_fn, grad_fn, rhs_fn = generate_mms_functions(expr, velocity=("0.0", "0.0"), mu=1.0, rho=0.0)
        bc_file = BC_files[tag]
        run_mms_test(
            structured_uniform["fine"],
            bc_file,
            u_fn, rhs_fn, 1.0, 0.0,
            tag_prefix=f"{tag}_structured",
            
        )
        run_mms_test(
            unstructured["fine"],
            bc_file,
            u_fn, rhs_fn, 1.0, 0.0,
            tag_prefix=f"{tag}_unstructured",
            
        )

        # Uncomment to run convergence studies
        run_convergence_study(
            [structured_uniform["coarse"], structured_uniform["medium"], structured_uniform["fine"]],
            bc_file,
            u_fn, rhs_fn, 1.0, 0.0,
            tag_prefix=f"{tag}_structured"
        )
        run_convergence_study(
            [unstructured["coarse"], unstructured["medium"], unstructured["fine"]],
            bc_file,
            u_fn, rhs_fn, 1.0, 0.0,
            tag_prefix=f"{tag}_unstructured"
        )
 