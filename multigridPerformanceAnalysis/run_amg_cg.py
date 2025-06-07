import argparse
from multigridPerformanceAnalysis.common_run import run_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMG-CG simulation with specified parameters.")
    parser.add_argument("--grid_size", type=int, default=63, help="Grid size (for nx and ny)")
    parser.add_argument("--reynolds", type=int, default=100, help="Reynolds number")
    parser.add_argument("--max_iterations", type=int, default=3000, help="Maximum number of SIMPLE iterations")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Convergence tolerance for SIMPLE")
    parser.add_argument("--pressure_tolerance", type=float, default=1e-5, help="Convergence tolerance for the pressure solver")
    parser.add_argument("--alpha_p", type=float, default=0.1, help="Pressure relaxation factor")
    parser.add_argument("--alpha_u", type=float, default=0.8, help="Velocity relaxation factor")
    parser.add_argument("--cycle_type", type=str, default='V', choices=['V', 'W'], help="AMG cycle type")
    parser.add_argument("--momentum_tol", type=float, default=1e-5, help="Tolerance for the momentum solver")
    parser.add_argument("--momentum_max_iter", type=int, default=10000, help="Max iterations for the momentum solver")
    parser.add_argument("--verbose_mg", action='store_true', help="Enable verbose output for the multigrid solver.")

    args = parser.parse_args()

    params = {
        "name": "amg_cg",
        "grid_size": args.grid_size,
        "reynolds": args.reynolds,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
        "pressure_tolerance": args.pressure_tolerance,
        "alpha_p": args.alpha_p,
        "alpha_u": args.alpha_u,
        "cycle_type": args.cycle_type,
        "momentum_tol": args.momentum_tol,
        "momentum_max_iter": args.momentum_max_iter,
        "verbose_mg": args.verbose_mg,
    }

    run_simulation('experiments/Staggered/07 AMG_CG/precond_AMG_CG.py', params) 