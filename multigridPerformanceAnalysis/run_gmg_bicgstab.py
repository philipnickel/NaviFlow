import argparse
from multigridPerformanceAnalysis.common_run import run_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GMG-BiCGSTAB simulation with specified parameters.")
    parser.add_argument("--grid_size", type=int, default=127, help="Grid size (for nx and ny)")
    parser.add_argument("--reynolds", type=int, default=100, help="Reynolds number")
    parser.add_argument("--max_iterations", type=int, default=35000, help="Maximum number of SIMPLE iterations")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Convergence tolerance for SIMPLE")
    parser.add_argument("--pressure_tolerance", type=float, default=1e-7, help="Convergence tolerance for the pressure solver")
    parser.add_argument("--alpha_p", type=float, default=0.3, help="Pressure relaxation factor")
    parser.add_argument("--alpha_u", type=float, default=0.7, help="Velocity relaxation factor")
    parser.add_argument("--pre_smoothing", type=int, default=1, help="Number of pre-smoothing steps for MG preconditioner")
    parser.add_argument("--post_smoothing", type=int, default=1, help="Number of post-smoothing steps for MG preconditioner")
    parser.add_argument("--cycle_type", type=str, default='v', choices=['v', 'w'], help="Multigrid cycle type for preconditioner")
    parser.add_argument("--momentum_tol", type=float, default=1e-9, help="Tolerance for the momentum solver")
    parser.add_argument("--momentum_max_iter", type=int, default=10000, help="Max iterations for the momentum solver")
    parser.add_argument("--verbose_mg", action='store_true', help="Enable verbose output for the multigrid solver.")

    args = parser.parse_args()

    params = {
        "name": "gmg_bicgstab",
        "grid_size": args.grid_size,
        "reynolds": args.reynolds,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
        "pressure_tolerance": args.pressure_tolerance,
        "alpha_p": args.alpha_p,
        "alpha_u": args.alpha_u,
        "pre_smoothing": args.pre_smoothing,
        "post_smoothing": args.post_smoothing,
        "cycle_type": args.cycle_type,
        "momentum_tol": args.momentum_tol,
        "momentum_max_iter": args.momentum_max_iter,
        "verbose_mg": args.verbose_mg,
    }

    run_simulation('experiments/Staggered/gmgbi/02 GMG BiCGSTAB/100/BiCGSTAB_cavity_steady_oo.py', params) 