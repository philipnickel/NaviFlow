import numpy as np
from scipy.sparse import csr_matrix
from petsc4py import PETSc

def petsc_solver(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    ksp: PETSc.KSP = None,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    solver_type: str = "bcgs",
    preconditioner: str = "hypre",
    remove_nullspace: bool = False
):
    """
    Solve A x = b using PETSc with optional KSP reuse.

    Parameters
    ----------
    A_csr : csr_matrix
        Sparse matrix in CSR format (ignored if ksp is provided).
    b_np : np.ndarray
        Right-hand side vector.
    ksp : PETSc.KSP, optional
        Reusable KSP solver object. If provided, will be reused without reinitializing matrix/preconditioner.
    tolerance : float
        Solver tolerance (absolute).
    max_iterations : int
        Maximum Krylov iterations.
    solver_type : str
        Type of PETSc solver (e.g. "cg", "gmres", "bcgs").
    preconditioner : str
        PETSc preconditioner type (e.g. "hypre", "gamg").
    remove_nullspace : bool
        If True, constant nullspace is removed from RHS.

    Returns
    -------
    x_np : np.ndarray
        Solution vector.
    residual_norm : float
        ||b - Ax||₂ norm.
    ksp : PETSc.KSP
        The KSP object (for reuse).
    """

    n = b_np.shape[0]

    # ─────────────────────────────────────────────────────────────────────
    # REUSE PATH: KSP object already constructed outside
    # ─────────────────────────────────────────────────────────────────────
    if ksp is not None:
        A_petsc, _ = ksp.getOperators()
        b_petsc = PETSc.Vec().createWithArray(b_np)
        x_petsc = b_petsc.duplicate()
        if remove_nullspace:
            nullspace = A_petsc.getNullSpace()
            if nullspace is not None:
                nullspace.remove(b_petsc)
        ksp.solve(b_petsc, x_petsc)

        # Residual norm: r = b - Ax
        r_petsc = b_petsc.duplicate()
        A_petsc.mult(x_petsc, r_petsc)
        r_petsc.aypx(-1.0, b_petsc)
        residual_norm = r_petsc.norm()

        x_np = x_petsc.getArray().copy()

        # Cleanup vectors
        b_petsc.destroy()
        x_petsc.destroy()
        r_petsc.destroy()

        return x_np, residual_norm, ksp

    # ─────────────────────────────────────────────────────────────────────
    # FIRST-TIME SETUP: Construct everything
    # ─────────────────────────────────────────────────────────────────────
    A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape,
                                    csr=(A_csr.indptr, A_csr.indices, A_csr.data))
    A_petsc.assemble()

    b_petsc = PETSc.Vec().createWithArray(b_np)
    x_petsc = PETSc.Vec().createSeq(n)

    # Handle nullspace (only needed once for A)
    if remove_nullspace:
        nullvec = A_petsc.createVecLeft()
        nullvec.set(1.0)
        nullvec.normalize()
        nullspace = PETSc.NullSpace().create(vectors=[nullvec])
        A_petsc.setNullSpace(nullspace)
        nullspace.remove(b_petsc)

    # KSP and PC setup
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType(solver_type)
    ksp.setTolerances(atol=float(tolerance), max_it=max_iterations)
    pc = ksp.getPC()
    pc.setType(preconditioner)

    if preconditioner == "hypre":
        PETSc.Options().setValue("pc_hypre_type", "boomeramg")
        PETSc.Options().setValue("pc_hypre_boomeramg_strong_threshold", "0.7")
        PETSc.Options().setValue("pc_hypre_boomeramg_coarsen_type", "HMIS")
        PETSc.Options().setValue("pc_hypre_boomeramg_interp_type", "ext+i")
    elif preconditioner == "gamg":
        PETSc.Options().setValue("pc_gamg_type", "agg")
        PETSc.Options().setValue("pc_gamg_coarse_eq_limit", "1000")
        PETSc.Options().setValue("pc_gamg_agg_nsmooths", "1")
        PETSc.Options().setValue("pc_gamg_square_graph", "1")
        PETSc.Options().setValue("pc_gamg_threshold", "0.02")

    ksp.setFromOptions()
    ksp.setUp()

    ksp.solve(b_petsc, x_petsc)

    if ksp.getConvergedReason() <= 0:
        raise RuntimeError(f"PETSc did not converge. Reason: {ksp.getConvergedReason()}")

    # Residual norm
    r_petsc = b_petsc.duplicate()
    A_petsc.mult(x_petsc, r_petsc)
    r_petsc.aypx(-1.0, b_petsc)
    residual_norm = r_petsc.norm()

    x_np = x_petsc.getArray().copy()

    # Cleanup (do NOT destroy A_petsc or ksp for reuse)
    b_petsc.destroy()
    x_petsc.destroy()
    r_petsc.destroy()
    if remove_nullspace:
        nullvec.destroy()
        nullspace.destroy()

    return x_np, residual_norm, ksp
