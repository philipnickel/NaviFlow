import numpy as np
from scipy.sparse import csr_matrix
from petsc4py import PETSc

def petsc_solver(A_csr: csr_matrix, b_np: np.ndarray, ksp=None):
    """
    Solve A x = b using PETSc with optional KSP reuse.

    Parameters
    ----------
    A_csr : csr_matrix
        Sparse matrix in CSR format.
    b_np : np.ndarray
        Right-hand side vector.
    ksp : PETSc.KSP, optional
        Reusable KSP solver object. If None, a new KSP is created.

    Returns
    -------
    x_np : np.ndarray
        Solution vector x.
    residual_norm : float
        Final 2-norm of the residual: ||b - A x||_2.
    ksp : PETSc.KSP
        KSP solver (returned for reuse).
    """
    n = A_csr.shape[0]

    # Create PETSc matrix from SciPy CSR
    A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape,
                                    csr=(A_csr.indptr, A_csr.indices, A_csr.data))
    A_petsc.assemble()

    # Create PETSc vectors
    b_petsc = PETSc.Vec().createWithArray(b_np)
    x_petsc = PETSc.Vec().createSeq(n)

    # Create or reuse KSP solver
    if ksp is None:
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setType("bcgs")
        ksp.setTolerances(atol=1e-5, rtol=1e-5, max_it=10000)
        pc = ksp.getPC()
        pc.setType("hypre")
        ksp.setFromOptions()


    # Solve
    ksp.solve(b_petsc, x_petsc)

    if ksp.getConvergedReason() <= 0:
        raise RuntimeError(f"PETSc did not converge. Reason: {ksp.getConvergedReason()}")

    # Compute residual: r = b - Ax
    r_petsc = b_petsc.duplicate()
    A_petsc.mult(x_petsc, r_petsc)
    r_petsc.aypx(-1.0, b_petsc)
    residual_norm = r_petsc.norm()

    # Convert result to NumPy
    x_np = x_petsc.getArray().copy()

    # Cleanup
    A_petsc.destroy()
    b_petsc.destroy()
    r_petsc.destroy()
    x_petsc.destroy()

    return x_np, residual_norm, ksp
