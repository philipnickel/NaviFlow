import numpy as np
from scipy.sparse import csr_matrix
from petsc4py import PETSc

def petsc_solver(A_csr: csr_matrix, b_np: np.ndarray):
    """
    Solve A x = b using PETSc (SciPy CSR input) and return final residual norm.

    Parameters
    ----------
    A_csr : csr_matrix
        Sparse matrix in CSR format.
    b_np : np.ndarray
        Right-hand side vector.

    Returns
    -------
    x_np : np.ndarray
        Solution vector x.
    residual_norm : float
        Final 2-norm of the residual: ||b - A x||_2.
    """
    assert isinstance(A_csr, csr_matrix)
    n = A_csr.shape[0]

    # PETSc matrix from CSR
    A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape,
                                    csr=(A_csr.indptr, A_csr.indices, A_csr.data))
    A_petsc.assemble()

    # PETSc vectors
    b_petsc = PETSc.Vec().createWithArray(b_np)
    x_petsc = PETSc.Vec().createSeq(n)

    # KSP solver from options
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setFromOptions()
    ksp.solve(b_petsc, x_petsc)

    if not ksp.getConvergedReason() > 0:
        raise RuntimeError(f"PETSc did not converge. Reason: {ksp.getConvergedReason()}")

    # Compute residual r = b - Ax
    r_petsc = b_petsc.duplicate()
    A_petsc.mult(x_petsc, r_petsc)  # r = A*x
    r_petsc.aypx(-1.0, b_petsc)     # r = -1*A*x + b = b - Ax
    residual_norm = r_petsc.norm() # L2 norm

    return x_petsc.getArray(), residual_norm
