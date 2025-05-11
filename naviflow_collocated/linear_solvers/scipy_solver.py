# Placeholder for SciPy linear solver implementation

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, isspmatrix_csr

# Try to import BaseLinearSolver, if it exists and is properly defined.
# This is for future extension and adherence to ABC pattern.
try:
    from naviflow_collocated.linear_solvers.base_solver import BaseLinearSolver

    # If BaseLinearSolver is an ABC, ScipyDirectSolver should inherit and implement abstract methods.
    # For now, assuming BaseLinearSolver might be a placeholder or not strictly enforced as ABC yet.
    BASE_SOLVER_EXISTS = True
except ImportError:
    BASE_SOLVER_EXISTS = False
    BaseLinearSolver = object  # Fallback to object if BaseLinearSolver is not found


class ScipyDirectSolver(BaseLinearSolver):
    """
    A direct linear solver using SciPy's spsolve for sparse matrices.
    Assumes the matrix A is in CSR format.
    """

    def __init__(self, config: dict = None):
        """
        Initializes the SciPy direct solver.

        Parameters:
        - config: Optional dictionary for solver-specific configurations (not used by this direct solver).
        """
        self.config = config if config is not None else {}
        # print("ScipyDirectSolver initialized.")

    def solve(self, A_matrix: csr_matrix, b_vector: np.ndarray) -> np.ndarray:
        """
        Solves the linear system A*x = b using SciPy's direct sparse solver (spsolve).

        Parameters:
        - A_matrix: The coefficient matrix in CSR (Compressed Sparse Row) format.
        - b_vector: The right-hand side vector (NumPy array).

        Returns:
        - solution: The solution vector x (NumPy array).

        Raises:
        - ValueError: If matrix and vector dimensions do not match, or if A is not CSR.
        - RuntimeError: If spsolve encounters an error.
        """
        if not isspmatrix_csr(A_matrix):
            # Attempt to convert to CSR if it's another sparse format, or raise error
            try:
                print(
                    "[Warning] ScipyDirectSolver: A_matrix is not CSR. Attempting conversion."
                )
                A_matrix = csr_matrix(A_matrix)
            except Exception as e:
                raise ValueError(
                    f"ScipyDirectSolver: A_matrix must be a CSR matrix or convertible to one. Conversion failed: {e}"
                )

        if A_matrix.shape[0] != b_vector.shape[0]:
            raise ValueError(
                f"ScipyDirectSolver: Matrix rows ({A_matrix.shape[0]}) and RHS vector length ({b_vector.shape[0]}) do not match."
            )
        if A_matrix.shape[0] == 0:
            # print("ScipyDirectSolver: System size is 0, returning empty solution.")
            return np.array([])
        if A_matrix.shape[0] != A_matrix.shape[1]:
            print(
                f"[Warning] ScipyDirectSolver: Matrix is not square ({A_matrix.shape}). Solving anyway if possible."
            )

        # print(f"ScipyDirectSolver: Solving system of size {A_matrix.shape} with nnz={A_matrix.nnz}")
        try:
            solution = spsolve(A_matrix, b_vector)
            # print("ScipyDirectSolver: System solved.")
            return solution
        except Exception as e:
            print(f"[ERROR] ScipyDirectSolver: SciPy spsolve failed. Exception: {e}")
            # Consider the implications of returning zeros vs. raising the error further.
            # For a direct solver, failure is usually critical.
            # Returning zeros might hide issues in the algorithm if the system is ill-conditioned or singular.
            raise RuntimeError(f"SciPy spsolve failed: {e}") from e


if __name__ == "__main__":
    # Example Usage (for testing this module directly)
    print("Testing ScipyDirectSolver...")

    # Create a simple sparse matrix (CSR)
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 1, 1, 0, 1, 2])
    data = np.array([4, -1, -1, -1, 5, -1])
    A = csr_matrix((data, (row, col)), shape=(3, 3))
    print("Matrix A:\n", A.toarray())

    # Create a RHS vector
    b = np.array([1, 2, 3])
    print("RHS b:", b)

    # Initialize and use the solver
    solver_config = {"solver_type": "direct_scipy"}
    scipy_solver = ScipyDirectSolver(config=solver_config)

    try:
        x = scipy_solver.solve(A, b)
        print("Solution x:", x)

        # Verify solution
        if np.allclose(A @ x, b):
            print("Solution verified: A@x = b")
        else:
            print("Solution verification failed: A@x != b")
            print("A@x = ", A @ x)

    except Exception as e:
        print(f"An error occurred during testing: {e}")

    print("\nTesting with an empty system:")
    A_empty = csr_matrix((0, 0), dtype=np.float64)
    b_empty = np.array([])
    try:
        x_empty = scipy_solver.solve(A_empty, b_empty)
        print("Solution for empty system x_empty:", x_empty)
    except Exception as e:
        print(f"An error occurred with empty system: {e}")

    print(
        "\nTesting with a singular matrix (expecting an error from spsolve via RuntimeError):"
    )
    A_singular = csr_matrix(np.array([[1, 1], [1, 1]]))
    b_singular = np.array([1, 2])
    try:
        scipy_solver.solve(A_singular, b_singular)
    except RuntimeError as e:
        print(f"Caught expected RuntimeError for singular matrix: {e}")
    except Exception as e:
        print(f"Caught unexpected error for singular matrix: {e}")
