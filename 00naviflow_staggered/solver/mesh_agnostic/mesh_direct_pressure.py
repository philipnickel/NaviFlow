import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..pressure_solver.base_pressure_solver import PressureSolver

class MeshAgnosticDirectPressureSolver(PressureSolver):
    """
    Direct sparse matrix-based pressure solver for collocated grids.
    Works for arbitrary meshes (structured or unstructured).
    """

    def __init__(self, tolerance=1e-10, max_iterations=1000):
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.residual_history = []

    def build_pressure_matrix(self, mesh, rho, d_u, d_v, pin_pressure=True):
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        owners, neighbors = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_normals = mesh.get_face_normals()

        data = []
        rows = []
        cols = []

        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]

            if owner < 0 or owner >= n_cells:
                continue

            area = face_areas[face_idx]
            normal = face_normals[face_idx]

            if neighbor >= 0 and neighbor < n_cells:
                # Internal face
                d_u_face = 0.5 * (d_u[owner] + d_u[neighbor])
                d_v_face = 0.5 * (d_v[owner] + d_v[neighbor])
                d_face = d_u_face * normal[0]**2 + d_v_face * normal[1]**2
                coeff = rho * d_face * area

                # Off-diagonal entries
                data.append(-coeff)
                rows.append(owner)
                cols.append(neighbor)

                data.append(-coeff)
                rows.append(neighbor)
                cols.append(owner)

                # Diagonal entries
                data.append(coeff)
                rows.append(owner)
                cols.append(owner)

                data.append(coeff)
                rows.append(neighbor)
                cols.append(neighbor)
            else:
                # Boundary face, add small stabilization diagonal term
                data.append(0.1)
                rows.append(owner)
                cols.append(owner)

        A = sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells)).tocsr()

        if pin_pressure:
            A[0, :] = 0.0
            A[0, 0] = 1.0

        return A

    def build_rhs(self, mesh, rho, u_star, v_star):
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        owners, neighbors = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_normals = mesh.get_face_normals()

        rhs = np.zeros(n_cells)

        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]

            if owner < 0 or owner >= n_cells:
                continue

            area = face_areas[face_idx]
            normal = face_normals[face_idx]

            if neighbor >= 0 and neighbor < n_cells:
                u_face = 0.5 * (u_star[owner] + u_star[neighbor])
                v_face = 0.5 * (v_star[owner] + v_star[neighbor])
            else:
                # Wall BC (zero normal velocity)
                u_face = 0.0
                v_face = 0.0

            mass_flux = rho * (u_face * normal[0] + v_face * normal[1]) * area
            rhs[owner] -= mass_flux
            if neighbor >= 0 and neighbor < n_cells:
                rhs[neighbor] += mass_flux

        rhs[0] = 0.0  # Pin pressure reference
        return rhs

    def solve(self, mesh, u_star, v_star, d_u, d_v, p_star, return_dict=True):
        rho = 1.0  # Assume constant density for now

        # Build system
        A = self.build_pressure_matrix(mesh, rho, d_u, d_v)
        rhs = self.build_rhs(mesh, rho, u_star, v_star)

        # Solve
        p_prime = spsolve(A, rhs)

        residual = rhs - A @ p_prime
        rel_norm = np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-10)

        self.residual_history.append(rel_norm)

        residual_info = {
            'rel_norm': rel_norm,
            'field': residual
        }

        if return_dict:
            return p_prime, residual_info
        else:
            return p_prime

    def get_solver_info(self):
        return {
            'name': 'MeshAgnosticDirectPressureSolver',
            'inner_iterations_history': [1] * len(self.residual_history),
            'total_inner_iterations': len(self.residual_history),
            'convergence_rate': None
        }
