import numpy as np
from scipy import sparse
from .base_momentum_solver import MomentumSolver
from .discretization import power_law, quick, upwind
import pyamg

class AMGMomentumSolver(MomentumSolver):
    def __init__(self, discretization_scheme='power_law', tolerance=1e-8, max_iterations=100):
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        schemes = {
            'power_law': power_law.PowerLawDiscretization,
            'quick': quick.QUICKDiscretization,
            'upwind': upwind.UpwindDiscretization
        }

        if discretization_scheme not in schemes:
            raise ValueError(f"Unsupported discretization scheme: {discretization_scheme}")
        self.discretization_scheme = schemes[discretization_scheme]()

    def _build_sparse_matrix_face_based(self, mesh, fluid, a_p, a_nb, source, is_u=True):
        from scipy.sparse import coo_matrix

        owners, neighbors = mesh.get_owner_neighbor()
        face_areas = mesh.get_face_areas()
        face_distances = mesh.get_face_distances()
        wall_flags = mesh.get_wall_flags()
        wall_velocities = mesh.get_wall_velocities(is_u=is_u)

        n_cells = mesh.n_cells
        n_faces = mesh.n_faces

        data, row, col = [], [], []

        # Track which cells have strong BC enforcement
        fixed_cells = set()

        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]

            if owner == -1:
                continue

            if neighbor != -1:
                # Internal face
                coeff = a_nb['face'][face_idx]
                data += [-coeff, -coeff, coeff, coeff]
                row += [owner, neighbor, owner, neighbor]
                col += [neighbor, owner, owner, neighbor]
            elif wall_flags[face_idx]:
                wall_vel = wall_velocities[face_idx]

                # Replace the matrix row to enforce u = wall_vel
                data.append(1.0)
                row.append(owner)
                col.append(owner)
                source[owner] = wall_vel

                fixed_cells.add(owner)
            
            

        # Insert diagonal entries for regular cells
        for cell_idx in range(n_cells):
            if cell_idx not in fixed_cells:
                data.append(a_p[cell_idx])
                row.append(cell_idx)
                col.append(cell_idx)

        A = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        return A, source




    def solve_u_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, return_dict=True):
        alpha = relaxation_factor
        n_cells = mesh.n_cells
        coeffs = self.discretization_scheme.calculate_u_coefficients(mesh, fluid, u, v, p)

        safe_ap = coeffs['a_p']
        u_flat = u.flatten()
        relaxed_ap = safe_ap / alpha
        relaxed_source = coeffs['source'] + (1.0 - alpha) * safe_ap * u_flat

        A, b = self._build_sparse_matrix_face_based(mesh, fluid, relaxed_ap, coeffs['a_nb'], relaxed_source, is_u=True)
        u_star = pyamg.smoothed_aggregation_solver(A).solve(b, x0=u_flat, tol=self.tolerance, maxiter=self.max_iterations)

        rel_norm = np.linalg.norm(b - A @ u_star) / max(np.linalg.norm(b), 1e-10)
        residual_field = b - A @ u_star 

        d_u = mesh.get_cell_volumes() / coeffs['a_p']
        residual_info = {'rel_norm': rel_norm, 'field': residual_field}

        return u_star, d_u, residual_info if return_dict else u_star

    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, return_dict=True):
        alpha = relaxation_factor
        n_cells = mesh.n_cells
        coeffs = self.discretization_scheme.calculate_v_coefficients(mesh, fluid, u, v, p)

        safe_ap = coeffs['a_p']
        v_flat = v.flatten()
        relaxed_ap = safe_ap / alpha
        relaxed_source = coeffs['source'] + (1.0 - alpha) * safe_ap * v_flat

        A, b = self._build_sparse_matrix_face_based(mesh, fluid, relaxed_ap, coeffs['a_nb'], relaxed_source, is_u=False)
        v_star = pyamg.smoothed_aggregation_solver(A).solve(b, x0=v_flat, tol=self.tolerance, maxiter=self.max_iterations)

        rel_norm = np.linalg.norm(b - A @ v_star) / max(np.linalg.norm(b), 1e-10)
        residual_field = b - A @ v_star

        d_v = mesh.get_cell_volumes() / coeffs['a_p']
        residual_info = {'rel_norm': rel_norm, 'field': residual_field}

        return v_star, d_v, residual_info if return_dict else v_star
