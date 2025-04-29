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

    def _build_sparse_matrix_face_based(self, mesh, a_p, a_nb, source, is_u=True):
        from scipy.sparse import coo_matrix

        owners, neighbors = mesh.get_owner_neighbor()
        face_normals = mesh.get_face_normals()
        face_areas = mesh.get_face_areas()
        face_distances = mesh.get_face_distances()
        wall_flags = mesh.get_wall_flags()
        wall_velocities = mesh.get_wall_velocities(is_u=is_u)

        n_cells = mesh.n_cells
        n_faces = mesh.n_faces

        # --- Sanity checks ---
        if len(a_p) != n_cells:
            raise ValueError(f"Length of a_p ({len(a_p)}) does not match number of cells ({n_cells})")
        if 'face' not in a_nb or len(a_nb['face']) != n_faces:
            raise ValueError(f"a_nb['face'] must exist and match number of faces ({n_faces})")

        data, row, col = [], [], []

        for face_idx in range(n_faces):
            owner = owners[face_idx]
            neighbor = neighbors[face_idx]

            # Skip faces that don't have a valid owner (e.g., boundary node faces)
            if owner == -1:
                continue

            # Validate owner (should be redundant now, but keep as sanity check)
            if not (0 <= owner < n_cells):
                # This should technically not be reached if the owner == -1 check works
                raise ValueError(f"Invalid owner index {owner} at face {face_idx}")

            if neighbor != -1:
                if not (0 <= neighbor < n_cells):
                    raise ValueError(f"Invalid neighbor index {neighbor} at face {face_idx}")
                coeff = a_nb['face'][face_idx]
                data += [-coeff, -coeff, coeff, coeff]
                row += [owner, neighbor, owner, neighbor]
                col += [neighbor, owner, owner, neighbor]

            elif wall_flags[face_idx]:
                Af = face_areas[face_idx]
                d = face_distances[face_idx]
                mu = mesh.get_viscosity()
                wall_vel = wall_velocities[face_idx]

                if not np.isfinite(d) or d <= 0:
                    raise ValueError(f"Non-finite or non-positive distance d={d} at wall face {face_idx}")

                tau = mu * wall_vel / d
                Fb = tau * Af
                source[owner] += Fb

        # Add diagonal entries (a_p)
        for cell_idx in range(n_cells):
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

        A, b = self._build_sparse_matrix_face_based(mesh, relaxed_ap, coeffs['a_nb'], relaxed_source, is_u=True)
        u_star = pyamg.smoothed_aggregation_solver(A).solve(b, x0=u_flat, tol=self.tolerance, maxiter=self.max_iterations)

        d_u = mesh.get_cell_volumes() / coeffs['a_p']
        residual_info = {'rel_norm': np.linalg.norm(b - A @ u_star) / max(np.linalg.norm(b), 1e-10)}

        return u_star, d_u, residual_info if return_dict else u_star

    def solve_v_momentum(self, mesh, fluid, u, v, p, relaxation_factor=0.7, return_dict=True):
        alpha = relaxation_factor
        n_cells = mesh.n_cells
        coeffs = self.discretization_scheme.calculate_v_coefficients(mesh, fluid, u, v, p)

        safe_ap = coeffs['a_p']
        v_flat = v.flatten()
        relaxed_ap = safe_ap / alpha
        relaxed_source = coeffs['source'] + (1.0 - alpha) * safe_ap * v_flat

        A, b = self._build_sparse_matrix_face_based(mesh, relaxed_ap, coeffs['a_nb'], relaxed_source, is_u=False)
        v_star = pyamg.smoothed_aggregation_solver(A).solve(b, x0=v_flat, tol=self.tolerance, maxiter=self.max_iterations)

        d_v = mesh.get_cell_volumes() / coeffs['a_p']
        residual_info = {'rel_norm': np.linalg.norm(b - A @ v_star) / max(np.linalg.norm(b), 1e-10)}

        return v_star, d_v, residual_info if return_dict else v_star
