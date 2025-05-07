import numpy as np


class Fields2D:
    """
    Holds all evolving simulation fields for the flow solver.

    Supports steady/transient SIMPLE, PISO, and can be extended for RANS/LES.

    Field categories:
    - velocity: u, v components (current, previous, predictor)
    - pressure: pressure, pressure correction
    - gradients: ∇u, ∇v, ∇p
    - residuals: for momentum and pressure equations
    - face values: interpolated face velocities, fluxes
    - auxiliary: inverse diagonal (d_u, d_v), turbulent quantities, etc.
    """

    def __init__(self, mesh):
        n_cells = len(mesh.cell_volumes)
        n_faces = len(mesh.face_areas)

        # === Cell-centered ===
        # Velocity
        self.u = np.zeros(n_cells)
        self.v = np.zeros(n_cells)
        self.u_prev = np.zeros(n_cells)  # for transient
        self.v_prev = np.zeros(n_cells)

        self.u_star = np.zeros(n_cells)  # predicted velocity (PISO)

        # Pressure
        self.p = np.zeros(n_cells)
        self.p_corr = np.zeros(n_cells)  # pressure correction

        # Scalars (temperature, turbulence, etc.)
        self.phi = np.zeros(n_cells)

        # Gradients
        self.grad_u = np.zeros((n_cells, 2))
        self.grad_v = np.zeros((n_cells, 2))
        self.grad_p = np.zeros((n_cells, 2))

        # Residuals
        self.res_u = np.zeros(n_cells)
        self.res_v = np.zeros(n_cells)
        self.res_p = np.zeros(n_cells)

        # Inverse diagonal coefficients (from momentum solver)
        self.d_u = np.zeros(n_cells)
        self.d_v = np.zeros(n_cells)

        # === Face-centered ===
        self.face_fluxes = np.zeros(n_faces)  # ρ u · n A
        self.face_velocities = np.zeros((n_faces, 2))  # Rhie-Chow, etc.

        # === Optional turbulence support ===
        self.nu_t = np.zeros(n_cells)  # turbulent viscosity
        self.k = np.zeros(n_cells)  # turbulent kinetic energy
        self.epsilon = np.zeros(n_cells)  # dissipation rate
