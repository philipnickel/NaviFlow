import numpy as np
from scipy.sparse import coo_matrix
from naviflow_collocated.assembly.convection_diffusion_matrix import assemble_diffusion_convection_matrix
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients
from naviflow_collocated.linear_solvers.petsc_solver import petsc_solver
from naviflow_collocated.assembly.rhie_chow import compute_face_fluxes, compute_rhie_chow_face_velocities
from naviflow_collocated.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix
from naviflow_collocated.assembly.divergence import compute_divergence_from_face_fluxes
def simple_algorithm(mesh, alpha_uv, alpha_p, reynolds_number, max_iter, tol, convection_scheme="TVD", limiter="MUSCL"):

    # Initialize fields
    n_cells = mesh.cell_volumes.shape[0]
    u = np.zeros(n_cells)
    v = np.zeros(n_cells)
    mdot = np.zeros(n_cells)
    u_field = np.zeros((n_cells, 2))
    p = np.zeros(n_cells)
    p_prime = np.zeros(n_cells)
    u_prime = np.zeros(n_cells)
    v_prime = np.zeros(n_cells)

    # calculate rho and mu from Reynolds number
    rho = 1.0
    mu = 1.0 / reynolds_number

    for i in range(max_iter):
        #=============================================================================
        # ASSEMBLE MOMENTUM EQUATIONS
        #=============================================================================
        grad_p = compute_cell_gradients(mesh, p)

        # u-momentum
        grad_phi_u = compute_cell_gradients(mesh, u)
        row, col, data, b_u = assemble_diffusion_convection_matrix(
            mesh, grad_phi_u, u_field, rho, mu, 0, phi=u, scheme=convection_scheme, limiter=limiter
        )
        A_u = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        diag_u = A_u.diagonal()
        Ap_u_cell = A_u.diagonal()
        rhs_u = b_u - grad_p[:, 0] * mesh.cell_volumes
        if alpha_uv < 1.0:
            A_u = A_u.multiply(1.0 / alpha_uv)
            rhs_u += (1.0 - alpha_uv) / alpha_uv * diag_u * u
        u_prime, res_u = petsc_solver(A_u, rhs_u)

        # v-momentum
        grad_phi_v = compute_cell_gradients(mesh, v)
        row, col, data, b_v = assemble_diffusion_convection_matrix(
            mesh, grad_phi_v, u_field, rho, mu, 1, phi=v, scheme=convection_scheme, limiter=limiter
        )
        A_v = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        diag_v = A_v.diagonal()
        Ap_v_cell = A_v.diagonal()
        rhs_v = b_v - grad_p[:, 1] * mesh.cell_volumes
        if alpha_uv < 1.0:
            A_v = A_v.multiply(1.0 / alpha_uv)
            rhs_v += (1.0 - alpha_uv) / alpha_uv * diag_v * v
        v_prime, res_v = petsc_solver(A_v, rhs_v)

        #=============================================================================
        # RHIE-CHOW FACE VELOCITIES AND FLUXES
        #=============================================================================
        face_velocity = compute_rhie_chow_face_velocities(mesh, u_prime, v_prime, p, Ap_u_cell, Ap_v_cell)
        face_fluxes = compute_face_fluxes(mesh, face_velocity, rho)

        #=============================================================================
        # PRESSURE CORRECTION EQUATION
        #=============================================================================
        rhs_p = compute_divergence_from_face_fluxes(mesh, face_fluxes) 
        # pin one pressure node
        rhs_p[0] = 0.0
        row_p, col_p, data_p, bcorr = assemble_pressure_correction_matrix(mesh, rho, Ap_u_cell, Ap_v_cell)
        A_p = coo_matrix((data_p, (row_p, col_p)), shape=(n_cells, n_cells)).tocsr()
        p_prime, res_p = petsc_solver(A_p, rhs_p)

        #=============================================================================
        # CORRECT PRESSURE AND VELOCITY
        #=============================================================================
        p += alpha_p * p_prime
        grad_p_prime = compute_cell_gradients(mesh, p_prime)
        u = u_prime - grad_p_prime[:, 0] * mesh.cell_volumes
        v = v_prime - grad_p_prime[:, 1] * mesh.cell_volumes

        u_field[:, 0] = u
        u_field[:, 1] = v


        #=============================================================================
        # CONVERGENCE CHECK
        #=============================================================================
        # print residual
        print(f"Residuals: u = {res_u}, v = {res_v}, p = {res_p}")
        if res_u + res_v + res_p < tol:
            break

    return u, v, p
