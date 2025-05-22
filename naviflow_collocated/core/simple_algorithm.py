import os
import numpy as np
from scipy.sparse import coo_matrix
from naviflow_collocated.assembly.convection_diffusion_matrix import assemble_diffusion_convection_matrix
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients
#from naviflow_collocated.discretization.gradient.gauss import compute_cell_gradients
from naviflow_collocated.linear_solvers.petsc_solver import petsc_solver
from naviflow_collocated.assembly.rhie_chow import mdot_calculation, mdot_correction
from naviflow_collocated.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix
from naviflow_collocated.assembly.divergence import compute_divergence_from_face_fluxes
from naviflow_collocated.core.corrections import velocity_correction
import matplotlib.pyplot as plt

from numba import njit

@njit
def enforce_boundary_conditions(mesh, u_field):
    boundary_faces = mesh.boundary_faces
    n_boundary = boundary_faces.shape[0]
    for i in range(n_boundary):
        f = boundary_faces[i]
        owner_cell = mesh.owner_cells[f]
        u_field[owner_cell, 0] = mesh.boundary_values[f, 0]
        u_field[owner_cell, 1] = mesh.boundary_values[f, 1]
    return u_field

def simple_algorithm(mesh, alpha_uv, alpha_p, reynolds_number, max_iter, tol, convection_scheme="TVD", limiter="MUSCL"):

    # Initialize fields
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    mdot = np.zeros(n_internal + n_boundary)
    u_field = np.zeros((n_cells, 2))
    u_field = enforce_boundary_conditions(mesh, u_field)
    u = u_field[:, 0]
    v = u_field[:, 1]
    p = np.zeros(n_cells)
    p_prime = np.zeros(n_cells)
    u_prime = np.zeros(n_cells)
    v_prime = np.zeros(n_cells)

    # calculate rho and mu from Reynolds number
    rho = 1.0
    mu = rho * 1.0 / reynolds_number
    mom_solver_u = None
    mom_solver_v = None
    pres_solver = None

    for i in range(max_iter):
        #=============================================================================
        # ASSEMBLE MOMENTUM EQUATIONS
        #=============================================================================
        grad_p = compute_cell_gradients(mesh, p)
        # plot grad_p


        # u-momentum
        grad_phi_u = compute_cell_gradients(mesh, u)


        row, col, data, b_u = assemble_diffusion_convection_matrix(
            mesh,mdot,  grad_phi_u, u_field, rho, mu, 0, phi=u, scheme=convection_scheme, limiter=limiter, pressure_field=p, grad_pressure_field=grad_p
        )
        A_u = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        A_u_for_residual = A_u.copy()
        
        rhs_u = b_u - grad_p[:, 0] * mesh.cell_volumes
        rhs_for_residual = rhs_u
        
        diag_u = A_u.diagonal()
        if alpha_uv < 1.0:
            A_u.setdiag(A_u.diagonal() * (1.0 / alpha_uv))
            rhs_u += (1.0 - alpha_uv) / alpha_uv * diag_u * u

        Ap_u_cell = A_u.diagonal().copy()
        # solve
        u_star, res_u, _= petsc_solver(A_u, rhs_u)
        res_u_for_residual = rhs_for_residual - A_u_for_residual @ u_star
        l2_norm_u = np.linalg.norm(res_u_for_residual)
        # v-momentum
        grad_phi_v = compute_cell_gradients(mesh, v)
        row, col, data, b_v = assemble_diffusion_convection_matrix(
            mesh,mdot, grad_phi_v, u_field, rho, mu, 1, phi=v, scheme=convection_scheme, limiter=limiter, pressure_field=p, grad_pressure_field=grad_p
        )
        A_v = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        A_v_for_residual = A_v.copy()
        

        rhs_v = b_v - grad_p[:, 1] * mesh.cell_volumes
        rhs_for_residual = rhs_v
        diag_v = A_v.diagonal()
        if alpha_uv < 1.0:
            A_v.setdiag(A_v.diagonal() * (1.0 / alpha_uv))
            rhs_v += (1.0 - alpha_uv) / alpha_uv * diag_v * v
        Ap_v_cell = A_v.diagonal().copy()

        v_star, res_v, _= petsc_solver(A_v, rhs_v)
        res_v_for_residual = rhs_for_residual - A_v_for_residual @ v_star
        l2_norm_v = np.linalg.norm(res_v_for_residual)

        # gather u_star and v_star
        u_field[:, 0] = u_star
        u_field[:, 1] = v_star

        #=============================================================================
        # RHIE-CHOW FLUXES
        #=============================================================================
        mdot_star = mdot_calculation(mesh, rho, u_star, v_star, p, grad_p, Ap_u_cell, Ap_v_cell, alpha_uv, u, v)

        #=============================================================================
        # PRESSURE CORRECTION EQUATION
        #=============================================================================
        rhs_p = compute_divergence_from_face_fluxes(mesh, mdot_star) 

        cont_error = np.sum(rhs_p)
        # pin one pressure node
        rhs_p[0] = 0.0
        row_p, col_p, data_p, bcorr = assemble_pressure_correction_matrix(mesh, rho)
        A_p = coo_matrix((data_p, (row_p, col_p)), shape=(n_cells, n_cells)).tocsr()

        
        p_prime, res_p, _= petsc_solver(A_p, -rhs_p)

        #=============================================================================
        # CORRECT PRESSURE, VELOCITY and MASS FLUX
        #=============================================================================
        grad_p_prime= compute_cell_gradients(mesh, p_prime)

        uv_field_2star =u_field + velocity_correction(mesh, grad_p_prime,Ap_u_cell, Ap_v_cell)
        mdot_2star = mdot_star + mdot_correction(mesh, rho, Ap_u_cell, Ap_v_cell, grad_p_prime)

        p += alpha_p * p_prime
        grad_p = compute_cell_gradients(mesh, p)
        #p = set_boundary_pressure(mesh, p, grad_p)


        mdot = mdot_2star
        u_field = uv_field_2star
        u = u_field[:, 0].copy()
        v = u_field[:, 1].copy()
        # Compute continuity error
        div_u = compute_divergence_from_face_fluxes(mesh, mdot_2star)
        """
        x = mesh.cell_centers[:, 0]
        y = mesh.cell_centers[:, 1]
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        
        # Plot div_u
        sc1 = ax1.scatter(x, y, c=div_u, cmap="viridis", s=8)
        ax1.set_title("Continuity Error")
        plt.colorbar(sc1, ax=ax1)
        ax1.set_aspect('equal')
        
        # Plot rhs_p 
        sc2 = ax2.scatter(x, y, c=rhs_p, cmap="viridis", s=8)
        ax2.set_title("RHS of Pressure Equation")
        plt.colorbar(sc2, ax=ax2)
        ax2.set_aspect('equal')

        # plot mdots
        sc3 = ax3.scatter(mesh.face_centers[:, 0], mesh.face_centers[:, 1], c=mdot_star, cmap="viridis", s=8)
        ax3.set_title("Mdot Star")
        plt.colorbar(sc3, ax=ax3)
        ax3.set_aspect('equal')
        
        # plot mdot_2star
        sc4 = ax4.scatter(mesh.face_centers[:, 0], mesh.face_centers[:, 1], c=mdot_2star, cmap="viridis", s=8)
        ax4.set_title("Mdot 2 Star")
        plt.colorbar(sc4, ax=ax4)
        ax4.set_aspect('equal')

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/div_u.png", dpi=300)
        plt.close()
        """


        #=============================================================================
        # CONVERGENCE CHECK
        #=============================================================================
        # print residual
        print(f"Iteration {i}: Residuals: u = {l2_norm_u:.3e}, v = {l2_norm_v:.3e}, continuity = {cont_error:.3e}")
        if l2_norm_u + l2_norm_v + cont_error < tol:
            break

    return u_field, p
