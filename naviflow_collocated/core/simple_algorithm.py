import os
import numpy as np
from scipy.sparse import coo_matrix
from naviflow_collocated.assembly.convection_diffusion_matrix import assemble_diffusion_convection_matrix
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients
#from naviflow_collocated.discretization.gradient.gauss import compute_cell_gradients
from naviflow_collocated.linear_solvers.petsc_solver import petsc_solver
from naviflow_collocated.assembly.rhie_chow import mdot_calculation, rhie_chow_velocity
from naviflow_collocated.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix, pressure_correction_loop_term
from naviflow_collocated.assembly.divergence import compute_divergence_from_face_fluxes
from naviflow_collocated.core.corrections import velocity_correction
import matplotlib.pyplot as plt
from naviflow_collocated.core.helpers import bold_Dv_calculation, interpolate_to_face, compute_residual, relax_momentum_equation
import time
from numba import njit
def piso_corrector_loop(mesh, A_p, ksp, mdot_start, rho, bold_D, U_star_rc, U_star, p, alpha_p, num_corrections=1):
    """
    Perform PISO pressure–velocity correction loops.

    Parameters
    ----------
    mesh : Mesh object
    A_p : sparse matrix
        Pressure correction matrix (typically constant).
    ksp : PETSc.KSP
        Reusable KSP object for solving pressure corrections.
    mdot_start : ndarray
        Initial mass flux from Rhie–Chow velocities (before correction).
    rho : float
        Fluid density.
    bold_D : ndarray
        Diagonal momentum inverse matrix, shape (n_cells, 2).
    U_star_rc : ndarray
        Face velocity from Rhie–Chow interpolation.
    U_star : ndarray
        Momentum velocity solution at cell centers.
    p : ndarray
        Current pressure field (updated in-place).
    alpha_p : float
        Pressure under-relaxation factor.
    num_corrections : int
        Number of PISO pressure–velocity correction loops.

    Returns
    -------
    U_new : ndarray
        Updated velocity field (after all corrections).
    p_new : ndarray
        Updated pressure field.
    mdot_new : ndarray
        Updated mass flux field at faces.
    """
    n_cells = mesh.cell_volumes.shape[0]
    U = U_star.copy()
    U_faces = U_star_rc.copy()
    mdot = mdot_start.copy()
    p_prime_total = np.zeros(n_cells)

    for _ in range(num_corrections):
        # Step 1: Solve pressure correction
        rhs_p = compute_divergence_from_face_fluxes(mesh, mdot)
        rhs_p[0] = 0.0  # pin pressure
        p_prime, _, _ = petsc_solver(A_p, -rhs_p, ksp=ksp)

        # Step 2: Compute velocity correction
        grad_p_prime = compute_cell_gradients(mesh, p_prime)
        U_prime = velocity_correction(mesh, grad_p_prime, bold_D)
        U_prime_face = interpolate_to_face(mesh, U_prime)

        # Step 3: Update
        U += U_prime
        U_faces += U_prime_face
        mdot += mdot_calculation(mesh, rho, U_prime_face)
        p_prime_total += p_prime

    # Step 4: Relax and update pressure
    p += alpha_p * p_prime_total

    return U, p, mdot, U_faces

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

def is_diagonally_dominant(A):
    # Convert sparse matrix to dense array if needed
    if hasattr(A, 'toarray'):
        A = A.toarray()
    else:
        A = np.asarray(A)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    abs_A = np.abs(A)
    diagonal = np.diag(abs_A)
    off_diagonal_sum = np.sum(abs_A, axis=1) - diagonal
    dominance = np.all(diagonal >= off_diagonal_sum)
    return dominance

def simple_algorithm(mesh, alpha_uv, alpha_p, rho, mu, max_iter, tol, convection_scheme="TVD", limiter="MUSCL", PISO=False, PISO_corrections=1):
    time_start = time.time()

    # cells and faces
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    n_faces = n_internal + n_boundary

    # Mass fluxes
    mdot = np.zeros(n_internal + n_boundary)
    mdot_star = np.zeros(n_internal + n_boundary)
    mdot_2star = np.zeros(n_internal + n_boundary)
    mdot_prime = np.zeros(n_internal + n_boundary)

    # Velocity fields
    U = np.zeros((n_cells, 2))
    U_old = np.zeros((n_cells, 2))
    U_prime = np.zeros((n_cells, 2))
    U_star = np.zeros((n_cells, 2))
    U_2star = np.zeros((n_cells, 2))
    U_old_faces = np.zeros((n_faces, 2))
    U_old_bar = np.zeros((n_faces, 2))
    U_star_rc = np.zeros((n_faces, 2))


    # Pressure field
    p = np.zeros(n_cells)
    p_prime = np.zeros(n_cells)

    # Initialize residual tracking lists
    u_l2norm = np.zeros(max_iter)
    v_l2norm = np.zeros(max_iter)
    continuity_l2norm = np.zeros(max_iter)

    # calculate rho and mu from Reynolds number
    rho = 1.0
    mu = mu 
    mom_solver_u = None
    mom_solver_v = None
    pres_solver = None

    for i in range(max_iter):
             #=============================================================================
        # PRECOMPUTE QUANTITIES
        #=============================================================================
        grad_p = compute_cell_gradients(mesh, p)
        grad_p_bar = interpolate_to_face(mesh, grad_p)
        U_old_bar = interpolate_to_face(mesh, U)
        grad_u = compute_cell_gradients(mesh, U[:,0])
        grad_v = compute_cell_gradients(mesh, U[:,1])


        #=============================================================================
        # ASSEMBLE and solve U-MOMENTUM EQUATIONS
        #=============================================================================
        row, col, data, b_u = assemble_diffusion_convection_matrix(
            mesh,mdot,  grad_u, U_old, rho, mu, 0, phi=U[:,0], scheme=convection_scheme, limiter=limiter, pressure_field=p, grad_pressure_field=grad_p
        )
        A_u = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        A_u_diag = A_u.diagonal()
        rhs_u = b_u - grad_p[:, 0] * mesh.cell_volumes
        rhs_u_unrelaxed = rhs_u.copy()

        # Relax
        relaxed_A_u_diag, rhs_u = relax_momentum_equation(rhs_u, A_u_diag, U_old[:,0], alpha_uv)
        A_u.setdiag(relaxed_A_u_diag)

        # solve
        U_star[:,0], _, _= petsc_solver(A_u, rhs_u)
        A_u.setdiag(A_u_diag)

        # compute normalized residual
        u_l2norm[i], u_residual= compute_residual(A_u.data, A_u.indices, A_u.indptr, U_star[:,0], rhs_u_unrelaxed)

        #=============================================================================
        # ASSEMBLE and solve V-MOMENTUM EQUATIONS
        #=============================================================================
        row, col, data, b_v = assemble_diffusion_convection_matrix(
            mesh,mdot, grad_v, U_old, rho, mu, 1, phi=U[:,1], scheme=convection_scheme, limiter=limiter, pressure_field=p, grad_pressure_field=grad_p
        )
        A_v = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        A_v_diag = A_v.diagonal()
        
        rhs_v = b_v - grad_p[:, 1] * mesh.cell_volumes
        rhs_v_unrelaxed = rhs_v.copy()

        # Relax
        relaxed_A_v_diag, rhs_v = relax_momentum_equation(rhs_v, A_v_diag, U_old[:,1], alpha_uv)
        A_v.setdiag(relaxed_A_v_diag)

        # solve
        U_star[:,1], _, _= petsc_solver(A_v, rhs_v)
        A_v.setdiag(A_v_diag)

        # compute normalized residual
        v_l2norm[i], v_residual = compute_residual(A_v.data, A_v.indices, A_v.indptr, U_star[:,1], rhs_v_unrelaxed)

    

        #=============================================================================
        # RHIE-CHOW VELOCITY
        #=============================================================================

        # Calculate bold D at centroids
        bold_D = bold_Dv_calculation(mesh, A_u_diag, A_v_diag)
        bold_D_bar = interpolate_to_face(mesh, bold_D)
        U_star_bar = interpolate_to_face(mesh, U_star)
        U_star_rc = rhie_chow_velocity(mesh, U_star, U_star_bar, U_old_bar, U_old_faces, grad_p_bar, p, alpha_uv, bold_D_bar)
        #=============================================================================
        # RHIE-CHOW FLUXES
        #=============================================================================
        mdot_star = mdot_calculation(mesh, rho, U_star_rc)

        #=============================================================================
        # PRESSURE CORRECTION EQUATION
        #=============================================================================
        rhs_p = compute_divergence_from_face_fluxes(mesh, mdot_star) 


        continuity_l2norm[i] = np.linalg.norm(rhs_p)

        # pin one pressure node
        rhs_p[0] = 0.0
        row_p, col_p, data_p = assemble_pressure_correction_matrix(mesh, rho)
        A_p = coo_matrix((data_p, (row_p, col_p)), shape=(n_cells, n_cells)).tocsr()

        # First solution of pressure correction equation (orthogonal)
        p_prime, res_p, ksp_1= petsc_solver(A_p, -rhs_p)
        grad_p_prime= compute_cell_gradients(mesh, p_prime)
        grad_p_prime_face = interpolate_to_face(mesh, grad_p_prime)
        # Second solution of pressure correction equation (non-orthogonal correction)
        rhs_p_2 = pressure_correction_loop_term(mesh, rho, grad_p_prime_face)
        p_prime2, res_p_2, _= petsc_solver(A_p, -(rhs_p_2), ksp=ksp_1)
        p_prime = p_prime + p_prime2

        #=============================================================================
        # CORRECT PRESSURE, VELOCITIES and MASS FLUXES
        #=============================================================================
        
        if PISO==True:
            U_2star, p, mdot_2star, U_2star_faces = piso_corrector_loop(
                mesh, A_p, ksp_1, mdot_star, rho, bold_D, U_star_rc, U_star, p, alpha_p, num_corrections=PISO_corrections
            )
            U_old_faces = U_2star_faces
            U = U_2star
            U_old = U_star
            mdot = mdot_2star
        else:
            grad_p_prime= compute_cell_gradients(mesh, p_prime)
            U_prime = velocity_correction(mesh, grad_p_prime, bold_D)
            U_prime_face = interpolate_to_face(mesh, U_prime)
            U_2star_faces = U_star_rc + U_prime_face
            U_2star = U_star + U_prime
            U_old_faces = U_2star_faces
            mdot_prime = mdot_calculation(mesh, rho, U_prime_face)
            mdot_2star = mdot_star + mdot_prime


        # Update fields
        p += alpha_p * p_prime
        U = U_2star
        U_old = U_star
        mdot = mdot_2star
        
        """
        
        x = mesh.cell_centers[:, 0]
        y = mesh.cell_centers[:, 1]
        
        # Create figure with 3x2 subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))
        
        # Plot div_u
        sc1 = ax1.scatter(x, y, c=rhs_p, cmap="viridis", s=8)
        ax1.set_title("RHS of Pressure Equation")
        plt.colorbar(sc1, ax=ax1)
        ax1.set_aspect('equal')
        
        # Plot mdot_star 
        sc2 = ax2.scatter(mesh.face_centers[:, 0], mesh.face_centers[:, 1], c=mdot_star, cmap="viridis", s=8)
        ax2.set_title("Mdot Star")
        plt.colorbar(sc2, ax=ax2)
        ax2.set_aspect('equal')

        # plot mdots correction
        sc3 = ax3.scatter(mesh.face_centers[:, 0], mesh.face_centers[:, 1], c=mdot_prime, cmap="viridis", s=8)
        ax3.set_title("Mdot Prime")
        plt.colorbar(sc3, ax=ax3)
        ax3.set_aspect('equal')
        
        # plot mdot_2star
        sc4 = ax4.scatter(mesh.face_centers[:, 0], mesh.face_centers[:, 1], c=mdot_2star, cmap="viridis", s=8)
        ax4.set_title("Mdot 2 Star") 
        plt.colorbar(sc4, ax=ax4)
        ax4.set_aspect('equal')

        # plot u_prime
        sc5 = ax5.scatter(x, y, c=U_prime[:,0], cmap="viridis", s=8)
        ax5.set_title("U Prime (u-component)")
        plt.colorbar(sc5, ax=ax5)
        ax5.set_aspect('equal')

        # plot u_2star
        sc6 = ax6.scatter(x, y, c=U_2star[:,0], cmap="viridis", s=8)
        ax6.set_title("U 2 Star (u-component)")
        plt.colorbar(sc6, ax=ax6)
        ax6.set_aspect('equal')

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/DEBUG.png", dpi=300)
        plt.close()
        """
        

        #=============================================================================
        # CONVERGENCE CHECK
        #=============================================================================
        print(f"Iteration {i}: u_residuals = {u_l2norm[i]:.3e}, v_residuals = {v_l2norm[i]:.3e}, continuity_residuals = {continuity_l2norm[i]:.3e}")
        if u_l2norm[i] < tol and v_l2norm[i] < tol:
            print(f"Converged at iteration {i}")
            break

        
    u_l2norm = u_l2norm[:i+1]
    v_l2norm = v_l2norm[:i+1]
    continuity_l2norm = continuity_l2norm[:i+1]

    time_end = time.time()
    elapsed_time = time_end - time_start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    return U, p, rhs_p, u_l2norm, v_l2norm, continuity_l2norm, u_residual, v_residual
