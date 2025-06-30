import os
import numpy as np
import numpy.linalg as la
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
from naviflow_collocated.core.helpers import bold_Dv_calculation, interpolate_to_face, compute_residual, relax_momentum_equation, apply_mean_zero_constraint, set_pressure_boundaries, compute_l2_norm, get_unique_cells_from_faces
import time
from numba import njit

# Set up persistent Numba cache for faster subsequent runs
os.environ['NUMBA_CACHE_DIR'] = os.path.expanduser('~/.numba_cache')
os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True)

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
    U_faces_new : ndarray
        Updated face velocities with Rhie-Chow correction.
    """
    n_cells = mesh.cell_volumes.shape[0]
    U = U_star.copy()
    U_faces = U_star_rc.copy()
    mdot = mdot_start.copy()
    p_prime_total = np.zeros(n_cells)

    for correction_iter in range(num_corrections):
        # Step 1: recompute RHS from current flux imbalance
        rhs_p = compute_divergence_from_face_fluxes(mesh, mdot)

        # Step 2: pressure correction solve
        p_prime, _, _ = petsc_solver(A_p, -rhs_p, ksp=ksp, remove_nullspace=True)

        # Step 3: velocity correction at cell centers
        grad_p_prime = compute_cell_gradients(mesh, p_prime, weighted=True, weight_exponent=0.5, use_limiter=False)
        U_prime = velocity_correction(mesh, grad_p_prime, bold_D)
        U += U_prime

        # Step 4: face velocity correction (maintaining Rhie-Chow)
        # Interpolate pressure correction gradient to faces
        grad_p_prime_face = interpolate_to_face(mesh, grad_p_prime)
        
        # Face velocity correction using the same bold_D interpolated to faces
        bold_D_face = interpolate_to_face(mesh, bold_D)
        U_prime_face = np.zeros_like(U_faces)
        U_prime_face[:, 0] = -bold_D_face[:, 0] * grad_p_prime_face[:, 0]
        U_prime_face[:, 1] = -bold_D_face[:, 1] * grad_p_prime_face[:, 1]
        
        # Update face velocities
        U_faces += U_prime_face
        
        # Step 5: correct mass flux directly
        mdot_prime = mdot_calculation(mesh, rho, U_prime_face, correction=True)
        mdot += mdot_prime

        # accumulate pressure correction
        p_prime_total += p_prime

    # Final pressure update (apply under-relaxation)
    p += alpha_p * p_prime_total

    return U, p, mdot, U_faces



@njit(cache=True)
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

def simple_algorithm(mesh, alpha_uv, alpha_p, rho, mu, max_iter, tol, convection_scheme="TVD", limiter="MUSCL", PISO=False, PISO_corrections=1, progress_callback=None, interruption_flag=lambda: False, linear_solver_settings=None, n_nonortho_corrections=2, transient=False, dt=0.0, end_time=0.0, time_scheme="Euler", results_dir=None, save_interval=0):
    # Convert tolerance from string to float if needed
    tol = float(tol)

    # Default linear solver settings if not provided
    if linear_solver_settings is None:
        linear_solver_settings = {
            'momentum': {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000},
            'pressure': {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000}
        }

    time_start = time.time()

    # cells and faces
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    n_faces = n_internal + n_boundary

    # Mass fluxes
    mdot = np.ascontiguousarray(np.zeros(n_faces))
    mdot_star = np.ascontiguousarray(np.zeros(n_faces))

    # Velocity fields
    U = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_prev_iter = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_star = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_old_time = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_old_faces = np.ascontiguousarray(np.zeros((n_faces, 2)))

    # Pressure field
    p = np.ascontiguousarray(np.zeros(n_cells))

    # Residuals
    u_l2norm = np.zeros(max_iter)
    v_l2norm = np.zeros(max_iter)
    continuity_l2norm = np.zeros(max_iter)
    max_u_l2norm = 0.0
    max_v_l2norm = 0.0
    max_mass_imbalance = 0.0
    
    # Initialize residual fields
    u_residual_field = np.zeros(n_cells)
    v_residual_field = np.zeros(n_cells)
    continuity_field = np.zeros(n_cells)
    
    internal_cells = get_unique_cells_from_faces(mesh, mesh.internal_faces)
    
    num_time_steps = int(end_time / dt) if transient and dt > 0 else 1
    if transient:
        print(f"Running transient simulation for {num_time_steps} time steps with dt={dt}")

    final_iter_count = 0
    is_converged = False
    interrupted = False

    for time_step in range(num_time_steps):
        if interrupted:
            break
            
        if transient:
            print(f"Time step {time_step + 1}/{num_time_steps}, Time: {time_step * dt:.4f}s")
            U_old_time = U.copy()

        explicit_crank_nic_u = np.zeros(n_cells)
        explicit_crank_nic_v = np.zeros(n_cells)

        if transient and time_scheme == "CrankNicolson":
            # Pre-calculate the explicit part of the Crank-Nicolson scheme
            grad_p_old = compute_cell_gradients(mesh, p, weighted=True, weight_exponent=0.5, use_limiter=False)
            grad_u_old = compute_cell_gradients(mesh, U_old_time[:,0], weighted=True, weight_exponent=0.5, use_limiter=True)
            grad_v_old = compute_cell_gradients(mesh, U_old_time[:,1], weighted=True, weight_exponent=0.5, use_limiter=True)

            # U-momentum spatial residual
            row_u, col_u, data_u, b_u = assemble_diffusion_convection_matrix(
                mesh, mdot, grad_u_old, U_old_time, rho, mu, 0, phi=U_old_time[:,0], 
                scheme=convection_scheme, limiter=limiter, pressure_field=p, 
                grad_pressure_field=grad_p_old, transient=False
            )
            A_u_old = coo_matrix((data_u, (row_u, col_u)), shape=(n_cells, n_cells)).tocsr()
            rhs_u_old_spatial = b_u - (-grad_p_old[:, 0] * mesh.cell_volumes)
            residual_u_spatial = rhs_u_old_spatial - (A_u_old @ U_old_time[:, 0])
            explicit_crank_nic_u = 0.5 * residual_u_spatial

            # V-momentum spatial residual
            row_v, col_v, data_v, b_v = assemble_diffusion_convection_matrix(
                mesh, mdot, grad_v_old, U_old_time, rho, mu, 1, phi=U_old_time[:,1], 
                scheme=convection_scheme, limiter=limiter, pressure_field=p, 
                grad_pressure_field=grad_p_old, transient=False
            )
            A_v_old = coo_matrix((data_v, (row_v, col_v)), shape=(n_cells, n_cells)).tocsr()
            rhs_v_old_spatial = b_v - (-grad_p_old[:, 1] * mesh.cell_volumes)
            residual_v_spatial = rhs_v_old_spatial - (A_v_old @ U_old_time[:, 1])
            explicit_crank_nic_v = 0.5 * residual_v_spatial

        for i in range(max_iter):
            final_iter_count = i + 1
            if interruption_flag():
                print(f"Interrupted at iteration {i}. Exiting solver loop.")
                interrupted = True
                break

            #=============================================================================
            # PRECOMPUTE QUANTITIES
            #=============================================================================
            grad_p = compute_cell_gradients(mesh, p, weighted=True, weight_exponent=0.5, use_limiter=False)
            grad_p_bar = interpolate_to_face(mesh, grad_p)
            U_old_bar = interpolate_to_face(mesh, U)
            grad_u = compute_cell_gradients(mesh, U[:,0], weighted=True, weight_exponent=0.5, use_limiter=True)
            grad_v = compute_cell_gradients(mesh, U[:,1], weighted=True, weight_exponent=0.5, use_limiter=True)

            #=============================================================================
            # ASSEMBLE and solve U-MOMENTUM EQUATIONS
            #=============================================================================
            phi_arg_u = U_old_time[:,0] if transient else U[:,0]
            row, col, data, b_u = assemble_diffusion_convection_matrix(
                mesh, mdot, grad_u, U_prev_iter, rho, mu, 0, phi=phi_arg_u, 
                scheme=convection_scheme, limiter=limiter, pressure_field=p, 
                grad_pressure_field=grad_p, dt=dt, transient=transient, time_scheme=time_scheme
            )
            A_u = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
            A_u_diag = A_u.diagonal()
            rhs_u = b_u - grad_p[:, 0] * mesh.cell_volumes
            if transient and time_scheme == "CrankNicolson":
                rhs_u += explicit_crank_nic_u
            rhs_u_unrelaxed = rhs_u.copy()

            # Relax
            relaxed_A_u_diag, rhs_u = relax_momentum_equation(rhs_u, A_u_diag, U_prev_iter[:,0], alpha_uv)
            A_u.setdiag(relaxed_A_u_diag)

            # solve
            U_star[:,0], _, _= petsc_solver(A_u, rhs_u, **linear_solver_settings['momentum'])
            A_u.setdiag(A_u_diag) # Restore original diagonal

            # Store residual field for u-momentum
            u_residual_field = A_u @ U_star[:,0] - rhs_u_unrelaxed

            # compute normalized residual
            u_l2norm[i], max_u_l2norm = compute_residual(A_u.data, A_u.indices, A_u.indptr, U_star[:,0], rhs_u_unrelaxed, max_u_l2norm, internal_cells)

            #=============================================================================
            # ASSEMBLE and solve V-MOMENTUM EQUATIONS
            #=============================================================================
            phi_arg_v = U_old_time[:,1] if transient else U[:,1]
            row, col, data, b_v = assemble_diffusion_convection_matrix(
                mesh, mdot, grad_v, U_prev_iter, rho, mu, 1, phi=phi_arg_v, 
                scheme=convection_scheme, limiter=limiter, pressure_field=p, 
                grad_pressure_field=grad_p, dt=dt, transient=transient, time_scheme=time_scheme
            )
            A_v = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
            A_v_diag = A_v.diagonal()
            rhs_v = b_v - grad_p[:, 1] * mesh.cell_volumes
            if transient and time_scheme == "CrankNicolson":
                rhs_v += explicit_crank_nic_v
            rhs_v_unrelaxed = rhs_v.copy()

            # Relax
            relaxed_A_v_diag, rhs_v = relax_momentum_equation(rhs_v, A_v_diag, U_prev_iter[:,1], alpha_uv)
            A_v.setdiag(relaxed_A_v_diag)

            # solve
            U_star[:,1], _, _= petsc_solver(A_v, rhs_v, **linear_solver_settings['momentum'])
            A_v.setdiag(A_v_diag) # Restore original diagonal

            # Store residual field for v-momentum
            v_residual_field = A_v @ U_star[:,1] - rhs_v_unrelaxed

            # compute normalized residual
            v_l2norm[i], max_v_l2norm = compute_residual(A_v.data, A_v.indices, A_v.indptr, U_star[:,1], rhs_v_unrelaxed, max_v_l2norm, internal_cells)

            #=============================================================================
            # RHIE-CHOW, MASS FLUX, and PRESSURE CORRECTION
            #=============================================================================
            bold_D = bold_Dv_calculation(mesh, A_u_diag, A_v_diag)
            bold_D_bar = interpolate_to_face(mesh, bold_D)
            U_star_bar = interpolate_to_face(mesh, U_star)
            U_star_rc = rhie_chow_velocity(mesh, U_star, U_star_bar, U_old_bar, U_old_faces, grad_p_bar, grad_p, p, alpha_uv, bold_D_bar)
            
            mdot_star = mdot_calculation(mesh, rho, U_star_rc)
            
            row, col, data = assemble_pressure_correction_matrix(mesh, rho)
            A_p = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
            rhs_p = -compute_divergence_from_face_fluxes(mesh, mdot_star)
            
            # Store continuity residual field
            continuity_field = rhs_p.copy()

            continuity_l2norm[i], max_mass_imbalance = compute_residual(A_p.data, A_p.indices, A_p.indptr, np.zeros_like(p), rhs_p, max_mass_imbalance, internal_cells)

            p_prime, _, ksp_1 = petsc_solver(A_p, rhs_p, remove_nullspace=True, **linear_solver_settings['pressure'])

            #=============================================================================
            # CORRECTIONS
            #=============================================================================
            if PISO:
                U_corrected, p_corrected, mdot_corrected, U_faces_corrected = piso_corrector_loop(
                    mesh, A_p, ksp_1, mdot_star, rho, bold_D, U_star_rc, U_star, p, alpha_p, num_corrections=PISO_corrections
                )
            else: # SIMPLE
                grad_p_prime = compute_cell_gradients(mesh, p_prime, weighted=True, weight_exponent=0.5, use_limiter=False)
                U_prime = velocity_correction(mesh, grad_p_prime, bold_D)
                U_corrected = U_star + U_prime
                
                U_prime_face = interpolate_to_face(mesh, U_prime)
                U_faces_corrected = U_star_rc + U_prime_face
                
                mdot_prime = mdot_calculation(mesh, rho, U_prime_face, correction=True)
                mdot_corrected = mdot_star + mdot_prime
                
                p_corrected = p + alpha_p * p_prime

            # Update fields for next iteration
            p = p_corrected
            U = U_corrected
            U_prev_iter = U_corrected.copy()
            mdot = mdot_corrected
            U_old_faces = U_faces_corrected

            if progress_callback:
                progress_callback.update(i, u_l2norm[i], v_l2norm[i], continuity_l2norm[i])

            is_converged = u_l2norm[i] < tol and v_l2norm[i] < tol #and continuity_l2norm[i] < tol)
            if is_converged:
                if not transient: # Only print for steady-state to avoid verbose output
                    print(f"Converged at iteration {i}")
                break
        
        # After the inner loop, if running a transient simulation, issue a warning
        # if the solution for the current time step did not converge.
        if transient and not is_converged and not interrupted:
            print(f"  > Warning: Time step {time_step + 1} did not converge within {max_iter} iterations. "
                  f"Final residuals: u={u_l2norm[i]:.2e}, v={v_l2norm[i]:.2e}, cont={continuity_l2norm[i]:.2e}")

        # For a steady-state simulation, convergence means we can exit the main loop.
        if is_converged and not transient:
            break

        if transient and results_dir and save_interval > 0 and (time_step + 1) % save_interval == 0:
            # Create dedicated directories for U and p transient data
            transient_base_dir = os.path.join(results_dir, "transient_data")
            p_dir = os.path.join(transient_base_dir, "p")
            U_dir = os.path.join(transient_base_dir, "U")
            os.makedirs(p_dir, exist_ok=True)
            os.makedirs(U_dir, exist_ok=True)

            time_val = (time_step + 1) * dt
            print(f"Saving transient solution at time {time_val:.4f}s (Time Step {time_step + 1})")
            
            p_path = os.path.join(p_dir, f"p_{time_step + 1:04d}.npy")
            U_path = os.path.join(U_dir, f"U_{time_step + 1:04d}.npy")
            np.save(p_path, p)
            np.save(U_path, U)

    time_end = time.time()
    print(f"Solver finished in {time_end - time_start:.2f} seconds.")

    # Save final state for compatibility with existing post-processing
    if results_dir:
        np.save(os.path.join(results_dir, "p_final.npy"), p)
        np.save(os.path.join(results_dir, "U_final.npy"), U)

    # Trim residual history
    u_l2norm = u_l2norm[:final_iter_count]
    v_l2norm = v_l2norm[:final_iter_count]
    continuity_l2norm = continuity_l2norm[:final_iter_count]

    return p, U, mdot, (u_l2norm, v_l2norm, continuity_l2norm), final_iter_count, is_converged, u_residual_field, v_residual_field, continuity_field 