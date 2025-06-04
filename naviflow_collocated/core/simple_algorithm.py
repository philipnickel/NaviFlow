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
from naviflow_collocated.core.helpers import bold_Dv_calculation, interpolate_to_face, compute_residual, relax_momentum_equation, apply_mean_zero_constraint, set_pressure_boundaries, compute_l2_norm, get_unique_cells_from_faces
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
        # Step 1: recompute RHS from current flux imbalance
        rhs_p = compute_divergence_from_face_fluxes(mesh, mdot)
        #rhs_p[0] = 0.0  # pin pressure

        # Step 2: pressure correction solve
        p_prime, _, _ = petsc_solver(A_p, -rhs_p)

        # Step 3: velocity correction
        grad_p_prime = compute_cell_gradients(mesh, p_prime)
        U_prime = velocity_correction(mesh, grad_p_prime, bold_D)
        U += U_prime
        U_faces = interpolate_to_face(mesh, U)
        mdot = mdot_calculation(mesh, rho, U_faces)

        # accumulate pressure correction
        p_prime_total += p_prime

    # Final pressure update
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

def simple_algorithm(mesh, alpha_uv, alpha_p, rho, mu, max_iter, tol, convection_scheme="TVD", limiter="MUSCL", PISO=False, PISO_corrections=1, progress_callback=None, interruption_flag=lambda: False, linear_solver_settings=None, n_nonortho_corrections=2):
    # Convert tolerance from string to float if needed
    tol = float(tol)

    # Default linear solver settings if not provided
    if linear_solver_settings is None:
        linear_solver_settings = {
            'momentum': {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000},
            'pressure': {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000}
        }

    # Track how many versions of the JIT-compiled functions have been compiled
    jitted_functions = [
        assemble_diffusion_convection_matrix,
        compute_cell_gradients,
        interpolate_to_face,
        relax_momentum_equation,
        compute_residual,
        bold_Dv_calculation,
        rhie_chow_velocity,
        mdot_calculation,
        compute_divergence_from_face_fluxes,
        assemble_pressure_correction_matrix,
        pressure_correction_loop_term,
        velocity_correction,
        calculate_cylinder_forces
    ]


    time_start = time.time()

    # cells and faces
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    n_faces = n_internal + n_boundary

    # Mass fluxes - ensure C-contiguous arrays
    mdot = np.ascontiguousarray(np.zeros(n_internal + n_boundary))
    mdot_star = np.ascontiguousarray(np.zeros(n_internal + n_boundary))
    mdot_2star = np.ascontiguousarray(np.zeros(n_internal + n_boundary))
    mdot_prime = np.ascontiguousarray(np.zeros(n_internal + n_boundary))

    # Velocity fields - ensure C-contiguous arrays
    U = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_old = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_prime = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_star = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_2star = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_old_faces = np.ascontiguousarray(np.zeros((n_faces, 2)))
    U_old_bar = np.ascontiguousarray(np.zeros((n_faces, 2)))
    U_star_rc = np.ascontiguousarray(np.zeros((n_faces, 2)))

    # Pressure field - ensure C-contiguous arrays
    p = np.ascontiguousarray(np.zeros(n_cells))
    p_prime = np.ascontiguousarray(np.zeros(n_cells))

    # Initialize residual tracking lists - ensure C-contiguous arrays
    u_l2norm = np.ascontiguousarray(np.zeros(max_iter))
    max_u_l2norm = 0.0
    v_l2norm = np.ascontiguousarray(np.zeros(max_iter))
    max_v_l2norm = 0.0 
    continuity_l2norm = np.ascontiguousarray(np.zeros(max_iter))
    max_continuity_l2norm = np.ascontiguousarray(np.zeros(max_iter))

    # Initialize arrays to store force coefficients history - ensure C-contiguous arrays
    cd_history = np.ascontiguousarray(np.zeros(max_iter))
    cl_history = np.ascontiguousarray(np.zeros(max_iter))

    # calculate rho and mu from Reynolds number
    rho = 1.0
    mu = mu 
    mom_solver_u = None
    mom_solver_v = None
    pres_solver = None
    max_mass_imbalance = 0.0

    # Get interior cell indices for L2 norm calculations
    internal_faces = mesh.internal_faces
    #boundary_faces = mesh.boundary_faces
    #all_faces = np.concatenate((internal_faces, boundary_faces))
    interior_cells = get_unique_cells_from_faces(mesh, internal_faces)

    for i in range(max_iter):
        if interruption_flag():
            print(f"Interrupted at iteration {i}. Exiting solver loop.")
            break



        #=============================================================================
        # PRECOMPUTE QUANTITIES
        #=============================================================================
        grad_p = compute_cell_gradients(mesh, np.ascontiguousarray(p))
        grad_p_bar = interpolate_to_face(mesh, np.ascontiguousarray(grad_p))
        U_old_bar = interpolate_to_face(mesh, np.ascontiguousarray(U))
        grad_u = compute_cell_gradients(mesh, np.ascontiguousarray(U[:,0]))
        grad_v = compute_cell_gradients(mesh, np.ascontiguousarray(U[:,1]))

        #=============================================================================
        # ASSEMBLE and solve U-MOMENTUM EQUATIONS
        #=============================================================================
        row, col, data, b_u = assemble_diffusion_convection_matrix(
            mesh, np.ascontiguousarray(mdot),  np.ascontiguousarray(grad_u), np.ascontiguousarray(U_old), rho, mu, 0, phi=np.ascontiguousarray(U[:,0]), scheme=convection_scheme, limiter=limiter, pressure_field=np.ascontiguousarray(p), grad_pressure_field=np.ascontiguousarray(grad_p)
        )
        A_u = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        A_u_diag = A_u.diagonal()
        rhs_u = b_u - grad_p[:, 0] * mesh.cell_volumes
        rhs_u_unrelaxed = rhs_u.copy()

        # Relax
        relaxed_A_u_diag, rhs_u = relax_momentum_equation(np.ascontiguousarray(rhs_u), np.ascontiguousarray(A_u_diag), np.ascontiguousarray(U_old[:,0]), alpha_uv)
        A_u.setdiag(relaxed_A_u_diag)

        # solve
        U_star[:,0], _, _= petsc_solver(A_u, rhs_u, 
            tolerance=linear_solver_settings['momentum']['tolerance'],
            max_iterations=linear_solver_settings['momentum']['max_iterations'],
            solver_type=linear_solver_settings['momentum']['type'],
            preconditioner=linear_solver_settings['momentum']['preconditioner'])
        A_u.setdiag(A_u_diag)

        # compute normalized residual
        u_l2norm[i], u_residual = compute_residual(
            np.ascontiguousarray(A_u.data), np.ascontiguousarray(A_u.indices), np.ascontiguousarray(A_u.indptr), np.ascontiguousarray(U_star[:,0]), np.ascontiguousarray(rhs_u_unrelaxed),
            max_residual=max_u_l2norm, norm_indices=np.ascontiguousarray(interior_cells))

        #=============================================================================
        # ASSEMBLE and solve V-MOMENTUM EQUATIONS
        #=============================================================================
        row, col, data, b_v = assemble_diffusion_convection_matrix(
            mesh, np.ascontiguousarray(mdot), np.ascontiguousarray(grad_v), np.ascontiguousarray(U_old), rho, mu, 1, phi=np.ascontiguousarray(U[:,1]), scheme=convection_scheme, limiter=limiter, pressure_field=np.ascontiguousarray(p), grad_pressure_field=np.ascontiguousarray(grad_p)
        )
        A_v = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        A_v_diag = A_v.diagonal()
        
        rhs_v = b_v - grad_p[:, 1] * mesh.cell_volumes
        rhs_v_unrelaxed = rhs_v.copy()

        # Relax
        relaxed_A_v_diag, rhs_v = relax_momentum_equation(np.ascontiguousarray(rhs_v), np.ascontiguousarray(A_v_diag), np.ascontiguousarray(U_old[:,1]), alpha_uv)
        A_v.setdiag(relaxed_A_v_diag)

        # solve
        U_star[:,1], _, _= petsc_solver(A_v, rhs_v,
            tolerance=linear_solver_settings['momentum']['tolerance'],
            max_iterations=linear_solver_settings['momentum']['max_iterations'],
            solver_type=linear_solver_settings['momentum']['type'],
            preconditioner=linear_solver_settings['momentum']['preconditioner'])
        A_v.setdiag(A_v_diag)

        # compute normalized residual
        v_l2norm[i], v_residual = compute_residual(
            np.ascontiguousarray(A_v.data), np.ascontiguousarray(A_v.indices), np.ascontiguousarray(A_v.indptr), np.ascontiguousarray(U_star[:,1]), np.ascontiguousarray(rhs_v_unrelaxed),
            max_residual=max_v_l2norm, norm_indices=np.ascontiguousarray(interior_cells))

        #=============================================================================
        # RHIE-CHOW VELOCITY
        #=============================================================================

        # Calculate bold D at centroids
        bold_D = bold_Dv_calculation(mesh, np.ascontiguousarray(A_u_diag), np.ascontiguousarray(A_v_diag))
        bold_D_bar = interpolate_to_face(mesh, np.ascontiguousarray(bold_D))
        U_star_bar = interpolate_to_face(mesh, np.ascontiguousarray(U_star))
        U_star_rc = rhie_chow_velocity(mesh, np.ascontiguousarray(U_star), np.ascontiguousarray(U_star_bar), np.ascontiguousarray(U_old_bar), np.ascontiguousarray(U_old_faces), np.ascontiguousarray(grad_p_bar), np.ascontiguousarray(grad_p), np.ascontiguousarray(p), alpha_uv, np.ascontiguousarray(bold_D_bar))
        #=============================================================================
        # RHIE-CHOW FLUXES
        #=============================================================================
        mdot_star = mdot_calculation(mesh, rho, np.ascontiguousarray(U_star_rc))

        #=============================================================================
        # PRESSURE CORRECTION EQUATION
        #=============================================================================
        rhs_p = compute_divergence_from_face_fluxes(mesh, np.ascontiguousarray(mdot_star)) 
        # Calculate total mass flow rate for normalization
        mass_imbalance = compute_l2_norm(rhs_p, interior_cells)
        continuity_l2norm[i] = mass_imbalance
        
        

        # pin one pressure node
        row_p, col_p, data_p = assemble_pressure_correction_matrix(mesh, rho)
        A_p = coo_matrix((data_p, (row_p, col_p)), shape=(n_cells, n_cells)).tocsr()
        #A_p.setdiag(A_p.diagonal() + 1e-20)
        #rhs_p = rhs_p + 1e-20 
        
        cell_centers= mesh.cell_centers
        #pinned_cell_coords = [2.2, 0.0]
        #pinned_cell = np.argmin(np.linalg.norm(cell_centers - pinned_cell_coords, axis=1))
        #pinned_cell = 0
        #random cell index:
        #pinned_cell = np.random.randint(0, n_cells)
        
        # Find rightmost boundary cells
        
        #BC_WALL = 0
        #right_boundary_mask = mesh.boundary_types[:,0] == BC_WALL
        #right_boundary_faces = mesh.boundary_faces[right_boundary_mask]
        #right_boundary_cells = mesh.owner_cells[right_boundary_faces]
        #right_boundary_cells = np.unique(right_boundary_cells)
        
        # Pin all cells on right boundary
        #pinned_cells = right_boundary_cells
        
        #A_p[pinned_cell, :] = 0.0
        #A_p[pinned_cell, pinned_cell] = 1#e-10 
        #A_p = A_p.tocsr()
        #rhs_p[pinned_cell] = 0
        #epsilon = np.max(np.abs(A_p.diagonal()))
        #epsilon = 1e-12 * np.max(np.abs(A_p.diagonal()))
        #A_p.setdiag(A_p.diagonal() + epsilon)
        #rhs_p = rhs_p + 100

            

        # First (orthogonal) solve
        p_prime, res_p, ksp_1 = petsc_solver(
            A_p, -rhs_p,
            tolerance=linear_solver_settings['pressure']['tolerance'],
            max_iterations=linear_solver_settings['pressure']['max_iterations'],
            solver_type=linear_solver_settings['pressure']['type'],
            preconditioner=linear_solver_settings['pressure']['preconditioner'],
            remove_nullspace=True
        )

        # Initialize accumulated correction term
        accumulated_correction = np.zeros_like(rhs_p)
        grad_p_prime = compute_cell_gradients(mesh, np.ascontiguousarray(p_prime))
        grad_p_prime_face = interpolate_to_face(mesh, np.ascontiguousarray(grad_p_prime))
        beta_nonortho = 0.6
        
        for _ in range(n_nonortho_corrections):

            
            # Calculate correction term for this iteration
            correction_term = pressure_correction_loop_term(mesh, rho, np.ascontiguousarray(grad_p_prime_face))
            
            # Accumulate the correction term
            accumulated_correction = correction_term
            
            # Use accumulated correction in RHS
            rhs_corrected = (rhs_p - beta_nonortho * accumulated_correction)

            # Solve with accumulated correction
            p_prime_new, _, _ = petsc_solver(A_p, -rhs_corrected, ksp=ksp_1, remove_nullspace=True)

            p_prime = p_prime_new #(1 - beta_nonortho) * p_prime + beta_nonortho * p_prime_new


            grad_p_prime = compute_cell_gradients(mesh, np.ascontiguousarray(p_prime))
            grad_p_prime_face = interpolate_to_face(mesh, np.ascontiguousarray(grad_p_prime))



   
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
            #grad_p_prime= compute_cell_gradients(mesh, p_prime)
            U_prime = velocity_correction(mesh, grad_p_prime, bold_D)
            U_prime_face = interpolate_to_face(mesh, U_prime)
            U_2star_faces = U_star_rc + U_prime_face
            U_2star = U_star +  U_prime
            U_old_faces = U_2star_faces + U_prime_face
            mdot_prime = mdot_calculation(mesh, rho, U_prime_face, correction=True)
            mdot_2star = mdot_star +  mdot_prime
            p += alpha_p * p_prime

        # Update fields
        #p = set_pressure_boundaries(mesh, p)
        U = U_2star
        U_old = U_2star
        mdot = mdot_2star

        # Calculate force coefficients
        cd, cl = calculate_cylinder_forces(mesh, p, U, mu, rho, U_inf=0.2, D=0.1)  # Using correct U_inf from config
        cd_history[i] = cd
        cl_history[i] = cl

        #=============================================================================
        # CONVERGENCE CHECK
        #=============================================================================
        if progress_callback is not None:
            progress_callback(i, u_l2norm[i], v_l2norm[i], continuity_l2norm[i])
            if getattr(progress_callback.__self__, "diverging", False):
                print(f"Divergence detected at iteration {i}. Aborting SIMPLE loop.")
                break
            if getattr(progress_callback.__self__, "converged", False):
                print(f"Converged at iteration {i}. Exiting solver loop.")
                break


    u_l2norm = u_l2norm[:i+1]
    v_l2norm = v_l2norm[:i+1]
    continuity_l2norm = continuity_l2norm[:i+1]
    cd_history = cd_history[:i+1]
    cl_history = cl_history[:i+1]

    time_end = time.time()
    elapsed_time = time_end - time_start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    return U, p, rhs_p, u_l2norm, v_l2norm, continuity_l2norm, u_residual, v_residual, mdot, cd_history, cl_history 

@njit(cache=True)
def calculate_cylinder_forces(mesh, p, U, mu, rho, U_inf, D):
    """
    Calculate lift and drag coefficients for flow past a cylinder.
    
    Parameters
    ----------
    mesh : Mesh object
    p : ndarray
        Pressure field
    U : ndarray
        Velocity field at cell centers
    mu : float
        Dynamic viscosity
    rho : float
        Fluid density
    U_inf : float
        Free stream velocity
    D : float
        Cylinder diameter
        
    Returns
    -------
    cd : float
        Drag coefficient
    cl : float
        Lift coefficient
    """
    # Find cylinder surface faces (obstacle boundary faces)
    cylinder_faces = []
    for i in range(mesh.boundary_faces.shape[0]):
        f = mesh.boundary_faces[i]
        if mesh.boundary_types[f, 0] == 4:  # Obstacle boundary (BC_OBSTACLE = 4)
            cylinder_faces.append(f)
    
    cylinder_faces = np.array(cylinder_faces)
    
    # Precompute gradients (outside the loop)
    grad_p = compute_cell_gradients(mesh, p)
    grad_u = compute_cell_gradients(mesh, U[:, 0])
    grad_v = compute_cell_gradients(mesh, U[:, 1])
    
    # Initialize force components
    Fx = 0.0  # Drag force
    Fy = 0.0  # Lift force
    
    # Calculate forces on each cylinder face
    for idx, f in enumerate(cylinder_faces):
        # Get face normal and area
        S_f = mesh.vector_S_f[f]
        S_f_norm = np.sqrt(S_f[0]**2 + S_f[1]**2 + 1e-14)
        n = S_f / S_f_norm  # Unit normal vector
        
        # Get pressure at face
        P = mesh.owner_cells[f]
        d_Cb = mesh.d_Cb[f]
        vec_Cb = d_Cb * n
        p_f = p[P] + np.dot(grad_p[P], vec_Cb)
        
        # Interpolate velocity gradients to face
        grad_u_f = grad_u[P] + np.dot(grad_u[P], vec_Cb)
        grad_v_f = grad_v[P] + np.dot(grad_v[P], vec_Cb)
        
        # Calculate strain rate tensor components
        du_dx = grad_u_f[0]
        du_dy = grad_u_f[1]
        dv_dx = grad_v_f[0]
        dv_dy = grad_v_f[1]
        
        # Viscous stress tensor
        tau_xx = 2 * mu * du_dx
        tau_xy = mu * (du_dy + dv_dx)
        tau_yx = tau_xy
        tau_yy = 2 * mu * dv_dy
        
        # Calculate force components
        # Pressure force (force on cylinder)
        Fx += p_f * S_f[0]  # Pressure contribution to drag
        Fy += p_f * S_f[1]  # Pressure contribution to lift
        
        # Viscous force (force on cylinder)
        Fx += -(tau_xx * n[0] + tau_xy * n[1]) * S_f_norm
        Fy += -(tau_yx * n[0] + tau_yy * n[1]) * S_f_norm
    
    # Calculate dynamic pressure
    q_inf = 0.5 * rho * U_inf**2
    
    # Calculate coefficients using D*L as reference area (L=1.0 for 2D)
    cd = Fx / (q_inf * D)
    cl = Fy / (q_inf * D)
    
    return cd, cl

