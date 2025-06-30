import numpy as np
from numba import njit
from naviflow_collocated.discretization.gradient.leastSquares import compute_cell_gradients

def calculate_pressure_difference(mesh, p):
    """
    Calculate pressure difference between front and back stagnation points
    using pb = pc + (∇p)_C^(n) · d_Cb for face pressure calculation
    """
    # Find cylinder surface faces
    cylinder_faces = []
    for i in range(mesh.boundary_faces.shape[0]):
        f = mesh.boundary_faces[i]
        if mesh.boundary_types[f, 0] == 4:  # Obstacle boundary
            cylinder_faces.append(f)
    
    cylinder_faces = np.array(cylinder_faces)
    face_centers = mesh.face_centers[cylinder_faces]
    
    # Get owner cells
    owner_cells = mesh.owner_cells[cylinder_faces]
    
    # Calculate pressure gradients
    grad_p = compute_cell_gradients(mesh, np.ascontiguousarray(p), weighted=True, weight_exponent=0.5, use_limiter=False)
    
    # Stagnation points
    front_point = np.array([0.15, 0.2])
    back_point = np.array([0.25, 0.2])
    
    # Find closest faces
    front_dists = np.sum((face_centers - front_point)**2, axis=1)
    back_dists = np.sum((face_centers - back_point)**2, axis=1)
    front_face_idx = np.argmin(front_dists)
    back_face_idx = np.argmin(back_dists)
    
    # Get cells and faces
    front_cell = owner_cells[front_face_idx]
    back_cell = owner_cells[back_face_idx]
    front_face = cylinder_faces[front_face_idx]
    back_face = cylinder_faces[back_face_idx]
    
    # Calculate d_Cb vectors
    d_Cb_front = mesh.face_centers[front_face] - mesh.cell_centers[front_cell]
    d_Cb_back = mesh.face_centers[back_face] - mesh.cell_centers[back_cell]
    
    # Calculate face pressures: pb = pc + (∇p)_C^(n) · d_Cb
    p_front = p[front_cell] + np.dot(grad_p[front_cell], d_Cb_front)
    p_back = p[back_cell] + np.dot(grad_p[back_cell], d_Cb_back)
    
    # Print diagnostic info
    print(f"\nPressure Difference Analysis:")
    print(f"Front stagnation face center: ({face_centers[front_face_idx,0]:.6f}, {face_centers[front_face_idx,1]:.6f})")
    print(f"Back stagnation face center: ({face_centers[back_face_idx,0]:.6f}, {face_centers[back_face_idx,1]:.6f})")
    print(f"Front cell pressure: {p[front_cell]:.6f}")
    print(f"Front gradient correction: {np.dot(grad_p[front_cell], d_Cb_front):.6f}")
    print(f"Front face pressure: {p_front:.6f}")
    print(f"Back cell pressure: {p[back_cell]:.6f}")
    print(f"Back gradient correction: {np.dot(grad_p[back_cell], d_Cb_back):.6f}")
    print(f"Back face pressure: {p_back:.6f}")
    print(f"Pressure difference: {p_front - p_back:.6f}")
    
    return p_front - p_back

@njit(cache=True)
def calculate_cylinder_forces(mesh, p, U, mu, rho, U_ref, D):
    """
    Calculate lift and drag coefficients for flow past a cylinder using Schäfer's formulation.
    
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
    U_ref : float
        Reference velocity (Um from Schäfer benchmark)
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
        if mesh.boundary_types[f, 0] == 4:  # Obstacle boundary
            cylinder_faces.append(f)
    
    cylinder_faces = np.array(cylinder_faces)
    
    # Initialize force components following Schäfer's formulation
    F_D_pressure = 0.0  # Pressure contribution to drag
    F_D_viscous = 0.0   # Viscous contribution to drag
    F_L_pressure = 0.0  # Pressure contribution to lift
    F_L_viscous = 0.0   # Viscous contribution to lift
    
    # Calculate pressure and velocity gradients using only internal faces
    grad_p = compute_cell_gradients(mesh, np.ascontiguousarray(p), weighted=True, weight_exponent=0.5, use_limiter=False)
    grad_u = compute_cell_gradients(mesh, np.ascontiguousarray(U[:,0]), weighted=True, weight_exponent=0.5, use_limiter=False)
    grad_v = compute_cell_gradients(mesh, np.ascontiguousarray(U[:,1]), weighted=True, weight_exponent=0.5, use_limiter=False)
    
    # Calculate forces on each cylinder face
    for f in cylinder_faces:
        # Get face area vector (points out of fluid domain)
        S_f = mesh.vector_S_f[f]
        dA = np.sqrt(S_f[0]**2 + S_f[1]**2)
        
        # Unit normal vector (pointing OUT of fluid domain)
        nx = S_f[0] / dA
        ny = S_f[1] / dA
        
        # Get owner cell
        P = mesh.owner_cells[f]
        
        # Calculate d_Cb vector from cell center to boundary face
        d_Cb = mesh.face_centers[f] - mesh.cell_centers[P]
        
        # Calculate face pressure using Taylor expansion: pb = pc + (∇p)_C^(n) · d_Cb
        p_f = p[P] + np.dot(grad_p[P], d_Cb)
        
        # Calculate velocity gradients at the wall
        # For velocity gradients, we need to account for the no-slip condition
        # The wall velocity is zero, so we can use this in our extrapolation
        # ∇u_wall = ∇u_cell + O(d)  (assuming linear variation)
        du_dx_wall = grad_u[P,0]
        du_dy_wall = grad_u[P,1]
        dv_dx_wall = grad_v[P,0]
        dv_dy_wall = grad_v[P,1]
        
        # Calculate tangential velocity gradient ∂vt/∂n at the wall
        # vt = v·t = u*ny - v*nx (dot product with tangent vector)
        # ∂vt/∂n = (∂u/∂x*nx + ∂u/∂y*ny)*ny - (∂v/∂x*nx + ∂v/∂y*ny)*nx
        dvt_dn = ((du_dx_wall*nx + du_dy_wall*ny)*ny - 
                  (dv_dx_wall*nx + dv_dy_wall*ny)*nx)
        
        # Calculate viscous stress term τ = ρν∂vt/∂n
        tau = rho * mu * dvt_dn
        
        # Add contributions exactly as per Schäfer's formulation:
        # FD = ∫(ρν∂vt/∂n·ny - P·nx) dS
        F_D_pressure += -p_f * nx * dA    # -P·nx term
        F_D_viscous += tau * ny * dA      # ρν∂vt/∂n·ny term
        
        # FL = -∫(ρν∂vt/∂n·nx + P·ny) dS
        F_L_pressure += -p_f * ny * dA    # -P·ny term
        F_L_viscous += -tau * nx * dA     # -ρν∂vt/∂n·nx term
    
    # Calculate coefficients exactly as per Schäfer
    # cD = 2FD/(ρŪ²D) where FD is force ON cylinder FROM fluid
    # cL = 2FL/(ρŪ²D) where FL is force ON cylinder FROM fluid
    # Since our forces are computed as acting FROM cylinder TO fluid, we need to flip signs
    denominator = rho * U_ref**2 * D
    cd = -2.0 * (F_D_pressure + F_D_viscous) / denominator  # Flip sign to get force ON cylinder
    cl = -2.0 * (F_L_pressure + F_L_viscous) / denominator  # Flip sign to get force ON cylinder
    
    return cd, cl 