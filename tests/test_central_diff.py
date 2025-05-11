import numpy as np
from naviflow_collocated.mesh.mesh_loader import load_mesh
from naviflow_collocated.discretization.diffusion.central_diff import compute_diffusive_flux

def u_sine(xy):
    return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])

def grad_sine(xy):
    gx = np.pi * np.cos(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])
    gy = np.pi * np.sin(np.pi * xy[:, 0]) * np.cos(np.pi * xy[:, 1])
    return np.stack([gx, gy], axis=1)

def test_single_face_diffusion_flux():
    mesh_path = "meshing/experiments/sanityCheck/structuredUniform/coarse/sanityCheck_uniform_coarse.msh"
    bc_path = "shared_configs/domain/sanityCheckDiffusion.yaml"
    mesh = load_mesh(mesh_path, bc_path)

    # Pick a face (ideally not at boundary)
    f = mesh.internal_faces[0]
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]
    
    mu = 1.0
    grad_u = grad_sine(mesh.cell_centers)
    
    # Evaluate face-center analytical value
    x_P = mesh.cell_centers[P]
    x_N = mesh.cell_centers[N]
    x_f = mesh.face_centers[f]
    phi_P = u_sine(mesh.cell_centers[[P]])[0]
    phi_N = u_sine(mesh.cell_centers[[N]])[0]
    delta = np.linalg.norm(x_N - x_P)

    # Analytical normal vector and flux
    S_f = mesh.face_normals[f]
    n_hat = S_f / (np.linalg.norm(S_f) + 1e-14)
    phi_grad_exact = (phi_N - phi_P) / (delta + 1e-14)
    flux_exact = -mu * phi_grad_exact * np.linalg.norm(S_f)

    # From your numerical scheme
    a_PP, a_PN, b_corr = compute_diffusive_flux(f, grad_u, mesh, mu)
    flux_num = a_PN * phi_N + a_PP * phi_P + b_corr

    print(f"Face {f}")
    print(f"Analytic flux:     {flux_exact:.6f}")
    print(f"Numerical flux:    {flux_num:.6f}")
    print(f"Difference:        {abs(flux_exact - flux_num):.2e}")

    assert np.isclose(flux_num, flux_exact, atol=1e-6), "Flux mismatch!"

def test_all_faces_diffusion_flux():
    mesh_path = "meshing/experiments/sanityCheck/structuredUniform/coarse/sanityCheck_uniform_coarse.msh"
    bc_path = "shared_configs/domain/sanityCheckDiffusion.yaml"
    mesh = load_mesh(mesh_path, bc_path)
    
    mu = 1.0
    grad_u = grad_sine(mesh.cell_centers)
    
    # Evaluate analytical solution at cell centers
    phi = u_sine(mesh.cell_centers)
    
    errors = []
    for f in mesh.internal_faces:
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]
        
        # Cell-center values and distance
        x_P = mesh.cell_centers[P]
        x_N = mesh.cell_centers[N]
        phi_P = phi[P]
        phi_N = phi[N]
        delta = np.linalg.norm(x_N - x_P)
        
        # Analytical normal vector and flux
        S_f = mesh.face_normals[f]
        phi_grad_exact = (phi_N - phi_P) / (delta + 1e-14)
        flux_exact = -mu * phi_grad_exact * np.linalg.norm(S_f)
        
        # Numerical flux using our scheme
        a_PP, a_PN, b_corr = compute_diffusive_flux(f, grad_u, mesh, mu)
        flux_num = a_PN * phi_N + a_PP * phi_P + b_corr
        
        error = abs(flux_exact - flux_num)
        errors.append(error)
    
    print(f"Total internal faces tested: {len(mesh.internal_faces)}")
    print(f"Max error: {np.max(errors):.2e}")
    print(f"Mean error: {np.mean(errors):.2e}")
    print(f"Median error: {np.median(errors):.2e}")
    
    # Check if all errors are within tolerance
    assert np.all(np.array(errors) < 1e-5), "Excessive flux errors found across mesh"

if __name__ == "__main__":
    test_single_face_diffusion_flux()
    print("\nTesting all internal faces...")
    test_all_faces_diffusion_flux()
