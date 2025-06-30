from numba import njit
import numpy as np

BC_WALL = 0
BC_DIRICHLET = 1
BC_INLET = 2
BC_OUTLET = 4
BC_NEUMANN = 3


@njit()
def compute_divergence_from_face_fluxes(mesh, face_fluxes):
    """
    Compute divergence (mass imbalance) per cell from face mass fluxes.
    
    Each face flux is assumed to be rho * u_f ⋅ S_f, pointing from owner to neighbor.
    """
    n_cells = mesh.cell_volumes.shape[0]
    divergence = np.zeros(n_cells)
    n_boundary = mesh.boundary_faces.shape[0]

    for f in range(len(face_fluxes)):
        C = mesh.owner_cells[f]
        F = mesh.neighbor_cells[f]

        flux = face_fluxes[f]

        divergence[C] += flux  # flux leaving C (owner)
        if F >= 0:
            divergence[F] -= flux  # flux entering F (neighbor)

    
    #for i in range(n_boundary):
    #    f = mesh.boundary_faces[i]
    #    if mesh.boundary_types[f,0] == BC_OUTLET:
    #        divergence[mesh.owner_cells[f]] = 0.0
    


    return divergence
