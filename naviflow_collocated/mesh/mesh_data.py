import numpy as np
from dataclasses import dataclass


@dataclass
class MeshData:
    face_areas: np.ndarray
    face_normals: np.ndarray
    face_centers: np.ndarray
    face_interp_factors: np.ndarray
    cell_volumes: np.ndarray
    cell_centers: np.ndarray
    owner_cells: np.ndarray
    neighbor_cells: np.ndarray
    boundary_face_to_name: dict
    boundary_name_to_cell_indices: dict
    boundary_name_to_cell_mask: dict
    global_dirichlet_mask: np.ndarray


def mesh_to_data(mesh) -> MeshData:
    owner_cells, neighbor_cells = mesh.get_owner_neighbor()
    face_interp_factors = np.array(
        [mesh.get_face_interpolation_factors(i) for i in range(mesh.n_faces)]
    )

    boundary_face_to_name = mesh.boundary_face_to_name.copy()
    boundary_name_to_cell_indices = {}

    for face_idx, name in boundary_face_to_name.items():
        if name not in boundary_name_to_cell_indices:
            boundary_name_to_cell_indices[name] = []
        boundary_name_to_cell_indices[name].append(owner_cells[face_idx])

    boundary_name_to_cell_mask = {}
    global_mask = np.zeros(mesh.n_cells, dtype=bool)

    for name, indices in boundary_name_to_cell_indices.items():
        arr = np.array(indices, dtype=np.int32)
        mask = np.zeros(mesh.n_cells, dtype=bool)
        mask[arr] = True
        boundary_name_to_cell_indices[name] = arr
        boundary_name_to_cell_mask[name] = mask
        global_mask[arr] = True

    return MeshData(
        face_areas=mesh.get_face_areas(),
        face_normals=mesh.get_face_normals(),
        face_centers=mesh.get_face_centers(),
        face_interp_factors=face_interp_factors,
        cell_volumes=mesh.get_cell_volumes(),
        cell_centers=mesh.get_cell_centers(),
        owner_cells=owner_cells,
        neighbor_cells=neighbor_cells,
        boundary_face_to_name=boundary_face_to_name,
        boundary_name_to_cell_indices=boundary_name_to_cell_indices,
        boundary_name_to_cell_mask=boundary_name_to_cell_mask,
        global_dirichlet_mask=global_mask,
    )


def save_mesh_data(path: str, mesh_data: MeshData):
    np.savez_compressed(
        path,
        face_areas=mesh_data.face_areas,
        face_normals=mesh_data.face_normals,
        face_centers=mesh_data.face_centers,
        face_interp_factors=mesh_data.face_interp_factors,
        cell_volumes=mesh_data.cell_volumes,
        cell_centers=mesh_data.cell_centers,
        owner_cells=mesh_data.owner_cells,
        neighbor_cells=mesh_data.neighbor_cells,
        boundary_face_to_name=np.array(
            list(mesh_data.boundary_face_to_name.items()), dtype=object
        ),
        boundary_name_to_cell_indices=np.array(
            list(mesh_data.boundary_name_to_cell_indices.items()), dtype=object
        ),
        boundary_name_to_cell_mask=np.array(
            list(mesh_data.boundary_name_to_cell_mask.items()), dtype=object
        ),
        global_dirichlet_mask=mesh_data.global_dirichlet_mask,
    )


def load_mesh_data(path: str) -> MeshData:
    data = np.load(path, allow_pickle=True)
    return MeshData(
        face_areas=data["face_areas"],
        face_normals=data["face_normals"],
        face_centers=data["face_centers"],
        face_interp_factors=data["face_interp_factors"],
        cell_volumes=data["cell_volumes"],
        cell_centers=data["cell_centers"],
        owner_cells=data["owner_cells"],
        neighbor_cells=data["neighbor_cells"],
        boundary_face_to_name=dict(data["boundary_face_to_name"]),
        boundary_name_to_cell_indices=dict(data["boundary_name_to_cell_indices"]),
        boundary_name_to_cell_mask=dict(data["boundary_name_to_cell_mask"]),
        global_dirichlet_mask=data["global_dirichlet_mask"],
    )
