# naviflow_collocated/mesh/mesh_data.py

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


def mesh_to_data(mesh) -> MeshData:
    return MeshData(
        face_areas=mesh.get_face_areas(),
        face_normals=mesh.get_face_normals(),
        face_centers=mesh.get_face_centers(),
        face_interp_factors=np.array(
            [mesh.get_face_interpolation_factors(i) for i in range(mesh.n_faces)]
        ),
        cell_volumes=mesh.get_cell_volumes(),
        cell_centers=mesh.get_cell_centers(),
        owner_cells=mesh.get_owner_neighbor()[0],
        neighbor_cells=mesh.get_owner_neighbor()[1],
        boundary_face_to_name=mesh.boundary_face_to_name.copy(),
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
    )
