"""Mesh generator for a structured, uniform quadrilateral grid
with correct physical tagging for the fluid domain and boundary curves.

Changelog
---------
* **v1.3** – Fixed Python `SyntaxError` on multiple annotated assignment
  and removed unused variable.  Now compatible with Python ≥3.8.
"""

import numpy as np
import gmsh
from .mesh_data import MeshData2D
from .mesh_helpers import (
    calculate_face_normals,
    build_owner_neighbor,
    calculate_cell_centers,
    calculate_face_centers,
    calculate_face_areas,
    calculate_cell_volumes,
)


def generate(
    L: float = 1.0,
    nx: int = 50,
    ny: int = 50,
    lc: float = 0.1,
    output_filename: str | None = None,
    model_name: str = "structured_uniform",
) -> MeshData2D:
    """Generate a recombined transfinite quadrilateral mesh on a square [0,L]²."""

    gmsh.clear()
    gmsh.model.add(model_name)

    # ------------------------------------------------------------------
    # 1. Geometry definition
    # ------------------------------------------------------------------
    def add_point(x: float, y: float):
        return gmsh.model.geo.addPoint(x, y, 0.0, lc)

    p0, p1, p2, p3 = add_point(0, 0), add_point(L, 0), add_point(L, L), add_point(0, L)
    l0 = gmsh.model.geo.addLine(p0, p1)  # bottom
    l1 = gmsh.model.geo.addLine(p1, p2)  # right
    l2 = gmsh.model.geo.addLine(p2, p3)  # top
    l3 = gmsh.model.geo.addLine(p3, p0)  # left

    cloop = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])
    surf = gmsh.model.geo.addPlaneSurface([cloop])

    # ------------------------------------------------------------------
    # 2. Transfinite & recombine settings (geo‑mesh namespace)
    # ------------------------------------------------------------------
    gmsh.model.geo.mesh.setTransfiniteCurve(l0, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surf)
    gmsh.model.geo.mesh.setRecombine(2, surf)

    gmsh.model.geo.synchronize()  # make entities visible for physical groups

    # ------------------------------------------------------------------
    # 3. Physical groups
    # ------------------------------------------------------------------
    for name, line in zip(["bottom", "right", "top", "left"], [l0, l1, l2, l3], strict=True):
        tag = gmsh.model.addPhysicalGroup(1, [line])
        gmsh.model.setPhysicalName(1, tag, f"{name}_boundary")

    fluid_tag = gmsh.model.addPhysicalGroup(2, [surf])
    gmsh.model.setPhysicalName(2, fluid_tag, "fluid_domain")

    # ------------------------------------------------------------------
    # 4. Mesh generation
    # ------------------------------------------------------------------
    gmsh.model.mesh.generate(2)

    # ------------------------------------------------------------------
    # 5. Optional export
    # ------------------------------------------------------------------
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
    gmsh.option.setNumber("Mesh.SaveElementTagType", 2)

    if output_filename:
        gmsh.write(output_filename)
        print(f"\u2713  Mesh saved to {output_filename}")

    # ------------------------------------------------------------------
    # 6. Convert to NumPy arrays
    # ------------------------------------------------------------------
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(coords).reshape(-1, 3)[:, :2]
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    elem_types, _, elem_nodes = gmsh.model.mesh.getElements()
    try:
        idx_quad = list(elem_types).index(3)
    except ValueError as e:
        raise RuntimeError("No quadrilateral elements found – check transfinite & recombine settings.") from e

    quads = np.fromiter((node_map[int(n)] for n in elem_nodes[idx_quad]), dtype=np.int32).reshape(-1, 4)

    # ------------------------------------------------------------------
    # 7. Boundary edges
    # ------------------------------------------------------------------
    edge_nodes: list[int] = []
    edge_tags: list[int] = []
    tag_map: dict[int, int] = {}

    for dim, entity in gmsh.model.getEntities(1):
        phys = gmsh.model.getPhysicalGroupsForEntity(dim, entity)
        if not phys:
            continue
        phys_tag = phys[0]
        e_types, tags_nested, nodes_nested = gmsh.model.mesh.getElements(dim, entity)
        if 1 not in e_types:
            continue
        idx_line = list(e_types).index(1)
        edge_tags.extend(tags_nested[idx_line])
        edge_nodes.extend(nodes_nested[idx_line])
        for t in tags_nested[idx_line]:
            tag_map[int(t)] = phys_tag

    edges_np = np.asarray([node_map[int(n)] for n in edge_nodes], dtype=np.int64).reshape(-1, 2)
    edge_tags_np = np.asarray(edge_tags, dtype=np.int64)

    # ------------------------------------------------------------------
    # 8. Build FVM geometric quantities
    # ------------------------------------------------------------------
    cell_centers = calculate_cell_centers(points, quads)
    face_centers = calculate_face_centers(points, edges_np)
    face_areas = calculate_face_areas(points, edges_np)
    cell_volumes = calculate_cell_volumes(points, quads)
    owner, neighbor = build_owner_neighbor(quads, edges_np)
    bface_idx = np.where(neighbor == -1)[0]
    btypes = np.asarray([tag_map.get(int(t), 0) for t in edge_tags_np[bface_idx]], dtype=np.int64)

    return MeshData2D(
        cell_volumes=cell_volumes,
        face_areas=face_areas,
        face_normals=calculate_face_normals(points, edges_np),
        face_centers=face_centers,
        cell_centers=cell_centers,
        owner_cells=owner,
        neighbor_cells=neighbor,
        boundary_faces=bface_idx,
        boundary_types=btypes,
        boundary_values=np.zeros((len(edges_np), 2)),
        boundary_patches=np.zeros(len(edges_np), dtype=np.int64),
        face_interp_factors=np.full(len(edges_np), 0.5),
        d_CF=np.zeros((len(edges_np), 2)),
        non_ortho_correction=np.zeros((len(edges_np), 2)),
        is_structured=True,
        is_orthogonal=True,
        is_conforming=True,
    )
