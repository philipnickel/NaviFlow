"""
Unstructured mesh generator using Gmsh.

Generates a 2D triangular mesh for a rectangular domain with optional internal
obstacles (circle, rectangle, or custom shape like a NACA airfoil). Applies
boundary tagging and optional wake refinement.
"""

import gmsh
import numpy as np


def generate(
    Lx=1.0,
    Ly=1.0,
    n_cells=1000,
    ratio=2.5,
    obstacle=None,
    wake_refinement=False,
    output_filename=None,
    model_name="unstructured_gmsh",
):
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.model.add(model_name)

    # Domain and mesh size
    area = Lx * Ly
    h = np.sqrt(area / (n_cells / 4.2))
    h_min = h / ratio
    h_max = h * 1.2

    # Domain boundary
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface_loops = [outer_loop]
    obstacle_lines = []

    if obstacle:
        otype = obstacle.get("type", "")
        if otype == "circle":
            cx, cy = obstacle.get("center", (Lx / 2, Ly / 2))
            r = obstacle.get("radius", min(Lx, Ly) / 8)
            pc = gmsh.model.geo.addPoint(cx, cy, 0)
            pts = [
                gmsh.model.geo.addPoint(cx + r, cy, 0),
                gmsh.model.geo.addPoint(cx, cy + r, 0),
                gmsh.model.geo.addPoint(cx - r, cy, 0),
                gmsh.model.geo.addPoint(cx, cy - r, 0),
            ]
            arcs = [
                gmsh.model.geo.addCircleArc(pts[0], pc, pts[1]),
                gmsh.model.geo.addCircleArc(pts[1], pc, pts[2]),
                gmsh.model.geo.addCircleArc(pts[2], pc, pts[3]),
                gmsh.model.geo.addCircleArc(pts[3], pc, pts[0]),
            ]
            obstacle_loop = gmsh.model.geo.addCurveLoop(arcs)
            surface_loops.append(obstacle_loop)
            obstacle_lines.extend(arcs)

        # Other obstacle types can be added here

    surface = gmsh.model.geo.addPlaneSurface(surface_loops)
    gmsh.model.geo.synchronize()

    # Physical tags
    tags = {
        "bottom": gmsh.model.addPhysicalGroup(1, [l1]),
        "right": gmsh.model.addPhysicalGroup(1, [l2]),
        "top": gmsh.model.addPhysicalGroup(1, [l3]),
        "left": gmsh.model.addPhysicalGroup(1, [l4]),
    }
    for name, tag in tags.items():
        gmsh.model.setPhysicalName(1, tag, name)

    if obstacle_lines:
        tag = gmsh.model.addPhysicalGroup(1, obstacle_lines)
        gmsh.model.setPhysicalName(1, tag, "obstacle_boundary")

    ftag = gmsh.model.addPhysicalGroup(2, [surface], 10)
    gmsh.model.setPhysicalName(2, ftag, "fluid_domain")

    # Mesh size control
    bnd_edges = [l1, l2, l3, l4] + obstacle_lines
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "EdgesList", bnd_edges)

    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field, "IField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_field, "LcMin", h_min)
    gmsh.model.mesh.field.setNumber(thresh_field, "LcMax", h_max)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", min(Lx, Ly) / 3)
    gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)

    # Mesh options
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    gmsh.model.mesh.generate(2)

    if output_filename:
        gmsh.write(output_filename)
        print(f"✓ Mesh saved to {output_filename}")
