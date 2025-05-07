"""
Unstructured mesh generator using Gmsh.

Generates a 2D triangular mesh for a rectangular domain with optional internal
obstacles (circle, rectangle, or custom shape like a NACA airfoil). Applies
boundary tagging.
"""

import gmsh
import numpy as np


def generate(
    Lx=1.0,
    Ly=1.0,
    n_cells=1000,
    ratio=2.5,
    obstacle=None,
    output_filename=None,
    model_name="unstructured_gmsh",
    refinement_factors={},
):
    if refinement_factors is None:
        raise ValueError("refinement_factors must be provided")

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.model.add(model_name)

    # Estimate global mesh size from target n_cells, adaptively correcting for refinement impact
    area = Lx * Ly
    # Compute effective h using maximum refinement bleed
    all_refinements = []

    boundary_refinement = refinement_factors.get("boundaries", {})
    corner_refinement = refinement_factors.get("corners", {})
    obstacle_refinement = refinement_factors.get("obstacle", None)

    for k, (factor, bleed) in boundary_refinement.items():
        all_refinements.append(factor * bleed)
    for k, (factor, bleed) in corner_refinement.items():
        all_refinements.append(factor * bleed)
    if obstacle_refinement:
        factor, bleed = obstacle_refinement
        all_refinements.append(factor * bleed)

    max_ref_factor = max(all_refinements) if all_refinements else 1.0
    correction_factor = 1 + 0.25 * max_ref_factor
    h_effective = np.sqrt(area / (n_cells / correction_factor))

    # Limit minimum size to avoid huge mesh counts
    h_min = max(h_effective / ratio, h_effective / 10)
    h_max = h_effective

    print(
        f"[Mesh Size] h_effective={h_effective:.5f}, h_min={h_min:.5f}, h_max={h_max:.5f}"
    )

    # Domain boundary points (no need for local mesh size here)
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

    # --- Boundary refinement
    boundary_edge_map = {
        "bottom": l1,
        "right": l2,
        "top": l3,
        "left": l4,
    }

    boundary_refinement = refinement_factors.get("boundaries", {})
    corner_refinement = refinement_factors.get("corners", {})

    corner_points = {
        "bottom_left": p1,
        "bottom_right": p2,
        "top_right": p3,
        "top_left": p4,
    }

    # Combine corner and boundary fields separately
    corner_fields = []
    boundary_fields = []
    diag = np.sqrt(Lx**2 + Ly**2)
    bleed_correction_boundary = diag * 2
    bleed_correction_corner = bleed_correction_boundary * 1

    for name, edge in boundary_edge_map.items():
        if name in boundary_refinement:
            ref_factor, bleed_frac = boundary_refinement[name]
            print(
                f"[Boundary Refinement] Processing {name}: factor={ref_factor}, bleed={bleed_frac}"
            )
            df = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(df, "EdgesList", [edge])
            gmsh.model.mesh.field.setNumber(df, "Sampling", 150)
            tf = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(tf, "IField", df)
            gmsh.model.mesh.field.setNumber(tf, "LcMin", h_min / ref_factor)
            gmsh.model.mesh.field.setNumber(tf, "LcMax", h_max)
            gmsh.model.mesh.field.setNumber(tf, "DistMin", 0.01 * Lx)
            gmsh.model.mesh.field.setNumber(
                tf, "DistMax", bleed_frac * bleed_correction_boundary
            )
            boundary_fields.append(tf)

    for name, pid in corner_points.items():
        if name in corner_refinement:
            ref_factor, bleed_frac = corner_refinement[name]
            print(
                f"[Corner Refinement] Processing {name}: factor={ref_factor}, bleed={bleed_frac}"
            )
            df = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(df, "PointsList", [pid])
            tf = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(tf, "IField", df)
            gmsh.model.mesh.field.setNumber(tf, "LcMin", h_min / ref_factor)
            gmsh.model.mesh.field.setNumber(tf, "LcMax", h_max)
            gmsh.model.mesh.field.setNumber(tf, "DistMin", 0.01 * Lx)
            gmsh.model.mesh.field.setNumber(
                tf, "DistMax", bleed_frac * bleed_correction_corner
            )
            corner_fields.append(tf)

    # --- Obstacle refinement
    if (
        obstacle
        and "refinement_factors" in refinement_factors
        and "obstacle" in refinement_factors["refinement_factors"]
    ):
        ref_factor, bleed_frac = refinement_factors["refinement_factors"]["obstacle"]
        print(f"[Obstacle Refinement] factor={ref_factor}, bleed={bleed_frac}")
        obs_edges = obstacle_lines
        if obs_edges:
            # Use Attractor + Threshold for better control
            attractor_field = gmsh.model.mesh.field.add("Attractor")
            gmsh.model.mesh.field.setNumbers(attractor_field, "EdgesList", obs_edges)
            gmsh.model.mesh.field.setNumber(attractor_field, "Sampling", 150)

            threshold_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_field, "IField", attractor_field)
            gmsh.model.mesh.field.setNumber(
                threshold_field, "LcMin", h_min / ref_factor
            )
            gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", h_max)
            gmsh.model.mesh.field.setNumber(
                threshold_field, "DistMin", 0.02 * obstacle.get("radius", 0.05)
            )
            gmsh.model.mesh.field.setNumber(
                threshold_field,
                "DistMax",
                bleed_frac * obstacle.get("radius", 0.05) * 2,
            )
            corner_fields.append(threshold_field)

    corner_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(corner_min, "FieldsList", corner_fields)

    boundary_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(boundary_min, "FieldsList", boundary_fields)

    # Use both boundary and corner refinement fields for final field
    final_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(
        final_field, "FieldsList", [boundary_min, corner_min]
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(final_field)

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

    # Mesh options
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    print("[Gmsh] Finalizing and generating mesh with applied refinement fields...")
    gmsh.model.mesh.generate(2)

    if output_filename:
        gmsh.write(output_filename)
        print(f"✓ Mesh saved to {output_filename}")
