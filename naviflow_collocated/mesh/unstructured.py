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
    use_quads=False,
):
    if refinement_factors is None:
        raise ValueError("refinement_factors must be provided")

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.model.add(model_name)

    # Select geometry kernel: OpenCASCADE (`occ`) for speed, or classical (`geo`)
    # geom = gmsh.model.occ
    geom = gmsh.model.occ

    # Estimate global mesh size from target n_cells, adaptively correcting for refinement impact
    area = Lx * Ly
    # Compute effective h using maximum refinement bleed
    all_refinements = []

    boundary_refinement = refinement_factors.get("boundaries", {})
    corner_refinement = refinement_factors.get("corners", {})
    if "obstacle" in refinement_factors:
        obstacle_refinement = refinement_factors["obstacle"]
    else:
        obstacle_refinement = None

    for k, (factor, bleed) in boundary_refinement.items():
        all_refinements.append(factor * bleed)
    for k, (factor, bleed) in corner_refinement.items():
        all_refinements.append(factor * bleed)
    if obstacle_refinement:
        factor, bleed = obstacle_refinement
        all_refinements.append(factor * bleed)

    total_refinement_intensity = sum(all_refinements)
    correction_factor = 1 + 0.5 * total_refinement_intensity
    h_effective = np.sqrt(area / (n_cells / correction_factor))

    # Limit minimum size to avoid huge mesh counts
    h_min = max(h_effective / ratio, h_effective / 10) * 2
    h_max = h_effective * 2

    print(
        f"[Mesh Size] h_effective={h_effective:.5f}, h_min={h_min:.5f}, h_max={h_max:.5f}"
    )

    # Domain boundary points (no need for local mesh size here)
    p1 = geom.addPoint(0, 0, 0)
    p2 = geom.addPoint(Lx, 0, 0)
    p3 = geom.addPoint(Lx, Ly, 0)
    p4 = geom.addPoint(0, Ly, 0)

    l1 = geom.addLine(p1, p2)
    l2 = geom.addLine(p2, p3)
    l3 = geom.addLine(p3, p4)
    l4 = geom.addLine(p4, p1)

    outer_loop = geom.addCurveLoop([l1, l2, l3, l4])
    surface_loops = [outer_loop]
    obstacle_lines = []

    if obstacle is not None:
        otype = obstacle.get("type", "")
        if otype == "circle":
            cx, cy = obstacle.get("center", (Lx / 2, Ly / 2))
            r = obstacle.get("radius", min(Lx, Ly) / 8)

            # OpenCASCADE can represent the circle as a single analytic edge
            circ = geom.addCircle(cx, cy, 0, r)
            obstacle_loop = geom.addCurveLoop([circ])
            surface_loops.append(obstacle_loop)
            obstacle_lines.extend([circ])

        elif otype == "arbitrary":
            coords = obstacle.get("object_geometry", [])
            scale = obstacle.get("scale", 1.0)
            center = obstacle.get("center", (Lx / 2, Ly / 2))
            print(f"[Arbitrary Obstacle] {len(coords)} coords received")
            print(f"[Arbitrary Obstacle] center = {center}, scale = {scale}")
            x0, y0 = center
            pts = []
            for x, y in coords:
                pts.append(geom.addPoint(x0 + x * scale, y0 + y * scale, 0))

            # Add the first point again at the end to close the loop
            if coords[0] != coords[-1]:
                pts.append(pts[0])  # Close the loop by adding first point again

            print(f"[Arbitrary Obstacle] Created {len(pts)} points")

            # Create a single spline for the entire airfoil shape
            # lines = [geom.addSpline(pts)]
            lines = []
            for i in range(len(pts) - 1):
                lines.append(gmsh.model.occ.addLine(pts[i], pts[i + 1]))
            obstacle_loop = gmsh.model.occ.addWire(lines)
            # lines = [geom.addPolyLine(pts)]
            print(f"[Arbitrary Obstacle] Creating loop with {len(lines)} lines")
            obstacle_loop = geom.addCurveLoop(lines)
            surface_loops.append(obstacle_loop)
            obstacle_lines.extend(lines)

    surface = geom.addPlaneSurface(surface_loops)
    geom.synchronize()

    gmsh.option.setNumber("Mesh.RecombineAll", 0)  # use triangles globally by default
    if use_quads and obstacle_lines:
        for line in obstacle_lines:
            gmsh.model.mesh.setRecombine(1, line)  # recombine only around obstacle
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

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
    if obstacle and obstacle_refinement:
        ref_factor, bleed_frac = obstacle_refinement
        print(f"[Obstacle Refinement] factor={ref_factor}, bleed={bleed_frac}")
        obs_edges = obstacle_lines
        if obs_edges:
            bl_field = gmsh.model.mesh.field.add("BoundaryLayer")
            gmsh.model.mesh.field.setNumbers(bl_field, "CurvesList", obs_edges)
            gmsh.model.mesh.field.setNumber(bl_field, "Size", h_min / ref_factor)
            gmsh.model.mesh.field.setNumber(bl_field, "SizeFar", h_max)
            bl_thickness = bleed_frac * 2 * obstacle.get("radius", 1.0)
            gmsh.model.mesh.field.setNumber(bl_field, "Thickness", bl_thickness)
            gmsh.model.mesh.field.setNumber(bl_field, "Ratio", 1.3)
            gmsh.model.mesh.field.setNumber(bl_field, "Quads", 1)
            print(
                f"[Obstacle BoundaryLayer] Size={h_min / ref_factor:.5f}, SizeFar={h_max:.5f}, Thickness={bl_thickness:.5f}"
            )
            corner_fields.append(bl_field)

            # Add Distance + Threshold refinement to extend beyond boundary layer
            df = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(df, "EdgesList", obs_edges)

            tf = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(tf, "IField", df)
            gmsh.model.mesh.field.setNumber(tf, "LcMin", h_min / ref_factor)
            gmsh.model.mesh.field.setNumber(tf, "LcMax", h_max)
            tf_dist = bleed_frac * diag
            gmsh.model.mesh.field.setNumber(tf, "DistMin", 0.01 * Lx)
            gmsh.model.mesh.field.setNumber(tf, "DistMax", tf_dist)
            print(f"[Obstacle Threshold Field] DistMax={tf_dist:.3f}")
            boundary_fields.append(tf)

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
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(2)

    if output_filename:
        gmsh.write(output_filename)
        print(f"✓ Mesh saved to {output_filename}")
