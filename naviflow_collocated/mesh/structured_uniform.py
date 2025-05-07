"""Structured uniform quadrilateral mesh generator using Gmsh.

Generates a transfinite, recombined quadrilateral mesh on [0, L]² with
physical tagging of fluid domain and each boundary segment (left, right, top, bottom).
"""

import gmsh


def generate(
    L: float = 1.0,
    nx: int = 50,
    ny: int = 50,
    output_filename: str | None = None,
    model_name: str = "structured_uniform",
) -> None:
    """Generate and save a structured quadrilateral mesh with physical tags.

    Parameters
    ----------
    L : float
        Domain size (square domain from (0,0) to (L,L)).
    nx, ny : int
        Number of cells in x and y directions.
    output_filename : str, optional
        Path to save the mesh file (must be .msh).
    model_name : str
        Name of the Gmsh model (used internally).
    """

    lc = L / max(nx, ny)

    gmsh.clear()
    gmsh.model.add(model_name)

    # 1. Geometry
    def add_point(x, y):
        return gmsh.model.geo.addPoint(x, y, 0.0, lc)

    p = [add_point(0, 0), add_point(L, 0), add_point(L, L), add_point(0, L)]
    lines = [
        gmsh.model.geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)
    ]  # bottom→left
    cloop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([cloop])

    # 2. Transfinite mesh + recombination (for quads)
    gmsh.model.geo.mesh.setTransfiniteCurve(lines[0], nx + 1)  # bottom
    gmsh.model.geo.mesh.setTransfiniteCurve(lines[2], nx + 1)  # top
    gmsh.model.geo.mesh.setTransfiniteCurve(lines[1], ny + 1)  # right
    gmsh.model.geo.mesh.setTransfiniteCurve(lines[3], ny + 1)  # left
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)

    gmsh.model.geo.synchronize()

    # 3. Physical tagging
    for name, line in zip(["bottom", "right", "top", "left"], lines):
        tag = gmsh.model.addPhysicalGroup(1, [line])
        gmsh.model.setPhysicalName(1, tag, name)

    fluid_tag = gmsh.model.addPhysicalGroup(2, [surface], 10)
    gmsh.model.setPhysicalName(2, fluid_tag, "fluid_domain")

    # 4. Mesh generation and export
    gmsh.model.mesh.generate(2)

    if output_filename:
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
        gmsh.write(output_filename)
        print(f"✓ Mesh saved to {output_filename}")
