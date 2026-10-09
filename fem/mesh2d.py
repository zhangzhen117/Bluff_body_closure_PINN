"""Build a 2D half-domain body-fitted mesh for a shape with gmsh-python and load
it into dolfinx (mesh + facet tags). Half domain x in [X0,X1], y in [0,YT];
body upper-half = wall; y=0 outside body = symmetry. Facet tags:
  inlet=1, outlet=2, top=3, symmetry=4, wall=5 ; cell tag fluid=10.
"""
import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio
import geom_np

X0, X1, YT = -10.0, 15.0, 10.0
TAGS = dict(inlet=1, outlet=2, top=3, symmetry=4, wall=5, fluid=10)


def build_mesh(name, hwall=0.02, hfar=0.8, wake_h=0.08, show=False):
    ch = geom_np.upper_chain(name)
    xd, xu = ch[0, 0], ch[-1, 0]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    g = gmsh.model.geo
    P = lambda x, y: g.addPoint(float(x), float(y), 0.0, hfar)
    pib, pit = P(X0, 0.0), P(X0, YT)
    pot, pob = P(X1, YT), P(X1, 0.0)
    bp = [P(x, y) for x, y in ch]
    L = g.addLine
    l_inlet = L(pib, pit); l_top = L(pit, pot); l_outlet = L(pot, pob)
    l_sd = L(pob, bp[0])
    l_wall = [L(bp[i], bp[i + 1]) for i in range(len(bp) - 1)]
    l_su = L(bp[-1], pib)
    loop = g.addCurveLoop([l_inlet, l_top, l_outlet, l_sd] + l_wall + [l_su])
    surf = g.addPlaneSurface([loop])
    g.synchronize()
    gmsh.model.addPhysicalGroup(1, [l_inlet], TAGS["inlet"])
    gmsh.model.addPhysicalGroup(1, [l_outlet], TAGS["outlet"])
    gmsh.model.addPhysicalGroup(1, [l_top], TAGS["top"])
    gmsh.model.addPhysicalGroup(1, [l_sd, l_su], TAGS["symmetry"])
    gmsh.model.addPhysicalGroup(1, l_wall, TAGS["wall"])
    gmsh.model.addPhysicalGroup(2, [surf], TAGS["fluid"])
    # size field: fine near wall + wake box, coarse far
    f = gmsh.model.mesh.field
    f.add("Distance", 1); f.setNumbers(1, "CurvesList", l_wall); f.setNumber(1, "Sampling", 200)
    f.add("Threshold", 2); f.setNumber(2, "InField", 1)
    f.setNumber(2, "SizeMin", hwall); f.setNumber(2, "SizeMax", hfar)
    f.setNumber(2, "DistMin", 0.05); f.setNumber(2, "DistMax", 4.0)
    f.add("Box", 3); f.setNumber(3, "VIn", wake_h); f.setNumber(3, "VOut", hfar)
    f.setNumber(3, "XMin", xu - 0.5); f.setNumber(3, "XMax", 8.0)
    f.setNumber(3, "YMin", 0.0); f.setNumber(3, "YMax", 2.0); f.setNumber(3, "Thickness", 1.5)
    f.add("Min", 4); f.setNumbers(4, "FieldsList", [2, 3]); f.setAsBackgroundMesh(4)
    for o in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber("Mesh." + o, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    res = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    mesh = res.mesh if hasattr(res, "mesh") else res[0]
    ct = res.cell_tags if hasattr(res, "cell_tags") else res[1]
    ft = res.facet_tags if hasattr(res, "facet_tags") else res[2]
    return mesh, ct, ft


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "tri_long_0_dns"
    mesh, ct, ft = build_mesh(name)
    nc = mesh.topology.index_map(2).size_local
    print(f"{name}: {nc} cells")
    for k, v in TAGS.items():
        if k == "fluid": continue
        n = int(np.sum(ft.values == v))
        print(f"  facet tag {k:9s}({v}): {n} facets")
