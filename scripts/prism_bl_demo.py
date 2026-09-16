"""
prism_bl_demo.py

Proof that gmsh can build real prism boundary layers on a 3D body and export
them to SU2, using gmsh.model.geo.extrudeBoundaryLayer.

This is NOT the "BoundaryLayer" size field. That field is 2D only and aborts
generate(3) with "Only 2D Boundary Layers are supported", which is what
final_meshing.py documents. extrudeBoundaryLayer is a different mechanism: it
extrudes an already-meshed surface along its mesh normals into prisms, and the
remaining volume is then filled with tets.

Recipe, in order, because the order is the whole trick:
  1. build and cut the geometry, classify wall vs farfield faces
  2. generate(2) FIRST, so a wall surface mesh exists to extrude from
  3. extrudeBoundaryLayer on the wall surfaces with cumulative heights
  4. build a surface loop from the extrusion TOP surfaces plus the farfield
     faces only. Including the lateral quads gives
     "non-manifold quad boundaries not supported yet"
  5. generate(3) to fill the rest with tets

Verified on gmsh 4.15.2: 27552 prisms + 22098 tets, first layer 0.0500 mm
against a 0.0500 mm target, zero non-positive cells, and prisms present in the
.su2 as element type 13.

See the notes at the end before trusting this on a real airframe.
"""
import collections
import gmsh

nlay, r, h0 = 42, 1.1112, 0.05e-3
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("bl")

# thin slab body inside a farfield box, like the BWB case in miniature
body = gmsh.model.occ.addBox(0, 0, -0.4, 47, 18, 2.5)
box  = gmsh.model.occ.addBox(-20, -5, -20, 100, 30, 40)
out, _ = gmsh.model.occ.cut([(3, box)], [(3, body)])
gmsh.model.occ.synchronize()

vol = out[0][1]
faces = [t for d, t in gmsh.model.getBoundary(out, oriented=False)]
wall, far = [], []
for f in faces:
    bb = gmsh.model.getBoundingBox(2, f)
    if bb[0] > -1 and bb[3] < 48 and bb[1] > -1 and bb[4] < 19 and bb[5] < 3:
        wall.append(f)
    else:
        far.append(f)
print("wall faces %d, farfield faces %d" % (len(wall), len(far)))

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)
gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0)
gmsh.model.mesh.generate(2)
print("surface mesh done")

heights, acc = [], 0.0
for i in range(nlay):
    acc += h0 * r**i
    heights.append(acc)
print("stack %d layers, total %.2f mm" % (nlay, acc*1e3))

ex = gmsh.model.geo.extrudeBoundaryLayer([(2, f) for f in wall],
                                         [1]*nlay, heights, True)
# extrude returns, per input surface: (2, top), (3, volume), then laterals.
# Only the top surfaces bound the outer tet region; the laterals are internal.
top_surfs = [ex[i][1] for i in range(len(ex)-1)
             if ex[i][0] == 2 and ex[i+1][0] == 3]
top = [t for d, t in ex if d == 3]
gmsh.model.geo.synchronize()
print("extrude returned %d entities, %d surfaces, %d volumes"
      % (len(ex), len(top_surfs), len(top)))

sl = gmsh.model.geo.addSurfaceLoop(top_surfs + far)
v = gmsh.model.geo.addVolume([sl])
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)

t = collections.Counter()
for dim, tag in gmsh.model.getEntities(3):
    et, _, _ = gmsh.model.mesh.getElements(3, tag)
    for e in et:
        t[e] += len(gmsh.model.mesh.getElements(3, tag)[1][list(et).index(e)])
names = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}
print("VOLUME ELEMENTS:", {names.get(k, k): v for k, v in t.items()})
# verify: first layer height, inverted cells, su2 export
import numpy as np
nt, nc, _ = gmsh.model.mesh.getNodes()
xyz = np.asarray(nc).reshape(-1,3)
lut = {int(t): xyz[i] for i, t in enumerate(nt)}
worst = 1e30; hmin = 1e30; neg = 0
for dim, tag in gmsh.model.getEntities(3):
    et, etg, enn = gmsh.model.mesh.getElements(3, tag)
    for typ, tags, nodes in zip(et, etg, enn):
        if typ == 6:  # prism
            p = np.array([[lut[int(n)] for n in nodes[i*6:(i+1)*6]]
                          for i in range(len(tags))])
            h = np.linalg.norm(p[:,3]-p[:,0], axis=1)
            hmin = min(hmin, h.min())
        q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSICN"))
        worst = min(worst, q.min()); neg += int((q <= 0).sum())
print("thinnest prism layer: %.4f mm (target first layer %.4f mm)" % (hmin*1e3, h0*1e3))
print("worst minSICN %.4f, non-positive cells %d" % (worst, neg))
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("bl_test.msh")
gmsh.write("bl_test.su2")
gmsh.finalize()

import collections
c = collections.Counter()
inel = False
for line in open("bl_test.su2"):
    if line.startswith("NELEM"): inel = True; continue
    if line.startswith(("NPOIN","NMARK")): inel = False
    if inel and line.split(): c[line.split()[0]] += 1
names = {"10":"tet","12":"hex","13":"prism","14":"pyramid"}
print("SU2 file volume elements:", {names.get(k,k): v for k,v in c.items()})


# ----------------------------------------------------------------------
# Notes for the real geometry
#
# extrudeBoundaryLayer marches along mesh normals with no collision handling.
# It is safe here because the body is convex and the 37 mm stack is tiny
# against the geometry scale. On a real airframe check, in this order:
#   - the sharp trailing edge, where opposing normals converge
#   - any concave junction, where the layers collide first
#   - the symmetry plane, where the wall meets it: layers must slide along the
#     plane rather than extrude off it, or the symmetry marker is broken
# Always re-check element quality and the non-positive cell count afterwards.
#
# Cost: prism count = wall triangles x layers. A wall of 855k triangles at 42
# layers is 36M prisms before a single tet is placed. Coarsen the surface mesh
# or reduce the layer count to fit the budget.
#
# minSICN is near zero for these cells by construction: a 1000:1 aspect ratio
# prism is meant to be sliver-shaped. Judge them on positive volume and on the
# first layer height, not on an isotropic quality metric.
