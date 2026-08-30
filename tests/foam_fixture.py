"""Write a small structured 2-D OpenFOAM case (polyMesh + hy2Foam-style
fields) so the reader and profile extractor can be tested without a solver."""
import os
import numpy as np

from pse import species as sp

HDR = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v1706                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       %s;
    location    "%s";
    object      %s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def _write_list(path, cls, loc, obj, items, fmt):
    with open(path, "w") as f:
        f.write(HDR % (cls, loc, obj))
        f.write("%d\n(\n" % len(items))
        for it in items:
            f.write(fmt(it) + "\n")
        f.write(")\n")


def write_mesh(case, x, y, dz=1e-3):
    d = os.path.join(case, "constant", "polyMesh")
    os.makedirs(d, exist_ok=True)
    NX, NY = len(x) - 1, len(y) - 1

    def pid(i, j, k):
        return i + (NX + 1) * (j + (NY + 1) * k)

    pts = np.zeros(((NX + 1) * (NY + 1) * 2, 3))
    for k in range(2):
        for j in range(NY + 1):
            for i in range(NX + 1):
                pts[pid(i, j, k)] = (x[i], y[j], k * dz)
    _write_list(os.path.join(d, "points"), "vectorField", "constant", "points",
                pts, lambda p: "(%.10g %.10g %.10g)" % tuple(p))

    def cid(i, j):
        return i * NY + j

    faces, owner, neigh = [], [], []
    for i in range(NX):
        for j in range(NY):
            c = cid(i, j)
            if j < NY - 1:                                # +y internal face
                faces.append([pid(i, j + 1, 0), pid(i, j + 1, 1),
                              pid(i + 1, j + 1, 1), pid(i + 1, j + 1, 0)])
                owner.append(c); neigh.append(cid(i, j + 1))
            if i < NX - 1:                                # +x internal face
                faces.append([pid(i + 1, j, 0), pid(i + 1, j + 1, 0),
                              pid(i + 1, j + 1, 1), pid(i + 1, j, 1)])
                owner.append(c); neigh.append(cid(i + 1, j))
    n_internal = len(faces)

    patches = []

    def add_patch(name, ptype, flist, olist):
        patches.append((name, ptype, len(faces), len(flist)))
        faces.extend(flist)
        owner.extend(olist)

    add_patch("inlet", "patch",
              [[pid(0, j, 0), pid(0, j, 1), pid(0, j + 1, 1), pid(0, j + 1, 0)]
               for j in range(NY)], [cid(0, j) for j in range(NY)])
    add_patch("outlet", "patch",
              [[pid(NX, j, 0), pid(NX, j + 1, 0), pid(NX, j + 1, 1),
                pid(NX, j, 1)] for j in range(NY)],
              [cid(NX - 1, j) for j in range(NY)])
    add_patch("wall", "wall",
              [[pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 0, 1), pid(i, 0, 1)]
               for i in range(NX)], [cid(i, 0) for i in range(NX)])
    add_patch("top", "patch",
              [[pid(i, NY, 0), pid(i, NY, 1), pid(i + 1, NY, 1),
                pid(i + 1, NY, 0)] for i in range(NX)],
              [cid(i, NY - 1) for i in range(NX)])
    fb, fo = [], []
    for i in range(NX):
        for j in range(NY):
            fb.append([pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0),
                       pid(i + 1, j, 0)])
            fo.append(cid(i, j))
    add_patch("back", "empty", fb, fo)
    ff, fo2 = [], []
    for i in range(NX):
        for j in range(NY):
            ff.append([pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j + 1, 1),
                       pid(i, j + 1, 1)])
            fo2.append(cid(i, j))
    add_patch("front", "empty", ff, fo2)

    _write_list(os.path.join(d, "faces"), "faceList", "constant", "faces",
                faces, lambda f: "4(%d %d %d %d)" % tuple(f))
    _write_list(os.path.join(d, "owner"), "labelList", "constant", "owner",
                owner, lambda v: str(v))
    _write_list(os.path.join(d, "neighbour"), "labelList", "constant",
                "neighbour", neigh, lambda v: str(v))
    with open(os.path.join(d, "boundary"), "w") as f:
        f.write(HDR % ("polyBoundaryMesh", "constant", "boundary"))
        f.write("%d\n(\n" % len(patches))
        for name, ptype, start, n in patches:
            f.write("    %s\n    {\n        type            %s;\n"
                    "        nFaces          %d;\n"
                    "        startFace       %d;\n    }\n"
                    % (name, ptype, n, start))
        f.write(")\n")
    return NX, NY, n_internal


def write_field(case, time, name, values, vector=False):
    d = os.path.join(case, time)
    os.makedirs(d, exist_ok=True)
    cls = "volVectorField" if vector else "volScalarField"
    with open(os.path.join(d, name), "w") as f:
        f.write(HDR % (cls, time, name))
        f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
        f.write("internalField   nonuniform List<%s>\n%d\n(\n"
                % ("vector" if vector else "scalar", len(values)))
        for v in values:
            f.write(("(%.10g %.10g %.10g)" % tuple(v)) if vector
                    else "%.10g" % v)
            f.write("\n")
        f.write(")\n;\n\nboundaryField\n{\n"
                "    inlet { type zeroGradient; }\n"
                "    outlet { type zeroGradient; }\n"
                "    wall { type zeroGradient; }\n"
                "    top { type zeroGradient; }\n"
                "    back { type empty; }\n"
                "    front { type empty; }\n}\n")


def build_case(case, bf, time="0.001", nx=90, ny=70):
    """Sample an existing BaseFlow onto a structured mesh and write it out."""
    x0, x1 = bf.x[0], bf.x[-1]
    xg = np.linspace(x0, x1, nx + 1)
    ymax = bf.grid.y_max
    t = np.linspace(0.0, 1.0, ny + 1)
    yg = ymax * (t ** 2.2)
    write_mesh(case, xg, yg)
    xc = 0.5 * (xg[1:] + xg[:-1])
    yc = 0.5 * (yg[1:] + yg[:-1])
    nc = nx * ny
    Q = np.zeros((nc, 16))
    for i, xv in enumerate(xc):
        k = int(np.clip(np.searchsorted(bf.x, xv) - 1, 0, bf.nx - 2))
        w = (xv - bf.x[k]) / (bf.x[k + 1] - bf.x[k])
        prof = (1 - w) * bf.q[k] + w * bf.q[k + 1]
        for j, yv in enumerate(yc):
            for v in range(16):
                Q[i * ny + j, v] = np.interp(yv, bf.grid.y, prof[v])
    rho = Q[:, :11].sum(axis=1)
    for s, nm in enumerate(sp.NAMES):
        write_field(case, time, nm, Q[:, s] / rho)
    write_field(case, time, "Tt", Q[:, 14])
    write_field(case, time, "Tv", Q[:, 15])
    write_field(case, time, "rho", rho)
    from pse import thermo as th
    p = th.pressure(Q[:, :11].T, Q[:, 14], Q[:, 15])
    write_field(case, time, "p", p)
    U = np.zeros((nc, 3))
    U[:, 0] = Q[:, 11]
    U[:, 1] = Q[:, 12]
    write_field(case, time, "U", U, vector=True)
    return case
