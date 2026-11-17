"""SU2 solution files: restart/volume CSV (ASCII) and the SU2 binary
restart format.

CSV   : header  "PointID","x","y",...  then one row per point
        (CSU2FileWriter; also what VOLUME_OUTPUT_FILES= CSV writes).
binary: 5 int32 [535532, nVar, nPoint, sizeof(double) | 0, 0], nVar names
        of 33 chars, then nPoint x nVar float64 row-major
        (CSU2BinaryFileWriter); no PointID column, rows in point order.

NEMO field names (CNEMOCompOutput):
    Density_0..Density_{ns-1}, Momentum_x/_y(/_z), Energy, Energy_ve,
    Pressure, Temperature_tr, Temperature_ve, Velocity_x/_y(/_z),
    MassFrac_k, Laminar_Viscosity, ...
"""
import glob
import os
import numpy as np

MAGIC = 535532
NAME_LEN = 33


def read_csv(path):
    with open(path) as f:
        header = f.readline()
    names = [h.strip().strip('"') for h in header.strip().split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    fields = {n: data[:, k] for k, n in enumerate(names)}
    if "PointID" in fields:
        order = np.argsort(fields["PointID"].astype(np.int64))
        fields = {k: v[order] for k, v in fields.items()}
    return fields


def read_binary(path):
    with open(path, "rb") as f:
        head = np.fromfile(f, dtype=np.int32, count=5)
        if head.size < 3 or head[0] != MAGIC:
            raise ValueError("%s is not an SU2 binary restart" % path)
        nvar, npts = int(head[1]), int(head[2])
        dsize = int(head[3]) if head[3] in (4, 8) else 8
        raw = f.read(nvar * NAME_LEN)
        names = [raw[k * NAME_LEN:(k + 1) * NAME_LEN].split(b"\0")[0]
                 .decode("ascii", "replace").strip() for k in range(nvar)]
        dt = np.float32 if dsize == 4 else np.float64
        data = np.fromfile(f, dtype=dt, count=nvar * npts).reshape(npts, nvar)
    return {n: data[:, k].astype(float) for k, n in enumerate(names)}


def read_solution(path):
    with open(path, "rb") as f:
        b = f.read(4)
    if len(b) == 4 and int(np.frombuffer(b, dtype=np.int32)[0]) == MAGIC:
        return read_binary(path)
    return read_csv(path)


def write_csv(path, fields, order=None):
    names = order or list(fields)
    n = len(fields[names[0]])
    with open(path, "w") as f:
        f.write(",".join(['"PointID"'] + ['"%s"' % k for k in names]) + "\n")
        for i in range(n):
            f.write("%d, " % i + ", ".join("%.15e" % fields[k][i] for k in names)
                    + "\n")


def write_binary(path, fields, order=None):
    names = order or list(fields)
    n = len(fields[names[0]])
    with open(path, "wb") as f:
        np.array([MAGIC, len(names), n, 8, 0], dtype=np.int32).tofile(f)
        for k in names:
            f.write(k.encode("ascii")[:NAME_LEN - 1].ljust(NAME_LEN, b"\0"))
        np.column_stack([fields[k] for k in names]).astype(np.float64).tofile(f)


def expand(spec):
    """File list from a path, a glob, or a list of either (sorted)."""
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            out.extend(expand(s))
        return out
    hits = sorted(glob.glob(spec))
    if not hits and os.path.exists(spec):
        hits = [spec]
    if not hits:
        raise FileNotFoundError("no SU2 solution file matches '%s'" % spec)
    return hits
