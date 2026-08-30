"""polyMesh reader with OpenFOAM-consistent cell-centre computation."""
import os
import re
import numpy as np
from .parser import read_raw, header, strip_comments, _after_header, _parse_list


class FoamMesh:
    def __init__(self, case, region=None, time=None):
        self.case = case
        base = os.path.join(case, "constant", "polyMesh")
        if time is not None and os.path.isdir(os.path.join(case, time, "polyMesh")):
            base = os.path.join(case, time, "polyMesh")
        self.dir = base
        self.points = self._read_points()
        self.faces = self._read_faces()
        self.owner = self._read_labels("owner")
        self.neighbour = self._read_labels("neighbour")
        self.boundary = self._read_boundary()
        self.ncells = int(max(self.owner.max(), self.neighbour.max()) + 1)
        self.Cf, self.Sf, self.magSf = self._face_geometry()
        self.C, self.V = self._cell_geometry()

    # ------------------------------------------------------------------ read
    def _read_points(self):
        raw = read_raw(os.path.join(self.dir, "points"))
        hdr = header(raw)
        binary = hdr.get("format", "ascii") == "binary"
        body = raw[_after_header(raw):] if binary else \
            strip_comments(raw[_after_header(raw):])
        arr, _ = _parse_list(body, 0, 3, binary)
        return np.asarray(arr, dtype=float)

    def _read_labels(self, name):
        p = os.path.join(self.dir, name)
        if not os.path.exists(p) and not os.path.exists(p + ".gz"):
            return np.zeros(0, dtype=np.int64)
        raw = read_raw(p)
        hdr = header(raw)
        binary = hdr.get("format", "ascii") == "binary"
        body = raw[_after_header(raw):] if binary else \
            strip_comments(raw[_after_header(raw):])
        dt = np.int32 if hdr.get("class", "").startswith("label") else np.int32
        arr, _ = _parse_list(body, 0, 1, binary, dtype=dt if binary else np.float64)
        return np.asarray(arr, dtype=np.int64)

    def _read_faces(self):
        raw = read_raw(os.path.join(self.dir, "faces"))
        hdr = header(raw)
        if hdr.get("format", "ascii") == "binary":
            raise RuntimeError(
                "binary polyMesh/faces is not supported; run "
                "`foamFormatConvert -ascii` or supply cell centres via "
                "`postProcess -func writeCellCentres`")
        body = strip_comments(raw[_after_header(raw):]).decode("ascii", "replace")
        m = re.search(r"(\d+)\s*\(", body)
        n = int(m.group(1))
        pos = m.end()
        faces = []
        pat = re.compile(r"\s*(\d+)\s*\(([^)]*)\)")
        for _ in range(n):
            mm = pat.match(body, pos)
            faces.append(np.array(mm.group(2).split(), dtype=int))
            pos = mm.end()
        return faces

    def _read_boundary(self):
        raw = strip_comments(read_raw(os.path.join(self.dir, "boundary")))
        txt = raw[_after_header(raw):].decode("ascii", "replace")
        out = {}
        for m in re.finditer(r"(\w[\w.\-]*)\s*\{([^}]*)\}", txt):
            d = dict(re.findall(r"(\w+)\s+([^;]+);", m.group(2)))
            if "nFaces" in d and "startFace" in d:
                out[m.group(1)] = dict(nFaces=int(d["nFaces"]),
                                       startFace=int(d["startFace"]),
                                       type=d.get("type", "").strip())
        return out

    # -------------------------------------------------------------- geometry
    def _face_geometry(self):
        nf = len(self.faces)
        Cf = np.zeros((nf, 3))
        Sf = np.zeros((nf, 3))
        P = self.points
        for i, f in enumerate(self.faces):
            pts = P[f]
            if len(f) == 3:
                Cf[i] = pts.mean(axis=0)
                Sf[i] = 0.5 * np.cross(pts[1] - pts[0], pts[2] - pts[0])
                continue
            pavg = pts.mean(axis=0)
            a = np.cross(pts - pavg, np.roll(pts, -1, axis=0) - pavg) * 0.5
            ct = (pts + np.roll(pts, -1, axis=0) + pavg) / 3.0
            aw = np.linalg.norm(a, axis=1)
            tot = aw.sum()
            Sf[i] = a.sum(axis=0)
            Cf[i] = (ct * aw[:, None]).sum(axis=0) / tot if tot > 0 else pavg
        return Cf, Sf, np.linalg.norm(Sf, axis=1)

    def _cell_geometry(self):
        nc = self.ncells
        cEst = np.zeros((nc, 3))
        cnt = np.zeros(nc)
        np.add.at(cEst, self.owner, self.Cf[:len(self.owner)])
        np.add.at(cnt, self.owner, 1.0)
        nb = len(self.neighbour)
        np.add.at(cEst, self.neighbour, self.Cf[:nb])
        np.add.at(cnt, self.neighbour, 1.0)
        cEst /= cnt[:, None]

        C = np.zeros((nc, 3))
        V = np.zeros(nc)
        for own, sgn, idx in ((self.owner, 1.0, np.arange(len(self.owner))),
                              (self.neighbour, -1.0, np.arange(nb))):
            if len(own) == 0:
                continue
            d = self.Cf[idx] - cEst[own]
            pyr3Vol = sgn * np.einsum("ij,ij->i", self.Sf[idx], d)
            pc = 0.75 * self.Cf[idx] + 0.25 * cEst[own]
            np.add.at(V, own, pyr3Vol)
            np.add.at(C, own, pyr3Vol[:, None] * pc)
        good = np.abs(V) > 1e-300
        C[good] /= V[good][:, None]
        C[~good] = cEst[~good]
        return C, V / 3.0

    # ---------------------------------------------------------------- access
    def patch_faces(self, name):
        b = self.boundary[name]
        return np.arange(b["startFace"], b["startFace"] + b["nFaces"])

    def patch_centres(self, name):
        return self.Cf[self.patch_faces(name)]

    def patch_normals(self, name, outward_into_fluid=True):
        idx = self.patch_faces(name)
        n = self.Sf[idx] / np.maximum(self.magSf[idx], 1e-300)[:, None]
        return -n if outward_into_fluid else n

    def patch_owner_cells(self, name):
        return self.owner[self.patch_faces(name)]
