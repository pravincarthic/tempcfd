"""Reader for the native SU2 mesh format (.su2, ASCII), single zone.

Only what the base-flow extraction needs is kept: point coordinates and the
boundary-marker connectivity.  Volume elements are skipped.
"""
import numpy as np

# VTK element types used by SU2 and their node counts
NODES = {3: 2, 5: 3, 9: 4, 10: 4, 12: 8, 13: 6, 14: 5}


class SU2Mesh:
    def __init__(self, path):
        self.path = path
        self.ndim = None
        self.points = None
        self.markers = {}
        self._read(path)

    def _read(self, path):
        with open(path) as f:
            lines = f.readlines()
        n = len(lines)
        i = 0

        def kv(line):
            k, v = line.split("=", 1)
            return k.strip().upper(), v.strip()

        while i < n:
            line = lines[i].split("%")[0].strip()
            i += 1
            if not line or "=" not in line:
                continue
            key, val = kv(line)
            if key == "NDIME":
                self.ndim = int(val)
            elif key == "NELEM":
                i += int(val.split()[0])
            elif key == "NPOIN":
                npoin = int(val.split()[0])
                pts = np.zeros((npoin, self.ndim))
                k = 0
                while k < npoin:
                    t = lines[i].split()
                    i += 1
                    if not t:
                        continue
                    pts[k] = [float(v) for v in t[:self.ndim]]
                    k += 1
                self.points = pts
            elif key == "NMARK":
                continue
            elif key == "MARKER_TAG":
                tag = val
                key2, val2 = kv(lines[i].split("%")[0].strip())
                i += 1
                if key2 != "MARKER_ELEMS":
                    raise ValueError("MARKER_ELEMS expected after MARKER_TAG "
                                     "%s" % tag)
                ne = int(val2)
                elems = []
                k = 0
                while k < ne:
                    t = lines[i].split()
                    i += 1
                    if not t:
                        continue
                    et = int(t[0])
                    nn = NODES[et]
                    elems.append(np.array([int(v) for v in t[1:1 + nn]]))
                    k += 1
                self.markers[tag] = elems
            elif key in ("NZONE", "IZONE"):
                if key == "NZONE" and int(val) > 1:
                    raise ValueError("multi-zone meshes are not supported")
        if self.points is None:
            raise ValueError("no NPOIN section in %s" % path)

    def marker_nodes(self, tags):
        ids = set()
        for t in tags:
            if t not in self.markers:
                raise KeyError("marker '%s' not in mesh (have %s)"
                               % (t, ", ".join(self.markers)))
            for e in self.markers[t]:
                ids.update(int(v) for v in e)
        return np.array(sorted(ids), dtype=int)

    def marker_edges(self, tags):
        """2-D boundary line segments (pairs of node ids)."""
        out = []
        for t in tags:
            for e in self.markers[t]:
                if len(e) == 2:
                    out.append((int(e[0]), int(e[1])))
        return out


def write_su2_mesh(path, points, quads, markers):
    """Minimal writer (tests and examples): 2-D quads + line markers."""
    with open(path, "w") as f:
        f.write("NDIME= 2\n")
        f.write("NELEM= %d\n" % len(quads))
        for k, qd in enumerate(quads):
            f.write("9 %d %d %d %d %d\n" % (qd[0], qd[1], qd[2], qd[3], k))
        f.write("NPOIN= %d\n" % len(points))
        for k, p in enumerate(points):
            f.write("%.15e %.15e %d\n" % (p[0], p[1], k))
        f.write("NMARK= %d\n" % len(markers))
        for tag, edges in markers.items():
            f.write("MARKER_TAG= %s\n" % tag)
            f.write("MARKER_ELEMS= %d\n" % len(edges))
            for a, b in edges:
                f.write("3 %d %d\n" % (a, b))
