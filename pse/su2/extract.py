"""Base flow from an SU2-NEMO (NEMO_NAVIER_STOKES) solution.

SU2 is vertex based: the solution lives on the mesh points, so the restart
file already holds the data at the grid nodes, wall nodes included.  The
extractor

1. reads one or several solution files (restart CSV/binary, or the CSV
   volume output) and, for an ILES/unsteady run, averages them;
2. converts SU2 variables to the PSE primitive vector
   [rho_s, u, v, w, T, Tve]: species densities Density_k (in the
   GAS_COMPOSITION order), temperatures from Temperature_tr/_ve when they
   were written, otherwise recovered from Energy/Energy_ve with the gas
   model's own energy inversion (the same relations SU2 uses);
3. projects 3-D data onto the meridional (axisymmetric body) or x-y (planar,
   spanwise-homogeneous) plane, merging coincident projected points;
4. builds the wall curve from the wall marker(s) of the .su2 mesh (or from
   the no-slip points if no mesh is given), smooths it, and samples the
   fields along the wall normals with Delaunay-linear interpolation.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator, make_smoothing_spline
from scipy.spatial import cKDTree, Delaunay

from .restart import read_solution, expand
from .mesh import SU2Mesh


class SU2Solution:
    def __init__(self, gas, solution, mesh=None, species_order=None,
                 plane="axisym", average="primitive", merge_tol=1e-9,
                 quiet=True):
        self.gas = gas
        self.files = expand(solution)
        self.mesh = mesh if (mesh is None or isinstance(mesh, SU2Mesh)) \
            else SU2Mesh(mesh)
        self.plane = plane
        order = species_order or list(gas.names)
        missing = [n for n in gas.names if n not in order]
        if missing:
            raise ValueError("species_order lacks %s" % missing)
        self.col_of = [order.index(n) for n in gas.names]
        self.quiet = quiet
        prims = [self._primitives(read_solution(f)) for f in self.files]
        if not quiet:
            print("su2 %d solution files with %d points"
                  % (len(prims), prims[0]["x"].size))
        self.data = self._average(prims, average)
        self._project(merge_tol)

    # --------------------------------------------------------------- input
    def _primitives(self, fld):
        g = self.gas
        ns = g.ns
        ndim = 3 if "z" in fld else 2
        rs = np.vstack([fld["Density_%d" % self.col_of[s]] for s in range(ns)])
        rs = np.maximum(rs, 0.0)
        rho = rs.sum(axis=0)
        vel = []
        for c in "xyz"[:ndim]:
            if "Velocity_" + c in fld:
                vel.append(fld["Velocity_" + c])
            else:
                vel.append(fld["Momentum_" + c] / rho)
        cons = {}
        if "Energy" in fld:
            cons = dict(E=fld["Energy"], Eve=fld["Energy_ve"])
        if "Temperature_tr" in fld and "Temperature_ve" in fld:
            T, Tv = fld["Temperature_tr"], fld["Temperature_ve"]
        else:
            ke = 0.5 * rho * sum(v * v for v in vel)
            T, Tv = g.temperatures(rs, cons["E"] - ke, cons["Eve"])
        out = dict(x=fld["x"], y=fld["y"], z=fld.get("z"), rs=rs, vel=vel,
                   T=T, Tv=Tv, ndim=ndim, rho=rho)
        out.update(cons)
        return out

    def _average(self, prims, how):
        if len(prims) == 1:
            return prims[0]
        n = len(prims)
        p0 = prims[0]
        out = dict(p0)
        if how == "conservative":
            rs = sum(p["rs"] for p in prims) / n
            rho = rs.sum(axis=0)
            mom = [sum(p["rho"] * p["vel"][k] for p in prims) / n
                   for k in range(p0["ndim"])]
            E = sum(p["E"] for p in prims) / n
            Eve = sum(p["Eve"] for p in prims) / n
            vel = [m / rho for m in mom]
            ke = 0.5 * rho * sum(v * v for v in vel)
            T, Tv = self.gas.temperatures(rs, E - ke, Eve)
            out.update(rs=rs, rho=rho, vel=vel, T=T, Tv=Tv, E=E, Eve=Eve)
        else:
            out["rs"] = sum(p["rs"] for p in prims) / n
            out["rho"] = out["rs"].sum(axis=0)
            out["vel"] = [sum(p["vel"][k] for p in prims) / n
                          for k in range(p0["ndim"])]
            out["T"] = sum(p["T"] for p in prims) / n
            out["Tv"] = sum(p["Tv"] for p in prims) / n
        return out

    # ---------------------------------------------------------- projection
    def _to_plane(self, x, y, z, vel):
        """(a, b) plane coordinates and (ua, ub, uc) velocities."""
        if z is None:
            w = np.zeros_like(x)
            return x, y, vel[0], vel[1], w
        if self.plane in ("axisym", "axisym3d"):
            r = np.sqrt(y * y + z * z)
            rs = np.where(r > 0, r, 1.0)
            ur = (vel[1] * y + vel[2] * z) / rs
            uphi = (vel[2] * y - vel[1] * z) / rs
            return x, r, vel[0], ur, uphi
        return x, y, vel[0], vel[1], vel[2]

    def _project(self, tol):
        d = self.data
        a, b, ua, ub, uc = self._to_plane(d["x"], d["y"], d.get("z"), d["vel"])
        self.idx_point = np.arange(a.size)
        if d.get("z") is not None:
            scale = max(np.ptp(a), np.ptp(b), 1e-30)
            key = np.round(np.column_stack([a, b]) / (tol * scale)).astype(np.int64)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.ravel()
            cnt = np.bincount(inv).astype(float)

            def avg(v):
                return np.bincount(inv, weights=v) / cnt
            a, b, ua, ub, uc = (avg(v) for v in (a, b, ua, ub, uc))
            rs = np.vstack([avg(r) for r in d["rs"]])
            T, Tv = avg(d["T"]), avg(d["Tv"])
            self.inv = inv
        else:
            rs, T, Tv = d["rs"], d["T"], d["Tv"]
            self.inv = np.arange(a.size)
        self.pts = np.column_stack([a, b])
        self.fields = np.vstack([rs, ua, ub, uc, T, Tv])          # (NV, n)
        self._tri = None

    @property
    def tri(self):
        if self._tri is None:
            self._tri = Delaunay(self.pts)
        return self._tri

    # -------------------------------------------------------------- wall
    def wall_curve(self, wall_markers=None, smooth=None, n_out=2000):
        """Ordered, smoothed wall curve:  WallCurve(s, x, y, nx, ny, kappa)."""
        pts = self.pts
        if self.mesh is not None and wall_markers:
            ids = self.mesh.marker_nodes(wall_markers)
            P = pts[self.inv[ids]]
            P = np.unique(np.round(P, 14), axis=0)
        else:
            sp = np.sqrt(sum(f * f for f in self.fields[self.gas.i_u:
                                                          self.gas.i_w + 1]))
            vmax = np.abs(self.fields[self.gas.i_u]).max()
            P = pts[sp < 1e-6 * vmax]
            if len(P) < 5:
                raise ValueError("cannot locate the wall: give the .su2 mesh "
                                 "and the wall marker")
        order = _chain(P)
        P = P[order]
        seg = np.hypot(*np.diff(P, axis=0).T)
        keep = np.concatenate([[True], seg > 1e-14])
        P = P[keep]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        lam = smooth
        sx = make_smoothing_spline(s, P[:, 0], lam=lam)
        sy = make_smoothing_spline(s, P[:, 1], lam=lam)
        S = np.linspace(s[0], s[-1], n_out)
        xs, ys = sx(S), sy(S)
        dx, dy = sx.derivative(1)(S), sy.derivative(1)(S)
        ddx, ddy = sx.derivative(2)(S), sy.derivative(2)(S)
        nrm = np.hypot(dx, dy)
        tx, ty = dx / nrm, dy / nrm
        nx, ny = -ty, tx
        kap = (dx * ddy - dy * ddx) / nrm ** 3
        # orient the normal into the fluid
        h = 1e-3 * (s[-1] - s[0])
        probe = np.column_stack([xs + h * nx, ys + h * ny])
        inside = self.tri.find_simplex(probe) >= 0
        sigma = 1.0
        if inside.mean() < 0.5:
            nx, ny, sigma = -nx, -ny, -1.0
        # Lame factor h = 1 + kappa n with n into the fluid: from
        # dX/ds = t (1 - sigma k n),  kappa = -sigma k  (> 0 on a convex body)
        return dict(s=S, x=xs, y=ys, nx=nx, ny=ny, tx=tx, ty=ty,
                    kappa=-sigma * kap)

    def interpolator(self):
        if not hasattr(self, "_interp"):
            self._interp = LinearNDInterpolator(self.tri, self.fields.T)
            self._tree = cKDTree(self.pts)
        return self._interp

    def sample(self, P):
        f = self.interpolator()(P)
        bad = ~np.all(np.isfinite(f), axis=1)
        if np.any(bad):
            _, k = self._tree.query(P[bad])
            f[bad] = self.fields.T[k]
        return f.T


def _chain(P):
    """Order unstructured wall points along the curve (nearest-neighbour
    walk from the upstream-most point)."""
    n = len(P)
    tree = cKDTree(P)
    start = int(np.argmin(P[:, 0]))
    used = np.zeros(n, bool)
    order = [start]
    used[start] = True
    cur = start
    for _ in range(n - 1):
        k = min(n, 16)
        while True:
            d, idx = tree.query(P[cur], k=k)
            idx = np.atleast_1d(idx)
            cand = [j for j in idx if not used[j]]
            if cand or k >= n:
                break
            k = min(n, 2 * k)
        if not cand:
            break
        nxt = cand[0]
        order.append(nxt)
        used[nxt] = True
        cur = nxt
    return np.array(order)
