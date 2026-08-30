"""Extraction of wall-normal boundary-layer profiles from a hy2Foam solution.

Field-name conventions follow hyStrath/hy2Foam (OpenFOAM v1706):
    species mass fractions : N2 O2 NO N O N2+ O2+ NO+ N+ O+ e-
    temperatures           : Tt (trans-rotational), Tv (vibrational-electronic)
    others                 : U, p, rho (optional), Mach, mu, kappatr, kappave
Alternate names are accepted through `field_map`.
"""
import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator

from .. import species as sp
from .parser import read_field, latest_time
from .mesh import FoamMesh

DEFAULT_MAP = {n: n for n in sp.NAMES}
DEFAULT_MAP.update({"T": "Tt", "Tv": "Tv", "U": "U", "p": "p", "rho": "rho"})


class ProfileExtractor:
    """Interpolates cell-centred data onto wall-normal rays."""

    def __init__(self, case, time=None, wall_patches=("wall",), plane="xy",
                 axis=(1, 0, 0), field_map=None, mesh=None, quiet=True):
        self.case = case
        self.time = time or latest_time(case)
        self.wall_patches = list(wall_patches)
        self.plane = plane
        self.axis = np.asarray(axis, dtype=float)
        self.map = dict(DEFAULT_MAP)
        if field_map:
            self.map.update(field_map)
        self.mesh = mesh if mesh is not None else FoamMesh(case)
        self.fields = self._load_fields(quiet)
        self._build_interpolators()
        self._build_wall()

    # ------------------------------------------------------------------ data
    def _path(self, name):
        return os.path.join(self.case, self.time, name)

    def _load_fields(self, quiet):
        out = {}
        nc = self.mesh.ncells
        for key, fname in self.map.items():
            p = self._path(fname)
            if not (os.path.exists(p) or os.path.exists(p + ".gz")):
                if not quiet:
                    print("  [extract] missing field %s" % fname)
                continue
            f = read_field(p)
            v = f["internal"]
            if np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0):
                v = np.full(nc, float(v))
            elif f["ncomp"] == 3 and v.ndim == 1:
                v = np.tile(v, (nc, 1))
            out[key] = np.asarray(v, dtype=float)
        for n in sp.NAMES:
            if n not in out:
                out[n] = np.zeros(nc)
        if "T" not in out:
            raise RuntimeError("temperature field %s not found" % self.map["T"])
        if "Tv" not in out:
            out["Tv"] = out["T"].copy()
        return out

    def _project(self, X):
        if self.plane == "axisym":
            a = self.axis / np.linalg.norm(self.axis)
            s = X @ a
            r = np.linalg.norm(X - np.outer(s, a), axis=1)
            return np.column_stack([s, r])
        i, j = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[self.plane]
        return X[:, [i, j]]

    def _build_interpolators(self):
        P = self._project(self.mesh.C)
        self.P2 = P
        self.tree = cKDTree(P)
        self._interp = {}
        self._lin = None
        try:
            from scipy.spatial import Delaunay
            self._tri = Delaunay(P)
        except Exception:
            self._tri = None

    def _interp_field(self, name, Q):
        v = self.fields[name]
        if v.ndim == 2:
            return np.column_stack([self._interp_scalar(v[:, k], Q)
                                    for k in range(v.shape[1])])
        return self._interp_scalar(v, Q)

    def _interp_scalar(self, v, Q):
        if self._tri is not None:
            f = LinearNDInterpolator(self._tri, v)
            out = f(Q)
            bad = ~np.isfinite(out)
            if bad.any():
                _, idx = self.tree.query(Q[bad])
                out[bad] = v[idx]
            return out
        _, idx = self.tree.query(Q)
        return v[idx]

    # ------------------------------------------------------------- wall data
    def _build_wall(self):
        Cf, Nf = [], []
        for pn in self.wall_patches:
            if pn not in self.mesh.boundary:
                raise KeyError("patch '%s' not in %s" %
                               (pn, list(self.mesh.boundary)))
            Cf.append(self.mesh.patch_centres(pn))
            Nf.append(self.mesh.patch_normals(pn))
        Cf = np.vstack(Cf)
        Nf = np.vstack(Nf)
        P = self._project(Cf)
        N = self._project(Cf + Nf) - P
        N /= np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
        order = np.argsort(P[:, 0])
        P, N = P[order], N[order]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0),
                                                      axis=1) > 1e-12])
        P, N = P[keep], N[keep]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0),
                                                            axis=1))])
        self.wall_xy = P
        self.wall_n = N
        self.wall_s = s

    def wall_point(self, s):
        x = np.interp(s, self.wall_s, self.wall_xy[:, 0])
        y = np.interp(s, self.wall_s, self.wall_xy[:, 1])
        nx = np.interp(s, self.wall_s, self.wall_n[:, 0])
        ny = np.interp(s, self.wall_s, self.wall_n[:, 1])
        m = np.hypot(nx, ny)
        return np.array([x, y]), np.array([nx / m, ny / m])

    # --------------------------------------------------------------- profile
    def profile(self, s, y):
        """Sample the solution along the wall-normal ray at arclength s.

        Returns dict with rho_s (ns, ny), u, v, T, Tv, p, plus wall metrics.
        """
        p0, n = self.wall_point(s)
        t = np.array([n[1], -n[0]])
        if np.dot(t, np.array([1.0, 0.0])) < 0.0:
            t = -t
        Q = p0[None, :] + np.outer(y, n)

        Y = np.stack([np.clip(self._interp_field(nm, Q), 0.0, 1.0)
                      for nm in sp.NAMES], axis=0)
        Ysum = Y.sum(axis=0)
        Ysum[Ysum <= 0.0] = 1.0
        Y = Y / Ysum
        T = self._interp_field("T", Q)
        Tv = self._interp_field("Tv", Q)
        Uv = self._interp_field("U", Q)
        pres = self._interp_field("p", Q)

        Rmix = np.zeros_like(T)
        for k, nm in enumerate(sp.NAMES):
            Ts = Tv if sp.IS_ELECTRON[k] else T
            Rmix = Rmix + Y[k] * sp.RS[k] * Ts / np.maximum(T, 1e-6)
        if "rho" in self.fields:
            rho = self._interp_field("rho", Q)
        else:
            rho = pres / np.maximum(Rmix * T, 1e-30)

        if self.plane == "axisym":
            a = self.axis / np.linalg.norm(self.axis)
            us = Uv @ a
            ur = np.linalg.norm(Uv - np.outer(us, a), axis=1)
            U2 = np.column_stack([us, ur])
        else:
            i, j = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[self.plane]
            U2 = Uv[:, [i, j]]
        u = U2 @ t
        v = U2 @ n
        return dict(y=y, rho_s=rho[None, :] * Y, rho=rho, u=u, v=v,
                    w=np.zeros_like(u), T=T, Tv=Tv, p=pres, Y=Y,
                    wall_xy=p0, normal=n, tangent=t, s=s)


def extract_profiles(case, stations, y, **kw):
    ex = ProfileExtractor(case, **kw)
    return ex, [ex.profile(s, y) for s in stations]
