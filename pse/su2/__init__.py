"""SU2-NEMO solution ingestion (native mesh, restart CSV/binary, ILES
time averaging, wall-normal profile extraction)."""
import numpy as np

from .mesh import SU2Mesh, write_su2_mesh
from .restart import read_solution, read_csv, read_binary, write_csv, \
    write_binary, expand
from .extract import SU2Solution
from ..baseflow import BaseFlow, WallGeometry
from ..grid import Grid


def su2_baseflow(gas, solution, stations, mesh=None, wall_markers=("wall",),
                 ny=201, y_max=None, y_half=None, y_max_factor=4.0,
                 y_half_factor=0.5, plane="axisym", axisymmetric=True,
                 species_order=None, average="primitive", smooth=1,
                 curve_smoothing=None, grid_kind="fd4", probe_ny=600,
                 quiet=True, extractor=None, s0=0.0, station_axis="s"):
    """BaseFlow sampled from an SU2-NEMO solution along wall normals.

    stations : positions [m] of the PSE stations: wall arclength measured
               from the upstream end of the wall marker, plus s0 (use s0
               when the meshed wall does not start at the tip / leading
               edge), or the axial coordinate x with station_axis='x'.
    plane    : 'axisym' for SU2 AXISYMMETRIC= YES (x axial, y radial) or a 3-D
               body of revolution about x; 'xy' for planar flows.
    """
    ex = extractor or SU2Solution(gas, solution, mesh=mesh,
                                  species_order=species_order, plane=plane,
                                  average=average, quiet=quiet)
    wc = ex.wall_curve(wall_markers=wall_markers if ex.mesh is not None
                       else None, smooth=curve_smoothing)
    wc["s"] = wc["s"] + float(s0)
    stations = np.asarray(stations, float)
    if station_axis == "x":
        stations = np.interp(stations, wc["x"], wc["s"])
    if stations.max() > wc["s"][-1] or stations.min() < wc["s"][0]:
        raise ValueError("stations outside the wall curve [%.4g, %.4g] m"
                         % (wc["s"][0], wc["s"][-1]))

    def wall_at(si):
        return {k: float(np.interp(si, wc["s"], v)) for k, v in wc.items()
                if k != "s"}

    def profile(si, yy):
        w = wall_at(si)
        P = np.column_stack([w["x"] + yy * w["nx"], w["y"] + yy * w["ny"]])
        F = ex.sample(P)
        out = F.copy()
        iu, iv = gas.i_u, gas.i_v
        out[iu] = F[iu] * w["tx"] + F[iv] * w["ty"]
        out[iv] = F[iu] * w["nx"] + F[iv] * w["ny"]
        return out

    if y_max is None or y_half is None:
        span = 0.25 * (wc["s"][-1] - wc["s"][0]) + 1e-9
        yp = np.linspace(0.0, span, probe_ny)
        d99 = []
        for si in stations:
            P = profile(si, yp)
            u = P[gas.i_u]
            ue = u[: max(int(0.8 * probe_ny), 3)].max()
            k = int(np.argmax(u >= 0.99 * ue))
            d99.append(yp[max(k, 1)])
        d99 = np.array(d99)
        y_max = y_max or float(y_max_factor * d99.max())
        y_half = y_half or float(y_half_factor * np.median(d99))
    grid = Grid(ny, y_max, min(y_half, 0.45 * y_max), kind=grid_kind)
    q = np.zeros((len(stations), gas.nv, ny))
    for i, si in enumerate(stations):
        P = profile(si, grid.y)
        P[:gas.ns] = np.maximum(P[:gas.ns], 0.0)
        q[i] = P
    ws = [wall_at(si) for si in stations]
    geo = WallGeometry(stations, [w["x"] for w in ws], [w["y"] for w in ws],
                       [w["nx"] for w in ws], [w["ny"] for w in ws],
                       [w["kappa"] for w in ws],
                       axisymmetric=(plane == "axisym" and axisymmetric))
    bf = BaseFlow(stations, grid, q, gas, geometry=geo,
                  meta=dict(source="su2", files=ex.files, plane=plane,
                            wall_markers=list(wall_markers or []),
                            average=average))
    if smooth:
        bf.smooth(n_pass=int(smooth), mode="y")
    return bf


__all__ = ["SU2Mesh", "SU2Solution", "su2_baseflow", "read_solution",
           "read_csv", "read_binary", "write_csv", "write_binary",
           "write_su2_mesh", "expand"]
