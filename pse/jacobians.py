"""Exact flux/source Jacobians by complex-step differentiation.

For a base state q (NV, ny) with gradients qx, qy, qz the routine returns
(ny, NV, NV) arrays

    Gam = dU/dq
    A, Ax, Ay, Az = d(lF)/d(q, qx, qy, qz)
    B, Bx, By, Bz = dG/d(...)
    C, Cx, Cy, Cz = dH/d(...)
    Eq, Ex, Ey, Ez = dE/d(...)

Complex step gives machine-precision derivatives with no subtractive
cancellation, which matters because the species densities span ~20
decades.
"""
import numpy as np
from .constants import CS_H
from . import fluxes as fx

KEYS = ("Gam", "A", "Ax", "Ay", "Az", "B", "Bx", "By", "Bz",
        "C", "Cx", "Cy", "Cz", "Eq", "Ex", "Ey", "Ez")


def _stack(lst):
    return np.asarray(lst)


def flux_jacobians(model, q, qx, qy, qz, metric, comm=None):
    q = np.asarray(q, dtype=float)
    nv, ny = q.shape
    if comm is not None and comm.size > 1:
        return _parallel(model, q, qx, qy, qz, metric, comm)
    h = CS_H
    out = {k: np.zeros((ny, nv, nv)) for k in KEYS}
    qc = q.astype(complex)
    qxc, qyc, qzc = (np.asarray(a, dtype=complex) for a in (qx, qy, qz))
    g = model.gas
    trans_real = (model.transport(q[:g.ns], q[g.i_T], q[g.i_Tv])
                  if model.viscous else None)

    for k in range(nv):
        p = qc.copy()
        p[k] = p[k] + 1j * h
        U = _stack(fx.conservative(model, p))
        F, G, H, E = fx.fluxes(model, p, qxc, qyc, qzc, metric)
        out["Gam"][:, :, k] = (U.imag / h).T
        out["A"][:, :, k] = (_stack(F).imag / h).T
        out["B"][:, :, k] = (_stack(G).imag / h).T
        out["C"][:, :, k] = (_stack(H).imag / h).T
        out["Eq"][:, :, k] = (_stack(E).imag / h).T

    for tag, idx in (("x", 0), ("y", 1), ("z", 2)):
        for k in range(nv):
            gg = [qxc, qyc, qzc]
            pert = gg[idx].copy()
            pert[k] = pert[k] + 1j * h
            gg[idx] = pert
            F, G, H, E = fx.fluxes(model, qc, gg[0], gg[1], gg[2], metric,
                                   trans=trans_real, with_source=False)
            out["A" + tag][:, :, k] = (_stack(F).imag / h).T
            out["B" + tag][:, :, k] = (_stack(G).imag / h).T
            out["C" + tag][:, :, k] = (_stack(H).imag / h).T
            out["E" + tag][:, :, k] = (_stack(E).imag / h).T
    return out


def residual_parts(model, q, qx, qy, qz, metric):
    """Real (lF, G, H, E, U) at a state, as (NV, ...) arrays."""
    F, G, H, E = fx.fluxes(model, q, qx, qy, qz, metric)
    U = fx.conservative(model, q)
    return tuple(_stack(a) for a in (F, G, H, E, U))


def _sub_metric(m, sl):
    return fx.Metric(len(range(*sl.indices(m.ny))), l=m.l[sl], kap=m.kap[sl],
                     a_s=m.a_s[sl], a_n=m.a_n[sl], dl_ds=m.dl_ds[sl],
                     r=None if m.r is None else m.r[sl])


def _parallel(model, q, qx, qy, qz, metric, comm):
    nv, ny = q.shape
    edges = np.linspace(0, ny, comm.size + 1).astype(int)
    j0, j1 = edges[comm.rank], edges[comm.rank + 1]
    sl = slice(j0, j1)
    if j1 > j0:
        model.gas.anchor(q[:model.gas.ns, sl], q[model.gas.i_T, sl],
                         q[model.gas.i_Tv, sl])
        loc = flux_jacobians(model, q[:, sl], qx[:, sl], qy[:, sl], qz[:, sl],
                             _sub_metric(metric, sl))
    else:
        loc = {k: np.zeros((0, nv, nv)) for k in KEYS}
    out = {}
    for k in KEYS:
        buf = np.zeros((ny, nv, nv))
        if j1 > j0:
            buf[sl] = loc[k]
        tot = np.zeros_like(buf)
        comm.Allreduce(buf, tot)
        out[k] = tot
    model.gas.anchor(q[:model.gas.ns], q[model.gas.i_T], q[model.gas.i_Tv])
    return out
