"""Exact flux/source Jacobians by complex-step differentiation.

For a base state q (NV, Ny) with gradients qx, qy, qz the routine returns the
sixteen coefficient matrices needed to assemble the linearised operators.
Complex-step gives machine-precision derivatives with no subtractive
cancellation, which matters because the species densities span ~20 decades.
"""
import numpy as np
from .constants import CS_H
from . import fluxes as fx

NV = fx.NV


def _stack(lst):
    return np.asarray(lst)


def flux_jacobians(q, qx, qy, qz, model, comm=None):
    """Return dict of (Ny, NV, NV) Jacobian arrays.

    With a communicator the wall-normal points are split across its ranks and
    the blocks are gathered, so the (dominant) assembly cost also scales at
    the group level."""
    q = np.asarray(q, dtype=float)
    ny = q.shape[1]
    if comm is not None and comm.size > 1:
        return _flux_jacobians_parallel(q, qx, qy, qz, model, comm)
    h = CS_H
    out = {k: np.zeros((ny, NV, NV)) for k in
           ("Gam", "A", "Ax", "Ay", "Az", "B", "Bx", "By", "Bz",
            "C", "Cx", "Cy", "Cz", "E")}

    qc = q.astype(complex)
    qxc, qyc, qzc = (np.asarray(a, dtype=complex) for a in (qx, qy, qz))
    trans_real = model.transport(q[:fx.NS], q[fx.I_T], q[fx.I_TV])

    for k in range(NV):
        p = qc.copy()
        p[k] = p[k] + 1j * h
        U = _stack(fx.conservative(p))
        F, G, H = fx.fluxes(p, qxc, qyc, qzc, model)
        S = _stack(fx.source(p, model))
        out["Gam"][:, :, k] = (U.imag / h).T
        out["A"][:, :, k] = (_stack(F).imag / h).T
        out["B"][:, :, k] = (_stack(G).imag / h).T
        out["C"][:, :, k] = (_stack(H).imag / h).T
        out["E"][:, :, k] = (S.imag / h).T

    if model.viscous:
        for tag, arr in (("x", qxc), ("y", qyc), ("z", qzc)):
            for k in range(NV):
                g = [qxc, qyc, qzc]
                idx = {"x": 0, "y": 1, "z": 2}[tag]
                gg = list(g)
                pert = arr.copy()
                pert[k] = pert[k] + 1j * h
                gg[idx] = pert
                F, G, H = fx.fluxes(qc, gg[0], gg[1], gg[2], model,
                                    trans=trans_real)
                out["A" + tag][:, :, k] = (_stack(F).imag / h).T
                out["B" + tag][:, :, k] = (_stack(G).imag / h).T
                out["C" + tag][:, :, k] = (_stack(H).imag / h).T
    return out


def dcoef_dy(J, D1):
    """d/dy of every Jacobian array using the wall-normal derivative matrix."""
    return {k: np.einsum("jk,kmn->jmn", D1, v) for k, v in J.items()}


def _flux_jacobians_parallel(q, qx, qy, qz, model, comm):
    ny = q.shape[1]
    edges = np.linspace(0, ny, comm.size + 1).astype(int)
    j0, j1 = edges[comm.rank], edges[comm.rank + 1]
    sl = slice(j0, j1)
    if j1 > j0:
        loc = flux_jacobians(q[:, sl], qx[:, sl], qy[:, sl], qz[:, sl], model)
    else:
        loc = {k: np.zeros((0, NV, NV)) for k in
               ("Gam", "A", "Ax", "Ay", "Az", "B", "Bx", "By", "Bz",
                "C", "Cx", "Cy", "Cz", "E")}
    out = {}
    for k in loc:
        buf = np.zeros((ny, NV, NV))
        if j1 > j0:
            buf[sl] = loc[k]
        tot = np.zeros_like(buf)
        comm.Allreduce(buf, tot)
        out[k] = tot
    return out
