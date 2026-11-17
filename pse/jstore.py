"""Base-flow Jacobians shared by every mode.

The 17 Jacobian arrays at a station depend only on the base flow, yet each
(frequency, wavenumber) march used to recompute them (~35 % of a mode's
cost).  The store computes every station once per node, the node's ranks
splitting the stations round-robin, into one MPI-3 shared-memory block
(nx * 17 * ny * NV^2 doubles, ~0.4 GB for 121 x 161 AIR-7), so memory does
not grow with ranks per node and no inter-node traffic is needed.
"""
import numpy as np

from . import jacobians as jac
from . import operators as op


class JacobianStore:
    def __init__(self, solver, par=None):
        self.s = solver
        self.par = par
        bf = solver.bf
        nv = solver.model.nv
        self.shape = (bf.nx, len(jac.KEYS), bf.ny, nv, nv)
        self.win = None

    def build(self):
        bf = self.s.bf
        node = self.par.node if self.par is not None else None
        if self.par is not None:
            self.data, self.win = self.par.shared_array(self.shape)
        else:
            self.data = np.zeros(self.shape)
        r, n = (node.rank, node.size) if (node is not None and self.win is
                                          not None) else (0, 1)
        if self.win is not None:
            self.win.Fence()
        from . import progress
        for i in range(r, bf.nx, n):
            progress.poke()
            q, qx, qy, qz = self.s.station(i)
            o = op.StationOperators(self.s.model, bf.grid, q, qx, qy, qz,
                                    bf.metric(i))
            for k, key in enumerate(jac.KEYS):
                self.data[i, k] = o.J[key]
        if self.win is not None:
            self.win.Fence()
        return self

    def get(self, i):
        return {key: self.data[i, k] for k, key in enumerate(jac.KEYS)}

    def free(self):
        if self.win is not None:
            self.win.Free()
            self.win = None
