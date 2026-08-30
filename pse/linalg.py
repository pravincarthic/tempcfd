"""Linear-solver back ends: serial sparse/dense and an MPI Schur-complement
substructuring solver that splits the wall-normal direction across the ranks
of a mode communicator.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg as sla


class SerialSolver:
    def __init__(self, A, dense=False):
        self.dense = dense or not sps.issparse(A)
        if self.dense:
            self.lu = sla.lu_factor(np.asarray(A.todense() if sps.issparse(A)
                                               else A))
        else:
            self.lu = spla.splu(A.tocsc(), permc_spec="COLAMD")

    def solve(self, b):
        return sla.lu_solve(self.lu, b) if self.dense else self.lu.solve(b)


class SchurSolver:
    """Distributed solve of a banded system whose unknowns are blocked by
    wall-normal node.

    Rank r owns node slab [j0, j1).  Nodes within `halo` of an internal slab
    boundary form the interface set Gamma; the interiors are eliminated
    locally and the (small) interface Schur complement is formed by an
    all-reduce and solved redundantly.  Exact, not an approximation.
    """

    def __init__(self, comm, ny, nv, halo=2):
        self.comm = comm
        self.ny, self.nv, self.halo = ny, nv, halo
        g = comm.size
        edges = np.linspace(0, ny, g + 1).astype(int)
        self.j0, self.j1 = edges[comm.rank], edges[comm.rank + 1]
        gamma = set()
        for e in edges[1:-1]:
            for j in range(e - halo, e + halo):
                if 0 <= j < ny:
                    gamma.add(j)
        self.gamma_nodes = np.array(sorted(gamma), dtype=int)
        self.gamma = np.concatenate([np.arange(j * nv, (j + 1) * nv)
                                     for j in self.gamma_nodes]) \
            if self.gamma_nodes.size else np.zeros(0, dtype=int)
        own = np.arange(self.j0, self.j1)
        int_nodes = np.array([j for j in own if j not in gamma], dtype=int)
        self.int_nodes = int_nodes
        self.interior = np.concatenate([np.arange(j * nv, (j + 1) * nv)
                                        for j in int_nodes]) \
            if int_nodes.size else np.zeros(0, dtype=int)
        self.n = ny * nv
        self.ng = self.gamma.size

    def factor(self, A):
        A = A.tocsr()
        I, G = self.interior, self.gamma
        self.A_II = A[I][:, I].tocsc()
        self.A_IG = A[I][:, G].tocsc()
        self.A_GI = A[G][:, I].tocsc()
        self.lu = spla.splu(self.A_II, permc_spec="COLAMD") \
            if I.size else None
        W = (self.A_GI @ np.asarray(self.lu.solve(self.A_IG.toarray()))) \
            if I.size else np.zeros((self.ng, self.ng), dtype=complex)
        S_local = -W
        S = np.zeros_like(S_local)
        self.comm.Allreduce(S_local, S)
        # A_GG is global; add it once (rank 0 contribution replicated)
        self.A_GG = np.asarray(A[G][:, G].todense())
        self.S_lu = sla.lu_factor(self.A_GG + S)
        return self

    def solve(self, b):
        I, G = self.interior, self.gamma
        bI = b[I]
        yI = self.lu.solve(bI) if I.size else np.zeros(0, dtype=complex)
        rhs_local = -(self.A_GI @ yI) if I.size else np.zeros(self.ng,
                                                              dtype=complex)
        rhs = np.zeros(self.ng, dtype=complex)
        self.comm.Allreduce(rhs_local, rhs)
        xG = sla.lu_solve(self.S_lu, b[G] + rhs)
        xI = self.lu.solve(bI - self.A_IG @ xG) if I.size else yI
        x_local = np.zeros(self.n, dtype=complex)
        if I.size:
            x_local[I] = xI
        x = np.zeros(self.n, dtype=complex)
        self.comm.Allreduce(x_local, x)
        x[G] = xG
        return x


def make_solver(A, comm=None, ny=None, nv=None, dense=False, halo=2):
    if comm is None or comm.size == 1:
        return SerialSolver(A, dense=dense)
    return SchurSolver(comm, ny, nv, halo=halo).factor(A)
