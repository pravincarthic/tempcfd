"""MPI decomposition.

Two levels:

1.  Mode level.  Each (frequency, spanwise wavenumber) pair is an independent
    march, so the outer loop is embarrassingly parallel.  Work is handed out
    by a lock-free shared counter (MPI-3 RMA fetch-and-add) rather than a
    static block decomposition, which keeps every rank busy even though the
    cost per mode varies by a factor of several (different numbers of alpha
    iterations, different station ranges, occasional mode re-searches).  No
    rank is sacrificed as a master.

2.  Wall-normal level.  When there are fewer modes than ranks, COMM_WORLD is
    split into groups of `group_size`; inside a group the wall-normal domain
    is partitioned and every linear solve is done by the exact Schur-complement
    substructuring solver in linalg.SchurSolver, while the Jacobian assembly
    is split over the same slabs and gathered.  This keeps 384 cores busy for
    a sweep of, say, 96 modes.
"""
import os
import time
import numpy as np

try:
    from mpi4py import MPI
    HAVE_MPI = True
except Exception:                                    # pragma: no cover
    MPI = None
    HAVE_MPI = False


class _SerialComm:
    rank = 0
    size = 1

    def Barrier(self):
        pass

    def bcast(self, obj, root=0):
        return obj

    def gather(self, obj, root=0):
        return [obj]

    def allgather(self, obj):
        return [obj]

    def Allreduce(self, a, b, op=None):
        b[...] = a

    def Split(self, color, key):
        return self

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1


class Parallel:
    """Hierarchical communicator set-up and the dynamic mode scheduler."""

    def __init__(self, group_size=1, comm=None):
        if HAVE_MPI:
            self.world = comm or MPI.COMM_WORLD
        else:
            self.world = _SerialComm()
        self.size = self.world.size
        self.rank = self.world.rank
        group_size = max(1, min(int(group_size), self.size))
        while self.size % group_size:
            group_size -= 1
        self.group_size = group_size
        self.n_groups = self.size // group_size
        self.group_id = self.rank // group_size
        if HAVE_MPI and group_size > 1:
            self.group = self.world.Split(self.group_id, self.rank)
            self.leaders = self.world.Split(
                0 if self.rank % group_size == 0 else 1, self.rank)
        else:
            self.group = _SerialComm()
            self.leaders = self.world
        self.group_size = group_size
        self.group_rank = self.group.rank if group_size > 1 else 0
        self.is_leader = self.group_rank == 0
        self._win = None
        self._counter = None

    # ------------------------------------------------------------- scheduler
    def _open_counter(self):
        if not HAVE_MPI or self.n_groups == 1:
            self._counter = np.zeros(1, dtype=np.int64)
            return
        if self.rank == 0:
            self._counter = np.zeros(1, dtype=np.int64)
            self._win = MPI.Win.Create(self._counter, disp_unit=8,
                                       comm=self.world)
        else:
            self._counter = np.zeros(0, dtype=np.int64)
            self._win = MPI.Win.Create(self._counter, disp_unit=8,
                                       comm=self.world)

    def _next_index(self):
        """Atomically fetch-and-add the global work counter."""
        if self._win is None:
            v = int(self._counter[0])
            self._counter[0] += 1
            return v
        out = np.zeros(1, dtype=np.int64)
        one = np.ones(1, dtype=np.int64)
        self._win.Lock(0, MPI.LOCK_EXCLUSIVE)
        self._win.Fetch_and_op(one, out, 0, 0, MPI.SUM)
        self._win.Unlock(0)
        return int(out[0])

    def map_modes(self, n_modes, work_fn, progress=None):
        """Run work_fn(index) over 0..n_modes-1 with dynamic scheduling.

        Only group leaders pull work; the other ranks of a group participate
        through the collective linear solves inside work_fn.
        Returns the list of (index, result, seconds) produced by this rank.
        """
        self._open_counter()
        out = []
        while True:
            if self.is_leader:
                idx = self._next_index()
            else:
                idx = 0
            if self.group_size > 1:
                idx = self.group.bcast(idx, root=0)
            if idx >= n_modes:
                break
            t0 = time.time()
            r = work_fn(idx)
            dt = time.time() - t0
            if self.is_leader:
                out.append((idx, r, dt))
                if progress is not None:
                    progress(idx, dt)
        if self._win is not None:
            self.world.Barrier()
            self._win.Free()
            self._win = None
        return out

    # ------------------------------------------------------------- utilities
    def gather_results(self, local, root=0):
        if not HAVE_MPI or self.size == 1:
            return local
        chunks = self.world.gather(local, root=root)
        if self.rank != root:
            return None
        merged = []
        for c in chunks:
            if c:
                merged.extend(c)
        merged.sort(key=lambda t: t[0])
        return merged

    def log(self, msg):
        if self.rank == 0:
            print(msg, flush=True)

    def summary(self):
        return dict(ranks=self.size, group_size=self.group_size,
                    groups=self.n_groups, mpi=HAVE_MPI,
                    threads=int(os.environ.get("OMP_NUM_THREADS", "1")))
