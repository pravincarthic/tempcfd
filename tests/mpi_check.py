"""Run under mpirun: verifies the Schur-complement solver and the dynamic
mode scheduler against their serial equivalents.

    mpirun -n 4 python3 tests/mpi_check.py
"""
import os
import sys
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpi4py import MPI                                            # noqa: E402
from pse.linalg import SchurSolver                                # noqa: E402
from pse.parallel import Parallel                                 # noqa: E402


def banded(n, nv, halo=2, seed=0):
    rng = np.random.default_rng(seed)
    A = sps.lil_matrix((n * nv, n * nv), dtype=complex)
    for j in range(n):
        for k in range(max(0, j - halo), min(n, j + halo + 1)):
            A[j * nv:(j + 1) * nv, k * nv:(k + 1) * nv] = (
                rng.normal(size=(nv, nv)) + 1j * rng.normal(size=(nv, nv)))
        A[j * nv:(j + 1) * nv, j * nv:(j + 1) * nv] += 30.0 * np.eye(nv)
    return A.tocsr()


def main():
    comm = MPI.COMM_WORLD
    ny, nv = 61, 16
    A = banded(ny, nv)
    b = np.linspace(1.0, 2.0, ny * nv) + 1j
    ref = spla.spsolve(A.tocsc(), b)
    x = SchurSolver(comm, ny, nv, halo=2).factor(A).solve(b)
    err = np.abs(x - ref).max() / np.abs(ref).max()
    ok_solve = err < 1e-10

    par = Parallel(group_size=1)
    n_modes = 4 * comm.size + 3
    seen = par.map_modes(n_modes, lambda k: k * k)
    merged = par.gather_results(seen)
    ok_sched = True
    if comm.rank == 0:
        idx = [i for i, _, _ in merged]
        ok_sched = sorted(idx) == list(range(n_modes)) and \
            all(r == i * i for i, r, _ in merged)
    ok_sched = comm.bcast(ok_sched, root=0)

    if comm.rank == 0:
        print("ranks=%d  schur_rel_err=%.3e  solve_ok=%s  scheduler_ok=%s"
              % (comm.size, err, ok_solve, ok_sched))
    return 0 if (ok_solve and ok_sched) else 1


if __name__ == "__main__":
    sys.exit(main())
