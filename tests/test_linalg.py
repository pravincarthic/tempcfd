import numpy as np
import scipy.sparse as sps
from pse.linalg import SerialSolver


def _banded(n, nv, halo=2, seed=0):
    rng = np.random.default_rng(seed)
    A = sps.lil_matrix((n * nv, n * nv), dtype=complex)
    for j in range(n):
        for k in range(max(0, j - halo), min(n, j + halo + 1)):
            A[j * nv:(j + 1) * nv, k * nv:(k + 1) * nv] = (
                rng.normal(size=(nv, nv)) + 1j * rng.normal(size=(nv, nv)))
        A[j * nv:(j + 1) * nv, j * nv:(j + 1) * nv] += 20.0 * np.eye(nv)
    return A.tocsr()


def test_serial_solver_dense_and_sparse():
    A = _banded(20, 4)
    b = np.arange(A.shape[0], dtype=complex) + 1.0
    x1 = SerialSolver(A).solve(b)
    x2 = SerialSolver(A, dense=True).solve(b)
    assert np.abs(A @ x1 - b).max() < 1e-9
    assert np.abs(x1 - x2).max() < 1e-9
