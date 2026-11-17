"""Linear-algebra back end: CPU (SuperLU) or GPU (CuPy, dense LU on device).

The PSE systems are block banded, n = ny*NV ~ 2000 complex unknowns.  On an
A100 a dense complex128 LU of that size runs at FP64 tensor-core rate
(~19.5 TFLOP/s) in about a millisecond, faster than SuperLU on a CPU core,
and every alpha iteration refactors.  The sparse matrix is shipped to the
device as CSR (a few MB) and densified there, so host-device traffic per
solve is O(nnz), not O(n^2).

Select with   parallel.backend = "cpu" | "gpu"   or env PSE_BACKEND.
Device binding: rank r on a node uses GPU (local_rank % n_gpus); run several
ranks per GPU under NVIDIA MPS (A100 SXM: 4-8 GPUs per node).
"""
import os
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg as sla

_STATE = {"name": "cpu", "xp": None, "la": None, "sparse": None,
          "device": None}


def configure(name=None, local_rank=0, xp=None, la=None, sparse=None):
    """Activate a back end.  xp/la/sparse override the modules (tests use
    numpy/scipy to exercise the device code path without a GPU)."""
    name = (name or os.environ.get("PSE_BACKEND", "cpu")).lower()
    if name == "cpu":
        _STATE.update(name="cpu", xp=None, la=None, sparse=None, device=None)
        return _STATE
    if name != "gpu":
        raise ValueError("unknown backend '%s'" % name)
    if xp is None:
        import cupy as cp
        import cupyx.scipy.linalg as cla
        import cupyx.scipy.sparse as csp
        ndev = cp.cuda.runtime.getDeviceCount()
        if ndev == 0:
            raise RuntimeError("backend gpu requested but no CUDA device")
        dev = int(local_rank) % ndev
        cp.cuda.Device(dev).use()
        xp, la, sparse = cp, cla, csp
        _STATE["device"] = dev
    _STATE.update(name="gpu", xp=xp, la=la, sparse=sparse)
    return _STATE


def name():
    return _STATE["name"]


class _CPU:
    def __init__(self, A):
        self.lu = spla.splu(sps.csc_matrix(A), permc_spec="COLAMD")

    def solve(self, b):
        return self.lu.solve(np.asarray(b))


class _Device:
    def __init__(self, A):
        xp, la, sp = _STATE["xp"], _STATE["la"], _STATE["sparse"]
        A = sps.csr_matrix(A)
        if sp is not None:
            M = sp.csr_matrix((xp.asarray(A.data), xp.asarray(A.indices),
                               xp.asarray(A.indptr)), shape=A.shape).toarray()
        else:
            M = xp.asarray(A.toarray())
        self.xp = xp
        self.lu = la.lu_factor(M, overwrite_a=True, check_finite=False)
        self.la = la

    def solve(self, b):
        xp = self.xp
        x = self.la.lu_solve(self.lu, xp.asarray(b), check_finite=False)
        return x.get() if hasattr(x, "get") else np.asarray(x)


def factor(A):
    """LU of a sparse matrix on the active back end; returns obj.solve(b)."""
    return _Device(A) if _STATE["name"] == "gpu" else _CPU(A)


def solve(A, b):
    return factor(A).solve(b)
