import numpy as np
import scipy.linalg as sla

from pse import backend
from pse.pse import PSESolver, PSEOptions


def test_shared_jacobians_equal_on_the_fly(bf_small, model):
    a = PSESolver(bf_small, model, options=PSEOptions(alpha_iter=10))
    b = PSESolver(bf_small, model, options=PSEOptions(alpha_iter=10))
    b.share_jacobians()
    J1, J2 = a.cache.jac(3), b.cache.jac(3)
    for k in J1:
        assert np.array_equal(J1[k], J2[k])
    r1 = a.march(165e3, i1=2)
    r2 = b.march(165e3, i1=2)
    assert np.allclose(r1.alpha, r2.alpha, rtol=1e-12)


def test_device_dense_lu_path_matches_cpu(bf_small, model):
    s = PSESolver(bf_small, model, options=PSEOptions(alpha_iter=10))
    s.share_jacobians()
    ref = s.march(165e3, i1=2)
    try:
        # the GPU code path with numpy/scipy standing in for cupy
        backend.configure("gpu", xp=np, la=sla, sparse=None)
        assert backend.name() == "gpu"
        dev = s.march(165e3, i1=2)
    finally:
        backend.configure("cpu")
    assert np.allclose(dev.alpha, ref.alpha, rtol=1e-8)
    assert np.allclose(dev.N, ref.N, atol=1e-8)


def test_block_tridiagonal_solver_with_cross_block_pivoting():
    from pse.batched import block_tridiag_solve
    rng = np.random.default_rng(0)
    B, nb, S = 2, 5, 6

    def c(*s):
        return rng.normal(size=s) + 1j * rng.normal(size=s)
    Lo, Dg, Up, r = c(B, nb, S, S), c(B, nb, S, S), c(B, nb, S, S), c(B, nb, S)
    Dg[:, 2] *= 1e-9
    x = block_tridiag_solve(np, Lo, Dg, Up, r)
    for b in range(B):
        A = np.zeros((nb * S, nb * S), complex)
        for k in range(nb):
            A[k*S:(k+1)*S, k*S:(k+1)*S] = Dg[b, k]
            if k:
                A[k*S:(k+1)*S, (k-1)*S:k*S] = Lo[b, k]
            if k < nb - 1:
                A[k*S:(k+1)*S, (k+1)*S:(k+2)*S] = Up[b, k]
        ref = np.linalg.solve(A, r[b].ravel())
        assert np.abs(x[b].ravel() - ref).max() < 1e-11 * np.abs(ref).max()


def test_lockstep_batch_equals_reference_march(bf_small, model):
    from pse.lockstep import LockstepMarcher
    s = PSESolver(bf_small, model, options=PSEOptions(alpha_iter=12))
    s.share_jacobians()
    modes = [(155e3, 0.0, None), (170e3, 0.0, None)]
    ref = [s.march(f, i1=3) for f, _, _ in modes]
    out = LockstepMarcher(s).march(modes, i1=3)
    for r, o in zip(ref, out):
        assert o.ok == r.ok and len(o.x) == len(r.x)
        assert np.allclose(o.alpha, r.alpha, rtol=1e-9)
        assert np.allclose(o.N, r.N, atol=1e-8)
