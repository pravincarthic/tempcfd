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
