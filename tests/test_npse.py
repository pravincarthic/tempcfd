import numpy as np

from pse.npse import NPSESolver, NPSEOptions
from pse.pse import PSESolver, PSEOptions

F = 165e3


def test_fft_roundtrip(bf_small, model):
    s = NPSESolver(bf_small, model, options=NPSEOptions(n_harmonics=2))
    modes = s.mode_set()
    rng = np.random.default_rng(3)
    coef = {mn: rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
            for mn in modes}
    coef[(0, 0)] = coef[(0, 0)].real + 0j
    f = s._to_physical(coef)
    back = s._to_modal(f, modes)
    for mn in modes:
        assert np.allclose(back[mn], coef[mn])


def test_small_amplitude_npse_is_lpse_and_harmonics_are_quadratic(bf_small,
                                                                  model):
    lin = PSESolver(bf_small, model, options=PSEOptions(order=2,
                                                        alpha_iter=15))
    r = lin.march(F, 0.0, i1=3)
    out = {}
    for A in (1e-6, 1e-5):
        s = NPSESolver(bf_small, model,
                       options=NPSEOptions(n_harmonics=2, order=2))
        out[A] = s.march_nonlinear(F, amplitudes={(1, 0): A}, i1=3)
    a = out[1e-6]["alpha"][(1, 0)]
    assert np.allclose(a, r.alpha[:4], rtol=1e-6)
    h1 = out[1e-6]["u_max"][(2, 0)][-1]
    h2 = out[1e-5]["u_max"][(2, 0)][-1]
    m1 = out[1e-6]["u_max"][(0, 0)][-1]
    m2 = out[1e-5]["u_max"][(0, 0)][-1]
    assert abs(h2 / h1 / 100.0 - 1.0) < 0.02
    assert abs(m2 / m1 / 100.0 - 1.0) < 0.02
    assert out[1e-6]["u_max"][(1, 0)][-1] > 1e-6
