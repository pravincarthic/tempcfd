import numpy as np
from pse import fluxes as fx, jacobians as jac, thermo as th, species as sp


def _state(n=4, T=1500.0, u=2000.0):
    q = np.zeros((16, n))
    Y = np.maximum(sp.mass_fraction_vector({"N2": 0.75, "O2": 0.2, "NO": 0.03,
                                            "N": 0.01, "O": 0.01}), 1e-14)
    q[:11] = Y[:, None] * 1e-2
    q[11] = u
    q[12] = 3.0
    q[14] = T
    q[15] = 0.85 * T
    return q


def test_inviscid_characteristic_speeds():
    m = fx.Model(viscous=False, reacting=False, thermal_noneq=False)
    q = _state(1)
    z = np.zeros_like(q)
    J = jac.flux_jacobians(q, z, z, z, m)
    lam = np.linalg.eigvals(np.linalg.solve(J["Gam"][0], J["A"][0]))
    a = float(th.sound_speed_frozen(q[:11], q[14], q[15])[0])
    u = q[11, 0]
    lam = np.sort_complex(lam).real
    assert abs(lam[0] - (u - a)) < 1e-6 * u
    assert abs(lam[-1] - (u + a)) < 1e-6 * u
    assert np.allclose(lam[1:-1], u, rtol=1e-9)


def test_complex_step_matches_finite_difference():
    m = fx.Model(reacting=True, thermal_noneq=True)
    q = _state(2, T=4000.0)
    z = np.zeros_like(q)
    z[14] = 5.0e4                       # a temperature gradient
    z[11] = 1.0e5
    J = jac.flux_jacobians(q, z, z, z, m)
    for k in (0, 11, 14, 15):
        h = 1e-6 * max(abs(q[k, 0]), 1.0)
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        Fp = np.array(fx.fluxes(qp, z, z, z, m)[0])
        Fm = np.array(fx.fluxes(qm, z, z, z, m)[0])
        fd = (Fp - Fm) / (2 * h)
        ref = J["A"][:, :, k].T
        scale = max(np.abs(ref).max(), 1e-30)
        assert np.abs(fd - ref).max() / scale < 2e-4


def test_viscous_jacobian_is_linear_in_gradients():
    m = fx.Model()
    q = _state(3)
    z = np.zeros_like(q)
    J0 = jac.flux_jacobians(q, z, z, z, m)
    z2 = np.zeros_like(q)
    z2[11] = 1e6
    J1 = jac.flux_jacobians(q, z, z2, z, m)
    assert np.allclose(J0["By"], J1["By"], rtol=1e-10, atol=1e-14)


def test_source_jacobian_signs():
    m = fx.Model(reacting=True, thermal_noneq=True)
    q = _state(1, T=6000.0)
    z = np.zeros_like(q)
    J = jac.flux_jacobians(q, z, z, z, m)
    # relaxation must damp a vibrational temperature perturbation
    assert J["E"][0, fx.I_TV, fx.I_TV] < 0.0
