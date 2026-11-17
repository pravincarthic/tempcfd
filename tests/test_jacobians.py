import numpy as np

from pse import fluxes as fx, jacobians as jac


def _state(g, n=4, T=1500.0, u=2000.0):
    q = np.zeros((g.nv, n))
    Y = g.mass_fraction_vector({"N2": 0.75, "O2": 0.2, "NO": 0.03, "N": 0.01,
                                "O": 0.01})
    Y = np.maximum(Y, 1e-12)
    q[:g.ns] = Y[:, None] * 1e-2
    q[g.i_u] = u
    q[g.i_v] = 3.0
    q[g.i_T] = T
    q[g.i_Tv] = 0.85 * T
    return q


def test_inviscid_characteristic_speeds(gas):
    m = fx.Model(gas, viscous=False)
    q = _state(gas, 1)
    z = np.zeros_like(q)
    J = jac.flux_jacobians(m, q, z, z, z, fx.Metric(1))
    lam = np.linalg.eigvals(np.linalg.solve(J["Gam"][0], J["A"][0]))
    a = float(gas.sound_speed(q[:gas.ns], q[gas.i_T], q[gas.i_Tv])[0])
    u = q[gas.i_u, 0]
    lam = np.sort(lam.real)
    assert abs(lam[0] - (u - a)) < 1e-6 * u
    assert abs(lam[-1] - (u + a)) < 1e-6 * u
    assert np.allclose(lam[1:-1], u, rtol=1e-9)


def test_complex_step_matches_finite_difference(gas):
    m = fx.Model(gas)
    q = _state(gas, 2, T=4000.0)
    z = np.zeros_like(q)
    z[gas.i_T] = 5.0e4
    z[gas.i_u] = 1.0e5
    met = fx.Metric(2)
    J = jac.flux_jacobians(m, q, z, z, z, met)
    for k in (1, gas.i_u, gas.i_T, gas.i_Tv):
        h = 1e-6 * max(abs(q[k, 0]), 1.0)
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        Fp = np.array(fx.fluxes(m, qp, z, z, z, met)[0])
        Fm = np.array(fx.fluxes(m, qm, z, z, z, met)[0])
        fd = (Fp - Fm) / (2 * h)
        ref = J["A"][:, :, k].T
        assert np.abs(fd - ref).max() / max(np.abs(ref).max(), 1e-30) < 2e-4


def test_viscous_jacobian_is_linear_in_gradients(gas):
    m = fx.Model(gas)
    q = _state(gas, 3)
    z = np.zeros_like(q)
    J0 = jac.flux_jacobians(m, q, z, z, z, fx.Metric(3))
    z2 = np.zeros_like(q)
    z2[gas.i_u] = 1e6
    J1 = jac.flux_jacobians(m, q, z, z2, z, fx.Metric(3))
    assert np.allclose(J0["By"], J1["By"], rtol=1e-10, atol=1e-14)


def test_vt_relaxation_damps_tv(gas):
    m = fx.Model(gas)
    q = _state(gas, 1, T=6000.0)
    z = np.zeros_like(q)
    J = jac.flux_jacobians(m, q, z, z, z, fx.Metric(1))
    assert J["Eq"][0, gas.i_Tv, gas.i_Tv] > 0.0     # E = ... - S


def _uniform_axial(gas, theta, r_wall, y):
    """Uniform stream along the body axis seen in a cone-surface frame."""
    n = y.size
    q = np.zeros((gas.nv, n))
    Y = np.maximum(gas.mass_fraction_vector({"N2": 0.767, "O2": 0.233}), 1e-12)
    q[:gas.ns] = (Y / Y.sum())[:, None] * 5e-3
    U = 3000.0
    q[gas.i_u] = U * np.cos(theta)
    q[gas.i_v] = -U * np.sin(theta)
    q[gas.i_T] = 400.0
    q[gas.i_Tv] = 400.0
    r = r_wall + y * np.cos(theta)
    met = fx.Metric(n, a_s=np.sin(theta) / r, a_n=np.cos(theta) / r, r=r)
    return q, met


def test_axisymmetric_metric_preserves_uniform_stream():
    from pse.gas import make_gas
    g = make_gas("su2tclib", frozen=True)
    y = np.linspace(0.0, 0.05, 5)
    q, met = _uniform_axial(g, np.radians(7.0), 0.1, y)
    z = np.zeros_like(q)
    m = fx.Model(g, thermal_noneq=False)
    lF, G, H, E = fx.fluxes(m, q, z, z, z, met)
    E = np.array(E)
    ref = np.abs(np.array(lF)).max(axis=1)[:, None] / 0.1
    assert np.all(np.abs(E) <= 1e-12 * ref + 1e-12)


def test_curvature_metric_preserves_uniform_stream():
    """Uniform stream past a convex circular wall (radius R): in the
    wall frame u = U cos(s/R), v = U sin(s/R); with h = 1 + n/R the
    stresses vanish and the residual is zero."""
    from pse.gas import make_gas
    g = make_gas("su2tclib", frozen=True)
    R = 0.3
    s = 0.07
    y = np.linspace(0.0, 0.02, 5)
    n = y.size
    q = np.zeros((g.nv, n))
    Y = np.maximum(g.mass_fraction_vector({"N2": 0.767, "O2": 0.233}), 1e-12)
    q[:g.ns] = (Y / Y.sum())[:, None] * 5e-3
    U = 2500.0
    u, v = U * np.cos(s / R), U * np.sin(s / R)
    q[g.i_u], q[g.i_v] = u, v
    q[g.i_T] = q[g.i_Tv] = 500.0
    qx = np.zeros_like(q)
    qx[g.i_u] = -v / R
    qx[g.i_v] = u / R
    h = 1.0 + y / R
    met = fx.Metric(n, l=1.0 / h, kap=(1.0 / R) / h)
    z = np.zeros_like(q)
    m = fx.Model(g, thermal_noneq=False)
    lF, G, H, E = fx.fluxes(m, q, qx, z, z, met)
    # R = d(lF)/ds + dG/dn + E; lF and G depend on s only through u, v
    qxx = np.zeros_like(q)
    qxx[g.i_u] = -u / R ** 2
    qxx[g.i_v] = -v / R ** 2
    lFs = np.array(fx.fluxes(m, q + 1e-30j * qx, qx + 1e-30j * qxx, z, z,
                             met)[0]).imag / 1e-30
    res = lFs + np.array(E)                       # dG/dn = 0 here
    scale = np.abs(np.array(lF)).max(axis=1)[:, None] / R
    assert np.all(np.abs(res) <= 1e-10 * scale + 1e-10)
