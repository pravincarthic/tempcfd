import numpy as np

from pse import bcs, fluxes as fx, lst, operators as op
from pse.pse import PSESolver, PSEOptions


def _ops(bf, i, model=None):
    model = model or fx.Model(reacting=False, thermal_noneq=True)
    q, qx, qy, qz = bf.station(i)
    q = np.array(q)
    rho = q[:11].sum(axis=0)
    q[:11] = np.maximum(q[:11], 1e-12 * rho)
    o = op.StationOperators(q, qx, qy, qz, model, bf.grid, metric=bf.metric(i))
    return o, bcs.BCOperator(bf.grid, q, bcs.WallBC()), q


def test_farfield_characteristic_split(bf_small):
    o, b, q = _ops(bf_small, 8)
    om = 2 * np.pi * 150e3
    L0, L1, L2 = o.farfield_coefficients(300.0 - 5j, 0.0, om)
    lam, V, Z = bcs.farfield_modes(L0, L1, L2)
    assert len(lam) == 31                      # 15 second-order + 1 first-order
    assert int((lam.real < 0).sum()) == 16     # -> 16 far-field conditions


def test_second_mode_is_trapped_and_wall_peaked(bf_small):
    o, b, q = _ops(bf_small, 8)
    e = bf_small.edge
    om = 2 * np.pi * 130e3
    r = lst.find_second_mode(o, b, 0.0, om, e["u_e"][8], e["T_e"][8],
                             e["delta99"][8])
    assert r is not None
    a, v, d = r
    assert d["converged"]
    assert d["far_fraction"] < 1e-4            # decays inside the layer
    assert d["p_wall"] > 0.8                   # pressure maximum at the wall
    assert 0.7 < d["c_phase"] < 1.05
    assert 1.5 < a.real * e["delta99"][8] < 6.0


def test_newton_is_grid_converged(bf_small):
    from pse.grid import Grid
    e = bf_small.edge
    om = 2 * np.pi * 130e3
    res = []
    for ny in (81, 121):
        g = Grid(ny, bf_small.grid.y_max, bf_small.grid.y_half, kind="fd4")
        q0 = bf_small.q[8]
        q = np.array([np.interp(g.y, bf_small.grid.y, q0[v])
                      for v in range(16)])
        rho = q[:11].sum(axis=0)
        q[:11] = np.maximum(q[:11], 1e-12 * rho)
        qy = (g.D1 @ q.T).T
        z = np.zeros_like(q)
        o = op.StationOperators(q, z, qy, z, fx.Model(reacting=False), g)
        b = bcs.BCOperator(g, q, bcs.WallBC())
        r = lst.find_second_mode(o, b, 0.0, om, e["u_e"][8], e["T_e"][8],
                                 e["delta99"][8])
        assert r is not None
        res.append(r[0])
    assert abs(res[0] - res[1]) / abs(res[1]) < 0.02


def test_pse_march_produces_growth_and_nfactor(bf_small):
    s = PSESolver(bf_small, fx.Model(reacting=False, thermal_noneq=True),
                  bcs.WallBC(), PSEOptions(order=2, alpha_iter=12))
    r = s.march(130e3, 0.0, i0=0)
    assert r.ok, r.message
    assert len(r.x) == bf_small.nx
    assert np.all(np.isfinite(r.alpha))
    assert r.N.max() > 0.5                      # an amplified band exists
    assert np.all(np.diff(r.N_alpha) == np.diff(r.N_alpha))   # no NaN
    assert np.all((r.c_phase > 0.6) & (r.c_phase < 1.1))


def test_pse_alpha_close_to_lst_at_first_station(bf_small):
    s = PSESolver(bf_small, fx.Model(reacting=False, thermal_noneq=True),
                  bcs.WallBC(), PSEOptions(order=1, alpha_iter=15))
    r = s.march(130e3, 0.0, i0=0, i1=3)
    o, b, q = _ops(bf_small, 0)
    e = bf_small.edge
    ref = lst.find_second_mode(o, b, 0.0, 2 * np.pi * 130e3, e["u_e"][0],
                               e["T_e"][0], e["delta99"][0])
    assert abs(r.alpha[0] - ref[0]) / abs(ref[0]) < 1e-6
