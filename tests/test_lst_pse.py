import numpy as np

from pse import bcs, fluxes as fx, lst, operators as op
from pse.grid import Grid
from pse.pse import PSESolver, PSEOptions

F = 165e3


def _ops(bf, i, model):
    q, qx, qy, qz = bf.station(i)
    q = np.array(q)
    ns = model.ns
    rho = q[:ns].sum(axis=0)
    q[:ns] = np.maximum(q[:ns], 1e-12 * rho)
    o = op.StationOperators(model, bf.grid, q, qx, qy, qz, bf.metric(i))
    return o, bcs.BCOperator(model, bf.grid, q, bcs.WallBC(), J=o.J), q


def test_farfield_characteristic_split(bf_small, model):
    o, b, q = _ops(bf_small, 3, model)
    L0, L1, L2 = o.farfield_coefficients(300.0 - 5j, 0.0, 2 * np.pi * 150e3)
    lam, _, _ = bcs.farfield_modes(L0, L1, L2, col=b.col)
    assert len(lam) == 23                    # 11 second-order + 1 first-order
    assert int((lam.real < 0).sum()) == 12   # -> 12 far-field conditions
    R0, R1, R2 = b.reduce_far(L0, L1, L2)
    lam_r, _, _ = bcs.farfield_modes(R0, R1, R2, col=b.col[b.active])
    na = len(b.active)
    assert len(lam_r) == 2 * na - 1 and int((lam_r.real < 0).sum()) == na


def test_trace_species_are_locked(bf_small, model):
    o, b, q = _ops(bf_small, 0, model)
    names = [model.gas.names[s] for s in b.trace]
    assert set(names) == {"e-", "NO", "N", "O", "NO+"}
    assert model.gas.names[b.keep] == "N2"


def test_second_mode_is_trapped_wall_peaked_and_amplified(bf_small, model):
    o, b, q = _ops(bf_small, 0, model)
    e = bf_small.edge
    r = lst.find_second_mode(o, b, 0.0, 2 * np.pi * F, e["u_e"][0],
                             e["T_e"][0], e["delta99"][0])
    assert r is not None
    a, v, d = r
    assert d["converged"]
    assert d["far_fraction"] < 1e-4
    assert d["p_wall"] > 0.8
    assert 1.0 - 1.0 / e["M_e"][0] < d["c_phase"] < 1.0
    assert a.imag < -3.0                      # amplified
    assert 1.0 < a.real * e["delta99"][0] < 3.0


def test_lst_is_grid_converged(bf_small, model):
    e = bf_small.edge
    res = []
    for ny in (91, 131):
        g = Grid(ny, bf_small.grid.y_max, bf_small.grid.y_half, kind="fd4")
        q0 = bf_small.q[0]
        q = np.array([np.interp(g.y, bf_small.grid.y, q0[v])
                      for v in range(model.nv)])
        rho = q[:model.ns].sum(axis=0)
        q[:model.ns] = np.maximum(q[:model.ns], 1e-12 * rho)
        qy = (g.D1 @ q.T).T
        z = np.zeros_like(q)
        o = op.StationOperators(model, g, q, z, qy, z,
                                bf_small.geometry.metric(0, g.y))
        b = bcs.BCOperator(model, g, q, bcs.WallBC(), J=o.J)
        r = lst.find_second_mode(o, b, 0.0, 2 * np.pi * F, e["u_e"][0],
                                 e["T_e"][0], e["delta99"][0])
        res.append(r[0])
    assert abs(res[0] - res[1]) / abs(res[1]) < 0.01


def test_pse_march_growth_and_first_station(bf_small, model):
    s = PSESolver(bf_small, model, options=PSEOptions(order=2, alpha_iter=15))
    r = s.march(F, 0.0)
    assert r.ok, r.message
    assert len(r.x) == bf_small.nx
    assert np.all(np.isfinite(r.alpha))
    assert r.N.max() > 0.9 and r.N_alpha.max() > 1.0
    assert np.all((r.c_phase > 0.85) & (r.c_phase < 1.0))
    o, b, q = _ops(bf_small, 0, model)
    e = bf_small.edge
    ref = lst.find_second_mode(o, b, 0.0, 2 * np.pi * F, e["u_e"][0],
                               e["T_e"][0], e["delta99"][0])
    assert abs(r.alpha[0] - ref[0]) / abs(ref[0]) < 1e-8
    # parallel LST tracking gives the same sign of growth
    rl = s.lst_march(F, 0.0)
    assert rl.ok and rl.N[-1] > 0.3


def test_stabilised_march_runs_below_the_step_limit(bf_small, model):
    s = PSESolver(bf_small, model, options=PSEOptions(order=1, alpha_iter=12,
                                                      stabilize=1.0))
    r = s.march(F, 0.0, i1=3)
    assert r.ok and np.all(np.isfinite(r.N))
