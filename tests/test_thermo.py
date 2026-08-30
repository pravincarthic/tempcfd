import numpy as np
from pse import species as sp, thermo as th, transport as tr


def _air(rho=1e-3, n=3):
    Y = sp.mass_fraction_vector({"N2": 0.767, "O2": 0.233})
    return np.maximum(Y[:, None] * rho, 1e-30) * np.ones((1, n))


def test_pressure_and_sound_speed():
    T = np.array([250.0, 1000.0, 3000.0])
    rs = _air()
    p = th.pressure(rs, T, T)
    R = p / (rs.sum(axis=0) * T)
    assert np.allclose(R, 288.06, rtol=1e-3)
    a = th.sound_speed_frozen(rs, T, T)
    assert abs(a[0] / np.sqrt(1.4 * 288.06 * 250.0) - 1.0) < 2e-3


def test_energy_inversion_roundtrip():
    T = np.array([300.0, 1500.0, 4000.0])
    Tv = np.array([280.0, 1200.0, 3500.0])
    rs = _air()
    e = th.energy_internal(rs, T, Tv)
    ev = th.energy_ve(rs, Tv)
    T2, Tv2 = th.T_from_energy(rs, e, ev)
    assert np.allclose(T2, T, rtol=1e-6)
    assert np.allclose(Tv2, Tv, rtol=1e-6)


def test_vibrational_energy_limits():
    lo = th.e_vib_s(np.array([50.0]), sp.IDX["N2"])[0]
    hi = th.e_vib_s(np.array([50000.0]), sp.IDX["N2"])[0]
    assert lo < 1.0
    assert abs(hi / (sp.RS[0] * 50000.0) - 1.0) < 0.05   # classical limit


def test_transport_prandtl_and_low_T_blend():
    T = np.array([250.0, 1000.0, 3000.0])
    rs = _air()
    mu, ktr, kve, rhoD = tr.mixture(rs, T, T)
    Pr = mu * th.cp_mix(rs, T, T) / (ktr + kve)
    assert np.all((Pr > 0.55) & (Pr < 0.85))
    suth = 1.458e-6 * 250.0 ** 1.5 / (250.0 + 110.4)
    assert abs(mu[0] / suth - 1.0) < 2e-3   # tanh blend leaves ~0.4% Blottner
