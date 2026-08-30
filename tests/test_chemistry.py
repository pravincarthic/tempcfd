import numpy as np
from pse import chemistry as ch, species as sp


def test_element_and_charge_balance():
    assert np.abs(ch.NU @ sp.MW).max() < 1e-15
    assert np.abs(ch.NU @ sp.CHARGE).max() == 0.0


def test_reaction_inventory():
    s = ch.summary()
    assert s["n_species"] == 11
    assert s["n_reactions"] == 49          # 31 diss + 13 exch + 3 assoc + 2 ei
    assert s["n_eimpact"] == 2


def test_source_conserves_mass():
    rho = 1e-3
    y = np.zeros((11, 1))
    y[0] = 0.6 * rho
    y[1] = 0.1 * rho
    y[3] = 0.15 * rho
    y[4] = 0.15 * rho
    y[2:3] = 1e-10
    T = np.array([5000.0])
    w, _ = ch.source(y, T, T * 0.8)
    assert abs(float(np.real(sum(np.ravel(x)[0] for x in w)))) < 1e-8


def test_equilibrium_trend():
    """O2 dissociates before N2 and both increase with temperature."""
    from scipy.integrate import solve_ivp
    rho = 1e-3

    def rhs(t, v, T):
        w, _ = ch.source(np.maximum(v, 0).reshape(11, 1),
                         np.array([T]), np.array([T]))
        return np.concatenate([np.real(np.atleast_1d(x)) for x in w])

    out = {}
    for T in (4000.0, 6000.0):
        y0 = np.zeros(11)
        y0[0], y0[1] = 0.767 * rho, 0.233 * rho
        y0[2:] = 1e-24
        s = solve_ivp(rhs, [0, 5.0], y0, method="BDF", rtol=1e-8, atol=1e-24,
                      args=(T,))
        ye = np.maximum(s.y[:, -1], 0.0)
        out[T] = ye / ye.sum()
    assert out[4000.0][sp.IDX["O2"]] < 0.01          # O2 gone at 4000 K
    assert out[4000.0][sp.IDX["N2"]] > 0.5           # N2 intact
    assert out[6000.0][sp.IDX["N"]] > out[4000.0][sp.IDX["N"]]
    assert out[6000.0][sp.IDX["e-"]] > out[4000.0][sp.IDX["e-"]]
