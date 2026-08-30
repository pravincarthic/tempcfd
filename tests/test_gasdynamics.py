import numpy as np
from pse.gasdynamics import oblique_shock, taylor_maccoll


def test_oblique_shock_m10_7deg():
    b, M2, p, r, T = oblique_shock(10.0, np.radians(7.0))
    assert abs(np.degrees(b) - 11.38) < 0.1
    assert abs(M2 - 7.62) < 0.05
    assert abs(p - 4.377) < 0.02


def test_normal_shock_limit():
    b, M2, p, r, T = oblique_shock(2.0, np.radians(1e-6))
    assert abs(np.degrees(b) - np.degrees(np.arcsin(0.5))) < 0.05


def test_cone_m10_7deg():
    c = taylor_maccoll(10.0, np.radians(7.0))
    assert abs(np.degrees(c["beta"]) - 9.45) < 0.15
    assert abs(c["M_cone"] - 8.14) < 0.1
    assert c["beta"] > np.radians(7.0)
