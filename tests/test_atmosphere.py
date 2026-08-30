from pse.atmosphere import us1976, freestream


def test_sea_level():
    a = us1976(z_m=0.0)
    assert abs(a["T"] - 288.15) < 1e-6
    assert abs(a["p"] - 101325.0) < 1e-3
    assert abs(a["rho"] - 1.225) < 2e-3


def _z_of_h(h):
    """Geometric altitude for a tabulated geopotential altitude."""
    from pse.constants import R_EARTH
    return R_EARTH * h / (R_EARTH - h)


def test_tropopause_and_11km():
    a = us1976(z_m=_z_of_h(11000.0))
    assert abs(a["T"] - 216.65) < 0.05
    assert abs(a["p"] - 22632.0) / 22632.0 < 2e-3


def test_20km_32km():
    assert abs(us1976(z_m=_z_of_h(20000.0))["p"] - 5474.9) / 5474.9 < 3e-3
    assert abs(us1976(z_m=_z_of_h(32000.0))["p"] - 868.02) / 868.02 < 3e-3


def test_130kft_mach10():
    f = freestream(10.0, z_ft=130000.0)
    assert abs(f["z"] - 39624.0) < 1e-6
    assert abs(f["T"] - 249.3) < 0.5          # 39.6 km, +2.8 K/km layer
    assert abs(f["p"] - 302.1) / 302.1 < 5e-3
    assert abs(f["rho"] - 4.222e-3) / 4.222e-3 < 5e-3
    assert abs(f["U"] - 3165.0) / 3165.0 < 5e-3
    assert 8.0e5 < f["Re_unit"] < 8.8e5
