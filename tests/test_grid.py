import numpy as np
from pse.grid import Grid


def _err(kind, ny):
    g = Grid(ny, 0.02, 0.004, kind=kind)
    f = np.exp(-((g.y - 0.003) / 0.002) ** 2)
    df = -2 * (g.y - 0.003) / 0.002 ** 2 * f
    d2 = (4 * (g.y - 0.003) ** 2 / 0.002 ** 4 - 2 / 0.002 ** 2) * f
    return (np.abs(g.D1 @ f - df).max() / np.abs(df).max(),
            np.abs(g.D2 @ f - d2).max() / np.abs(d2).max())


def test_fd4_convergence():
    e1 = _err("fd4", 81)[0]
    e2 = _err("fd4", 161)[0]
    assert e2 < e1 / 8.0            # at least 3rd order in practice
    assert e2 < 1e-4


def test_chebyshev_spectral():
    d1, d2 = _err("chebyshev", 161)
    assert d1 < 1e-10 and d2 < 1e-7


def test_grid_maps_to_wall_and_edge():
    for kind in ("fd4", "chebyshev"):
        g = Grid(101, 0.05, 0.01, kind=kind)
        assert abs(g.y[0]) < 1e-14
        assert abs(g.y[-1] - 0.05) < 1e-12
        assert np.all(np.diff(g.y) > 0)
        assert abs((g.w.sum() - 0.05) / 0.05) < 1e-12
        # half the points below y_half
        assert abs(np.searchsorted(g.y, 0.01) - 50) < 6 or kind == "chebyshev"


def test_integration_weights():
    g = Grid(401, 1.0, 0.2, kind="fd4")
    assert abs(g.w @ np.sin(np.pi * g.y) - 2.0 / np.pi) < 1e-4
