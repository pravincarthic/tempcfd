import os
import numpy as np
import pytest

from su2_fixture import build_case
from pse.su2 import su2_baseflow, SU2Mesh, read_solution, SU2Solution


@pytest.fixture(scope="module")
def case(tmp_path_factory, bf_small):
    d = str(tmp_path_factory.mktemp("su2"))
    mesh, sol = build_case(d, bf_small, ns_cells=40, nn_cells=90)
    _, sol_b = build_case(d, bf_small, ns_cells=40, nn_cells=90, binary=True,
                          temperatures=False)
    return d, mesh, sol, sol_b


def test_mesh_and_restart_readers(case, bf_small):
    d, mesh, sol, sol_b = case
    m = SU2Mesh(mesh)
    assert m.ndim == 2
    assert set(m.markers) == {"wall", "farfield", "inlet", "outlet"}
    assert len(m.marker_nodes(["wall"])) == 41
    a = read_solution(sol)
    b = read_solution(sol_b)
    assert "Temperature_tr" in a and "Temperature_tr" not in b
    for k in ("x", "y", "Density_1", "Momentum_x", "Energy", "Energy_ve"):
        assert np.allclose(a[k], b[k], rtol=1e-14)


@pytest.mark.parametrize("which", ["csv", "binary"])
def test_su2_baseflow_matches_source(case, bf_small, which):
    d, mesh, sol, sol_b = case
    x = np.linspace(bf_small.x[0] + 0.05, bf_small.x[-1] - 0.05, 4)
    bf = su2_baseflow(bf_small.gas, sol if which == "csv" else sol_b, x,
                      mesh=mesh, ny=101, y_max=bf_small.grid.y_max,
                      y_half=bf_small.grid.y_half, smooth=0,
                      s0=bf_small.x[0])
    g = bf.gas
    for i, xv in enumerate(x):
        ue = np.interp(xv, bf_small.x, bf_small.edge["u_e"])
        assert abs(bf.edge["u_e"][i] / ue - 1.0) < 1e-6
        assert abs(bf.q[i, g.i_T, 0] - 1000.0) < 1e-6
        assert abs(bf.q[i, g.i_Tv, 0] - 1000.0) < 1e-6
        d99 = np.interp(xv, bf_small.x, bf_small.edge["delta99"])
        assert abs(bf.edge["delta99"][i] / d99 - 1.0) < 0.05
    geo = bf.geometry
    assert geo.axisymmetric
    assert np.allclose(geo.ny, np.cos(np.radians(7.0)), atol=1e-8)
    assert np.allclose(geo.yw, x * np.sin(np.radians(7.0)), atol=1e-8)
    assert np.all(np.abs(geo.kappa) < 1e-6)


def test_time_average_of_snapshots(tmp_path, bf_small):
    d = str(tmp_path)
    files = []
    for k, eps in enumerate((0.02, -0.02)):
        _, f = build_case(d, bf_small, ns_cells=20, nn_cells=60,
                          fname="restart_flow_%05d.csv" % k)
        fl = read_solution(f)
        fl["Velocity_x"] = fl["Velocity_x"] * (1.0 + eps)
        from pse.su2 import write_csv
        write_csv(f, fl, order=[n for n in fl if n != "PointID"])
        files.append(f)
    ex = SU2Solution(bf_small.gas, os.path.join(d, "restart_flow_*.csv"))
    one = SU2Solution(bf_small.gas, files[0])
    assert len(ex.files) == 2
    g = bf_small.gas
    ua = ex.fields[g.i_u]
    u1 = one.fields[g.i_u]
    assert np.allclose(ua, u1 / 1.02, rtol=1e-12, atol=1e-9)
