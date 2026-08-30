import os
import numpy as np
import pytest

from foam_fixture import build_case
from pse import baseflow as bfm
from pse.foam import FoamMesh, ProfileExtractor


@pytest.fixture(scope="module")
def case(tmp_path_factory, bf_small):
    d = str(tmp_path_factory.mktemp("foam"))
    build_case(d, bf_small, nx=60, ny=80)
    return d, bf_small


def test_mesh_geometry(case):
    d, bf = case
    m = FoamMesh(d)
    assert m.ncells == 60 * 80
    assert set(["inlet", "outlet", "wall", "top"]).issubset(m.boundary)
    expect = (bf.x[-1] - bf.x[0]) * bf.grid.y_max * 1e-3
    assert abs(m.V.sum() / expect - 1.0) < 1e-10
    assert np.all(m.V > 0.0)
    n = m.patch_normals("wall")
    assert np.allclose(n[:, 1], 1.0, atol=1e-12)     # points into the fluid


def test_field_reading_and_profiles(case):
    d, bf = case
    ex = ProfileExtractor(d, wall_patches=("wall",), plane="xy")
    assert ex.time in os.listdir(d)
    P = ex.profile(1.0, np.linspace(0.0, bf.grid.y_max, 200))
    j = int(np.argmin(np.abs(bf.x - (bf.x[0] + 1.0))))
    assert abs(P["u"][-1] / bf.edge["u_e"][j] - 1.0) < 5e-3
    assert abs(P["T"][0] - bf.q[j, 14, 0]) / bf.q[j, 14, 0] < 5e-3
    assert abs(P["Y"].sum(axis=0).max() - 1.0) < 1e-10
    assert np.all(P["rho"] > 0)


def test_foam_baseflow_matches_source(case):
    d, bf = case
    x = np.linspace(bf.x[0] + 0.15, bf.x[-1] - 0.15, 7)
    bf2 = bfm.foam_baseflow(d, x, ny=101, wall_patches=("wall",), plane="xy",
                            smooth=0, y_max=bf.grid.y_max,
                            y_half=bf.grid.y_half)
    for i, xv in enumerate(x):
        j = int(np.argmin(np.abs(bf.x - xv)))
        assert abs(bf2.edge["u_e"][i] / bf.edge["u_e"][j] - 1.0) < 1e-3
        assert abs(bf2.q[i, 14, 0] / bf.q[j, 14, 0] - 1.0) < 1e-2
