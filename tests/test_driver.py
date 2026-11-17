import json
import os
import numpy as np

from su2_fixture import build_case
from pse import config as C
from pse.driver import main


def test_driver_lpse_on_su2_case(tmp_path, bf_small):
    d = str(tmp_path)
    mesh, sol = build_case(d, bf_small, ns_cells=30, nn_cells=90)
    cfg = {
        "baseflow": {"source": "su2",
                     "su2": {"mesh": mesh, "solution": sol, "s0": 0.6},
                     "x_start": 0.62, "x_end": 0.72, "n_stations": 3,
                     "ny": 91},
        "analysis": {"type": "lpse"},
        "modes": {"f_min_hz": 160e3, "f_max_hz": 170e3, "n_frequencies": 2},
        "pse": {"alpha_iter": 10, "mode_check": False},
        "output": {"directory": os.path.join(d, "out"), "file": "m.npz"},
    }
    p = os.path.join(d, "cfg.json")
    with open(p, "w") as f:
        json.dump(cfg, f)
    assert main([p]) == 0
    env = np.loadtxt(os.path.join(d, "out", "n_factor_envelope.csv"),
                     delimiter=",", comments="#", skiprows=3)
    assert env.shape[1] == 3 and np.all(np.isfinite(env[:, 1]))
    assert os.path.exists(os.path.join(d, "out", "baseflow.npz"))


def test_config_defaults_and_modes():
    cfg = C.load(None, overrides={"modes": {"n_frequencies": 4,
                                            "n_azimuthal": [0, 10]}})
    m = C.mode_list(cfg)
    assert len(m) == 8 and m[1][2] == 10
    assert C.parse_amplitudes({"1,0": 1e-3, "(1,1)": 2e-4}) == \
        {(1, 0): 1e-3, (1, 1): 2e-4}
