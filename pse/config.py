"""Configuration handling (JSON)."""
import json
import numpy as np

DEFAULTS = {
    "case": {
        "name": "mach10_130kft",
        "mach": 10.0,
        "altitude_ft": 130000.0,
        "geometry": "cone",           # cone | flatplate
        "half_angle_deg": 7.0,
    },
    "baseflow": {
        "source": "similarity",       # similarity | openfoam
        "foam_case": None,
        "foam_time": None,
        "wall_patches": ["wall"],
        "plane": "axisym",            # xy | xz | yz | axisym
        "axis": [1.0, 0.0, 0.0],
        "field_map": {},
        "x_start": 0.05,
        "x_end": 2.0,
        "n_stations": 161,
        "ny": 161,
        "y_max": None,
        "y_half": None,
        "y_max_factor": 12.0,
        "y_half_factor": 1.0,
        "wall_temperature": 1000.0,
        "adiabatic_wall": False,
        "freestream_Y": {"N2": 0.767, "O2": 0.233},
        "smooth_passes": 1,
    },
    "physics": {
        "viscous": True,
        "reacting": False,
        "thermal_nonequilibrium": True,
        "diffusion": True,
        "lewis": 1.2,
        "schmidt": None,
        "low_T_blend": True,
        "species_floor": 1e-12,
    },
    "wall_bc": {
        "thermal": "isothermal",      # isothermal | adiabatic
        "vibrational": "equilibrium",  # equilibrium | adiabatic
        "catalytic": "noncatalytic",  # noncatalytic | supercatalytic
    },
    "modes": {
        "f_min_hz": 40.0e3,
        "f_max_hz": 400.0e3,
        "n_frequencies": 96,
        "beta_min": 0.0,
        "beta_max": 0.0,
        "n_beta": 1,
        "frequency_spacing": "linear",  # linear | log
        "c_guess": 0.92,
        "schedule": "center_out",     # center_out | sequential
    },
    "pse": {
        "order": 2,
        "nonparallel": True,
        "far_field": "asymptotic",    # asymptotic | dirichlet
        "alpha_tol": 1.0e-8,
        "alpha_iter": 20,
        "min_step_factor": 1.0,
        "stabilize": 0.0,
        "store_modes": False,
        "energy_norm": "kinetic",
        "i_start": 0,
        "mode_check": True,
    },
    "parallel": {
        "group_size": 1,              # ranks per mode (wall-normal splitting)
        "threads": 1,
    },
    "output": {
        "directory": "results",
        "file": "pse_modes.h5",
        "envelope_csv": "n_factor_envelope.csv",
        "save_baseflow": True,
    },
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path=None, overrides=None):
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    if path:
        with open(path) as f:
            user = json.load(f)
        cfg = _merge(cfg, user)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def mode_list(cfg):
    m = cfg["modes"]
    n = int(m["n_frequencies"])
    lo, hi = float(m["f_min_hz"]), float(m["f_max_hz"])
    f = (np.geomspace(lo, hi, n) if m["frequency_spacing"] == "log"
         else np.linspace(lo, hi, n))
    nb = max(1, int(m["n_beta"]))
    b = (np.array([float(m["beta_min"])]) if nb == 1
         else np.linspace(float(m["beta_min"]), float(m["beta_max"]), nb))
    return [(float(fi), float(bi)) for fi in f for bi in b]


def schedule_order(modes, mode="center_out"):
    """Order in which modes are handed to the dynamic scheduler.

    Cost per mode varies by several times: modes inside the unstable band
    march every station and take the most alpha iterations, modes far outside
    it stop early.  Serving the expensive (band-centre) modes first is the
    longest-processing-time-first heuristic and cuts the tail imbalance when
    there are only a few modes per rank.
    """
    n = len(modes)
    if mode == "sequential" or n < 3:
        return list(range(n))
    f = np.array([m[0] for m in modes])
    fc = 0.5 * (f.min() + f.max())
    return list(np.argsort(np.abs(f - fc), kind="stable"))


def dump(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return path
