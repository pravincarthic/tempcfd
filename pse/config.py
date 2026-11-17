"""Configuration handling (JSON).  A config file only needs to override the
keys it changes; everything else falls back to DEFAULTS."""
import json
import numpy as np

DEFAULTS = {
    "case": {
        "name": "mach10_130kft_cone7",
        "mach": 10.0,
        "altitude_ft": 130000.0,
        "geometry": "cone",            # cone | wedge | flatplate
        "half_angle_deg": 7.0,
    },
    "gas": {
        # SU2: FLUID_MODEL= SU2_NONEQ -> "su2tclib", GAS_MODEL= AIR-7
        #      FLUID_MODEL= MUTATIONPP -> "mutationpp", GAS_MODEL= air_7
        "library": "su2tclib",
        "gas_model": None,             # AIR-7 / air_7 by default
        "transport": "WILKE",          # TRANSPORT_COEFF_MODEL
        "frozen": False,               # FROZEN_MIXTURE
        "species_order": None,         # GAS_COMPOSITION order if not the library's
        "mpp_data_dir": None,          # else $MPP_DATA_DIRECTORY, else bundled
        "transport_fallback": "su2tclib",
        "gy_partner_bug": False,       # True reproduces SU2's Gupta-Yos k mixing rule
    },
    "baseflow": {
        "source": "similarity",        # similarity | su2
        "su2": {
            "mesh": None,              # .su2 mesh (for the wall marker)
            "solution": None,          # restart/volume file, glob or list
            "wall_markers": ["wall"],
            "plane": "axisym",         # axisym | xy
            "average": "primitive",    # primitive | conservative (ILES snapshots)
            "curve_smoothing": None,   # smoothing-spline lambda for the wall
            "s0": 0.0,                 # arclength of the first wall node
            "station_axis": "s",       # s (wall arclength) | x (axial)
        },
        "x_start": 0.6,                # wall arclength [m]
        "x_end": 1.8,
        "n_stations": 121,
        "ny": 161,
        "grid": "fd4",                 # fd4 | chebyshev
        "y_max": None,
        "y_half": None,
        "y_max_factor": 4.0,
        "y_half_factor": 0.5,
        "wall_temperature": 1000.0,
        "adiabatic_wall": False,
        "freestream_Y": {"N2": 0.767, "O2": 0.233},
        "smooth_passes": 1,
    },
    "physics": {
        "viscous": True,
        "diffusion": True,
        "thermal_nonequilibrium": True,
        "species_floor": 1e-12,
        "trace_threshold": 1e-6,       # lock species with Y_max below this
    },
    "wall_bc": {
        "thermal": "isothermal",       # isothermal | adiabatic
        "vibrational": "isothermal",   # isothermal | adiabatic
        "catalytic": "noncatalytic",   # noncatalytic | supercatalytic
    },
    "analysis": {
        "type": "lpse",                # lst | lpse | npse
    },
    "modes": {
        "f_min_hz": 80.0e3,
        "f_max_hz": 260.0e3,
        "n_frequencies": 96,
        "frequency_spacing": "linear",  # linear | log
        "beta_min": 0.0,               # planar spanwise wavenumber [1/m]
        "beta_max": 0.0,
        "n_beta": 1,
        "n_azimuthal": None,           # list of azimuthal mode numbers (axisym)
        "c_guess": 0.92,
        "schedule": "center_out",      # center_out | sequential
    },
    "pse": {
        "order": 2,
        "nonparallel": True,
        "far_field": "asymptotic",     # asymptotic | dirichlet
        "alpha_tol": 1.0e-8,
        "alpha_iter": 20,
        "stabilize": 0.0,              # Andersson/Krypton s-coefficient factor k
        "mixed_derivatives": True,
        "min_step_factor": 1.0,
        "store_modes": False,
        "energy_norm": "kinetic",      # kinetic | umax
        "i_start": 0,
        "mode_check": True,
    },
    "npse": {
        "f0_hz": 165.0e3,
        "beta0": 0.0,
        "n_az0": None,
        "n_harmonics": 3,
        "n_spanwise": 0,
        "amplitudes": {"1,0": 1.0e-3},  # max|u'|/u_e of the seeded modes
        "pad": 2,
        "nl_iter": 15,
        "nl_tol": 1.0e-6,
        "relax": 1.0,
        "mean_flow_distortion": True,
    },
    "parallel": {
        "group_size": 1,
        "threads": 1,
        "backend": "cpu",              # cpu | gpu (CuPy; A100: dense LU on device)
        "share_jacobians": True,       # base-flow Jacobians once per node
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
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        with open(path) as f:
            cfg = _merge(cfg, json.load(f))
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def mode_list(cfg):
    """[(f_hz, beta, n_az)] for the linear sweeps."""
    m = cfg["modes"]
    n = int(m["n_frequencies"])
    lo, hi = float(m["f_min_hz"]), float(m["f_max_hz"])
    f = (np.geomspace(lo, hi, n) if m["frequency_spacing"] == "log"
         else np.linspace(lo, hi, n))
    if m.get("n_azimuthal"):
        span = [(0.0, int(k)) for k in m["n_azimuthal"]]
    else:
        nb = max(1, int(m["n_beta"]))
        b = (np.array([float(m["beta_min"])]) if nb == 1
             else np.linspace(float(m["beta_min"]), float(m["beta_max"]), nb))
        span = [(float(bi), None) for bi in b]
    return [(float(fi), bi, na) for fi in f for bi, na in span]


def schedule_order(modes, mode="center_out"):
    """Band-centre-first ordering for the dynamic scheduler (longest
    processing time first)."""
    n = len(modes)
    if mode == "sequential" or n < 3:
        return list(range(n))
    f = np.array([m[0] for m in modes])
    fc = 0.5 * (f.min() + f.max())
    return list(np.argsort(np.abs(f - fc), kind="stable"))


def parse_amplitudes(d):
    out = {}
    for k, v in d.items():
        m, n = (int(t) for t in str(k).replace("(", "").replace(")", "")
                .split(","))
        out[(m, n)] = float(v)
    return out


def dump(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return path
