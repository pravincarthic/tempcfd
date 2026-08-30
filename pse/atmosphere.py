"""US Standard Atmosphere 1976, 0-86 km geometric, plus freestream builder."""
import numpy as np
from .constants import R_UNIV, G0, R_EARTH, M_AIR, FT_TO_M

_HB = np.array([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.85202]) * 1e3  # geopotential m
_LB = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0]) * 1e-3         # K/m
_TB = np.zeros(8)
_PB = np.zeros(8)
_TB[0], _PB[0] = 288.15, 101325.0
for _i in range(7):
    _dh = _HB[_i + 1] - _HB[_i]
    _TB[_i + 1] = _TB[_i] + _LB[_i] * _dh
    if _LB[_i] == 0.0:
        _PB[_i + 1] = _PB[_i] * np.exp(-G0 * M_AIR * _dh / (R_UNIV * _TB[_i]))
    else:
        _PB[_i + 1] = _PB[_i] * (_TB[_i] / _TB[_i + 1]) ** (G0 * M_AIR / (R_UNIV * _LB[_i]))

R_AIR = R_UNIV / M_AIR
GAMMA_AIR = 1.4


def geopotential(z_geometric):
    return R_EARTH * z_geometric / (R_EARTH + z_geometric)


def us1976(z_m=None, z_ft=None):
    """Return dict with T [K], p [Pa], rho [kg/m3], a [m/s], mu [Pa s], nu [m2/s]."""
    if z_m is None:
        z_m = z_ft * FT_TO_M
    h = geopotential(z_m)
    if h > _HB[-1]:
        raise ValueError("US1976 implementation limited to 86 km geometric")
    i = int(np.searchsorted(_HB, h, side="right") - 1)
    i = min(max(i, 0), 6)
    dh = h - _HB[i]
    T = _TB[i] + _LB[i] * dh
    if _LB[i] == 0.0:
        p = _PB[i] * np.exp(-G0 * M_AIR * dh / (R_UNIV * _TB[i]))
    else:
        p = _PB[i] * (_TB[i] / T) ** (G0 * M_AIR / (R_UNIV * _LB[i]))
    rho = p / (R_AIR * T)
    a = np.sqrt(GAMMA_AIR * R_AIR * T)
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    return dict(z=z_m, z_ft=z_m / FT_TO_M, T=T, p=p, rho=rho, a=a, mu=mu, nu=mu / rho)


def freestream(mach, z_ft=None, z_m=None, Y=None):
    """Freestream state for a given Mach and altitude.

    Y: dict of species mass fractions (default: dissociation-free air, 76.7/23.3)."""
    at = us1976(z_m=z_m, z_ft=z_ft)
    U = mach * at["a"]
    out = dict(at)
    out.update(mach=mach, U=U, Re_unit=at["rho"] * U / at["mu"],
               T0=at["T"] * (1.0 + 0.2 * mach ** 2),
               p0=at["p"] * (1.0 + 0.2 * mach ** 2) ** 3.5)
    out["Y"] = {"N2": 0.767, "O2": 0.233} if Y is None else dict(Y)
    return out
