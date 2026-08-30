"""Two-temperature thermodynamics for the 11-species Park air model.

All routines are written to be analytic in their arguments so that exact
Jacobians can be obtained by complex-step differentiation (see jacobians.py).
Never introduce abs/max/where on the differentiated path.

State convention used everywhere:
    q = [rho_1..rho_ns, u, v, w, T, Tv]      (primitive, ns+5 entries)
T  : translational-rotational temperature
Tv : vibrational-electronic-electron temperature (= Te)
"""
import numpy as np
from .constants import R_UNIV, K_BOLTZ, N_AVO, H_PLANCK
from . import species as sp

NS = sp.NS
NV = NS + 5
I_U, I_V, I_W, I_T, I_TV = NS, NS + 1, NS + 2, NS + 3, NS + 4


def _bcast(T):
    return np.asarray(T)


def e_vib_s(Tv, s):
    """Harmonic-oscillator vibrational energy of species s [J/kg]."""
    th = sp.THETA_V[s]
    if th == 0.0:
        return np.zeros_like(Tv)
    return sp.RS[s] * th / (np.exp(th / Tv) - 1.0)


def e_el_s(Tv, s):
    """Electronic excitation energy of species s [J/kg]."""
    g, th = sp.G_EL[s], sp.T_EL[s]
    if g.size == 1:
        return np.zeros_like(Tv)
    num = 0.0
    den = 0.0
    for gi, ti in zip(g, th):
        e = gi * np.exp(-ti / Tv)
        den = den + e
        num = num + e * ti
    return sp.RS[s] * num / den


def e_ve_s(Tv, s):
    """Vibrational + electronic (+ electron translational) energy [J/kg]."""
    if sp.IS_ELECTRON[s]:
        return 1.5 * sp.RS[s] * Tv
    return e_vib_s(Tv, s) + e_el_s(Tv, s)


def e_ve_all(Tv):
    """(ns, ...) array of species vibrational-electronic energies."""
    return np.stack([e_ve_s(Tv, s) for s in range(NS)], axis=0)


def e_tr_s(T, Tv, s):
    """Translational + rotational energy of species s [J/kg] (no formation)."""
    if sp.IS_ELECTRON[s]:
        return np.zeros_like(T)          # electron translation lives in e_ve
    e = 1.5 * sp.RS[s] * T
    if sp.IS_MOLECULE[s]:
        e = e + sp.RS[s] * T             # diatomic, fully excited rotation
    return e


def h_s(T, Tv, s):
    """Species static enthalpy [J/kg] (all modes + formation)."""
    if sp.IS_ELECTRON[s]:
        return 2.5 * sp.RS[s] * Tv
    return e_tr_s(T, Tv, s) + e_ve_s(Tv, s) + sp.HF0[s] + sp.RS[s] * T


def h_all(T, Tv):
    return np.stack([h_s(T, Tv, s) for s in range(NS)], axis=0)


def pressure(rho_s, T, Tv):
    p = 0.0
    for s in range(NS):
        Ts = Tv if sp.IS_ELECTRON[s] else T
        p = p + rho_s[s] * sp.RS[s] * Ts
    return p


def density(rho_s):
    r = rho_s[0]
    for s in range(1, NS):
        r = r + rho_s[s]
    return r


def mass_fractions(rho_s):
    rho = density(rho_s)
    return rho_s / rho


def energy_ve(rho_s, Tv):
    """Mixture vibrational-electronic energy per unit mass [J/kg]."""
    rho = density(rho_s)
    e = 0.0
    for s in range(NS):
        e = e + rho_s[s] * e_ve_s(Tv, s)
    return e / rho


def energy_internal(rho_s, T, Tv):
    """Mixture internal energy per unit mass [J/kg], incl. heats of formation."""
    rho = density(rho_s)
    e = 0.0
    for s in range(NS):
        e = e + rho_s[s] * (e_tr_s(T, Tv, s) + e_ve_s(Tv, s) + sp.HF0[s])
    return e / rho


def cv_tr_mix(rho_s):
    rho = density(rho_s)
    c = 0.0
    for s in range(NS):
        if sp.IS_ELECTRON[s]:
            continue
        c = c + rho_s[s] * sp.RS[s] * (2.5 if sp.IS_MOLECULE[s] else 1.5)
    return c / rho


def cv_ve_mix(rho_s, Tv, dT=1.0e-4):
    """d e_ve/d Tv via complex step (exact, analytic-safe)."""
    from .constants import CS_H
    Tc = Tv + 1j * CS_H
    return np.imag(energy_ve(rho_s.astype(complex), Tc)) / CS_H


def cp_mix(rho_s, T, Tv):
    """Frozen mixture cp (both modes lumped at T) used for transport closures."""
    rho = density(rho_s)
    c = 0.0
    for s in range(NS):
        if sp.IS_ELECTRON[s]:
            c = c + rho_s[s] * 2.5 * sp.RS[s]
        else:
            c = c + rho_s[s] * sp.RS[s] * (3.5 if sp.IS_MOLECULE[s] else 2.5)
    return c / rho


def sound_speed_frozen(rho_s, T, Tv):
    """Frozen speed of sound (chemistry and Tv frozen)."""
    rho = density(rho_s)
    p = pressure(rho_s, T, Tv)
    cv = cv_tr_mix(rho_s)
    R = p / (rho * T)
    return np.sqrt((1.0 + R / cv) * p / rho)


def gibbs_partition(T):
    """ln of the per-volume partition function of each species, including the
    formation-energy factor.  Used for equilibrium constants:
        Kc_n = prod_s exp(lnZ_s)^nu_s      [number density basis, 1/m^3]
    """
    lnZ = []
    for s in range(NS):
        m = sp.MW[s] / N_AVO
        ltr = 1.5 * np.log(2.0 * np.pi * m * K_BOLTZ * T / H_PLANCK ** 2)
        lrot = 0.0
        if sp.IS_MOLECULE[s]:
            lrot = np.log(T / (sp.SIGMA_ROT[s] * sp.THETA_R[s]))
        lvib = 0.0
        if sp.THETA_V[s] > 0.0:
            lvib = -np.log(1.0 - np.exp(-sp.THETA_V[s] / T))
        g, th = sp.G_EL[s], sp.T_EL[s]
        qel = 0.0
        for gi, ti in zip(g, th):
            qel = qel + gi * np.exp(-ti / T)
        lel = np.log(qel)
        lform = -sp.HF0[s] * sp.MW[s] / (R_UNIV * T)
        lnZ.append(ltr + lrot + lvib + lel + lform)
    return lnZ


def T_from_energy(rho_s, e_int, e_ve_target, T0=2000.0, Tv0=2000.0, iters=60):
    """Invert (e, e_ve) -> (T, Tv).  Real-valued utility (not differentiated)."""
    rho_s = np.asarray(rho_s, dtype=float)
    T = np.full(np.shape(e_int), float(T0))
    Tv = np.full(np.shape(e_ve_target), float(Tv0))
    for _ in range(iters):
        ev = energy_ve(rho_s, Tv)
        cvv = cv_ve_mix(rho_s, Tv)
        dTv = (e_ve_target - ev) / np.maximum(cvv, 1e-3)
        Tv = np.clip(Tv + np.clip(dTv, -0.3 * Tv, 0.3 * Tv), 50.0, 90000.0)
        ei = energy_internal(rho_s, T, Tv)
        cvt = cv_tr_mix(rho_s)
        dT = (e_int - ei) / np.maximum(cvt, 1e-3)
        T = np.clip(T + np.clip(dT, -0.3 * T, 0.3 * T), 50.0, 90000.0)
    return T, Tv
