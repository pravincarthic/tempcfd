"""Mixture transport: Blottner viscosity, Eucken conductivity, Wilke mixing,
and multicomponent diffusion via a constant-Lewis / constant-Schmidt closure.

Analytic (complex-step safe).
"""
import numpy as np
from . import species as sp
from . import thermo as th

NS = sp.NS
_A, _B, _C = sp.BLOTTNER[:, 0], sp.BLOTTNER[:, 1], sp.BLOTTNER[:, 2]


def mu_species(T, Tv):
    """Blottner species viscosities [Pa s]; electrons evaluated at Tv."""
    out = []
    for s in range(NS):
        Ts = Tv if sp.IS_ELECTRON[s] else T
        lnT = np.log(Ts)
        out.append(0.1 * np.exp((_A[s] * lnT + _B[s]) * lnT + _C[s]))
    return out


def k_species(T, Tv, mus):
    """Eucken conductivities, split into tr and ve parts [W/(m K)]."""
    ktr, kve = [], []
    for s in range(NS):
        R = sp.RS[s]
        if sp.IS_ELECTRON[s]:
            ktr.append(np.zeros_like(T))
            kve.append(mus[s] * 2.5 * 1.5 * R)
        elif sp.IS_MOLECULE[s]:
            ktr.append(mus[s] * (2.5 * 1.5 * R + 1.0 * R))       # trans + rot
            cvve = _cv_ve_s(Tv, s)
            kve.append(mus[s] * cvve)
        else:
            ktr.append(mus[s] * 2.5 * 1.5 * R)
            kve.append(mus[s] * _cv_ve_s(Tv, s))
    return ktr, kve


def _cv_ve_s(Tv, s):
    """d e_ve,s / d Tv, analytic."""
    c = 0.0
    thv = sp.THETA_V[s]
    if thv > 0.0:
        x = np.exp(thv / Tv)
        c = c + sp.RS[s] * (thv / Tv) ** 2 * x / (x - 1.0) ** 2
    g, te = sp.G_EL[s], sp.T_EL[s]
    if g.size > 1:
        q = 0.0
        q1 = 0.0
        q2 = 0.0
        for gi, ti in zip(g, te):
            e = gi * np.exp(-ti / Tv)
            q = q + e
            q1 = q1 + e * ti
            q2 = q2 + e * ti * ti
        c = c + sp.RS[s] * (q2 / q - (q1 / q) ** 2) / Tv ** 2
    return c


def _wilke_phi(mus, X, s, r):
    ms, mr = sp.MW[s], sp.MW[r]
    num = (1.0 + np.sqrt(mus[s] / mus[r]) * (mr / ms) ** 0.25) ** 2
    den = np.sqrt(8.0 * (1.0 + ms / mr))
    return num / den


def _sutherland(T):
    return 1.458e-6 * T ** 1.5 / (T + 110.4)


def mixture(rho_s, T, Tv, Le=1.2, Sc=None, low_T_blend=True):
    """Return (mu, k_tr, k_ve, rhoD) with rhoD an (ns,) list [kg/(m s)].

    low_T_blend smoothly reverts to Sutherland air below ~800 K where the
    Blottner fits (calibrated for T > 1000 K) are ~10% high and where the
    mixture is undissociated anyway.  The blend is analytic.
    """
    Msum = 0.0
    for s in range(NS):
        Msum = Msum + rho_s[s] / sp.MW[s]
    X = [(rho_s[s] / sp.MW[s]) / Msum for s in range(NS)]

    mus = mu_species(T, Tv)
    ktr_s, kve_s = k_species(T, Tv, mus)

    phi = [0.0] * NS
    for s in range(NS):
        acc = 0.0
        for r in range(NS):
            acc = acc + X[r] * _wilke_phi(mus, X, s, r)
        phi[s] = acc

    mu = 0.0
    ktr = 0.0
    kve = 0.0
    for s in range(NS):
        w = X[s] / phi[s]
        mu = mu + w * mus[s]
        ktr = ktr + w * ktr_s[s]
        kve = kve + w * kve_s[s]

    if low_T_blend:
        wgt = 0.5 * (1.0 + np.tanh((T - 800.0) / 200.0))
        mu_s = _sutherland(T)
        scale = wgt + (1.0 - wgt) * mu_s / mu
        mu = mu * scale
        ktr = ktr * scale
        kve = kve * scale

    cp = th.cp_mix(rho_s, T, Tv)
    if Sc is None:
        rhoD_scalar = Le * (ktr + kve) / cp
    else:
        rhoD_scalar = mu / Sc
    rhoD = [rhoD_scalar for _ in range(NS)]
    # electrons diffuse ambipolarly with the ions; the simple closure below
    # keeps the mixture mass-consistent and is adequate for the trace
    # ionisation levels reached at these conditions.
    return mu, ktr, kve, rhoD


def diffusion_fluxes(rho_s, gradY, rhoD):
    """Fickian fluxes with the mass-conservation correction.

    gradY : (ns,) list of dY_s/dx_i for one direction.
    Returns list of J_s (same direction).
    """
    rho = th.density(rho_s)
    Y = [rho_s[s] / rho for s in range(NS)]
    Jraw = [-rhoD[s] * gradY[s] for s in range(NS)]
    Jc = 0.0
    for s in range(NS):
        Jc = Jc + Jraw[s]
    return [Jraw[s] - Y[s] * Jc for s in range(NS)]
