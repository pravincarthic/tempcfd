"""Park (1993) 11-species air kinetics, two-temperature.

Base reaction set: 3 dissociation reactions expanded over all 11 third bodies
(33 elementary steps), 2 Zeldovich exchange, 3 associative-ionisation,
11 charge-exchange and 2 electron-impact ionisation reactions -> 51
elementary reactions.  Forward rates are Arrhenius in a rate-controlling
temperature Ta (Park's T^q Tv^(1-q) for heavy-impact dissociation, Tv=Te for
electron-driven steps); backward rates come from the equilibrium constant
evaluated from the same partition functions used by thermo.py, so the model
is thermodynamically consistent by construction.

Everything is analytic -> complex-step differentiable.
"""
import numpy as np
from .constants import R_UNIV, N_AVO, EV_TO_J
from . import species as sp
from . import thermo as th

NS = sp.NS
I = sp.IDX
CM3_TO_M3 = 1.0e-6            # cm^3/(mol s) -> m^3/(mol s)

MOLECULES = ["N2", "O2", "NO", "N2+", "O2+", "NO+"]
ATOMS = ["N", "O", "N+", "O+"]

# ---------------------------------------------------------------- reactions
# each entry: (reactants dict, products dict, C [cm3/mol/s], n, theta [K], mode)
# mode: 'dissA' heavy-impact dissociation (Ta = T^0.7 Tv^0.3)
#       'diss_e' electron-impact dissociation (Ta = Tv)
#       'exch'   heavy-particle exchange / charge exchange (Ta = T)
#       'assoc'  associative ionisation (Ta = T)
#       'eimp'   electron-impact ionisation (Ta = Tv)
_REACTIONS = []


def _merge(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _add(r, p, C, n, th_, mode):
    _REACTIONS.append((r, p, C, n, th_, mode))


def _build():
    _REACTIONS.clear()
    # --- dissociation ------------------------------------------------------
    diss = [("N2", {"N": 2}, 113200.0, -1.6, 7.0e21, 3.0e22, 1.2e25),
            ("O2", {"O": 2}, 59500.0, -1.5, 2.0e21, 1.0e22, None),
            ("NO", {"N": 1, "O": 1}, 75500.0, 0.0, 5.0e15, 1.1e17, None)]
    for name, prod, theta, n, Cmol, Catm, Cele in diss:
        for m in sp.NAMES:
            if m == "e-":
                if Cele is None:
                    continue
                _add(_merge({name: 1}, {m: 1}), _merge(prod, {m: 1}), Cele, n, theta, "diss_e")
            else:
                C = Cmol if m in MOLECULES else Catm
                _add(_merge({name: 1}, {m: 1}), _merge(prod, {m: 1}), C, n, theta, "dissA")
    # --- Zeldovich exchange -----------------------------------------------
    _add({"N2": 1, "O": 1}, {"NO": 1, "N": 1}, 6.4e17, -1.0, 38400.0, "exch")
    _add({"NO": 1, "O": 1}, {"O2": 1, "N": 1}, 8.4e12, 0.0, 19450.0, "exch")
    # --- associative ionisation -------------------------------------------
    _add({"N": 1, "O": 1}, {"NO+": 1, "e-": 1}, 8.8e8, 1.0, 31900.0, "assoc")
    _add({"O": 2}, {"O2+": 1, "e-": 1}, 7.1e2, 2.7, 80600.0, "assoc")
    _add({"N": 2}, {"N2+": 1, "e-": 1}, 4.4e7, 1.5, 67500.0, "assoc")
    # --- charge exchange ---------------------------------------------------
    ce = [({"NO+": 1, "O": 1}, {"N+": 1, "O2": 1}, 1.0e12, 0.5, 77200.0),
          ({"N+": 1, "N2": 1}, {"N2+": 1, "N": 1}, 1.0e12, 0.5, 12200.0),
          ({"O2+": 1, "N": 1}, {"N+": 1, "O2": 1}, 8.7e13, 0.14, 28600.0),
          ({"O+": 1, "NO": 1}, {"N+": 1, "O2": 1}, 1.4e5, 1.90, 26600.0),
          ({"O2+": 1, "N2": 1}, {"N2+": 1, "O2": 1}, 9.9e12, 0.0, 40700.0),
          ({"O2+": 1, "O": 1}, {"O+": 1, "O2": 1}, 4.0e12, -0.09, 18000.0),
          ({"NO+": 1, "N": 1}, {"O+": 1, "N2": 1}, 3.4e13, -1.08, 12800.0),
          ({"NO+": 1, "O2": 1}, {"O2+": 1, "NO": 1}, 2.4e13, 0.41, 32600.0),
          ({"NO+": 1, "O": 1}, {"O2+": 1, "N": 1}, 7.2e12, 0.29, 48600.0),
          ({"O+": 1, "N2": 1}, {"N2+": 1, "O": 1}, 9.1e11, 0.36, 22800.0),
          ({"NO+": 1, "N": 1}, {"N2+": 1, "O": 1}, 7.2e13, 0.0, 35500.0)]
    for r, p, C, n, t in ce:
        _add(r, p, C, n, t, "exch")
    # --- electron-impact ionisation ---------------------------------------
    _add({"O": 1, "e-": 1}, {"O+": 1, "e-": 2}, 3.9e33, -3.78, 158500.0, "eimp")
    _add({"N": 1, "e-": 1}, {"N+": 1, "e-": 2}, 2.5e34, -3.82, 168600.0, "eimp")


_build()
NR = len(_REACTIONS)

# pre-compiled stoichiometry
NU_R = np.zeros((NR, NS))
NU_P = np.zeros((NR, NS))
for _j, (_r, _p, *_rest) in enumerate(_REACTIONS):
    for k, v in _r.items():
        NU_R[_j, I[k]] += v
    for k, v in _p.items():
        NU_P[_j, I[k]] += v
NU = NU_P - NU_R
ARR_C = np.array([r[2] for r in _REACTIONS]) * CM3_TO_M3 ** 0  # scaled below
ARR_N = np.array([r[3] for r in _REACTIONS])
ARR_TH = np.array([r[4] for r in _REACTIONS])
MODE = [r[5] for r in _REACTIONS]
# convert C from cm^3^(order-1)/(mol^(order-1) s) to SI
_ORDER = NU_R.sum(axis=1)
ARR_C = np.array([r[2] for r in _REACTIONS]) * (1.0e-6) ** (_ORDER - 1.0)
DNU = NU.sum(axis=1)

_MASK_DISSA = np.array([m == "dissA" for m in MODE])
_MASK_E = np.array([m in ("diss_e", "eimp") for m in MODE])
PARK_Q = 0.7


# sparse stoichiometry: per reaction, index lists with multiplicity expanded
_RIDX = [np.repeat(np.arange(NS), NU_R[j].astype(int)).tolist() for j in range(NR)]
_PIDX = [np.repeat(np.arange(NS), NU_P[j].astype(int)).tolist() for j in range(NR)]
_NZ = [np.nonzero(NU[j])[0].tolist() for j in range(NR)]
_SPEC_RXN = [[j for j in range(NR) if NU[j, s] != 0.0] for s in range(NS)]

_TKEY = np.array([0 if _MASK_DISSA[j] else (1 if _MASK_E[j] else 2) for j in range(NR)])


def rate_temperatures(T, Tv):
    """The three Park rate-controlling temperatures used by the model."""
    return (T ** PARK_Q * Tv ** (1.0 - PARK_Q), Tv, T)


def source(rho_s, T, Tv):
    """Mass production rates omega_s [kg/(m^3 s)] and net reaction rates.

    Returns (omega list of ns, rates list of NR)."""
    conc = [rho_s[s] / sp.MW[s] for s in range(NS)]
    Tt = rate_temperatures(T, Tv)
    lnZ = [th.gibbs_partition(t) for t in Tt]

    rates = [None] * NR
    for j in range(NR):
        k = _TKEY[j]
        Taj = Tt[k]
        kf = ARR_C[j] * Taj ** ARR_N[j] * np.exp(-ARR_TH[j] / Taj)
        lnkc = -DNU[j] * np.log(N_AVO)
        Z = lnZ[k]
        for s in _NZ[j]:
            lnkc = lnkc + NU[j, s] * Z[s]
        fwd = kf
        for s in _RIDX[j]:
            fwd = fwd * conc[s]
        bwd = kf * np.exp(-lnkc)
        for s in _PIDX[j]:
            bwd = bwd * conc[s]
        rates[j] = fwd - bwd

    omega = []
    for s in range(NS):
        acc = 0.0
        for j in _SPEC_RXN[s]:
            acc = acc + NU[j, s] * rates[j]
        omega.append(acc * sp.MW[s])
    return omega, rates


# ------------------------------------------------------------ VT relaxation
_MW_G = sp.MW * 1000.0            # g/mol


def _tau_MW(T, p, s, r):
    mu_sr = _MW_G[s] * _MW_G[r] / (_MW_G[s] + _MW_G[r])
    A = 1.16e-3 * np.sqrt(mu_sr) * sp.THETA_V[s] ** (4.0 / 3.0)
    B = 0.015 * mu_sr ** 0.25
    return np.exp(A * (T ** (-1.0 / 3.0) - B) - 18.42) * 101325.0 / p


def source_ve(rho_s, T, Tv, omega, rates):
    """Vibrational-electronic energy source [W/m^3].

    Q = Q_{T-V} (Millikan-White + Park limiting cross section)
      + Q_{C-V} (non-preferential chemistry coupling)
      + Q_{e-i} (electron-impact ionisation energy sink).
    """
    p = th.pressure(rho_s, T, Tv)
    Msum = 0.0
    for s in range(NS):
        Msum = Msum + rho_s[s] / sp.MW[s]
    X = [(rho_s[s] / sp.MW[s]) / Msum for s in range(NS)]

    Q = 0.0
    for s in range(NS):
        if sp.THETA_V[s] <= 0.0:
            continue
        inv = 0.0
        for r in range(NS):
            if sp.IS_ELECTRON[r]:
                continue
            inv = inv + X[r] / _tau_MW(T, p, s, r)
        Xh = 0.0
        for r in range(NS):
            if not sp.IS_ELECTRON[r]:
                Xh = Xh + X[r]
        tau_mw = Xh / inv
        # Park high-temperature limiting relaxation time
        c_s = np.sqrt(8.0 * R_UNIV * T / (np.pi * sp.MW[s]))
        n_tot = p / (1.380649e-23 * T)
        sigma = sp.PARK_SIGMA_V * (sp.PARK_SIGMA_TREF / T) ** 2
        tau_p = 1.0 / (sigma * n_tot * c_s)
        tau = tau_mw + tau_p
        Q = Q + rho_s[s] * (th.e_vib_s(T, s) - th.e_vib_s(Tv, s)) / tau

    # chemistry -> vibrational-electronic coupling (non-preferential)
    for s in range(NS):
        Q = Q + omega[s] * th.e_ve_s(Tv, s)

    # electron-impact ionisation removes electron energy
    for j in range(NR):
        if MODE[j] != "eimp":
            continue
        for nm, ev in sp.ION_ENERGY_EV.items():
            if NU_R[j, I[nm]] > 0.0:
                Q = Q - rates[j] * ev * EV_TO_J * N_AVO
    return Q


def summary():
    return dict(n_species=NS, n_reactions=NR,
                n_dissociation=int(_MASK_DISSA.sum() + sum(m == "diss_e" for m in MODE)),
                n_exchange=sum(m == "exch" for m in MODE),
                n_assoc=sum(m == "assoc" for m in MODE),
                n_eimpact=sum(m == "eimp" for m in MODE))
