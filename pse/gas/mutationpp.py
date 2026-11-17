"""Mutation++ (FLUID_MODEL= MUTATIONPP) two-temperature model for AIR-7.

SU2's CMutationTCLib hands everything to a Mutation::Mixture built with the
ChemNonEqTTv state model.  The pieces that enter the stability operator are
re-implemented here, analytically, from the same Mutation++ data:

* RRHO thermodynamics (RrhoDB): translation at T (electrons at Te = Tv),
  rigid-rotor rotation at T, harmonic oscillator vibration and electronic
  levels at Tv, formation enthalpies at 298.15 K re-referenced to 0 K with
  the sensible enthalpy at 298.15 K, exactly as RrhoDB::hF does;
* kinetics from the mixture's mechanism file: Arrhenius forward rates,
  backward rates from the RRHO Gibbs energies at the backward-rate
  temperature (RateManager), Mutation++'s rate-controlling temperatures
  (DISSOCIATION_M: sqrt(T Tv) forward / T backward, DISSOCIATION_E: Te both,
  ASSOCIATIVE_IONIZATION: T forward / Te backward, EXCHANGE: T), third-body
  efficiencies with electrons excluded (ThirdbodyManager);
* energy transfer (ChemNonEqTTvStateModel): OmegaVT (Millikan-White with the
  VT.xml constants, X-weighted arithmetic mean of the partner times,
  Park's correction with the vibrator's own number density), OmegaCV,
  OmegaCElec and OmegaCE (non-preferential).  OmegaET and OmegaI need the
  electron-heavy collision integrals; they are omitted (they scale with the
  electron number density, zero to round-off at the shipped condition, and
  the air_7 mechanism has no electron-impact ionisation).

Transport (viscosity, frozen conductivity vector, mixture-averaged
diffusion coefficients) comes from the Mutation++ Python module when it is
importable: at each station the library is evaluated at the base state and
its first derivatives with respect to (rho_s, T, Tv) are taken by central
differences, and the properties are represented by that local first-order
expansion.  The linear stability operator only needs first derivatives, so
this is exact for LST/LPSE up to the finite-difference error.  Without the
module the SU2TCLib formulas of the same TRANSPORT_COEFF_MODEL are used and
a warning says so.
"""
import warnings
import numpy as np

from .base import GasModel
from . import mpp_data as MD
from ..constants import MPP_RU, MPP_KB, MPP_NA, MPP_HP, MPP_ONEATM

_TWOPI = 2.0 * np.pi


class MutationppAir7(GasModel):
    library = "mutationpp"

    def __init__(self, mixture="air_7", transport="WILKE", frozen=False,
                 data_dir=None, transport_fallback="su2tclib",
                 fd_rel=1.0e-6):
        self.gas_model = mixture
        mix_path = MD.mixture_path(mixture, data_dir)
        names, mech = MD.read_mixture(mix_path)
        if "e-" in names and names[0] != "e-":
            names = ["e-"] + [n for n in names if n != "e-"]   # Mutation++ rule
        self.names = names
        self.RU = MPP_RU
        self.MW = np.array([MD.species_mw(n) for n in names])
        self.is_electron = [n == "e-" for n in names]
        self.charge = np.array([-1.0 if n == "e-" else
                                (1.0 if n.endswith("+") else 0.0)
                                for n in names])
        super().__init__()
        self.frozen = bool(frozen)
        self.transport_model = transport.upper()
        rrho, ab, om, src = MD.load(names, data_dir)
        self.data_source = src
        self._setup_rrho(rrho)
        self._setup_vt(ab, om)
        self.mechanism = mech
        self._setup_kinetics(MD.read_mechanism(MD.mechanism_path(mech, data_dir),
                                               names))
        self._mpp = None
        self._fallback = None
        self._taylor = None
        self.fd_rel = fd_rel
        try:
            import mutationpp as mpp                      # noqa: F401
            self._mpp = _MppTransport(mpp, mixture, self.transport_model,
                                      names)
        except Exception as exc:                          # pragma: no cover
            if transport_fallback != "su2tclib":
                raise RuntimeError("Mutation++ Python module unavailable (%s) "
                                   "and no transport fallback" % exc)
            from .su2tclib import SU2TCLibAir7
            if sorted(names) != sorted(SU2TCLibAir7.names):
                raise RuntimeError("transport fallback needs the AIR-7 set")
            fb = SU2TCLibAir7(transport=("WILKE" if self.transport_model
                                         in ("CHAPMANN-ENSKOG",)
                                         else self.transport_model))
            self._fallback = (fb, [fb.idx[n] for n in names])
            warnings.warn("Mutation++ Python module not importable (%s): "
                          "using the SU2TCLib %s transport formulas"
                          % (exc, fb.transport_model), RuntimeWarning)

    # ------------------------------------------------------------- RRHO
    def _setup_rrho(self, rrho):
        ns = self.ns
        self.linear = np.zeros(ns)
        self.ln_omega_t = np.zeros(ns)
        self.theta_v_list = []
        self.el = []
        self.hform = np.zeros(ns)
        for s, n in enumerate(self.names):
            d = rrho[n]
            is_mol = len(d["theta_v"]) > 0
            self.linear[s] = (1.0 if d["linear"] else 1.5) if is_mol else 0.0
            if is_mol:
                lin = 2 if d["linear"] else 3
                self.ln_omega_t[s] = np.log(d["theta_r"]) + 2.0 / lin * \
                    np.log(d["sigma"])
            self.theta_v_list.append(np.array(d["theta_v"], float))
            g = np.array([l[0] for l in d["levels"]], float)
            t = np.array([l[1] for l in d["levels"]], float) * MD.CM1_TO_K
            self.el.append((g, t))
            self.hform[s] = d["hf"] / MPP_RU                  # K
        self.is_molecule = self.linear > 0.0
        self.cvtr = np.array([0.0 if self.is_electron[s] else
                              (1.5 + self.linear[s]) * self.Rs[s]
                              for s in range(ns)])
        Tref = 298.15
        sst = self._h_sensible_K(Tref, Tref)
        self.hF = np.array([self.hform[s] - sst[s] for s in range(ns)])  # K
        self.ef = self.hF * self.Rs
        self.hf = self.ef
        qt = 2.5 * np.log(MPP_KB) + 1.5 * np.log(_TWOPI / (MPP_NA * MPP_HP ** 2))
        self.lnqtmw = qt + 1.5 * np.log(self.MW)

    def _hV_K(self, s, T):
        tv = self.theta_v_list[s]
        out = 0.0 * T
        for th in tv:
            out = out + th / (np.exp(th / T) - 1.0)
        return out

    def _el_facs(self, s, T):
        g, t = self.el[s]
        q0 = 0.0 * T
        q1 = 0.0 * T
        q2 = 0.0 * T
        for gi, ti in zip(g, t):
            e = gi * np.exp(-ti / T)
            q0 = q0 + e
            q1 = q1 + e * ti
            q2 = q2 + e * ti * ti
        return q0, q1, q2

    def _hE_K(self, s, T):
        if self.is_electron[s] or len(self.el[s][0]) == 0:
            return 0.0 * T
        q0, q1, _ = self._el_facs(s, T)
        return q1 / q0

    def _h_sensible_K(self, T, Tv):
        out = []
        for s in range(self.ns):
            if self.is_electron[s]:
                out.append(2.5 * Tv)
                continue
            out.append(2.5 * T + self.linear[s] * T + self._hV_K(s, Tv)
                       + self._hE_K(s, Tv))
        return out

    def e_tr_species(self, T):
        return [self.cvtr[s] * T for s in range(self.ns)]

    def e_vib_species(self, Tv):
        return [self.Rs[s] * self._hV_K(s, Tv) for s in range(self.ns)]

    def e_el_species(self, Tv):
        return [self.Rs[s] * self._hE_K(s, Tv) for s in range(self.ns)]

    def e_ve_species(self, Tv):
        out = []
        for s in range(self.ns):
            if self.is_electron[s]:
                out.append(1.5 * self.Rs[s] * Tv)
            else:
                out.append(self.Rs[s] * (self._hV_K(s, Tv) + self._hE_K(s, Tv)))
        return out

    def cv_ve_species(self, Tv):
        out = []
        for s in range(self.ns):
            if self.is_electron[s]:
                out.append(1.5 * self.Rs[s] + 0.0 * Tv)
                continue
            c = 0.0 * Tv
            for th in self.theta_v_list[s]:
                x = np.exp(th / Tv)
                c = c + (th / Tv) ** 2 * x / (x - 1.0) ** 2
            if len(self.el[s][0]) > 1:
                q0, q1, q2 = self._el_facs(s, Tv)
                c = c + (q2 * q0 - q1 * q1) / (Tv * Tv * q0 * q0)
            out.append(self.Rs[s] * c)
        return out

    def h_species(self, T, Tv):
        """getEnthalpiesMass in TTv: hT(T | Te) + hR(T) + hV(Tv) + hE(Tv) + hF."""
        hs = self._h_sensible_K(T, Tv)
        return [self.Rs[s] * (hs[s] + self.hF[s]) for s in range(self.ns)]

    def g_over_RT(self, T):
        """Standard-state (1 atm) Gibbs energy over RT, all modes at T."""
        P = MPP_ONEATM
        out = []
        hs = self._h_sensible_K(T, T)
        for s in range(self.ns):
            h = (hs[s] + self.hF[s]) / T
            st = 2.5 * (1.0 + np.log(T)) - np.log(P) + self.lnqtmw[s]
            if self.is_molecule[s]:
                sr = self.linear[s] * (1.0 + np.log(T) - self.ln_omega_t[s])
                sv = 0.0 * T
                for th in self.theta_v_list[s]:
                    f = np.exp(th / T)
                    sv = sv + th / (f - 1.0) / T - np.log(1.0 - 1.0 / f)
            else:
                sr = 0.0 * T
                sv = 0.0 * T
            se = 0.0 * T
            if not self.is_electron[s] and len(self.el[s][0]) > 0:
                q0, q1, _ = self._el_facs(s, T)
                se = q1 / (q0 * T) + np.log(q0)
            out.append(h - (st + sr + sv + se))
        return out

    # ------------------------------------------------------------ kinetics
    def _setup_kinetics(self, rxns):
        self.rxns = []
        for r in rxns:
            R = {self.idx[k]: v for k, v in r["reac"].items()}
            P = {self.idx[k]: v for k, v in r["prod"].items()}
            e = self.i_e
            eR = R.get(e, 0) if e is not None else 0
            eP = P.get(e, 0) if e is not None else 0
            if r["thirdbody"]:
                kind = "DISSOCIATION_M"
            elif eR and eP and eP > eR:
                kind = "IONIZATION_E"
            elif eR and eP:
                kind = "DISSOCIATION_E"
            elif eP and not eR:
                kind = "ASSOCIATIVE_IONIZATION"
            else:
                kind = "EXCHANGE"
            eff = np.ones(self.ns)
            if e is not None:
                eff[e] = 0.0
            for k, v in r["eff"].items():
                eff[self.idx[k]] = v
            self.rxns.append(dict(R=R, P=P, A=r["A"], n=r["n"],
                                  theta=r["theta"], tb=r["thirdbody"],
                                  eff=eff, kind=kind, formula=r["formula"]))
        self.nr = len(self.rxns)

    def _rate_T(self, kind, T, Tv):
        if kind == "DISSOCIATION_M":
            return np.sqrt(T * Tv), T
        if kind == "DISSOCIATION_E":
            return Tv, Tv
        if kind == "ASSOCIATIVE_IONIZATION":
            return T, Tv
        if kind == "IONIZATION_E":
            return Tv, T
        return T, T

    def production_rates(self, rho_s, T, Tv):
        zero = 0.0 * T
        if self.frozen:
            return [zero] * self.ns
        c = [rho_s[s] / self.MW[s] for s in range(self.ns)]   # mol/m^3
        w = [zero] * self.ns
        gcache = {}
        for r in self.rxns:
            Tf, Tb = self._rate_T(r["kind"], T, Tv)
            kf = r["A"] * Tf ** r["n"] * np.exp(-r["theta"] / Tf)
            kfb = r["A"] * Tb ** r["n"] * np.exp(-r["theta"] / Tb)
            key = id(Tb)
            if key not in gcache:
                gcache[key] = self.g_over_RT(Tb)
            g = gcache[key]
            val = np.log(MPP_ONEATM / (MPP_RU * Tb))
            lnK = 0.0 * T
            for s, nu in r["R"].items():
                lnK = lnK + nu * (g[s] - val)
            for s, nu in r["P"].items():
                lnK = lnK - nu * (g[s] - val)
            kb = kfb / np.exp(lnK)
            f = kf
            for s, nu in r["R"].items():
                f = f * c[s] ** nu
            b = kb
            for s, nu in r["P"].items():
                b = b * c[s] ** nu
            rate = f - b
            if r["tb"]:
                m = 0.0
                for s in range(self.ns):
                    if r["eff"][s] != 0.0:
                        m = m + r["eff"][s] * c[s]
                rate = rate * m
            for s, nu in r["P"].items():
                w[s] = w[s] + nu * self.MW[s] * rate
            for s, nu in r["R"].items():
                w[s] = w[s] - nu * self.MW[s] * rate
        return w

    # ------------------------------------------------------ energy transfer
    def _setup_vt(self, ab, om):
        self.vib = []
        heavy = [s for s in range(self.ns) if not self.is_electron[s]]
        for s in range(self.ns):
            if not self.is_molecule[s]:
                continue
            n = self.names[s]
            th = self.theta_v_list[s][0]
            a = np.zeros(len(heavy))
            b = np.zeros(len(heavy))
            for k, r in enumerate(heavy):
                mu = 1000.0 * self.MW[s] * self.MW[r] / (self.MW[s] + self.MW[r])
                a[k] = 1.16e-3 * np.sqrt(mu) * th ** (4.0 / 3.0)
                b[k] = 0.015 * mu ** 0.25
                pr = ab.get(n, {}).get(self.names[r])
                if pr is not None:
                    a[k], b[k] = pr
            self.vib.append(dict(s=s, a=a, b=b, omegav=om.get(n, 1.0e-20)))
        self._heavy = heavy

    def source_ve(self, rho_s, T, Tv, omega=None):
        X, conc = self.molar_fractions(rho_s)
        n_tot = conc * MPP_NA
        p_atm = self.pressure(rho_s, T, Tv) / MPP_ONEATM
        Tfac = T ** (-1.0 / 3.0)
        Xh = 0.0
        for r in self._heavy:
            Xh = Xh + X[r]
        Q = 0.0 * T
        for v in self.vib:
            s = v["s"]
            acc = 0.0 * T
            for k, r in enumerate(self._heavy):
                acc = acc + X[r] * np.exp(v["a"][k] * (Tfac - v["b"][k]) - 18.42)
            tau_mw = acc / (Xh * p_atm)
            ci = np.sqrt(8.0 * MPP_RU * T / (np.pi * self.MW[s]))
            sig = v["omegav"] * 2.5e9 / (T * T)
            tau_p = 1.0 / (n_tot * X[s] * ci * sig)
            de = self.Rs[s] * (self._hV_K(s, T) - self._hV_K(s, Tv))
            Q = Q + rho_s[s] * de / (tau_mw + tau_p)
        if not self.frozen:
            if omega is None:
                omega = self.production_rates(rho_s, T, Tv)
            eve = self.e_ve_species(Tv)
            for s in range(self.ns):
                Q = Q + omega[s] * eve[s]
        return Q

    # ------------------------------------------------------------ transport
    def anchor(self, rho_s, T, Tv):
        if self._mpp is None:
            return None
        self._taylor = self._mpp.expand(np.real(rho_s), np.real(T),
                                        np.real(Tv), self.fd_rel)
        return self._taylor

    def transport(self, rho_s, T, Tv):
        if self._fallback is not None:
            fb, perm = self._fallback
            rs = [None] * fb.ns
            for k, j in enumerate(perm):
                rs[j] = rho_s[k]
            mu, ktr, kve, Dfb = fb.transport(rs, T, Tv)
            return mu, ktr, kve, [Dfb[j] for j in perm]
        if self._taylor is None:
            raise RuntimeError("Mutation++ transport used before anchor()")
        return self._taylor.evaluate(rho_s, T, Tv)


class _TaylorTransport:
    """First-order expansion of (mu, ktr, kve, D_s) about base nodes."""

    def __init__(self, base, f0, df):
        self.base = base                        # (ns+2, n)
        self.f0 = f0                            # (3+ns, n)
        self.df = df                            # (3+ns, ns+2, n)

    def evaluate(self, rho_s, T, Tv):
        ns = self.base.shape[0] - 2
        n = self.base.shape[1]
        nd = np.ndim(T)
        shp = (n,) + (1,) * max(nd - 1, 0)
        x = [rho_s[s] for s in range(ns)] + [T, Tv]
        out = []
        for k in range(self.f0.shape[0]):
            val = self.f0[k].reshape(shp) + 0.0 * T
            for j in range(ns + 2):
                d = self.df[k, j].reshape(shp)
                val = val + d * (x[j] - self.base[j].reshape(shp))
            out.append(val)
        return out[0], out[1], out[2], out[3:]


class _MppTransport:
    """Thin wrapper over the Mutation++ Python Mixture."""

    _ALG = {"WILKE": "Wilke", "GUPTA-YOS": "Gupta-Yos",
            "CHAPMANN-ENSKOG": "Chapmann-Enskog_LDLT"}

    def __init__(self, mpp, mixture, model, names):
        opt = mpp.MixtureOptions(mixture)
        opt.setStateModel("ChemNonEqTTv")
        alg = self._ALG.get(model, "Wilke")
        opt.setViscosityAlgorithm(alg)
        opt.setThermalConductivityAlgorithm(alg)
        self.mix = mpp.Mixture(opt)
        got = [self.mix.speciesName(i) for i in range(self.mix.nSpecies())]
        if got != list(names):
            raise RuntimeError("Mutation++ species order %s differs from %s"
                               % (got, names))
        self.ns = len(names)

    def _eval(self, rs, T, Tv):
        m = self.mix
        m.setState(np.asarray(rs, float), np.array([T, Tv], float), 1)
        lam = np.asarray(m.frozenThermalConductivityVector(), float)
        D = np.asarray(m.averageDiffusionCoeffs(), float)
        return np.concatenate([[m.viscosity(), lam[0], lam[1]], D])

    def expand(self, rho_s, T, Tv, rel):
        ns = self.ns
        n = np.size(T)
        rho_s = np.asarray(rho_s, float).reshape(ns, n)
        T = np.asarray(T, float).reshape(n)
        Tv = np.asarray(Tv, float).reshape(n)
        base = np.vstack([rho_s, T, Tv])
        f0 = np.zeros((3 + ns, n))
        df = np.zeros((3 + ns, ns + 2, n))
        for i in range(n):
            x = base[:, i]
            f0[:, i] = self._eval(x[:ns], x[ns], x[ns + 1])
            for j in range(ns + 2):
                h = rel * max(abs(x[j]), 1e-30 if j < ns else 1.0)
                xp = x.copy()
                xm = x.copy()
                xp[j] += h
                xm[j] = max(xm[j] - h, 0.0) if j < ns else xm[j] - h
                hh = xp[j] - xm[j]
                fp = self._eval(xp[:ns], xp[ns], xp[ns + 1])
                fm = self._eval(xm[:ns], xm[ns], xm[ns + 1])
                df[:, j, i] = (fp - fm) / hh
        return _TaylorTransport(base, f0, df)
