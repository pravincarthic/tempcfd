"""Common interface of the two-temperature (TTv) gas models.

State convention used everywhere in the solver:

    q = [rho_1 .. rho_ns, u, v, w, T, Tve]          (ns + 5 entries)

Species arrays are indexed on the first axis, shape (ns, ...); temperatures
have shape (...).  Every routine on the differentiated path must be analytic
in its arguments (complex-step Jacobians): no abs/max/where/clip on values
that depend on the state.  Branching on the *real part* of a quantity (table
look-ups) is allowed, because it is locally constant.
"""
import numpy as np


class GasModel:
    """Base class.  Subclasses fill the species data and the physics."""

    library = "base"
    gas_model = ""

    def __init__(self):
        self.ns = len(self.names)
        self.nv = self.ns + 5
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.i_u, self.i_v, self.i_w = self.ns, self.ns + 1, self.ns + 2
        self.i_T, self.i_Tv = self.ns + 3, self.ns + 4
        self.Rs = self.RU / self.MW                       # J/(kg K)
        self.heavy = np.array([not e for e in self.is_electron])
        self.i_e = (int(np.nonzero(self.is_electron)[0][0])
                    if np.any(self.is_electron) else None)
        self.frozen = False
        self.diffusion = True

    # ------------------------------------------------------------ helpers
    def mass_fraction_vector(self, Y):
        """dict {name: Y} or array -> normalised (ns,) array."""
        if isinstance(Y, dict):
            v = np.zeros(self.ns)
            for k, val in Y.items():
                if k not in self.idx:
                    raise KeyError("unknown species '%s' (have %s)"
                                   % (k, ", ".join(self.names)))
                v[self.idx[k]] = float(val)
        else:
            v = np.asarray(Y, dtype=float).copy()
            if v.size != self.ns:
                raise ValueError("expected %d mass fractions" % self.ns)
        v = np.maximum(v, 0.0)
        tot = v.sum()
        if tot <= 0.0:
            raise ValueError("zero mass fractions")
        return v / tot

    def density(self, rho_s):
        r = rho_s[0]
        for s in range(1, self.ns):
            r = r + rho_s[s]
        return r

    def molar_fractions(self, rho_s):
        c = [rho_s[s] / self.MW[s] for s in range(self.ns)]
        tot = c[0]
        for s in range(1, self.ns):
            tot = tot + c[s]
        return [cs / tot for cs in c], tot

    def pressure(self, rho_s, T, Tv):
        p = 0.0
        for s in range(self.ns):
            p = p + rho_s[s] * self.Rs[s] * (Tv if self.is_electron[s] else T)
        return p

    # ------------------------------------------------------- mixture energy
    def energies(self, rho_s, T, Tv):
        """(rho*e, rho*e_ve): internal energy per volume including formation,
        and vibrational-electronic(-electron) energy per volume."""
        etr = self.e_tr_species(T)
        eve = self.e_ve_species(Tv)
        rE = 0.0
        rEve = 0.0
        for s in range(self.ns):
            rE = rE + rho_s[s] * (etr[s] + eve[s] + self.ef[s])
            rEve = rEve + rho_s[s] * eve[s]
        return rE, rEve

    def cv_tr(self, rho_s):
        """rho * cv_tr of the heavy particles."""
        c = 0.0
        for s in range(self.ns):
            c = c + rho_s[s] * self.cvtr[s]
        return c

    def cv_ve(self, rho_s, Tv):
        cv = self.cv_ve_species(Tv)
        c = 0.0
        for s in range(self.ns):
            c = c + rho_s[s] * cv[s]
        return c

    def sound_speed(self, rho_s, T, Tv):
        """Frozen speed of sound, CNEMOGas::ComputeSoundSpeed."""
        rho = self.density(rho_s)
        conc = 0.0
        for s in range(self.ns):
            if not self.is_electron[s]:
                conc = conc + rho_s[s] / self.MW[s]
        rcv = self.cv_tr(rho_s)
        p = self.pressure(rho_s, T, Tv)
        return np.sqrt((1.0 + self.RU * conc / rcv) * p / rho)

    def gamma_frozen(self, rho_s):
        rR = 0.0
        for s in range(self.ns):
            rR = rR + rho_s[s] * self.Rs[s]
        return rR / self.cv_tr(rho_s) + 1.0

    def cp_frozen(self, rho_s, T, Tv):
        """Frozen cp per unit mass (tr + ve modes), for Prandtl numbers."""
        rho = self.density(rho_s)
        rR = 0.0
        for s in range(self.ns):
            rR = rR + rho_s[s] * self.Rs[s]
        return (self.cv_tr(rho_s) + self.cv_ve(rho_s, Tv) + rR) / rho

    # ------------------------------------------------------ inversions
    def temperatures(self, rho_s, rhoE_int, rhoEve, T0=None, Tv0=None,
                     iters=100):
        """Invert (rho e, rho e_ve) -> (T, Tve).  Real-valued utility."""
        rho_s = np.asarray(rho_s, dtype=float)
        rf = 0.0
        for s in range(self.ns):
            rf = rf + rho_s[s] * self.ef[s]
        T = (rhoE_int - rhoEve - rf) / self.cv_tr(rho_s)
        Tv = np.array(T if Tv0 is None else Tv0, dtype=float, copy=True)
        Tv = np.clip(Tv, 50.0, 8.0e4)
        for _ in range(iters):
            f = self.energies(rho_s, T, Tv)[1] - rhoEve
            df = self.cv_ve(rho_s, Tv)
            dT = -f / np.maximum(df, 1e-12)
            Tv = np.clip(Tv + np.clip(dT, -0.3 * Tv, 0.3 * Tv), 50.0, 8.0e4)
            if np.max(np.abs(dT) / Tv) < 1e-13:
                break
        return T, Tv

    def T_from_h_equilibrium(self, Y, h_target, T0, iters=80):
        """Thermal-equilibrium temperature for a frozen composition and a
        static enthalpy (similarity solutions)."""
        T = np.array(T0, dtype=float, copy=True)
        Yc = np.asarray(Y, dtype=float)
        for _ in range(iters):
            hs = self.h_species(T, T)
            h = 0.0
            for s in range(self.ns):
                h = h + Yc[s] * hs[s]
            Tc = T + 1j * 1e-30
            hc = 0.0
            hsc = self.h_species(Tc, Tc)
            for s in range(self.ns):
                hc = hc + Yc[s] * hsc[s]
            cp = np.imag(hc) / 1e-30
            dT = (h_target - h) / np.maximum(cp, 1.0)
            T = np.clip(T + np.clip(dT, -0.4 * T, 0.4 * T), 30.0, 6.0e4)
            if np.max(np.abs(dT)) < 1e-10:
                break
        return T

    # ---------------------------------------------------- anchoring hooks
    def anchor(self, rho_s, T, Tv):
        """Called once per station before the Jacobians are formed.  Models
        whose properties come from an external library build their local
        expansion here; analytic models ignore it."""
        return None

    def summary(self):
        return dict(library=self.library, gas_model=self.gas_model,
                    species=list(self.names), n_species=self.ns,
                    transport=getattr(self, "transport_model", ""),
                    frozen=self.frozen)
