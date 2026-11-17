"""SU2-NEMO built-in thermochemical library (FLUID_MODEL= SU2_NONEQ),
GAS_MODEL= AIR-7, reproduced from CSU2TCLib / CNEMOGas.

What is reproduced, function by function:

* energies, enthalpies, Cv             ComputeMixtureEnergies,
                                       ComputeSpeciesEve, ComputeSpeciesEnthalpy,
                                       ComputeSpeciesCvVibEle, GetSpeciesCvTraRot
* pressure, frozen sound speed         CNEMOGas::ComputePressure/ComputeSoundSpeed
* finite-rate chemistry                ComputeNetProductionRates, ComputeKeqConstants
  (22 reactions, Park Keq curve fits interpolated in mixture number density,
  smoothed rate-controlling temperatures with T_min = 800 K, eps = 80 K)
* vibrational source                   ComputeEveSourceTerm
  (Landau-Teller, Millikan-White harmonic mixture rule over all partners,
  Park limiting cross section 3e-21 (5e4/T)^2 m^2 with the total number
  density, non-preferential chemistry-vibration coupling on the
  vibrational energy only)
* transport                            WILKE   (ViscosityWBE, ThermalConductivitiesWBE,
                                                DiffusionCoeffWBE)
                                       GUPTA-YOS (ViscosityGY, ThermalConductivitiesGY,
                                                  DiffusionCoeffGY)
                                       SUTHERLAND (ViscositySuth,
                                                   ThermalConductivitiesSuth,
                                                   DiffusionCoeffWBE)

Reproduced as SU2 has it: NO+ is given the molar mass of NO, so the
associative ionisation N + O -> NO+ + e- creates one electron mass per mole
of reaction (a relative mass defect of 1.8e-5 of that reaction's rate).

Deliberate deviations (all scale with the NO+ or e- mass fraction, < 1e-10
at the shipped flight condition, and are listed so nobody chases them):

* SU2 loops over `nEl..nHeavy` when it sets the trans-rotational Cv, which
  leaves NO+ (the last species) with Cv_tr = 0 in the temperature inversion.
  Here NO+ carries its (3/2 + 1) R trans-rotational energy, as in
  ComputeMixtureEnergies.
* SU2's species-diffusion loop in CNEMONumerics::GetViscousProjFlux runs
  over indices 0..nHeavy-1, i.e. it includes e- and skips NO+.  Here every
  species diffuses (see fluxes.py), which keeps the diffusive fluxes summing
  to zero exactly and the species equations all second order.
"""
import numpy as np

from .base import GasModel
from . import su2tclib_data as DAT
from ..constants import SU2_RU, SU2_KB, SU2_NA, SU2_E_CGS, SU2_PI


def _real(x):
    return np.real(x)


class SU2TCLibAir7(GasModel):
    library = "su2tclib"
    gas_model = "AIR-7"
    names = list(DAT.NAMES)
    RU = SU2_RU

    def __init__(self, transport="WILKE", frozen=False, gy_partner_bug=True):
        self.MW = DAT.MOLAR_MASS * 1.0e-3                     # kg/mol
        self.MM = DAT.MOLAR_MASS                              # kg/kmol (SU2)
        self.charge = DAT.CHARGE.copy()
        self.is_electron = [n == "e-" for n in DAT.NAMES]
        super().__init__()
        self.transport_model = transport.upper()
        if self.transport_model not in ("WILKE", "GUPTA-YOS", "SUTHERLAND"):
            raise ValueError("TRANSPORT_COEFF_MODEL %s not available in "
                             "SU2TCLib" % transport)
        self.frozen = bool(frozen)
        self.gy_partner_bug = bool(gy_partner_bug)
        self.rot = DAT.ROTATION_MODES
        self.theta_v = DAT.CHAR_VIB_TEMP
        self.hf = DAT.ENTHALPY_FORMATION.copy()
        self.ef = self.hf - self.Rs * DAT.REF_TEMPERATURE
        self.cvtr = np.array([0.0 if self.is_electron[s] else
                              (1.5 + 0.5 * self.rot[s]) * self.Rs[s]
                              for s in range(self.ns)])
        self.is_molecule = self.rot > 0.0
        self._rx = [([self.idx[a] for a in r], [self.idx[b] for b in p])
                    for r, p in DAT.REACTIONS]
        self.nr = len(self._rx)
        nu = np.zeros((self.nr, self.ns))
        for j, (r, p) in enumerate(self._rx):
            for s in r:
                nu[j, s] -= 1.0
            for s in p:
                nu[j, s] += 1.0
        self.NU = nu

    # ------------------------------------------------------ thermodynamics
    def e_vib_species(self, Tv):
        out = []
        for s in range(self.ns):
            th = self.theta_v[s]
            if th == 0.0 or self.is_electron[s]:
                out.append(0.0 * Tv)
            else:
                out.append(self.Rs[s] * th / (np.exp(th / Tv) - 1.0))
        return out

    def e_el_species(self, Tv):
        out = []
        for s in range(self.ns):
            if self.is_electron[s]:
                out.append(0.0 * Tv)
                continue
            th, g = DAT.EL_THETA[s], DAT.EL_DEG[s]
            den = g[0] * np.exp(-th[0] / Tv)
            num = 0.0 * Tv
            for k in range(1, len(th)):
                e = g[k] * np.exp(-th[k] / Tv)
                num = num + th[k] * e
                den = den + e
            out.append(self.Rs[s] * num / den)
        return out

    def e_ve_species(self, Tv):
        ev = self.e_vib_species(Tv)
        ee = self.e_el_species(Tv)
        out = []
        for s in range(self.ns):
            if self.is_electron[s]:
                out.append(1.5 * self.Rs[s] * Tv)
            else:
                out.append(ev[s] + ee[s])
        return out

    def e_tr_species(self, T):
        return [self.cvtr[s] * T for s in range(self.ns)]

    def cv_ve_species(self, Tv):
        out = []
        for s in range(self.ns):
            if self.is_electron[s]:
                out.append(1.5 * self.Rs[s] + 0.0 * Tv)
                continue
            c = 0.0 * Tv
            th = self.theta_v[s]
            if th != 0.0:
                x = np.exp(th / Tv)
                c = c + self.Rs[s] * (th / Tv) ** 2 * x / (x - 1.0) ** 2
            tl, g = DAT.EL_THETA[s], DAT.EL_DEG[s]
            if len(tl) > 1:
                q0 = 0.0 * Tv
                q1 = 0.0 * Tv
                q2 = 0.0 * Tv
                for k in range(len(tl)):
                    e = g[k] * np.exp(-tl[k] / Tv)
                    q0 = q0 + e
                    q1 = q1 + e * tl[k]
                    q2 = q2 + e * tl[k] ** 2
                c = c + self.Rs[s] * (q2 / q0 - (q1 / q0) ** 2) / Tv ** 2
            out.append(c)
        return out

    def h_species(self, T, Tv):
        """ComputeSpeciesEnthalpy:  R_s T + Cv_tr,s T + h_f,s + e_ve,s."""
        eve = self.e_ve_species(Tv)
        return [self.Rs[s] * T + self.cvtr[s] * T + self.hf[s] + eve[s]
                for s in range(self.ns)]

    # ----------------------------------------------------------- chemistry
    def _keq_coeffs(self, n_cm3):
        """ComputeKeqConstants: per-reaction-group A[0..4] at density N.

        The table row follows SU2 exactly, including its unsigned-short
        arithmetic: N < 1e14 cm^-3 wraps around and takes the 1e19 row, and
        1e14 <= N < 1e15 takes the 1e14 row without interpolation."""
        Nr = _real(n_cm3)
        shape = np.shape(Nr)
        pwr = np.floor(np.log10(np.maximum(Nr, 1e-300))).astype(int)
        idx = pwr - 14
        lo = np.clip(idx, 0, 4)
        tabs = DAT.KEQ_TABLES                                  # (g, 6, 5)
        tmp1 = 10.0 ** pwr.astype(float)
        tmp2 = 10.0 * tmp1
        out = []
        for g in range(tabs.shape[0]):
            t = tabs[g]
            A = []
            for c in range(5):
                a_lo = t[lo, c]
                a_hi = t[np.minimum(lo + 1, 5), c]
                interp = (a_hi - a_lo) / (tmp2 - tmp1) * (n_cm3 - tmp1) + a_lo
                val = np.where(idx == 0, t[0, c],
                               np.where((idx >= 5) | (idx < 0), t[5, c],
                                        0.0)) + 0.0 * n_cm3
                inner = (idx > 0) & (idx < 5)
                val = val + np.where(inner, 1.0, 0.0) * interp
                A.append(val if shape else val)
            out.append(A)
        return out

    @staticmethod
    def _smooth_T(Tr):
        Tm, eps = DAT.RATE_T_MIN, DAT.RATE_EPS
        return 0.5 * (Tr + Tm + np.sqrt((Tr - Tm) ** 2 + eps * eps))

    def production_rates(self, rho_s, T, Tv):
        """Net mass production rates [kg/(m^3 s)] (list of ns)."""
        zero = 0.0 * T
        if self.frozen:
            return [zero] * self.ns
        conc = [rho_s[s] / (self.MW[s] * 1.0e6) for s in range(self.ns)]
        n_cm3 = 0.0
        for s in range(self.ns):
            n_cm3 = n_cm3 + rho_s[s] / self.MM[s] * (SU2_NA * 1.0e3)
        n_cm3 = n_cm3 * 1.0e-6
        A = self._keq_coeffs(n_cm3)
        w = [zero] * self.ns
        for j, (rr, pp) in enumerate(self._rx):
            Trf = T ** DAT.TCF_A[j] * Tv ** DAT.TCF_B[j]
            Trb = T ** DAT.TCB_A[j] * Tv ** DAT.TCB_B[j]
            Thf = self._smooth_T(Trf)
            Thb = self._smooth_T(Trb)
            a = A[DAT.KEQ_GROUP[j]]
            x = 1.0e4 / Thb
            keq = np.exp(a[0] / x + a[1] + a[2] * np.log(x) + a[3] * x
                         + a[4] * x * x)
            C, eta, th = (DAT.ARRHENIUSCOEFFICIENT[j], DAT.ARRHENIUSETA[j],
                          DAT.ARRHENIUSTHETA[j])
            kf = C * np.exp(eta * np.log(Thf)) * np.exp(-th / Thf)
            kb = C * np.exp(eta * np.log(Thb)) * np.exp(-th / Thb) / keq
            f = kf
            for s in rr:
                f = f * conc[s]
            b = kb
            for s in pp:
                b = b * conc[s]
            rate = (f - b) * 1.0e6                      # mol/(m^3 s)
            for s in pp:
                w[s] = w[s] + self.MW[s] * rate
            for s in rr:
                w[s] = w[s] - self.MW[s] * rate
        return w

    def source_ve(self, rho_s, T, Tv, omega=None):
        """ComputeEveSourceTerm [W/m^3]."""
        X, conc = self.molar_fractions(rho_s)
        n_tot = conc * SU2_NA
        p = self.pressure(rho_s, T, Tv)
        ev_eq = self.e_vib_species(T)
        ev = self.e_vib_species(Tv)
        sig = DAT.PARK_SIGMA * 2.5e9 / (T * T)
        Q = 0.0 * T
        for s in range(self.ns):
            th = self.theta_v[s]
            if th == 0.0 or self.is_electron[s]:
                continue                                  # eve_eq - eve == 0
            den = 0.0 * T
            for r in range(self.ns):
                mu = self.MM[s] * self.MM[r] / (self.MM[s] + self.MM[r])
                Asr = 1.16e-3 * np.sqrt(mu) * th ** (4.0 / 3.0)
                Bsr = 0.015 * mu ** 0.25
                tau = 101325.0 / p * np.exp(Asr * (T ** (-1.0 / 3.0) - Bsr)
                                            - 18.42)
                den = den + X[r] / tau
            num = 0.0
            for r in range(self.ns):
                num = num + X[r]
            tau_mw = num / den
            cs = np.sqrt(8.0 * SU2_RU * T / (SU2_PI * self.MW[s]))
            tau_p = 1.0 / (sig * cs * n_tot)
            Q = Q + rho_s[s] * (ev_eq[s] - ev[s]) / (tau_mw + tau_p)
        if not self.frozen:
            if omega is None:
                omega = self.production_rates(rho_s, T, Tv)
            for s in range(self.ns):
                Q = Q + omega[s] * ev[s]
        return Q

    # ----------------------------------------------------------- transport
    def transport(self, rho_s, T, Tv):
        """(mu, k_tr, k_ve, D_s list [m^2/s])."""
        m = self.transport_model
        if m == "GUPTA-YOS":
            mu = self._viscosity_gy(rho_s, T, Tv)
            ktr, kve = self._conductivity_gy(rho_s, T, Tv)
            Ds = self._diffusion_gy(rho_s, T, Tv)
        else:
            X, _ = self.molar_fractions(rho_s)
            mus, phis = self._blottner_wilke(X, T)
            if m == "WILKE":
                mu = 0.0
                for s in range(self.ns):
                    mu = mu + X[s] * mus[s] / phis[s]
                cvve = self.cv_ve_species(Tv)
                ktr = 0.0
                kve = 0.0
                for s in range(self.ns):
                    ks = mus[s] * (15.0 / 4.0 + 0.5 * self.rot[s]) * self.Rs[s]
                    ktr = ktr + X[s] * ks / phis[s]
                    kve = kve + X[s] * mus[s] * cvve[s] / phis[s]
            else:
                Tn = T / DAT.SUTH_TREF
                mu = DAT.SUTH_MU_REF * Tn * np.sqrt(Tn) * \
                    (DAT.SUTH_TREF + DAT.SUTH_SM) / (T + DAT.SUTH_SM)
                ktr = DAT.SUTH_K_REF * Tn * np.sqrt(Tn) * \
                    (DAT.SUTH_TREF + DAT.SUTH_SK) / (T + DAT.SUTH_SK)
                kve = self.cv_ve(rho_s, Tv) / self.cv_tr(rho_s) * ktr
            Ds = self._diffusion_wbe(rho_s, X, T, Tv)
        return mu, ktr, kve, Ds

    def _blottner_wilke(self, X, T):
        lnT = np.log(T)
        B = DAT.BLOTTNER
        mus = [0.1 * np.exp((B[s, 0] * lnT + B[s, 1]) * lnT + B[s, 2])
               for s in range(self.ns)]
        M = self.MM
        phis = []
        for i in range(self.ns):
            acc = 0.0 * T
            for j in range(self.ns):
                t1 = 1.0 + np.sqrt(mus[i] / mus[j]) * (M[j] / M[i]) ** 0.25
                t2 = np.sqrt(8.0 * (1.0 + M[i] / M[j]))
                acc = acc + X[j] * t1 * t1 / t2
            phis.append(acc)
        return mus, phis

    def _cross_section(self, i, j, T, d1, rho_e):
        """ComputeCollisionCrossSection [m^2] (non-Coulomb pairs include pi
        in the fit; Coulomb pairs use the shielded-Coulomb potential)."""
        O11, O22 = DAT.OMEGA11, DAT.OMEGA22
        coulomb = abs(O11[i, j, 0]) == 1.0
        if coulomb:
            kb_cgs = SU2_KB * 1.0e7
            ne = SU2_NA * 1.0e3 * rho_e / self.MM[self.i_e] * 1.0e-6
            debye = np.sqrt(kb_cgs * T / 4.0 / SU2_PI / ne / SU2_E_CGS ** 2)
            Ts = debye / (SU2_E_CGS ** 2 / (kb_cgs * T))
            fac = 1.0e-20 * 5.0e15 * SU2_PI * (debye / T) ** 2
            if O11[i, j, 0] == 1.0 and d1:
                Dd, Cc, cc = DAT.D1_A, DAT.C1_A, DAT.c1_A
            elif O11[i, j, 0] == -1.0 and d1:
                Dd, Cc, cc = DAT.D1_R, DAT.C1_R, DAT.c1_R
            elif O22[i, j, 0] == 1.0 and not d1:
                Dd, Cc, cc = DAT.D2_A, DAT.C2_A, DAT.c2_A
            else:
                Dd, Cc, cc = DAT.D2_R, DAT.C2_R, DAT.c2_R
            return fac * np.log(Dd * Ts * (1.0 - Cc * np.exp(-cc * Ts)) + 1.0)
        O = O11 if d1 else O22
        lnT = np.log(T)
        return 1.0e-20 * O[i, j, 3] * np.exp(
            (O[i, j, 0] * lnT * lnT + O[i, j, 1] * lnT + O[i, j, 2]) * lnT)

    def _diffusion_wbe(self, rho_s, X, T, Tv):
        rho = self.density(rho_s)
        M = 0.0
        for s in range(self.ns):
            M = M + self.MM[s] * X[s]
        M = M * 1.0e-3
        rho_e = rho_s[self.i_e]
        Dij = {}
        for i in range(self.ns):
            Mi = self.MM[i] * 1.0e-3
            for j in range(i, self.ns):
                Mj = self.MM[j] * 1.0e-3
                Tc = Tv if i == self.i_e else T
                Om = self._cross_section(i, j, Tc, True, rho_e) / SU2_PI
                d = 7.1613e-25 * M * np.sqrt(T * (1.0 / Mi + 1.0 / Mj)) / (rho * Om)
                Dij[(i, j)] = d
                Dij[(j, i)] = d
        out = []
        for i in range(self.ns):
            den = 0.0 * T
            for j in range(self.ns):
                if j != i:
                    den = den + X[j] / Dij[(i, j)]
            out.append((1.0 - X[i]) / den)
        return out

    def _delta(self, i, j, Mi, Mj, T, d1, rho_e):
        Om = self._cross_section(i, j, T, d1, rho_e)
        c = (8.0 / 3.0) if d1 else (16.0 / 5.0)
        dl = c * np.sqrt((2.0 * Mi * Mj) / (SU2_PI * SU2_RU * 1.0e3 * T
                                            * (Mi + Mj))) * Om
        # fmin(delta, 1e16) on the real part
        return np.where(_real(dl) > 1e16, 1e16 + 0.0 * dl, dl)

    def _viscosity_gy(self, rho_s, T, Tv):
        rho = self.density(rho_s)
        rho_e = rho_s[self.i_e]
        Na = SU2_NA * 1.0e3
        mu = 0.0 * T
        for i in range(self.ns):
            Mi = self.MM[i]
            gi = rho_s[i] / (rho * Mi)
            den = 0.0 * T
            for j in range(self.ns):
                Mj = self.MM[j]
                gj = rho_s[j] / (rho * Mj)
                Tc = Tv if i == self.i_e else T
                den = den + gj * self._delta(i, j, Mi, Mj, Tc, False, rho_e)
            mu = mu + (Mi / Na * gi) / den
        return mu

    def _conductivity_gy(self, rho_s, T, Tv):
        """ThermalConductivitiesGY.

        SU2 writes the partner concentration as rho_i/(rho M_j) instead of
        rho_j/(rho M_j).  With an electron in the mixture the e- partner
        term then scales like 1/sqrt(m_e) whatever the electron density is,
        and k_tr comes out several times too small (Pr ~ 5).  The default
        reproduces SU2 so that the operator matches the base flow; set
        gy_partner_bug=False for the textbook Gupta-Yos mixing rule."""
        rho = self.density(rho_s)
        rho_e = rho_s[self.i_e]
        Na = SU2_NA * 1.0e3
        kb = SU2_KB
        Cvve = self.cv_ve(rho_s, Tv) / rho
        R = 0.0
        for s in range(self.ns):
            R = R + self.Rs[s] * rho_s[s] / rho
        ktr = 0.0 * T
        kve = 0.0 * T
        for i in range(self.ns):
            Mi = self.MM[i]
            mi = Mi / Na
            gi = rho_s[i] / (rho * Mi)
            dt = 0.0 * T
            dr = 0.0 * T
            dre = 0.0 * T
            for j in range(self.ns):
                Mj = self.MM[j]
                mj = Mj / Na
                gj = (rho_s[i] if self.gy_partner_bug else rho_s[j]) / (rho * Mj)
                a = 1.0 + (1.0 - mi / mj) * (0.45 - 2.54 * mi / mj) / \
                    ((1.0 + mi / mj) ** 2)
                Tc = Tv if (i == self.i_e or j == self.i_e) else T
                d1 = self._delta(i, j, Mi, Mj, Tc, True, rho_e)
                d2 = self._delta(i, j, Mi, Mj, Tc, False, rho_e)
                dt = dt + (3.54 if j == self.i_e else a) * gj * d2
                dr = dr + gj * d1
                dre = dre + gj * d2
            if i != self.i_e:
                ktr = ktr + 15.0 / 4.0 * kb * gi / dt
            if self.rot[i] != 0.0:
                ktr = ktr + kb * gi / dr
                kve = kve + kb * Cvve / R * gi / dr
            if i == self.i_e:
                kve = kve + 15.0 / 4.0 * kb * gi / (1.45 * dre)
        return ktr, kve

    def _diffusion_gy(self, rho_s, T, Tv):
        rho = self.density(rho_s)
        rho_e = rho_s[self.i_e]
        p = self.pressure(rho_s, T, Tv)
        gt = 0.0
        for s in range(self.ns):
            gt = gt + rho_s[s] / (rho * self.MM[s])
        out = []
        for i in range(self.ns):
            Mi = self.MM[i]
            gi = rho_s[i] / (rho * Mi)
            den = 0.0 * T
            for j in range(self.ns):
                if j == i:
                    continue
                Mj = self.MM[j]
                gj = rho_s[j] / (rho * Mj)
                Tc = Tv if i == self.i_e else T
                d1 = self._delta(i, j, Mi, Mj, Tc, True, rho_e)
                den = den + gj / (SU2_KB * Tc / (p * d1))
            out.append(gt * gt * Mi * (1.0 - Mi * gi) / den)
        return out
