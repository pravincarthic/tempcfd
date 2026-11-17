"""Two-temperature (TTv) multi-species Navier-Stokes system of SU2-NEMO
(NEMO_NAVIER_STOKES) in body-fitted orthogonal coordinates.

Coordinates:  s along the wall (x), n normal to it (y), and z spanwise
(planar) or z = r*phi (axisymmetric).  Lame factors h_s = h = 1 + kappa n,
h_n = 1, h_phi = r(s, n).  The residual is written as

    R = dU/dt + d(l F)/ds + dG/dn + dH/dz + E(q, grad q) = 0

with l = 1/h, F, G, H the physical-component fluxes and E collecting the
geometric terms of the divergence in orthogonal coordinates, the curvature
and hoop-stress terms of the momentum equations, -F dl/ds, and minus the
chemical and vibrational sources.  Gradients are passed as

    qx : d q / ds   (computational; multiplied by l inside),
    qy : d q / dn,
    qz : d q / dz   (physical, = i beta(n) q for a Fourier mode).

Viscous fluxes follow CNEMONumerics::GetViscousProjFlux:

    J_s   = -( rho D_s grad Y_s - Y_s sum_k rho D_k grad Y_k )
    q_tr  = -k_tr grad T - k_ve grad Tve + sum_s J_s h_s
    q_ve  = -k_ve grad Tve + sum_s J_s e_ve,s
    tau   = mu (grad u + grad u^T - 2/3 div u I)

Everything is analytic in (q, qx, qy, qz): complex-step Jacobians are exact.
"""
import numpy as np


class Metric:
    """Geometric coefficients at the wall-normal nodes of one station.

    l     : 1/h
    kap   : kappa / h      (= dh/dn / h)
    a_s   : l (dr/ds) / r  (0 for planar)
    a_n   : (dr/dn) / r    (0 for planar)
    dl_ds : d(1/h)/ds
    r     : radius (axisymmetric) or None
    """

    def __init__(self, ny, l=None, kap=None, a_s=None, a_n=None, dl_ds=None,
                 r=None):
        one = np.ones(ny)
        zero = np.zeros(ny)
        self.ny = ny
        self.l = one if l is None else np.asarray(l, float)
        self.kap = zero if kap is None else np.asarray(kap, float)
        self.a_s = zero if a_s is None else np.asarray(a_s, float)
        self.a_n = zero if a_n is None else np.asarray(a_n, float)
        self.dl_ds = zero if dl_ds is None else np.asarray(dl_ds, float)
        self.r = None if r is None else np.asarray(r, float)

    @property
    def axisymmetric(self):
        return self.r is not None

    def broadcast(self, ndim):
        """Copy whose arrays broadcast against (ny, *extra) arrays."""
        shp = (self.ny,) + (1,) * (ndim - 1)
        m = Metric.__new__(Metric)
        m.ny = self.ny
        for k in ("l", "kap", "a_s", "a_n", "dl_ds"):
            setattr(m, k, getattr(self, k).reshape(shp))
        m.r = None if self.r is None else self.r.reshape(shp)
        return m


class Model:
    """Physics switches shared by base flow and stability operators."""

    def __init__(self, gas, viscous=True, diffusion=True,
                 thermal_noneq=True):
        self.gas = gas
        self.viscous = viscous
        self.diffusion = diffusion
        self.thermal_noneq = thermal_noneq
        self.ns = gas.ns
        self.nv = gas.nv

    @property
    def reacting(self):
        return not self.gas.frozen

    def transport(self, rho_s, T, Tv):
        return self.gas.transport(rho_s, T, Tv)


def _unpack(model, q):
    g = model.gas
    return q[:g.ns], q[g.i_u], q[g.i_v], q[g.i_w], q[g.i_T], q[g.i_Tv]


def conservative(model, q):
    """U = [rho_s, rho u, rho v, rho w, rho E, rho Eve]."""
    g = model.gas
    rho_s, u, v, w, T, Tv = _unpack(model, q)
    rho = g.density(rho_s)
    rE, rEve = g.energies(rho_s, T, Tv)
    U = [rho_s[s] for s in range(g.ns)]
    U += [rho * u, rho * v, rho * w,
          rE + 0.5 * rho * (u * u + v * v + w * w), rEve]
    return U


def _grad_Y(g, rho_s, rho, dq):
    drho = dq[0]
    for s in range(1, g.ns):
        drho = drho + dq[s]
    return [dq[s] / rho - rho_s[s] * drho / (rho * rho) for s in range(g.ns)]


def fluxes(model, q, qx, qy, qz, metric, trans=None, with_source=True):
    """Return (lF, G, H, E) as lists of NV arrays."""
    g = model.gas
    ns = g.ns
    iu, iv, iw, iT, iV = g.i_u, g.i_v, g.i_w, g.i_T, g.i_Tv
    rho_s, u, v, w, T, Tv = _unpack(model, q)
    m = metric
    rho = g.density(rho_s)
    p = g.pressure(rho_s, T, Tv)
    rE, rEve = g.energies(rho_s, T, Tv)
    ke = 0.5 * (u * u + v * v + w * w)
    rH = rE + rho * ke + p                    # rho * total enthalpy
    eve_m = rEve / rho

    # physical gradients
    gx = [m.l * qx[k] for k in range(g.nv)]
    gy = [qy[k] for k in range(g.nv)]
    gz = [qz[k] for k in range(g.nv)]

    F = [rho_s[s] * u for s in range(ns)]
    G = [rho_s[s] * v for s in range(ns)]
    H = [rho_s[s] * w for s in range(ns)]
    # momentum flux tensor Pi (inviscid part first)
    Pss = rho * u * u + p
    Pnn = rho * v * v + p
    Ppp = rho * w * w + p
    Psn = rho * u * v
    Psp = rho * u * w
    Pnp = rho * v * w
    Fe = rH * u
    Ge = rH * v
    He = rH * w
    Fv = rho * eve_m * u
    Gv = rho * eve_m * v
    Hv = rho * eve_m * w

    if model.viscous:
        mu, ktr, kve, Ds = model.transport(rho_s, T, Tv) if trans is None \
            else trans
        ux, vx, wx = gx[iu], gx[iv], gx[iw]
        uy, vy, wy = gy[iu], gy[iv], gy[iw]
        uz, vz, wz = gz[iu], gz[iv], gz[iw]
        ess = ux + m.kap * v
        enn = vy
        epp = wz + m.a_s * u + m.a_n * v
        esn = 0.5 * (uy - m.kap * u + vx)
        esp = 0.5 * (wx - m.a_s * w + uz)
        enp = 0.5 * (wy - m.a_n * w + vz)
        div = ess + enn + epp
        lam = -2.0 / 3.0 * mu * div
        tss = 2.0 * mu * ess + lam
        tnn = 2.0 * mu * enn + lam
        tpp = 2.0 * mu * epp + lam
        tsn = 2.0 * mu * esn
        tsp = 2.0 * mu * esp
        tnp = 2.0 * mu * enp
        Pss = Pss - tss
        Pnn = Pnn - tnn
        Ppp = Ppp - tpp
        Psn = Psn - tsn
        Psp = Psp - tsp
        Pnp = Pnp - tnp
        Fe = Fe - (u * tss + v * tsn + w * tsp) - ktr * gx[iT] - kve * gx[iV]
        Ge = Ge - (u * tsn + v * tnn + w * tnp) - ktr * gy[iT] - kve * gy[iV]
        He = He - (u * tsp + v * tnp + w * tpp) - ktr * gz[iT] - kve * gz[iV]
        Fv = Fv - kve * gx[iV]
        Gv = Gv - kve * gy[iV]
        Hv = Hv - kve * gz[iV]
        if model.diffusion:
            hs = g.h_species(T, Tv)
            evs = g.e_ve_species(Tv)
            Y = [rho_s[s] / rho for s in range(ns)]
            for gr, flux, kind in ((gx, F, 0), (gy, G, 1), (gz, H, 2)):
                dY = _grad_Y(g, rho_s, rho, gr)
                vs = [rho * Ds[s] * dY[s] for s in range(ns)]
                vsum = vs[0]
                for s in range(1, ns):
                    vsum = vsum + vs[s]
                eh = 0.0
                ev = 0.0
                for s in range(ns):
                    J = -(vs[s] - Y[s] * vsum)
                    flux[s] = flux[s] + J
                    eh = eh + J * hs[s]
                    ev = ev + J * evs[s]
                if kind == 0:
                    Fe, Fv = Fe + eh, Fv + ev
                elif kind == 1:
                    Ge, Gv = Ge + eh, Gv + ev
                else:
                    He, Hv = He + eh, Hv + ev

    F += [Pss, Psn, Psp, Fe, Fv]
    G += [Psn, Pnn, Pnp, Ge, Gv]
    H += [Psp, Pnp, Ppp, He, Hv]

    # geometric terms of the divergence (scalar part)
    ksum = m.kap + m.a_n
    E = [m.a_s * F[k] + ksum * G[k] - m.dl_ds * F[k] for k in range(g.nv)]
    # tensor (curvature / hoop) terms of the momentum equations
    E[iu] = E[iu] + m.kap * Psn - m.a_s * Ppp
    E[iv] = E[iv] - m.kap * Pss - m.a_n * Ppp
    E[iw] = E[iw] + m.a_s * Psp + m.a_n * Pnp

    if with_source:
        S = source(model, q)
        E = [E[k] - S[k] for k in range(g.nv)]
    lF = [m.l * F[k] for k in range(g.nv)]
    return lF, G, H, E


def source(model, q):
    g = model.gas
    rho_s, u, v, w, T, Tv = _unpack(model, q)
    zero = 0.0 * T
    S = [zero] * g.nv
    omega = None
    if not g.frozen:
        omega = g.production_rates(rho_s, T, Tv)
        for s in range(g.ns):
            S[s] = omega[s]
    if model.thermal_noneq:
        S[g.i_Tv] = g.source_ve(rho_s, T, Tv, omega)
    return S
