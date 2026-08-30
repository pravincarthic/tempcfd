"""Conservative state, total (inviscid + viscous) fluxes and sources for the
11-species two-temperature Navier-Stokes system.

Residual form used by the stability operators:

    R = dU/dt + dF/dx + dG/dy + dH/dz - S = 0

with F, G, H functions of (q, dq/dx, dq/dy, dq/dz) and S a function of q only.
Everything is analytic so that complex-step Jacobians are exact.
"""
import numpy as np
from . import species as sp
from . import thermo as th
from . import transport as tp
from . import chemistry as ch

NS = sp.NS
NV = th.NV
I_U, I_V, I_W, I_T, I_TV = th.I_U, th.I_V, th.I_W, th.I_T, th.I_TV


class Model:
    """Physics switches shared by base flow and stability operators."""

    def __init__(self, viscous=True, reacting=True, thermal_noneq=True,
                 Le=1.2, Sc=None, low_T_blend=True, diffusion=True):
        self.viscous = viscous
        self.reacting = reacting
        self.thermal_noneq = thermal_noneq
        self.Le = Le
        self.Sc = Sc
        self.low_T_blend = low_T_blend
        self.diffusion = diffusion

    def transport(self, rho_s, T, Tv):
        return tp.mixture(rho_s, T, Tv, Le=self.Le, Sc=self.Sc,
                          low_T_blend=self.low_T_blend)


def conservative(q):
    rho_s, u, v, w, T, Tv = q[:NS], q[I_U], q[I_V], q[I_W], q[I_T], q[I_TV]
    rho = th.density(rho_s)
    e = th.energy_internal(rho_s, T, Tv)
    eve = th.energy_ve(rho_s, Tv)
    U = [rho_s[s] for s in range(NS)]
    U.append(rho * u)
    U.append(rho * v)
    U.append(rho * w)
    U.append(rho * (e + 0.5 * (u * u + v * v + w * w)))
    U.append(rho * eve)
    return U


def _gradY(rho_s, dq):
    rho = th.density(rho_s)
    drho = dq[0]
    for s in range(1, NS):
        drho = drho + dq[s]
    return [dq[s] / rho - rho_s[s] * drho / rho ** 2 for s in range(NS)]


def fluxes(q, qx, qy, qz, model, trans=None):
    """Return (F, G, H) as lists of NV entries."""
    rho_s, u, v, w, T, Tv = q[:NS], q[I_U], q[I_V], q[I_W], q[I_T], q[I_TV]
    rho = th.density(rho_s)
    p = th.pressure(rho_s, T, Tv)
    e = th.energy_internal(rho_s, T, Tv)
    eve = th.energy_ve(rho_s, Tv)
    ke = 0.5 * (u * u + v * v + w * w)
    H0 = e + ke + p / rho

    F = [rho_s[s] * u for s in range(NS)]
    G = [rho_s[s] * v for s in range(NS)]
    Hf = [rho_s[s] * w for s in range(NS)]
    F += [rho * u * u + p, rho * u * v, rho * u * w, rho * H0 * u, rho * eve * u]
    G += [rho * v * u, rho * v * v + p, rho * v * w, rho * H0 * v, rho * eve * v]
    Hf += [rho * w * u, rho * w * v, rho * w * w + p, rho * H0 * w, rho * eve * w]

    if not model.viscous:
        return F, G, Hf

    mu, ktr, kve, rhoD = model.transport(rho_s, T, Tv) if trans is None else trans

    ux, vx, wx = qx[I_U], qx[I_V], qx[I_W]
    uy, vy, wy = qy[I_U], qy[I_V], qy[I_W]
    uz, vz, wz = qz[I_U], qz[I_V], qz[I_W]
    div = ux + vy + wz
    txx = mu * (2.0 * ux - (2.0 / 3.0) * div)
    tyy = mu * (2.0 * vy - (2.0 / 3.0) * div)
    tzz = mu * (2.0 * wz - (2.0 / 3.0) * div)
    txy = mu * (uy + vx)
    txz = mu * (uz + wx)
    tyz = mu * (vz + wy)

    Tx, Ty, Tz = qx[I_T], qy[I_T], qz[I_T]
    Vx, Vy, Vz = qx[I_TV], qy[I_TV], qz[I_TV]
    qhx = -ktr * Tx - kve * Vx
    qhy = -ktr * Ty - kve * Vy
    qhz = -ktr * Tz - kve * Vz
    qvx, qvy, qvz = -kve * Vx, -kve * Vy, -kve * Vz

    F[I_U] = F[I_U] - txx
    F[I_V] = F[I_V] - txy
    F[I_W] = F[I_W] - txz
    G[I_U] = G[I_U] - txy
    G[I_V] = G[I_V] - tyy
    G[I_W] = G[I_W] - tyz
    Hf[I_U] = Hf[I_U] - txz
    Hf[I_V] = Hf[I_V] - tyz
    Hf[I_W] = Hf[I_W] - tzz

    F[I_T] = F[I_T] - (u * txx + v * txy + w * txz) + qhx
    G[I_T] = G[I_T] - (u * txy + v * tyy + w * tyz) + qhy
    Hf[I_T] = Hf[I_T] - (u * txz + v * tyz + w * tzz) + qhz
    F[I_TV] = F[I_TV] + qvx
    G[I_TV] = G[I_TV] + qvy
    Hf[I_TV] = Hf[I_TV] + qvz

    if model.diffusion:
        Jx = tp.diffusion_fluxes(rho_s, _gradY(rho_s, qx), rhoD)
        Jy = tp.diffusion_fluxes(rho_s, _gradY(rho_s, qy), rhoD)
        Jz = tp.diffusion_fluxes(rho_s, _gradY(rho_s, qz), rhoD)
        hs = th.h_all(T, Tv)
        evs = th.e_ve_all(Tv)
        ehx = ehy = ehz = 0.0
        evx = evy = evz = 0.0
        for s in range(NS):
            F[s] = F[s] + Jx[s]
            G[s] = G[s] + Jy[s]
            Hf[s] = Hf[s] + Jz[s]
            ehx = ehx + Jx[s] * hs[s]
            ehy = ehy + Jy[s] * hs[s]
            ehz = ehz + Jz[s] * hs[s]
            evx = evx + Jx[s] * evs[s]
            evy = evy + Jy[s] * evs[s]
            evz = evz + Jz[s] * evs[s]
        F[I_T] = F[I_T] + ehx
        G[I_T] = G[I_T] + ehy
        Hf[I_T] = Hf[I_T] + ehz
        F[I_TV] = F[I_TV] + evx
        G[I_TV] = G[I_TV] + evy
        Hf[I_TV] = Hf[I_TV] + evz

    return F, G, Hf


def source(q, model):
    rho_s, T, Tv = q[:NS], q[I_T], q[I_TV]
    zero = np.zeros_like(T)
    S = [zero] * NS + [zero, zero, zero, zero, zero]
    if not model.reacting:
        if model.thermal_noneq:
            w0 = [np.zeros_like(T)] * NS
            S = list(S)
            S[I_TV] = ch.source_ve(rho_s, T, Tv, w0, [np.zeros_like(T)] * ch.NR)
        return S
    omega, rates = ch.source(rho_s, T, Tv)
    S = list(omega) + [zero, zero, zero, zero]
    S.append(ch.source_ve(rho_s, T, Tv, omega, rates)
             if model.thermal_noneq else zero)
    return S
