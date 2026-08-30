"""Oblique shock and Taylor-Maccoll relations for edge conditions."""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp


def oblique_shock(M1, theta, gamma=1.4, weak=True):
    """Return (beta, M2, p2/p1, rho2/rho1, T2/T1) for deflection theta [rad]."""
    def f(b):
        return (2.0 / np.tan(b) * (M1 ** 2 * np.sin(b) ** 2 - 1.0) /
                (M1 ** 2 * (gamma + np.cos(2 * b)) + 2.0)) - np.tan(theta)
    bmin = np.arcsin(1.0 / M1) + 1e-9
    bmax = np.pi / 2 - 1e-9
    bs = np.linspace(bmin, bmax, 4000)
    vals = np.array([f(b) for b in bs])
    roots = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if roots.size == 0:
        raise ValueError("detached shock for M=%g theta=%g deg" %
                         (M1, np.degrees(theta)))
    k = roots[0] if weak else roots[-1]
    beta = brentq(f, bs[k], bs[k + 1])
    Mn1 = M1 * np.sin(beta)
    p = (2 * gamma * Mn1 ** 2 - (gamma - 1)) / (gamma + 1)
    r = (gamma + 1) * Mn1 ** 2 / ((gamma - 1) * Mn1 ** 2 + 2)
    T = p / r
    Mn2 = np.sqrt(((gamma - 1) * Mn1 ** 2 + 2) /
                  (2 * gamma * Mn1 ** 2 - (gamma - 1)))
    M2 = Mn2 / np.sin(beta - theta)
    return beta, M2, p, r, T


def taylor_maccoll(M1, theta_c, gamma=1.4):
    """Sharp-cone shock solution.  Returns dict with shock angle and surface
    Mach number, pressure/density/temperature ratios at the cone surface."""
    def surface_angle(beta):
        theta = np.arctan(2.0 / np.tan(beta) * (M1 ** 2 * np.sin(beta) ** 2 - 1)
                          / (M1 ** 2 * (gamma + np.cos(2 * beta)) + 2))
        Mn1 = M1 * np.sin(beta)
        Mn2 = np.sqrt(((gamma - 1) * Mn1 ** 2 + 2) /
                      (2 * gamma * Mn1 ** 2 - (gamma - 1)))
        M2 = Mn2 / np.sin(beta - theta)
        V = 1.0 / np.sqrt(2.0 / ((gamma - 1) * M2 ** 2) + 1.0)
        Vr = V * np.cos(beta - theta)
        Vt = -V * np.sin(beta - theta)

        def rhs(t, y):
            vr, vt = y
            num = (vr * vt ** 2 - (gamma - 1) / 2 * (1 - vr ** 2 - vt ** 2) *
                   (2 * vr + vt / np.tan(t)))
            den = (gamma - 1) / 2 * (1 - vr ** 2 - vt ** 2) - vt ** 2
            return [vt, num / den]

        def ev(t, y):
            return y[1]
        ev.terminal = True
        ev.direction = 1.0
        sol = solve_ivp(rhs, [beta, 1e-6], [Vr, Vt], events=ev,
                        rtol=1e-10, atol=1e-12, dense_output=True)
        if sol.t_events[0].size == 0:
            return None, sol
        return sol.t_events[0][0], sol

    lo = np.arcsin(1.0 / M1) + 1e-6
    hi = np.pi / 2 - 1e-4

    def g(b):
        tc, _ = surface_angle(b)
        return (tc if tc is not None else np.pi / 2) - theta_c

    bs = np.linspace(lo + 1e-6, hi, 200)
    vals = []
    for b in bs:
        try:
            vals.append(g(b))
        except Exception:
            vals.append(np.nan)
    vals = np.array(vals)
    ok = np.isfinite(vals)
    idx = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    idx = [i for i in idx if ok[i] and ok[i + 1]]
    if not idx:
        raise ValueError("no cone solution for M=%g, theta_c=%g deg"
                         % (M1, np.degrees(theta_c)))
    beta = brentq(g, bs[idx[0]], bs[idx[0] + 1], xtol=1e-12)
    tc, sol = surface_angle(beta)
    vr, vt = sol.y[:, -1]
    Vc = np.hypot(vr, vt)
    Mc = np.sqrt(2.0 / ((gamma - 1) * (1.0 / Vc ** 2 - 1.0)))
    theta_s = np.arctan(2.0 / np.tan(beta) * (M1 ** 2 * np.sin(beta) ** 2 - 1)
                        / (M1 ** 2 * (gamma + np.cos(2 * beta)) + 2))
    _, M2, pr, rr, Tr = oblique_shock(M1, theta_s, gamma)
    # isentropic compression from just behind the shock to the cone surface
    f = (1 + (gamma - 1) / 2 * M2 ** 2) / (1 + (gamma - 1) / 2 * Mc ** 2)
    return dict(beta=beta, M_cone=Mc, p_ratio=pr * f ** (gamma / (gamma - 1)),
                rho_ratio=rr * f ** (1 / (gamma - 1)), T_ratio=Tr * f,
                M_behind_shock=M2, theta_c=theta_c)
