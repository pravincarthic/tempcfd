"""Base-flow container, wall geometry metrics and the Levy-Lees similarity
generator (verification path).  SU2-NEMO ingestion lives in pse.su2."""
import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import make_interp_spline

from .fluxes import Metric
from .grid import Grid


class WallGeometry:
    """Wall curve in the meridional (x, r) or planar (x, y) plane, as
    functions of the arclength s: position, unit normal (into the fluid) and
    curvature kappa, with the Lame factor h = 1 + kappa n (n into the fluid),
    so kappa > 0 on a convex body."""

    def __init__(self, s, xw, yw, nx, ny, kappa, axisymmetric):
        self.s = np.asarray(s, float)
        self.xw, self.yw = np.asarray(xw, float), np.asarray(yw, float)
        self.nx, self.ny = np.asarray(nx, float), np.asarray(ny, float)
        self.kappa = np.asarray(kappa, float)
        self.axisymmetric = bool(axisymmetric)

    @classmethod
    def straight(cls, s, angle=0.0, x0=0.0, y0=0.0, axisymmetric=False):
        s = np.asarray(s, float)
        ca, sa = np.cos(angle), np.sin(angle)
        return cls(s, x0 + s * ca, y0 + s * sa, -sa * np.ones_like(s),
                   ca * np.ones_like(s), np.zeros_like(s), axisymmetric)

    def metric(self, i, y):
        """Metric at station i for wall-normal distances y."""
        k = self.kappa[i]
        h = 1.0 + k * y
        l = 1.0 / h
        dk = np.gradient(self.kappa, self.s)[i] if self.s.size > 1 else 0.0
        dl = -l * l * dk * y
        if not self.axisymmetric:
            return Metric(len(y), l=l, kap=k * l, dl_ds=dl)
        r = np.maximum(self.yw[i] + y * self.ny[i], 1e-12)
        drw = np.gradient(self.yw, self.s)[i] if self.s.size > 1 else 0.0
        dnr = np.gradient(self.ny, self.s)[i] if self.s.size > 1 else 0.0
        r_s = drw + y * dnr
        return Metric(len(y), l=l, kap=k * l, a_s=l * r_s / r,
                      a_n=self.ny[i] / r, dl_ds=dl, r=r)


class BaseFlow:
    """Base state on a fixed wall-normal grid at nx stations.

    q[i] is the primitive vector [rho_s, u, v, w, T, Tve] at station i,
    shape (NV, ny); u is tangential and v normal to the wall.
    """

    def __init__(self, x, grid, q, gas, geometry=None, meta=None):
        self.x = np.asarray(x, float)
        self.grid = grid
        self.q = np.asarray(q, float)
        self.gas = gas
        self.nx, _, self.ny = self.q.shape
        self.geometry = geometry or WallGeometry.straight(self.x)
        self.meta = meta or {}
        self._derivatives()
        self.edge = self._edge_quantities()

    def _derivatives(self):
        D1 = self.grid.D1
        self.qy = np.einsum("jk,ivk->ivj", D1, self.q)
        self.qx = np.zeros_like(self.q)
        if self.nx > 2:
            x = self.x
            for i in range(self.nx):
                j = min(max(i, 1), self.nx - 2)
                xm, x0, xp = x[j - 1], x[j], x[j + 1]
                xi = x[i]
                cm = (2 * xi - x0 - xp) / ((xm - x0) * (xm - xp))
                c0 = (2 * xi - xm - xp) / ((x0 - xm) * (x0 - xp))
                cp = (2 * xi - xm - x0) / ((xp - xm) * (xp - x0))
                self.qx[i] = cm * self.q[j - 1] + c0 * self.q[j] + cp * self.q[j + 1]
        elif self.nx == 2:
            d = (self.q[1] - self.q[0]) / (self.x[1] - self.x[0])
            self.qx[0] = d
            self.qx[1] = d
        self.qz = np.zeros_like(self.q)

    def smooth(self, n_pass=1, mode="y"):
        """Light Shuman filter to remove CFD interpolation noise."""
        for _ in range(n_pass):
            if "y" in mode:
                q = self.q.copy()
                q[:, :, 1:-1] = 0.25 * (self.q[:, :, :-2] + 2 * self.q[:, :, 1:-1]
                                        + self.q[:, :, 2:])
                self.q = q
            if "x" in mode and self.nx > 2:
                q = self.q.copy()
                q[1:-1] = 0.25 * (self.q[:-2] + 2 * self.q[1:-1] + self.q[2:])
                self.q = q
        self._derivatives()
        self.edge = self._edge_quantities()

    def _edge_quantities(self):
        g = self.gas
        ns = g.ns
        out = {k: np.zeros(self.nx) for k in
               ("u_e", "T_e", "Tv_e", "rho_e", "p_e", "mu_e", "a_e", "M_e",
                "delta99", "delta_star", "theta", "Re_dstar", "Re_x",
                "delta_ll", "T_w", "Tv_w", "p_w", "y_edge")}
        y = self.grid.y
        trapz = getattr(np, "trapezoid", None) or np.trapz
        for i in range(self.nx):
            q = self.q[i]
            rs = q[:ns]
            rho = rs.sum(axis=0)
            u = q[g.i_u]
            T, Tv = q[g.i_T], q[g.i_Tv]
            # edge: where the total enthalpy / velocity first levels off
            ue = float(np.max(u[: max(int(0.8 * self.ny), 3)]))
            k99 = int(np.argmax(u >= 0.99 * ue))
            k99 = max(k99, 2)
            d99 = float(np.interp(0.99 * ue, u[:k99 + 1], y[:k99 + 1]))
            ke = min(int(np.searchsorted(y, 1.5 * d99)), self.ny - 1)
            rhoe = rho[ke]
            ue_e = u[ke]
            f1 = 1.0 - rho[:ke + 1] * u[:ke + 1] / (rhoe * ue_e)
            f2 = rho[:ke + 1] * u[:ke + 1] / (rhoe * ue_e) * \
                (1.0 - u[:ke + 1] / ue_e)
            ds = float(trapz(f1, y[:ke + 1]))
            tt = float(trapz(f2, y[:ke + 1]))
            rse = rs[:, ke:ke + 1]
            g.anchor(rse, T[ke:ke + 1], Tv[ke:ke + 1])
            mue = float(np.ravel(g.transport(rse, T[ke:ke + 1],
                                             Tv[ke:ke + 1])[0])[0])
            ae = float(np.ravel(g.sound_speed(rse, T[ke:ke + 1],
                                              Tv[ke:ke + 1]))[0])
            out["u_e"][i], out["T_e"][i], out["rho_e"][i] = ue_e, T[ke], rhoe
            out["Tv_e"][i] = Tv[ke]
            out["p_e"][i] = float(np.ravel(g.pressure(rse, T[ke:ke + 1],
                                                      Tv[ke:ke + 1]))[0])
            out["p_w"][i] = float(np.ravel(g.pressure(rs[:, :1], T[:1],
                                                      Tv[:1]))[0])
            out["mu_e"][i], out["a_e"][i] = mue, ae
            out["M_e"][i] = ue_e / ae
            out["delta99"][i] = d99
            out["delta_star"][i] = ds
            out["theta"][i] = tt
            out["Re_dstar"][i] = rhoe * ue_e * ds / mue
            out["Re_x"][i] = rhoe * ue_e * max(self.x[i], 1e-12) / mue
            out["delta_ll"][i] = np.sqrt(mue * max(self.x[i], 1e-12)
                                         / (rhoe * ue_e))
            out["T_w"][i] = T[0]
            out["Tv_w"][i] = Tv[0]
            out["y_edge"][i] = y[ke]
        return out

    def station(self, i):
        return self.q[i], self.qx[i], self.qy[i], self.qz[i]

    def metric(self, i):
        return self.geometry.metric(i, self.grid.y)

    def second_mode_frequency(self, i):
        """f ~ u_e / (2 delta99): the classical second-mode estimate [Hz]."""
        return self.edge["u_e"][i] / (2.0 * max(self.edge["delta99"][i], 1e-12))


# --------------------------------------------------------------- similarity
def similarity_profile(gas, Me, Te, pe, Y, Tw=None, adiabatic=False,
                       eta_max=12.0, n=401, tol=1e-8, Y_floor=1e-10):
    """Levy-Lees compressible similarity solution with frozen composition,
    thermal equilibrium (Tve = T) and the gas model's own transport."""
    Yc = gas.mass_fraction_vector(Y)
    Yc = np.maximum(Yc, Y_floor)
    Yc = Yc / Yc.sum()
    Rmix = float(np.sum(Yc * gas.Rs))
    rho_e = pe / (Rmix * Te)
    one = np.array([Te])
    rs_e = Yc[:, None] * rho_e
    ae = float(gas.sound_speed(rs_e, one, one)[0])
    ue = Me * ae
    he = float(sum(Yc[s] * gas.h_species(one, one)[s][0] for s in range(gas.ns)))
    He = he + 0.5 * ue ** 2
    gas.anchor(rs_e, one, one)
    mue = float(np.ravel(gas.transport(rs_e, one, one)[0])[0])

    def props(gg, fp):
        h = gg * He - 0.5 * ue ** 2 * fp ** 2
        T = gas.T_from_h_equilibrium(Yc, h, np.full_like(h, Te))
        rho = pe / (Rmix * T)
        rs = Yc[:, None] * rho[None, :]
        gas.anchor(rs, T, T)
        mu, kt, kv, _ = gas.transport(rs, T, T)
        cp = gas.cp_frozen(rs, T, T)
        Pr = mu * cp / (kt + kv)
        C = rho * mu / (rho_e * mue)
        return T, rho, mu, Pr, C

    def rhs(eta, y):
        f, fp, Cfpp, gg, y5 = y
        T, rho, mu, Pr, C = props(gg, fp)
        fpp = Cfpp / C
        gp = (Pr / C) * (y5 - C * (1.0 - 1.0 / Pr) * (ue ** 2 / He) * fp * fpp)
        return np.vstack([fp, fpp, -f * fpp, gp, -f * gp])

    gw = None
    if not adiabatic:
        a = np.array([Tw])
        gw = float(sum(Yc[s] * gas.h_species(a, a)[s][0]
                       for s in range(gas.ns))) / He

    def bc(ya, yb):
        c = [ya[0], ya[1], yb[1] - 1.0, yb[3] - 1.0]
        c.append(ya[4] if adiabatic else ya[3] - gw)
        return np.array(c)

    eta = np.linspace(0.0, eta_max, n)
    y0 = np.zeros((5, n))
    y0[0] = eta - (1.0 - np.exp(-eta))
    y0[1] = 1.0 - np.exp(-eta)
    y0[2] = 0.47 * np.exp(-eta)
    g0 = gw if gw is not None else 1.0
    y0[3] = g0 + (1.0 - g0) * (1.0 - np.exp(-eta))
    sol = solve_bvp(rhs, bc, eta, y0, tol=tol, max_nodes=200000, verbose=0)
    if not sol.success:
        raise RuntimeError("similarity BVP failed: " + sol.message)
    f, fp, Cfpp, gg, y5 = sol.sol(eta)
    T, rho, mu, Pr, C = props(gg, fp)
    return dict(eta=eta, f=f, fp=fp, g=gg, T=T, rho=rho, mu=mu, Pr=Pr, C=C,
                u_e=ue, T_e=Te, rho_e=rho_e, mu_e=mue, p_e=pe, H_e=He,
                a_e=ae, M_e=Me, Y=Yc, Rmix=Rmix)


def similarity_baseflow(gas, x, Me, Te, pe, Y, Tw=None, adiabatic=False,
                        cone=False, theta_c=0.0, ny=201, y_max=None,
                        y_half=None, eta_max=12.0, grid_kind="fd4",
                        axisymmetric_metric=True):
    """BaseFlow from the self-similar solution.  cone=True applies the
    Mangler transformation for a sharp cone of half-angle theta_c and, with
    axisymmetric_metric, the axisymmetric terms in the stability operator."""
    S = similarity_profile(gas, Me, Te, pe, Y, Tw=Tw, adiabatic=adiabatic,
                           eta_max=eta_max)
    x = np.asarray(x, float)
    ue, rho_e, mue = S["u_e"], S["rho_e"], S["mu_e"]
    Yc = S["Y"]
    ns = gas.ns
    inv_rho = rho_e / S["rho"]
    Ieta = np.concatenate([[0.0], cumulative_trapezoid(inv_rho, S["eta"])])
    j = 1 if cone else 0
    st = np.sin(theta_c) if cone else 1.0
    y_of_x = []
    xis = []
    for xi in x:
        r = max(xi * st, 1e-12) if cone else 1.0
        xi_ll = rho_e * mue * ue * (r ** (2 * j)) * xi / (3.0 if cone else 1.0)
        scale = np.sqrt(2.0 * xi_ll) / (ue * r ** j * rho_e)
        y_of_x.append(scale * Ieta)
        xis.append((r, xi_ll))
    y_of_x = np.array(y_of_x)
    d99_last = float(np.interp(0.99, S["fp"], y_of_x[-1]))
    if y_max is None:
        y_max = 12.0 * d99_last
    if y_half is None:
        y_half = 1.0 * d99_last
    grid = Grid(ny, y_max, min(y_half, 0.45 * y_max), kind=grid_kind)

    def smooth_map(yy, vals, edge):
        spl = make_interp_spline(yy, vals, k=5)
        return np.where(grid.y <= yy[-1], spl(np.minimum(grid.y, yy[-1])), edge)

    q = np.zeros((len(x), gas.nv, ny))
    spl_f = make_interp_spline(S["eta"], S["f"], k=5)
    spl_fp = make_interp_spline(S["eta"], S["fp"], k=5)
    for i, xi in enumerate(x):
        yy = y_of_x[i]
        T = smooth_map(yy, S["T"], S["T"][-1])
        u = ue * smooth_map(yy, S["fp"], 1.0)
        rho = smooth_map(yy, S["rho"], S["rho"][-1])
        q[i, :ns] = Yc[:, None] * rho[None, :]
        q[i, gas.i_u] = u
        q[i, gas.i_T] = T
        q[i, gas.i_Tv] = T
        r, xi_ll = xis[i]
        slope = (S["eta"][-1] - S["eta"][-2]) / max(yy[-1] - yy[-2], 1e-30)
        spl_e = make_interp_spline(yy, S["eta"], k=5)
        eta_i = np.where(grid.y <= yy[-1], spl_e(np.minimum(grid.y, yy[-1])),
                         S["eta"][-1] + (grid.y - yy[-1]) * slope)
        ec = np.minimum(eta_i, S["eta"][-1])
        f_i = np.where(eta_i <= S["eta"][-1], spl_f(ec),
                       S["f"][-1] + (eta_i - S["eta"][-1]))
        fp_i = np.where(eta_i <= S["eta"][-1], spl_fp(ec), 1.0)
        q[i, gas.i_v] = (rho_e * mue * ue * r ** j / np.sqrt(2.0 * xi_ll)) \
            * (eta_i * fp_i - f_i) / rho
    geo = WallGeometry.straight(x, angle=0.0, axisymmetric=False)
    if cone and axisymmetric_metric:
        # wall in the (axial, radial) plane: x_w = s cos, r_w = s sin
        geo = WallGeometry.straight(x, angle=theta_c, axisymmetric=True)
    return BaseFlow(x, grid, q, gas, geometry=geo,
                    meta=dict(source="similarity", cone=cone, theta_c=theta_c,
                              M_e=Me, T_e=Te, p_e=pe, Y=Yc, similarity=S))
