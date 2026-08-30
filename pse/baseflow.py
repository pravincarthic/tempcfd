"""Base-flow container, self-similar generator and OpenFOAM ingestion."""
import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import make_interp_spline

from . import species as sp
from . import thermo as th
from . import transport as tr
from .grid import Grid

NS, NV = sp.NS, th.NV
I_U, I_V, I_W, I_T, I_TV = th.I_U, th.I_V, th.I_W, th.I_T, th.I_TV


class BaseFlow:
    """Base state on a fixed physical wall-normal grid at nx stations.

    q[i] is the primitive vector at station i, shape (NV, ny).
    metric holds the axisymmetric divergence terms (dr/dx)/r and (dr/dy)/r.
    """

    def __init__(self, x, grid, q, metric=None, meta=None):
        self.x = np.asarray(x, float)
        self.grid = grid
        self.q = np.asarray(q)
        self.nx, _, self.ny = self.q.shape
        self.meta = meta or {}
        if metric is None:
            metric = (np.zeros((self.nx, self.ny)), np.zeros((self.nx, self.ny)))
        self.rx_over_r, self.ry_over_r = metric
        self._derivatives()
        self.edge = self._edge_quantities()

    def _derivatives(self):
        D1 = self.grid.D1
        self.qy = np.einsum("jk,ivk->ivj", D1, self.q)
        self.qx = np.zeros_like(self.q)
        if self.nx > 1:
            self.qx[1:-1] = ((self.q[2:] - self.q[:-2]) /
                             (self.x[2:] - self.x[:-2])[:, None, None])
            self.qx[0] = (self.q[1] - self.q[0]) / (self.x[1] - self.x[0])
            self.qx[-1] = (self.q[-1] - self.q[-2]) / (self.x[-1] - self.x[-2])
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

    # -------------------------------------------------------------- metrics
    def _edge_quantities(self):
        out = {k: np.zeros(self.nx) for k in
               ("u_e", "T_e", "rho_e", "p_e", "mu_e", "a_e", "M_e", "H_e",
                "delta99", "delta_star", "theta", "Re_dstar", "Re_x",
                "delta_ll", "T_w", "y_edge")}
        y = self.grid.y
        for i in range(self.nx):
            q = self.q[i]
            rho = q[:NS].sum(axis=0)
            u = q[I_U]
            T, Tv = q[I_T], q[I_TV]
            H = th.energy_internal(q[:NS], T, Tv) + \
                th.pressure(q[:NS], T, Tv) / rho + 0.5 * u ** 2
            ue, rhoe, Te = u[-1], rho[-1], T[-1]
            k99 = int(np.argmax(u >= 0.99 * ue))
            k99 = max(k99, 2)
            d99 = float(np.interp(0.99 * ue, u[:k99 + 1], y[:k99 + 1]))
            je = k99
            f1 = 1.0 - rho * u / (rhoe * ue)
            f2 = rho * u / (rhoe * ue) * (1.0 - u / ue)
            ds = float(np.trapezoid(f1, y)) if hasattr(np, "trapezoid") else \
                float(np.trapz(f1, y))
            tt = float(np.trapezoid(f2, y)) if hasattr(np, "trapezoid") else \
                float(np.trapz(f2, y))
            mue, ktr, kve, _ = tr.mixture(q[:NS, -1:], T[-1:], Tv[-1:])
            ae = th.sound_speed_frozen(q[:NS, -1:], T[-1:], Tv[-1:])[0]
            out["u_e"][i], out["T_e"][i], out["rho_e"][i] = ue, Te, rhoe
            out["p_e"][i] = th.pressure(q[:NS, -1:], T[-1:], Tv[-1:])[0]
            out["mu_e"][i], out["a_e"][i] = float(mue[0]), float(ae)
            out["M_e"][i] = ue / ae
            out["H_e"][i] = H[-1]
            out["delta99"][i] = d99
            out["delta_star"][i] = ds
            out["theta"][i] = tt
            out["Re_dstar"][i] = rhoe * ue * ds / float(mue[0])
            out["Re_x"][i] = rhoe * ue * max(self.x[i], 1e-12) / float(mue[0])
            out["delta_ll"][i] = np.sqrt(float(mue[0]) * max(self.x[i], 1e-12)
                                         / (rhoe * ue))
            out["T_w"][i] = T[0]
            out["y_edge"][i] = y[je]
        return out

    def station(self, i):
        return self.q[i], self.qx[i], self.qy[i], self.qz[i]

    def metric(self, i):
        return self.rx_over_r[i], self.ry_over_r[i]

    def second_mode_frequency(self, i):
        """f ~ u_e / (2 delta99): the classical second-mode estimate [Hz]."""
        return self.edge["u_e"][i] / (2.0 * max(self.edge["delta99"][i], 1e-12))


# --------------------------------------------------------------- similarity
def _T_from_h(h_target, Y, T0):
    """Invert frozen-composition static enthalpy (thermal equilibrium)."""
    T = np.array(T0, dtype=float, copy=True)
    for _ in range(80):
        hh = 0.0
        cp = 0.0
        for s in range(NS):
            hh = hh + Y[s] * th.h_s(T, T, s)
            cp = cp + Y[s] * (3.5 * sp.RS[s] if sp.IS_MOLECULE[s]
                              else 2.5 * sp.RS[s])
        cvv = np.zeros_like(T)
        for s in range(NS):
            cvv = cvv + Y[s] * tr._cv_ve_s(T, s)
        dT = (h_target - hh) / np.maximum(cp + cvv, 1.0)
        T = np.clip(T + np.clip(dT, -0.4 * T, 0.4 * T), 30.0, 60000.0)
        if np.max(np.abs(dT)) < 1e-9:
            break
    return T


def similarity_profile(Me, Te, pe, Y, Tw=None, adiabatic=False, eta_max=12.0,
                       n=401, gamma_ref=1.4, tol=1e-8):
    """Levy-Lees self-similar compressible boundary layer with the 11-species
    frozen-composition thermodynamics and Blottner/Wilke/Eucken transport.

    Returns dict with eta, f, fp, g, T, rho, mu, and the edge state.
    """
    Yc = sp.mass_fraction_vector(Y)
    rs_e = np.array([Yc[s] for s in range(NS)])[:, None]
    Rmix = float(sum(Yc[s] * sp.RS[s] for s in range(NS)))
    rho_e = pe / (Rmix * Te)
    ae = float(th.sound_speed_frozen(rs_e * rho_e, np.array([Te]),
                                     np.array([Te]))[0])
    ue = Me * ae
    he = float(sum(Yc[s] * th.h_s(np.array([Te]), np.array([Te]), s)[0]
                   for s in range(NS)))
    He = he + 0.5 * ue ** 2
    mue = float(tr.mixture(rs_e * rho_e, np.array([Te]), np.array([Te]))[0][0])
    ktr, kve = tr.mixture(rs_e * rho_e, np.array([Te]), np.array([Te]))[1:3]
    cpe = float(np.ravel(th.cp_mix(rs_e * rho_e, np.array([Te]), np.array([Te])))[0])
    Pre = mue * cpe / float(np.ravel(ktr + kve)[0])

    def props(g, fp):
        h = g * He - 0.5 * ue ** 2 * fp ** 2
        T = _T_from_h(h, Yc, np.full_like(h, Te))
        rho = pe / (Rmix * T)
        rs = Yc[:, None] * rho[None, :]
        mu, kt, kv, _ = tr.mixture(rs, T, T)
        cp = th.cp_mix(rs, T, T)
        Pr = mu * cp / (kt + kv)
        C = rho * mu / (rho_e * mue)
        return T, rho, mu, Pr, C

    def rhs(eta, y):
        f, fp, Cfpp, g, y5 = y
        T, rho, mu, Pr, C = props(g, fp)
        fpp = Cfpp / C
        gp = (Pr / C) * (y5 - C * (1.0 - 1.0 / Pr) * (ue ** 2 / He) * fp * fpp)
        return np.vstack([fp, fpp, -f * fpp, gp, -f * gp])

    gw = None if adiabatic else _gw(Tw, Yc, He)

    def bc(ya, yb):
        c = [ya[0], ya[1], yb[1] - 1.0, yb[3] - 1.0]
        c.append(ya[4] if adiabatic else ya[3] - gw)
        return np.array(c)

    eta = np.linspace(0.0, eta_max, n)
    y0 = np.zeros((5, n))
    y0[0] = eta - (1.0 - np.exp(-eta))
    y0[1] = 1.0 - np.exp(-eta)
    y0[2] = 0.47 * np.exp(-eta)
    y0[3] = (gw if gw is not None else 1.0) + \
        (1.0 - (gw if gw is not None else 1.0)) * (1.0 - np.exp(-eta))
    y0[4] = 0.0
    sol = solve_bvp(rhs, bc, eta, y0, tol=tol, max_nodes=200000, verbose=0)
    if not sol.success:
        raise RuntimeError("similarity BVP failed: " + sol.message)
    eta = np.linspace(0.0, eta_max, n)
    Ysol = sol.sol(eta)
    f, fp, Cfpp, g, y5 = Ysol
    T, rho, mu, Pr, C = props(g, fp)
    return dict(eta=eta, f=f, fp=fp, g=g, T=T, rho=rho, mu=mu, Pr=Pr, C=C,
                u_e=ue, T_e=Te, rho_e=rho_e, mu_e=mue, p_e=pe, H_e=He,
                a_e=ae, M_e=Me, Y=Yc, Pr_e=Pre, Rmix=Rmix)


def _gw(Tw, Yc, He):
    hw = float(sum(Yc[s] * th.h_s(np.array([Tw]), np.array([Tw]), s)[0]
                   for s in range(NS)))
    return hw / He


def similarity_baseflow(x, Me, Te, pe, Y, Tw=None, adiabatic=False, cone=False,
                        theta_c=0.0, ny=201, y_max=None, y_half=None,
                        eta_max=12.0, cone_metric=False):
    """Build a BaseFlow from the self-similar solution at the given stations.

    cone=True applies the Mangler transformation for a sharp cone of
    half-angle theta_c (r = x sin theta_c).
    """
    S = similarity_profile(Me, Te, pe, Y, Tw=Tw, adiabatic=adiabatic,
                           eta_max=eta_max)
    x = np.asarray(x, float)
    ue, rho_e, mue = S["u_e"], S["rho_e"], S["mu_e"]
    Yc = S["Y"]

    # eta -> y at each station
    inv_rho = rho_e / S["rho"]
    Ieta = np.concatenate([[0.0], cumulative_trapezoid(inv_rho, S["eta"])])
    j = 1 if cone else 0
    st = np.sin(theta_c) if cone else 1.0
    y_of_x = []
    for xi in x:
        r = max(xi * st, 1e-12) if cone else 1.0
        xi_ll = rho_e * mue * ue * (r ** (2 * j)) * xi / (3.0 if cone else 1.0)
        scale = np.sqrt(2.0 * xi_ll) / (ue * r ** j * rho_e)
        y_of_x.append(scale * Ieta)
    y_of_x = np.array(y_of_x)

    d99_last = float(np.interp(0.99, S["fp"], y_of_x[-1]))
    if y_max is None:
        y_max = 12.0 * d99_last
    if y_half is None:
        y_half = 1.0 * d99_last
    grid = Grid(ny, y_max, min(y_half, 0.45 * y_max), kind="fd4")

    def smooth_map(yy, vals, edge):
        """C4 interpolation onto the PSE grid; base-flow smoothness controls
        the quality of the linearised operator, so linear interpolation is
        not adequate here."""
        spl = make_interp_spline(yy, vals, k=5)
        out = np.where(grid.y <= yy[-1], spl(np.minimum(grid.y, yy[-1])), edge)
        return out

    q = np.zeros((len(x), NV, ny))
    for i, xi in enumerate(x):
        yy = y_of_x[i]
        T = smooth_map(yy, S["T"], S["T"][-1])
        u = ue * smooth_map(yy, S["fp"], 1.0)
        rho = smooth_map(yy, S["rho"], S["rho"][-1])
        q[i, :NS] = Yc[:, None] * rho[None, :]
        q[i, I_U] = u
        q[i, I_T] = T
        q[i, I_TV] = T
    # wall-normal velocity from the Levy-Lees stream function
    #   rho u r^j = d(psi)/dy,  rho v r^j = -d(psi)/dx,  psi = sqrt(2 xi) f
    # keeping the leading term, which is the flat-plate form in the
    # Mangler-transformed variable and is bounded outside the layer.
    for i, xi in enumerate(x):
        rho = q[i, :NS].sum(axis=0)
        r = max(xi * st, 1e-12) if cone else 1.0
        xi_ll = rho_e * mue * ue * (r ** (2 * j)) * xi / (3.0 if cone else 1.0)
        yy = y_of_x[i]
        slope = (S["eta"][-1] - S["eta"][-2]) / max(yy[-1] - yy[-2], 1e-30)
        spl_e = make_interp_spline(yy, S["eta"], k=5)
        eta_i = np.where(grid.y <= yy[-1], spl_e(np.minimum(grid.y, yy[-1])),
                         S["eta"][-1] + (grid.y - yy[-1]) * slope)
        spl_f = make_interp_spline(S["eta"], S["f"], k=5)
        spl_fp = make_interp_spline(S["eta"], S["fp"], k=5)
        ec = np.minimum(eta_i, S["eta"][-1])
        f_i = np.where(eta_i <= S["eta"][-1], spl_f(ec),
                       S["f"][-1] + (eta_i - S["eta"][-1]))
        fp_i = np.where(eta_i <= S["eta"][-1], spl_fp(ec), 1.0)
        q[i, I_V] = (rho_e * mue * ue * r ** j / np.sqrt(2.0 * xi_ll)) \
            * (eta_i * fp_i - f_i) / rho

    metric = _cone_metric(x, grid.y, theta_c) if (cone and cone_metric) else None
    bf = BaseFlow(x, grid, q, metric=metric,
                  meta=dict(source="similarity", cone=cone, theta_c=theta_c,
                            M_e=Me, T_e=Te, p_e=pe, Y=Yc, similarity=S))
    return bf


def _cone_metric(x, y, theta_c):
    st, ct = np.sin(theta_c), np.cos(theta_c)
    X, Yg = np.meshgrid(x, y, indexing="ij")
    r = np.maximum(X * st + Yg * ct, 1e-12)
    return (st / r, ct / r)


# ------------------------------------------------------------------- OpenFOAM
def foam_baseflow(case, stations, ny=201, y_max=None, y_half=None,
                  wall_patches=("wall",), plane="xy", axis=(1, 0, 0),
                  time=None, field_map=None, cone_axis=None, smooth=1,
                  y_max_factor=12.0, y_half_factor=1.0, probe_ny=600,
                  extractor=None, quiet=True):
    """Build a BaseFlow by sampling a hy2Foam solution along wall normals.

    stations: arclengths [m] measured from the leading edge / stagnation point.
    """
    from .foam.extract import ProfileExtractor
    ex = extractor or ProfileExtractor(case, time=time, wall_patches=wall_patches,
                                       plane=plane, axis=axis,
                                       field_map=field_map, quiet=quiet)
    stations = np.asarray(stations, float)

    # probe once with a generous ray to size the wall-normal domain
    if y_max is None or y_half is None:
        probe_len = 0.35 * (ex.wall_s[-1] - ex.wall_s[0]) + 1e-6
        yp = np.linspace(0.0, probe_len, probe_ny)
        d99 = []
        for s in stations:
            P = ex.profile(s, yp)
            ue = P["u"][-1]
            k = np.argmax(P["u"] >= 0.99 * ue)
            d99.append(yp[max(k, 1)])
        d99 = np.array(d99)
        y_max = y_max or float(y_max_factor * d99.max())
        y_half = y_half or float(y_half_factor * np.median(d99))
    grid = Grid(ny, y_max, min(y_half, 0.45 * y_max), kind="fd4")

    q = np.zeros((len(stations), NV, ny))
    for i, s in enumerate(stations):
        P = ex.profile(s, grid.y)
        q[i, :NS] = np.maximum(P["rho_s"], 1e-30)
        q[i, I_U] = P["u"]
        q[i, I_V] = P["v"]
        q[i, I_W] = P["w"]
        q[i, I_T] = np.maximum(P["T"], 30.0)
        q[i, I_TV] = np.maximum(P["Tv"], 30.0)

    metric = None
    if plane == "axisym":
        p0 = np.array([ex.wall_point(s)[0] for s in stations])
        nrm = np.array([ex.wall_point(s)[1] for s in stations])
        R = p0[:, 1][:, None] + np.outer(nrm[:, 1], grid.y)
        R = np.maximum(R, 1e-12)
        drdx = np.gradient(p0[:, 1], stations)[:, None] * np.ones((1, ny))
        metric = (drdx / R, np.tile(nrm[:, 1][:, None], (1, ny)) / R)

    bf = BaseFlow(stations, grid, q, metric=metric,
                  meta=dict(source="openfoam", case=case, time=ex.time,
                            plane=plane, wall_patches=list(wall_patches)))
    if smooth:
        bf.smooth(n_pass=smooth, mode="y")
    return bf
