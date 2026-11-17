"""Boundary conditions of the multi-species two-temperature stability system.

Order of the system: the ns species-continuity equations sum exactly to
mixture continuity, which carries no diffusion and is first order in y.
With NV = ns + 5 unknowns the system is (NV-1) second-order + 1 first-order
equations and takes 2 NV - 1 conditions: NV at the far field and NV - 1 at
the wall; the remaining wall row keeps the mixture-continuity equation.
For AIR-7 that is 23 conditions, 12 + 11.

Wall (SU2 MARKER_ISOTHERMAL / MARKER_HEATFLUX, non-catalytic or
super-catalytic):
    non-catalytic   : J'_s,n = 0 for every species except the most abundant
                      one (whose row keeps mixture continuity; its flux
                      condition follows from sum_s J_s = 0), linearised from
                      the same diffusive flux as the base flow (complex-step
                      rows of dG/dq and dG/dq_y at the wall)
    super-catalytic : Y'_s = 0
    u' = v' = w' = 0
    T'  = 0 (isothermal) or dT'/dn = 0 (adiabatic)
    Tv' = 0 (isothermal) or dTv'/dn = 0

Far field: the asymptotic decay (Riccati) condition dq/dy = W q built from
the decaying wall-normal characteristics of the frozen far-field operator,
or Dirichlet, or Neumann (used for the mean-flow-distortion mode).

BCs are applied as  A_bc = P @ A + B  with P a row projector that also forms
the retained mixture-continuity row.
"""
import numpy as np
import scipy.sparse as sps
import scipy.linalg as sla


class WallBC:
    def __init__(self, thermal="isothermal", vibrational="isothermal",
                 catalytic="noncatalytic"):
        if vibrational == "equilibrium":
            vibrational = "isothermal"
        self.thermal = thermal
        self.vibrational = vibrational
        self.catalytic = catalytic


def wall_rows(model, grid, q, bc, J=None, skip=()):
    g = model.gas
    ns, nv = g.ns, g.nv
    D1, ny = grid.D1, grid.ny
    rho_s = np.real(q[:ns, 0])
    rho = rho_s.sum()
    Y = rho_s / rho
    keep = max((s for s in range(ns) if s not in skip), key=lambda s: rho_s[s])
    rows = []
    for s in range(ns):
        if s == keep or s in skip:
            continue
        cols = {}
        if bc.catalytic == "supercatalytic":
            cols[s] = 1.0
            for r in range(ns):
                cols[r] = cols.get(r, 0.0) - Y[s]
        else:
            if J is None:
                raise ValueError("non-catalytic wall needs the Jacobians")
            B0 = J["B"][0, s].copy()
            B0[g.i_v] = 0.0                    # remove rho_s v' (v' = 0)
            By0 = J["By"][0, s]
            for c in range(nv):
                if B0[c] != 0.0:
                    cols[c] = cols.get(c, 0.0) + B0[c]
            for k in range(ny):
                if D1[0, k] == 0.0:
                    continue
                for c in range(nv):
                    if By0[c] != 0.0:
                        cols[k * nv + c] = cols.get(k * nv + c, 0.0) \
                            + By0[c] * D1[0, k]
            sc = max(abs(v) for v in cols.values()) if cols else 1.0
            cols = {c: v / sc for c, v in cols.items()}
        rows.append((s, cols))
    for iv in (g.i_u, g.i_v, g.i_w):
        rows.append((iv, {iv: 1.0}))
    for iv, kind in ((g.i_T, bc.thermal), (g.i_Tv, bc.vibrational)):
        if kind == "adiabatic":
            rows.append((iv, {k * nv + iv: D1[0, k] for k in range(ny)
                              if D1[0, k] != 0.0}))
        else:
            rows.append((iv, {iv: 1.0}))
    return rows


def _equilibrate(L0, L1, L2, col):
    """Row/column scaling: the unknowns span ~20 decades (trace species vs
    temperatures), which wrecks a dense QZ unless the pencil is balanced.
    Exponents are invariant; eigenvectors are mapped back by `col`."""
    c = np.asarray(col, float)
    Ls = [L * c[None, :] for L in (L0, L1, L2)]
    rn = np.max(np.stack([np.abs(L).max(axis=1) for L in Ls]), axis=0)
    rn[rn == 0.0] = 1.0
    return [L / rn[:, None] for L in Ls]


def farfield_modes(L0, L1, L2, col=None):
    """Wall-normal exponents of (L0 + lam L1 + lam^2 L2) v = 0.

    col: optional column (variable) scales used to balance the pencil."""
    n = L0.shape[0]
    if col is None:
        col = np.ones(n)
    S0, S1, S2 = _equilibrate(L0, L1, L2, col)
    # balance the lambda scale as well
    s1 = np.abs(S1).max()
    s2 = np.abs(S2).max()
    sl = s1 / s2 if s2 > 0 and s1 > 0 else 1.0
    I = np.eye(n, dtype=complex)
    Z = np.zeros((n, n), dtype=complex)
    P = np.block([[S0, S1 * sl], [Z, I]])
    Q = np.block([[Z, -S2 * sl * sl], [I, Z]])
    w, V = sla.eig(P, Q)
    good = np.isfinite(w) & (np.abs(w) < 1e14)
    w = w[good] * sl
    V = V[:, good]
    Vq = V[:n] * np.asarray(col, float)[:, None]
    return w, Vq, V


def farfield_rows(L0, L1, L2, col=None):
    """Riccati form dqhat/dy = W qhat, W = V Lam V^-1 over the NV most
    rapidly decaying characteristics.  Returns (C0, C1) with
    C0 qhat + C1 dqhat/dy = 0."""
    lam, V, _ = farfield_modes(L0, L1, L2, col=col)
    n = L0.shape[0]
    order = np.argsort(np.real(lam))
    idx = order[:n]
    Vd = V[:, idx]
    if Vd.shape[1] != n or np.linalg.cond(Vd) > 1e12:
        return np.eye(n, dtype=complex), np.zeros((n, n), dtype=complex)
    W = Vd @ np.diag(lam[idx]) @ np.linalg.inv(Vd)
    return -W, np.eye(n, dtype=complex)


def trace_species(model, q, threshold):
    """Species whose mass fraction never exceeds `threshold` on the profile.
    Their perturbations are locked to the mixture (Y'_s = 0): they carry no
    momentum or energy worth resolving, and left free they only add
    degenerate passive-scalar (and, for e-, electron-pressure) modes."""
    if threshold is None or threshold <= 0.0:
        return []
    ns = model.ns
    rs = np.real(q[:ns])
    Y = rs / rs.sum(axis=0)
    return [s for s in range(ns) if Y[s].max() < threshold]


class BCOperator:
    def __init__(self, model, grid, q, bc, J=None, far="asymptotic",
                 trace_threshold=1e-6):
        self.model = model
        self.grid = grid
        self.far = far
        nv, ns = model.nv, model.ns
        self.nv = nv
        ny = grid.ny
        g = model.gas
        qe = np.abs(np.real(q[:, -1]))
        rho_e = qe[:ns].sum()
        self.col = np.maximum(qe, np.concatenate([np.full(ns, 1e-20 * rho_e),
                                                  np.full(5, 1e-30)]))
        self.col[g.i_u:g.i_w + 1] = max(qe[g.i_u], 1.0)
        n = ny * nv
        self.n = n
        self.trace = trace_species(model, q, trace_threshold)
        self.active = [k for k in range(nv) if k not in self.trace]
        rs = np.real(q[:ns])
        Yall = rs / rs.sum(axis=0)
        # reduced -> full map at the far field (trace rho_s' = Y_s rho')
        act_sp = [s for s in range(ns) if s not in self.trace]
        Tm = np.zeros((nv, len(self.active)))
        for a, k in enumerate(self.active):
            Tm[k, a] = 1.0
        for s in self.trace:
            for a, k in enumerate(self.active):
                if k in act_sp:
                    Tm[s, a] = Yall[s, -1]
        self.Tmap = Tm
        wrows = wall_rows(model, grid, q, bc, J=J, skip=self.trace)
        keep = max((s for s in act_sp), key=lambda s: rs[s, 0])
        self.keep = keep
        trows = []
        for j in range(ny):
            for s in self.trace:
                cols = {j * nv + s: 1.0}
                for k in range(ns):
                    cols[j * nv + k] = cols.get(j * nv + k, 0.0) - Yall[s, j]
                trows.append((j * nv + s, cols))
        far_rows = [(ny - 1) * nv + v for v in self.active]
        excluded = set(r for r, _ in wrows) | set(far_rows) | \
            set(r for r, _ in trows) | {keep}
        pr, pc, pv = [], [], []
        for r in range(n):
            if r in excluded:
                continue
            pr.append(r)
            pc.append(r)
            pv.append(1.0)
        for s in range(ns):
            pr.append(keep)
            pc.append(s)
            pv.append(1.0)
        self.P = sps.coo_matrix((pv, (pr, pc)), shape=(n, n)).tocsr()
        br, bcol, bv = [], [], []
        for r, cols in wrows + trows:
            for c, v in cols.items():
                br.append(r)
                bcol.append(c)
                bv.append(v)
        self.B_wall = sps.coo_matrix((bv, (br, bcol)), shape=(n, n)).tocsr()
        self.far_idx = np.array(far_rows, dtype=int)
        self.B_far_dirichlet = sps.coo_matrix(
            (np.ones(len(far_rows)), (far_rows, far_rows)),
            shape=(n, n)).tocsr()
        D1l = grid.D1[ny - 1]
        r, c, v = [], [], []
        for k, fr in zip(self.active, far_rows):
            for j in range(ny):
                if D1l[j] != 0.0:
                    r.append(fr)
                    c.append(j * nv + k)
                    v.append(D1l[j])
        self.B_far_neumann = sps.coo_matrix((v, (r, c)), shape=(n, n)).tocsr()
        self.B = (self.B_wall + (self.B_far_neumann if far == "neumann"
                                 else self.B_far_dirichlet)).tocsr()

    def reduce_far(self, L0f, L1f, L2f):
        a = self.active
        T = self.Tmap
        return [L[np.ix_(a, range(L.shape[1]))] @ T for L in (L0f, L1f, L2f)]

    def far_block(self, L0f, L1f, L2f):
        ny, nv = self.grid.ny, self.nv
        R0, R1, R2 = self.reduce_far(L0f, L1f, L2f)
        C0, C1 = farfield_rows(R0, R1, R2, col=self.col[self.active])
        D1last = self.grid.D1[ny - 1]
        r, c, v = [], [], []
        base = (ny - 1) * nv
        act = self.active
        for kk, k in enumerate(act):
            for vv_, vv in enumerate(act):
                if C0[kk, vv_] != 0.0:
                    r.append(base + k)
                    c.append(base + vv)
                    v.append(C0[kk, vv_])
                if C1[kk, vv_] != 0.0:
                    for j in range(ny):
                        if D1last[j] != 0.0:
                            r.append(base + k)
                            c.append(j * nv + vv)
                            v.append(C1[kk, vv_] * D1last[j])
        return sps.coo_matrix((v, (r, c)), shape=(self.n, self.n)).tocsr()

    def apply(self, A, far_block=None):
        B = self.B if far_block is None else (self.B_wall + far_block)
        return (self.P @ A + B).tocsr()

    def apply_rhs(self, b):
        return self.P @ b

    def apply_pencil(self, A0, A1, A2):
        return ((self.P @ A0 + self.B).tocsr(), (self.P @ A1).tocsr(),
                (self.P @ A2).tocsr())
