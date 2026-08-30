"""Boundary conditions for the multi-species two-temperature stability system.

Order of the system: the 11 species-continuity equations sum exactly to the
mixture continuity equation, which carries no diffusion and is therefore first
order in y.  The system is 15 second-order + 1 first-order equations and
admits 31 boundary conditions: 16 at the far field, 15 at the wall.  The
remaining wall row keeps the mixture-continuity equation itself.

BCs are applied as  A_bc = P @ A + B  with P a sparse row-projector (it also
forms the retained mixture-continuity row as the sum of the species rows) and
B a sparse matrix holding the explicit condition rows.  Both depend only on
the base flow, so they are built once per station and reused for every
alpha iteration.
"""
import numpy as np
import scipy.sparse as sps
import scipy.linalg as sla
from . import fluxes as fx

NV, NS = fx.NV, fx.NS
I_U, I_V, I_W, I_T, I_TV = fx.I_U, fx.I_V, fx.I_W, fx.I_T, fx.I_TV


class WallBC:
    def __init__(self, thermal="isothermal", vibrational="equilibrium",
                 catalytic="noncatalytic"):
        self.thermal = thermal              # isothermal | adiabatic
        self.vibrational = vibrational      # equilibrium (Tv' = 0) | adiabatic
        self.catalytic = catalytic          # noncatalytic | supercatalytic


def wall_rows(grid, q, bc):
    """(equation row index, {column: coefficient}) for the 15 wall conditions."""
    D1, ny = grid.D1, grid.ny
    rho_s = np.real(q[:NS, 0])
    rho = rho_s.sum()
    Y = rho_s / rho
    drho_dy = float(sum(D1[0] @ np.real(q[s]) for s in range(NS)))
    rows = []
    for s in range(NS - 1):
        cols = {}
        if bc.catalytic == "supercatalytic":
            cols[s] = 1.0
        else:
            for r in range(NS):
                d = (1.0 if r == s else 0.0) - Y[s]
                if d == 0.0:
                    continue
                for k in range(ny):
                    if D1[0, k] != 0.0:
                        cols[k * NV + r] = cols.get(k * NV + r, 0.0) \
                            + d * D1[0, k] / rho
                cols[r] = cols.get(r, 0.0) - d * drho_dy / rho ** 2
        rows.append((s, cols))
    for iv in (I_U, I_V, I_W):
        rows.append((iv, {iv: 1.0}))
    if bc.thermal == "adiabatic":
        rows.append((I_T, {k * NV + I_T: D1[0, k] for k in range(ny)
                           if D1[0, k] != 0.0}))
    else:
        rows.append((I_T, {I_T: 1.0}))
    if bc.vibrational == "adiabatic":
        rows.append((I_TV, {k * NV + I_TV: D1[0, k] for k in range(ny)
                            if D1[0, k] != 0.0}))
    else:
        rows.append((I_TV, {I_TV: 1.0}))
    return rows


def farfield_modes(L0, L1, L2):
    """Wall-normal characteristic exponents of the constant-coefficient
    far-field operator:  (L0 + lam L1 + lam^2 L2) v = 0.

    Returns (lam, V) with V the (NV, m) matrix of eigenvectors."""
    n = L0.shape[0]
    I = np.eye(n, dtype=complex)
    Z = np.zeros((n, n), dtype=complex)
    P = np.block([[L0, L1], [Z, I]])
    Q = np.block([[Z, -L2], [I, Z]])
    w, V = sla.eig(P, Q)
    good = np.isfinite(w) & (np.abs(w) < 1e14)
    return w[good], V[:n, good], V[:, good]


def farfield_rows(L0, L1, L2, n_keep=None):
    """Riccati form of the far-field decay condition.

    With V the matrix of the n_keep most rapidly decaying wall-normal
    characteristics and Lam their exponents, requiring that the solution at
    y_max contain no growing characteristic is exactly

        dqhat/dy = W qhat,      W = V Lam V^{-1}.

    W is a smooth function of alpha (unlike an SVD basis of the complementary
    subspace), which matters because dA/dalpha is formed by finite difference.
    Returns (C0, C1) so that the rows read  C0 qhat + C1 dqhat/dy = 0.
    """
    lam, V, _ = farfield_modes(L0, L1, L2)
    n = L0.shape[0]
    if n_keep is None:
        n_keep = n
    order = np.argsort(np.real(lam))
    idx = order[:n_keep]
    Vd = V[:, idx]
    Lam = np.diag(lam[idx])
    if Vd.shape[1] != n or np.linalg.cond(Vd) > 1e12:
        return np.eye(n, dtype=complex), np.zeros((n, n), dtype=complex)
    W = Vd @ Lam @ np.linalg.inv(Vd)
    return -W, np.eye(n, dtype=complex)


class BCOperator:
    """P (row projector / combiner), B (wall + far-field condition rows).

    far='asymptotic' builds the far-field rows from the local characteristic
    exponents (alpha dependent, rebuilt through `far_block`); far='dirichlet'
    simply sets qhat(y_max) = 0.
    """

    def __init__(self, grid, q, bc, far="dirichlet"):
        self.grid = grid
        self.far = far
        ny = grid.ny
        n = ny * NV
        self.n = n
        rows = wall_rows(grid, q, bc)
        bc_rows = set(r for r, _ in rows)
        far = [(ny - 1) * NV + v for v in range(NV)]

        pr, pc, pv = [], [], []
        for r in range(n):
            if r in bc_rows or r in far:
                continue
            if r == NS - 1:
                continue
            pr.append(r); pc.append(r); pv.append(1.0)
        for s in range(NS):                       # retained mixture continuity
            pr.append(NS - 1); pc.append(s); pv.append(1.0)
        self.P = sps.coo_matrix((pv, (pr, pc)), shape=(n, n)).tocsr()

        br, bc_, bv = [], [], []
        for r, cols in rows:
            for c, v in cols.items():
                br.append(r); bc_.append(c); bv.append(v)
        self.B_wall = sps.coo_matrix((bv, (br, bc_)), shape=(n, n)).tocsr()
        self.far_idx = np.array(far, dtype=int)
        d = sps.coo_matrix((np.ones(len(far)), (far, far)),
                           shape=(n, n)).tocsr()
        self.B_far_dirichlet = d
        self.B = (self.B_wall + d).tocsr()
        self.zero_rows = np.array(sorted(bc_rows | set(far)), dtype=int)

    def far_block(self, L0f, L1f, L2f):
        """Sparse far-field rows from the asymptotic decay condition."""
        ny = self.grid.ny
        C0, C1 = farfield_rows(L0f, L1f, L2f)
        D1last = self.grid.D1[ny - 1]
        r, c, v = [], [], []
        base = (ny - 1) * NV
        for k in range(NV):
            for vv in range(NV):
                if C0[k, vv] != 0.0:
                    r.append(base + k); c.append(base + vv); v.append(C0[k, vv])
                if C1[k, vv] != 0.0:
                    for j in range(ny):
                        if D1last[j] != 0.0:
                            r.append(base + k); c.append(j * NV + vv)
                            v.append(C1[k, vv] * D1last[j])
        return sps.coo_matrix((v, (r, c)), shape=(self.n, self.n)).tocsr()

    def apply(self, A, far_block=None):
        B = self.B if far_block is None else (self.B_wall + far_block)
        return (self.P @ A + B).tocsr()

    def apply_rhs(self, b):
        return self.P @ b

    def apply_pencil(self, A0, A1, A2):
        return (self.P @ A0 + self.B).tocsr(), (self.P @ A1).tocsr(), \
               (self.P @ A2).tocsr()
