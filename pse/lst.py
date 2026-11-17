"""Local (quasi-parallel) linear stability theory: the spatial eigenvalue
problem, quadratic in the streamwise wavenumber,

    ( A0 + alpha A1 + alpha^2 A2 ) qhat = 0,

solved by shift-invert Arnoldi on the companion pencil with the far-field
rows frozen at the shift, structural filtering for the trapped acoustic
(Mack) modes, and a bordered Newton polish with the fully alpha-dependent
asymptotic far-field condition.  The result seeds the PSE march.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg as sla

from .constants import CS_H
from .operators import assemble
from . import backend
from . import progress


class Scaling:
    """Diagonal row/column equilibration (variables span ~10 decades)."""

    def __init__(self, model, grid, q, u_e, T_e):
        ns = model.ns
        rho = np.real(q[:ns]).sum(axis=0).max()
        qref = np.array([rho] * ns + [u_e] * 3 + [T_e] * 2)
        self.c = np.tile(qref, grid.ny)
        self.r = None

    def calibrate(self, A):
        As = A.multiply(self.c[None, :]).tocsr()
        rn = np.abs(As).max(axis=1).toarray().ravel()
        rn[rn == 0.0] = 1.0
        self.r = 1.0 / rn
        return self

    def apply(self, A):
        return (sps.diags(self.r) @ A.multiply(self.c[None, :])).tocsr()

    def unscale_vector(self, v):
        return v * self.c


def reference_values(model, q, u_e, T_e):
    rho = np.real(q[:model.ns]).sum(axis=0).max()
    return np.array([rho] * model.ns + [u_e] * 3 + [T_e] * 2)


def build_operator(ops, bcop, alpha, beta, omega, nonparallel=False,
                   dalpha_dx=0.0, far="asymptotic", **kw):
    L0, L1, L2, M0, M1 = ops.coefficients(alpha, beta, omega,
                                          nonparallel=nonparallel,
                                          dalpha_dx=dalpha_dx, **kw)
    A = assemble(L0, L1, L2, ops.grid)
    fb = None
    if far == "asymptotic":
        b = np.atleast_1d(beta)[-1]
        fb = bcop.far_block(*ops.farfield_coefficients(alpha, b, omega))
    return bcop.apply(A, far_block=fb), (M0, M1), A


def newton_alpha(ops, bcop, alpha0, beta, omega, q0=None, tol=1e-9,
                 itmax=40, far="asymptotic", d_rel=1e-6, scaling=None,
                 u_e=None, T_e=None, damping=1.0, beta_y=None):
    """Bordered Newton on the spatial eigenvalue with the alpha-dependent
    far-field condition (dA/dalpha by finite difference)."""
    a = complex(alpha0)
    kw = dict(beta_y=beta_y)
    if scaling is None:
        scaling = Scaling(ops.model, ops.grid, ops.q, u_e, T_e)
        A0, _, _ = build_operator(ops, bcop, a, beta, omega, far=far, **kw)
        scaling.calibrate(A0)

    def mat(av):
        A, _, _ = build_operator(ops, bcop, av, beta, omega, far=far, **kw)
        return scaling.apply(A)

    A = mat(a)
    n = A.shape[0]
    if q0 is None:
        lu = backend.factor(A)
        rng = np.random.default_rng(1)
        q = rng.normal(size=n) + 1j * rng.normal(size=n)
        for _ in range(4):
            q = lu.solve(q)
            q /= np.linalg.norm(q)
    else:
        q = np.array(q0, dtype=complex) / scaling.c
        q /= np.linalg.norm(q)
    c = q.conj().copy()
    da = np.inf
    hist = []
    for it in range(itmax):
        A = mat(a)
        h = d_rel * max(abs(a), 1.0)
        dA = (mat(a + h) - A) / h
        r = A @ q
        rc = c @ q - 1.0
        Kb = sps.bmat([[A, (dA @ q)[:, None]],
                       [sps.csr_matrix(c[None, :]), None]], format="csc")
        try:
            sol = backend.solve(Kb, -np.concatenate([r, [rc]]))
        except Exception:
            break
        if not np.all(np.isfinite(sol)):
            break
        q = q + damping * sol[:n]
        da = damping * sol[n]
        a = a + da
        hist.append(abs(da))
        if abs(da) < tol * max(abs(a), 1.0) and it >= 1:
            break
    A = mat(a)
    r = A @ q
    resf = float(np.linalg.norm(r) / max(np.linalg.norm(q), 1e-300))
    info = dict(res=resf, dalpha=float(abs(da)), iters=len(hist),
                scaling=scaling,
                converged=bool(abs(da) < 1e-6 * max(abs(a), 1.0)
                               and resf < 1e-6))
    return a, scaling.unscale_vector(q / np.linalg.norm(q)), info


def pencil(ops, bcop, beta, omega, far_alpha=None, beta_y=None):
    A0, A1, A2 = ops.lst_pencil(beta, omega, beta_y=beta_y)
    P0, P1, P2 = bcop.apply_pencil(A0, A1, A2)
    if far_alpha is not None:
        b = np.atleast_1d(beta)[-1]
        fb = bcop.far_block(*ops.farfield_coefficients(far_alpha, b, omega))
        n = P0.shape[0]
        d = np.ones(n)
        d[bcop.far_idx] = 0.0
        keep = sps.diags(d)
        P0 = (keep @ P0 + fb).tocsr()
        P1 = (keep @ P1).tocsr()
        P2 = (keep @ P2).tocsr()
    return P0, P1, P2


def spatial_modes(ops, bcop, beta, omega, sigma, k=12, far_alpha=None,
                  beta_y=None):
    """Eigenvalues alpha near `sigma` of the quadratic pencil."""
    A0, A1, A2 = pencil(ops, bcop, beta, omega, far_alpha=far_alpha,
                        beta_y=beta_y)
    n = A0.shape[0]
    I = sps.identity(n, format="csr", dtype=complex)
    Z = sps.csr_matrix((n, n), dtype=complex)
    P = sps.bmat([[A0, A1], [Z, I]], format="csc")
    Q = sps.bmat([[Z, -A2], [I, Z]], format="csc")
    lu = backend.factor(P - sigma * Q)
    cnt = [0]

    def mv(v):
        cnt[0] += 1
        if cnt[0] % 50 == 0:
            progress.poke()
        return lu.solve(Q @ v)
    OP = spla.LinearOperator((2 * n, 2 * n), matvec=mv, dtype=complex)
    vals, vecs = spla.eigs(OP, k=k, which="LM",
                           v0=np.random.default_rng(0).normal(size=2 * n))
    return sigma + 1.0 / vals, vecs[:n]


def temporal_modes(ops, bcop, alpha, beta, dense=True, k=30, sigma=None):
    """Temporal spectrum omega(alpha, beta): A_a q = i omega Gam q."""
    A_a, Gam = ops.temporal_pencil(alpha, beta)
    A_a = bcop.apply(A_a)
    Gam = (bcop.P @ Gam).tocsr()
    if dense:
        w, V = sla.eig(A_a.toarray(), 1j * Gam.toarray())
        good = np.isfinite(w) & (np.abs(w) < 1e12)
        return w[good], V[:, good]
    n = A_a.shape[0]
    M = (1j * Gam).tocsc()
    lu = backend.factor(A_a - sigma * M)
    OP = spla.LinearOperator((n, n), matvec=lambda v: lu.solve(M @ v),
                             dtype=complex)
    vals, vecs = spla.eigs(OP, k=k, which="LM")
    return sigma + 1.0 / vals, vecs


def pressure_gradient_matrix(model, q):
    """dp/dq at every wall-normal node (ny, NV)."""
    g = model.gas
    nv, ny = q.shape
    out = np.zeros((ny, nv))
    qc = np.asarray(q, dtype=complex)
    for k in range(nv):
        p = qc.copy()
        p[k] = p[k] + 1j * CS_H
        out[:, k] = np.imag(g.pressure(p[:g.ns], p[g.i_T], p[g.i_Tv])) / CS_H
    return out


def mode_diagnostics(model, v, grid, q, u_e, T_e, d_ref, dpdq=None):
    """Far-field energy fraction, wall pressure amplitude and the number of
    interior |p| minima (Mack's mode index)."""
    nv = model.nv
    ny = grid.ny
    V = v.reshape(ny, nv)
    sc = 1.0 / reference_values(model, q, u_e, T_e)
    E = (np.abs(V * sc) ** 2).sum(axis=1)
    tot = float((E * grid.w).sum())
    kc = int(np.searchsorted(grid.y, 4.0 * d_ref))
    ke = int(np.searchsorted(grid.y, 2.0 * d_ref))
    ff = float((E[kc:] * grid.w[kc:]).sum() / max(tot, 1e-300))
    if dpdq is None:
        dpdq = pressure_gradient_matrix(model, q)
    ph = np.einsum("jk,jk->j", dpdq, V)
    A = np.abs(ph) / max(np.abs(ph).max(), 1e-300)
    nmin = sum(1 for j in range(1, max(ke, 2))
               if A[j] < A[j - 1] and A[j] <= A[j + 1])
    return dict(far_fraction=ff, p_wall=float(A[0]), n_pmin=int(nmin),
                p_profile=A)


def find_second_mode(ops, bcop, beta, omega, u_e, T_e, d_ref, c_guess=0.92,
                     growth_guess=0.02, k=25, ff_max=1e-3, refine_it=30,
                     alpha_guess=None, q_guess=None, c_band=(0.6, 1.1),
                     beta_y=None):
    """Locate the trapped discrete mode (Mack's second mode) at a station."""
    model = ops.model
    dpdq = pressure_gradient_matrix(model, ops.q)
    if alpha_guess is None:
        alpha_guess = omega / (c_guess * u_e) * (1.0 - 1j * growth_guess)
    if q_guess is None:
        alphas, vecs = spatial_modes(ops, bcop, beta, omega, alpha_guess, k=k,
                                     far_alpha=alpha_guess, beta_y=beta_y)
        best = None
        for j in range(len(alphas)):
            a = alphas[j]
            if not np.isfinite(a) or a.real <= 0:
                continue
            c = omega / a.real / u_e
            if not (c_band[0] <= c <= c_band[1]):
                continue
            d = mode_diagnostics(model, vecs[:, j], ops.grid, ops.q, u_e, T_e,
                                 d_ref, dpdq)
            if d["far_fraction"] > ff_max or d["p_wall"] < 0.3:
                continue
            # most amplified trapped candidate (Krypton/Mack convention)
            score = a.imag / max(abs(a.real), 1e-30)
            if best is None or score < best[0]:
                best = (score, a, vecs[:, j])
        if best is None:
            return None
        a0, v0 = best[1], best[2]
    else:
        a0, v0 = alpha_guess, q_guess
    a, vp, info = newton_alpha(ops, bcop, a0, beta, omega, q0=v0,
                               itmax=refine_it, u_e=u_e, T_e=T_e,
                               beta_y=beta_y)
    d = mode_diagnostics(model, vp, ops.grid, ops.q, u_e, T_e, d_ref, dpdq)
    d.update(alpha=a, converged=info["converged"], res=info["res"],
             c_phase=omega / a.real / u_e)
    return a, vp, d
