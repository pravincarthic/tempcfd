"""Local (quasi-parallel) linear stability: spatial eigenvalue problem.

The quasi-parallel operator is quadratic in the streamwise wavenumber,

    ( A0 + alpha A1 + alpha^2 A2 ) qhat = 0,

which is solved either by shift-invert Arnoldi on the companion pencil or,
given a good guess, by a bordered-Newton iteration.  The result seeds the PSE
march.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg as sla

from . import fluxes as fx

NV, NS = fx.NV, fx.NS
I_U, I_V, I_W, I_T, I_TV = fx.I_U, fx.I_V, fx.I_W, fx.I_T, fx.I_TV


class Scaling:
    """Diagonal row/column equilibration of the discrete operator.

    The 16 equations and 16 variables span ~10 decades (species densities vs
    temperatures vs energy fluxes), so the raw matrix is hopeless for both
    iterative refinement and residual-based convergence tests.  Columns are
    scaled by reference values of the primitive variables and rows by the
    resulting row norms, both fixed for the life of a station so that
    dA/dalpha is scaled consistently.
    """

    def __init__(self, grid, q, u_e, T_e):
        rho = q[:NS].sum(axis=0).max()
        qref = np.array([rho] * NS + [u_e] * 3 + [T_e] * 2)
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


def build_operator(ops, bcop, alpha, beta, omega, nonparallel=False,
                   dalpha_dx=0.0, far="asymptotic"):
    """Assemble the BC-constrained quasi-parallel (or nonparallel) operator."""
    L0, L1, L2, M0 = ops.coefficients(alpha, beta, omega,
                                      nonparallel=nonparallel,
                                      dalpha_dx=dalpha_dx)
    from .operators import assemble
    A = assemble(L0, L1, L2, ops.grid)
    fb = None
    if far == "asymptotic":
        fb = bcop.far_block(*ops.farfield_coefficients(alpha, beta, omega))
    return bcop.apply(A, far_block=fb), M0


def newton_alpha(ops, bcop, alpha0, beta, omega, q0=None, tol=1e-9,
                 itmax=40, far="asymptotic", d_rel=1e-6, scaling=None,
                 u_e=None, T_e=None, damping=1.0):
    """Bordered Newton on the spatial eigenvalue with the (alpha-dependent)
    asymptotic far-field condition.  dA/dalpha by finite difference.

    Returns (alpha, qhat_scaled, info)."""
    a = complex(alpha0)
    if scaling is None:
        scaling = Scaling(ops.grid, ops.q, u_e, T_e)
        A0, _ = build_operator(ops, bcop, a, beta, omega, far=far)
        scaling.calibrate(A0)

    def mat(av):
        A, _ = build_operator(ops, bcop, av, beta, omega, far=far)
        return scaling.apply(A)

    A = mat(a)
    n = A.shape[0]
    if q0 is None:
        lu = spla.splu(A.tocsc())
        rng = np.random.default_rng(1)
        q = rng.normal(size=n) + 1j * rng.normal(size=n)
        for _ in range(4):
            q = lu.solve(q)
            q /= np.linalg.norm(q)
    else:
        q = np.array(q0, dtype=complex)
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
            sol = spla.spsolve(Kb, -np.concatenate([r, [rc]]))
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
    return a, q / np.linalg.norm(q), info


def pencil(ops, bcop, beta, omega, far_alpha=None):
    """Quadratic-in-alpha pencil.  If far_alpha is given, the asymptotic
    far-field rows are frozen at that alpha so the pencil stays polynomial
    while still suppressing the continuous spectrum."""
    A0, A1, A2 = ops.lst_pencil(beta, omega)
    P0, P1, P2 = bcop.apply_pencil(A0, A1, A2)
    if far_alpha is not None:
        fb = bcop.far_block(*ops.farfield_coefficients(far_alpha, beta, omega))
        n = P0.shape[0]
        d = np.ones(n)
        d[bcop.far_idx] = 0.0
        keep = sps.diags(d)
        P0 = (keep @ P0 + fb).tocsr()
        P1 = (keep @ P1).tocsr()
        P2 = (keep @ P2).tocsr()
    return P0, P1, P2


def temporal_modes(ops, bcop, alpha, beta, dense=True, k=30, sigma=None):
    """Temporal spectrum omega(alpha, beta).  Linear generalised problem."""
    A_a, Gam = ops.temporal_pencil(alpha, beta)
    A_a = bcop.apply(A_a)
    Gam = (bcop.P @ Gam).tocsr()
    if dense:
        w, V = sla.eig(A_a.toarray(), 1j * Gam.toarray())
        good = np.isfinite(w) & (np.abs(w) < 1e12)
        return w[good], V[:, good]
    n = A_a.shape[0]
    M = (1j * Gam).tocsc()
    lu = spla.splu((A_a - sigma * M).tocsc())
    OP = spla.LinearOperator((n, n), matvec=lambda v: lu.solve(M @ v),
                             dtype=complex)
    vals, vecs = spla.eigs(OP, k=k, which="LM")
    return sigma + 1.0 / vals, vecs


def discrete_filter(vecs, grid, scale, y_cut, tol=1e-4):
    """Keep only modes whose energy above y_cut is a negligible fraction:
    this removes the discretised continuous spectra."""
    ny = grid.ny
    kcut = int(np.searchsorted(grid.y, y_cut))
    w = grid.w
    E = (np.abs(vecs.reshape(ny, NV, -1) * scale[None, :, None]) ** 2).sum(axis=1)
    tot = np.einsum("jk,j->k", E, w)
    top = np.einsum("jk,j->k", E[kcut:], w[kcut:])
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = top / tot
    return np.isfinite(frac) & (frac < tol), frac


def reference_scale(q, u_e, T_e):
    rho = q[:NS].sum(axis=0).max()
    return np.array([1.0 / rho] * NS + [1.0 / u_e] * 3 + [1.0 / T_e] * 2)


def spatial_modes(ops, bcop, beta, omega, sigma, k=12, far_alpha=None):
    """Eigenvalues alpha near `sigma` of the quadratic pencil."""
    A0, A1, A2 = pencil(ops, bcop, beta, omega, far_alpha=far_alpha)
    n = A0.shape[0]
    I = sps.identity(n, format="csr", dtype=complex)
    Z = sps.csr_matrix((n, n), dtype=complex)
    P = sps.bmat([[A0, A1], [Z, I]], format="csc")
    Q = sps.bmat([[Z, -A2], [I, Z]], format="csc")
    lu = spla.splu((P - sigma * Q).tocsc())
    OP = spla.LinearOperator((2 * n, 2 * n), matvec=lambda v: lu.solve(Q @ v),
                             dtype=complex)
    vals, vecs = spla.eigs(OP, k=k, which="LM",
                           v0=np.random.default_rng(0).normal(size=2 * n))
    alpha = sigma + 1.0 / vals
    return alpha, vecs[:n]


def refine(ops, bcop, alpha0, beta, omega, q0=None, tol=1e-11, itmax=40):
    """Bordered-Newton refinement of a single spatial eigenvalue."""
    A0, A1, A2 = pencil(ops, bcop, beta, omega)
    n = A0.shape[0]
    a = complex(alpha0)
    q = np.ones(n, dtype=complex) if q0 is None else np.array(q0, dtype=complex)
    q /= np.linalg.norm(q)
    c = q.conj().copy()
    for _ in range(itmax):
        A = (A0 + a * A1 + a * a * A2).tocsc()
        dA = (A1 + 2.0 * a * A2).tocsc()
        r = A @ q
        rc = c @ q - 1.0
        Kb = sps.bmat([[A, (dA @ q)[:, None]],
                       [sps.csr_matrix(c[None, :]), None]], format="csc")
        sol = spla.spsolve(Kb, -np.concatenate([r, [rc]]))
        q = q + sol[:n]
        da = sol[n]
        a = a + da
        if abs(da) < tol * max(abs(a), 1.0):
            break
    return a, q / np.linalg.norm(q)


def _phase_speed(alpha, omega):
    return np.real(omega / alpha)


def pick_mode(alphas, vecs, omega, u_e, grid=None, c_range=(0.55, 1.05),
              alpha_r_min=0.0, target=None):
    """Select the second Mack mode: downstream-running, phase speed in the
    trapped-acoustic band, largest spatial growth (or nearest to `target`)."""
    ok = []
    for i, a in enumerate(alphas):
        if not np.isfinite(a) or a.real <= alpha_r_min:
            continue
        c = _phase_speed(a, omega) / u_e
        if not (c_range[0] <= c <= c_range[1]):
            continue
        ok.append(i)
    if not ok:
        return None, None
    if target is not None:
        j = min(ok, key=lambda i: abs(alphas[i] - target))
    else:
        j = min(ok, key=lambda i: alphas[i].imag)
    return alphas[j], vecs[:, j]


def second_mode_guess(u_e, delta, omega, c_ratio=0.92, growth=0.02):
    """alpha guess for the trapped second mode."""
    ar = omega / (c_ratio * u_e)
    return ar - 1j * growth * ar


def solve_station(ops, bcop, beta, omega, u_e, alpha_guess, k=16,
                  c_range=(0.55, 1.05), refine_result=True):
    """Full local solve: Arnoldi around the guess, mode selection, Newton."""
    alphas, vecs = spatial_modes(ops, bcop, beta, omega, alpha_guess, k=k)
    a, v = pick_mode(alphas, vecs, omega, u_e, c_range=c_range)
    if a is None:
        a, v = alphas[np.argmin(np.abs(alphas - alpha_guess))], None
        v = vecs[:, int(np.argmin(np.abs(alphas - alpha_guess)))]
    if refine_result:
        a, v = refine(ops, bcop, a, beta, omega, q0=v)
    return a, v, alphas


def mode_diagnostics(v, grid, q, u_e, T_e, d_ref, dpdq=None):
    """Structure metrics used to identify the trapped (Mack) modes:
    far-field energy fraction, wall pressure amplitude and the number of
    interior |p| minima (mode index)."""
    ny = grid.ny
    V = v.reshape(ny, NV)
    sc = reference_scale(q, u_e, T_e)
    E = (np.abs(V * sc) ** 2).sum(axis=1)
    tot = float((E * grid.w).sum())
    kc = int(np.searchsorted(grid.y, 4.0 * d_ref))
    ke = int(np.searchsorted(grid.y, 2.0 * d_ref))
    ff = float((E[kc:] * grid.w[kc:]).sum() / max(tot, 1e-300))
    if dpdq is None:
        dpdq = pressure_gradient_matrix(q)
    ph = np.einsum("jk,jk->j", dpdq, V)
    A = np.abs(ph) / max(np.abs(ph).max(), 1e-300)
    nmin = sum(1 for j in range(1, max(ke, 2))
               if A[j] < A[j - 1] and A[j] <= A[j + 1])
    return dict(far_fraction=ff, p_wall=float(A[0]), n_pmin=int(nmin),
                p_profile=A)


def pressure_gradient_matrix(q):
    """dp/dq at every wall-normal node (ny, NV)."""
    from .constants import CS_H
    from . import thermo as th
    ny = q.shape[1]
    out = np.zeros((ny, NV))
    qc = np.asarray(q, dtype=complex)
    for k in range(NV):
        p = qc.copy()
        p[k] = p[k] + 1j * CS_H
        out[:, k] = np.imag(th.pressure(p[:NS], p[I_T], p[I_TV])) / CS_H
    return out


def find_second_mode(ops, bcop, beta, omega, u_e, T_e, d_ref, c_guess=0.92,
                     growth_guess=0.02, k=25, ff_max=1e-3, refine_it=30,
                     alpha_guess=None, q_guess=None):
    """Locate the trapped discrete mode (Mack's second mode) at one station.

    Strategy: shift-invert Arnoldi on the alpha-quadratic pencil with the
    far-field Riccati rows frozen at the shift (this suppresses the
    continuous spectra), structural filtering, then bordered Newton with the
    fully alpha-dependent far-field condition.
    """
    dpdq = pressure_gradient_matrix(ops.q)
    if alpha_guess is None:
        alpha_guess = omega / (c_guess * u_e) * (1.0 - 1j * growth_guess)
    best = None
    if q_guess is None:
        alphas, vecs = spatial_modes(ops, bcop, beta, omega, alpha_guess, k=k,
                                     far_alpha=alpha_guess)
        for j in range(len(alphas)):
            a = alphas[j]
            if not np.isfinite(a) or a.real <= 0:
                continue
            c = omega / a.real / u_e
            if not (0.6 <= c <= 1.1):
                continue
            d = mode_diagnostics(vecs[:, j], ops.grid, ops.q, u_e, T_e, d_ref,
                                 dpdq)
            if d["far_fraction"] > ff_max or d["p_wall"] < 0.3:
                continue
            score = d["far_fraction"] + abs(a - alpha_guess) / abs(alpha_guess)
            if best is None or score < best[0]:
                best = (score, a, vecs[:, j], d)
        if best is None:
            return None
        a0, v0 = best[1], best[2]
    else:
        a0, v0 = alpha_guess, q_guess

    a, v, info = newton_alpha(ops, bcop, a0, beta, omega, q0=v0,
                              itmax=refine_it, u_e=u_e, T_e=T_e)
    vp = info["scaling"].unscale_vector(v)
    d = mode_diagnostics(vp, ops.grid, ops.q, u_e, T_e, d_ref, dpdq)
    d.update(alpha=a, converged=info["converged"], res=info["res"],
             c_phase=omega / a.real / u_e)
    return a, vp, d


def scale_vector(q):
    """Normalise an eigenfunction so that max |u_hat| = 1."""
    u = q[I_U::NV]
    k = int(np.argmax(np.abs(u)))
    return q / u[k]
