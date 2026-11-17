"""Linear PSE marching (and a local-LST tracker) for one Fourier mode.

    ( L0 + L1 D + L2 D^2 ) qhat + ( M0 + M1 D ) dqhat/dx = 0

is marched in x with backward differences (BDF1/BDF2 on the actual,
possibly non-uniform, station spacing).  After each solve the wavenumber is
updated from the normalisation

    < qhat , dqhat/dx > = 0

(weighted by reference values of the primitive variables) until alpha
converges.  As in Krypton, the step-size restriction dx > 1/|alpha_r| of the
parabolised system can be relaxed by the Andersson-Henningson-Hanifi
stabilisation, which replaces M by M + s L with

    s = k * max(0.5/|alpha_r| - dx, 0).

N-factors are reported both as int -alpha_i dx and with the growth of the
amplitude function,  N = int ( -alpha_i + d ln||qhat||_E / dx ) dx.
"""
import numpy as np
import scipy.sparse.linalg as spla

from . import lst
from . import operators as op
from .bcs import BCOperator, WallBC
from .linalg import make_solver
from . import backend
from . import progress


class PSEOptions:
    def __init__(self, alpha_tol=1e-8, alpha_iter=25, order=2,
                 nonparallel=True, far="asymptotic", stabilize=0.0,
                 mixed_derivatives=True, store_modes=False, verbose=False,
                 species_floor=1e-12, energy_norm="kinetic",
                 min_step_factor=1.0, trace_threshold=1e-6):
        self.alpha_tol = alpha_tol
        self.alpha_iter = alpha_iter
        self.order = order
        self.nonparallel = nonparallel
        self.far = far
        self.stabilize = stabilize
        self.mixed_derivatives = mixed_derivatives
        self.store_modes = store_modes
        self.verbose = verbose
        self.species_floor = species_floor
        self.energy_norm = energy_norm
        self.min_step_factor = min_step_factor
        self.trace_threshold = trace_threshold


class PSEResult:
    FIELDS = ("x", "alpha", "sigma", "N", "N_alpha", "amplitude", "c_phase")

    def __init__(self):
        for k in self.FIELDS:
            setattr(self, k, [])
        self.diag = []
        self.modes = []
        self.f_hz = None
        self.beta = None
        self.n_az = None
        self.ok = True
        self.message = ""

    def append(self, **kw):
        for k, v in kw.items():
            getattr(self, k).append(v)

    def finalize(self):
        for k in self.FIELDS:
            setattr(self, k, np.asarray(getattr(self, k)))
        return self

    def to_dict(self):
        d = {k: np.asarray(getattr(self, k)) for k in self.FIELDS}
        d.update(f_hz=self.f_hz, beta=self.beta, ok=self.ok,
                 message=self.message,
                 n_az=-1 if self.n_az is None else self.n_az)
        if self.modes:
            d["modes"] = np.asarray(self.modes)
        return d


def bdf_coefficients(xs, order):
    """d/dx at xs[-1] from the last order+1 points: sum c_k q_k."""
    x = np.asarray(xs[-(order + 1):], float)
    n = len(x)
    c = np.zeros(n)
    x0 = x[-1]
    for j in range(n):
        # derivative of the Lagrange basis l_j at x0
        num = 0.0
        for m in range(n):
            if m == j:
                continue
            prod = 1.0 / (x[j] - x[m])
            for k in range(n):
                if k != j and k != m:
                    prod *= (x0 - x[k]) / (x[j] - x[k])
            num += prod
        c[j] = num
    return c                                       # c[-1] multiplies q_i


class StationCache:
    """Jacobians per station, with central x-derivatives for the
    non-parallel terms."""

    def __init__(self, solver, keep=4):
        self.s = solver
        self.J = {}
        self.keep = keep
        self.store = None

    def jac(self, i):
        if self.store is not None:
            return self.store.get(i)
        if i not in self.J:
            q, qx, qy, qz = self.s.station(i)
            o = op.StationOperators(self.s.model, self.s.bf.grid, q, qx, qy,
                                    qz, self.s.bf.metric(i), comm=self.s.comm)
            self.J[i] = o.J
            for k in sorted(self.J):
                if len(self.J) > self.keep and k < i - 2:
                    del self.J[k]
        return self.J[i]

    def ops(self, i, nonparallel=True):
        bf = self.s.bf
        q, qx, qy, qz = self.s.station(i)
        J = self.jac(i)
        dJdx = None
        if nonparallel and bf.nx > 1:
            im, ip = max(i - 1, 0), min(i + 1, bf.nx - 1)
            Jm, Jp = self.jac(im), self.jac(ip)
            dx = bf.x[ip] - bf.x[im]
            dJdx = {k: (Jp[k] - Jm[k]) / dx for k in J}
        self.s.model.gas.anchor(q[:self.s.model.ns], q[self.s.model.gas.i_T],
                                q[self.s.model.gas.i_Tv])
        return op.StationOperators(self.s.model, bf.grid, q, qx, qy, qz,
                                   bf.metric(i), J=J, dJdx=dJdx), q


class PSESolver:
    def __init__(self, baseflow, model, wallbc=None, options=None, comm=None):
        self.bf = baseflow
        self.model = model
        self.comm = comm if (comm is not None and comm.size > 1) else None
        self.wallbc = wallbc or WallBC()
        self.opt = options or PSEOptions()
        self.cache = StationCache(self)

    def share_jacobians(self, par=None):
        """Precompute all station Jacobians once (per node, in shared
        memory under MPI) and reuse them for every mode."""
        from .jstore import JacobianStore
        self.cache.store = JacobianStore(self, par).build()
        return self.cache.store

    # ------------------------------------------------------------ utilities
    def station(self, i):
        q, qx, qy, qz = self.bf.station(i)
        q = np.array(q, dtype=float)
        ns = self.model.ns
        rho = q[:ns].sum(axis=0)
        q[:ns] = np.maximum(q[:ns], self.opt.species_floor * rho)
        return q, qx, qy, qz

    def _w2(self, q, ue, Te):
        return 1.0 / lst.reference_values(self.model, q, ue, Te) ** 2

    def _inner(self, a, b, w2):
        g = self.bf.grid
        A = a.reshape(g.ny, -1)
        B = b.reshape(g.ny, -1)
        return np.einsum("jv,jv,v,j->", A.conj(), B, w2, g.w)

    def energy(self, v, q):
        g = self.bf.grid
        gm = self.model.gas
        V = v.reshape(g.ny, -1)
        if self.opt.energy_norm == "umax":
            return float(np.abs(V[:, gm.i_u]).max())
        rho = q[:gm.ns].sum(axis=0)
        E = rho * (np.abs(V[:, gm.i_u]) ** 2 + np.abs(V[:, gm.i_v]) ** 2
                   + np.abs(V[:, gm.i_w]) ** 2)
        return float(np.sqrt(np.abs((E * g.w).sum())))

    def spanwise(self, i, beta, n_az):
        """beta(y), d beta/dy, d beta/dx at station i."""
        m = self.bf.metric(i)
        ny = self.bf.grid.ny
        if n_az is not None and m.axisymmetric:
            b = n_az / m.r
            return b, self.bf.grid.D1 @ b, -b * m.a_s / m.l
        return np.full(ny, float(beta)), np.zeros(ny), np.zeros(ny)

    def bcop(self, i, ops, q, far=None):
        return BCOperator(self.model, self.bf.grid, q, self.wallbc, J=ops.J,
                          far=far or self.opt.far,
                          trace_threshold=self.opt.trace_threshold)

    # ---------------------------------------------------------- initial mode
    def initial_mode(self, f_hz, beta=0.0, n_az=None, i0=0, c_guess=0.92,
                     alpha0=None, q0=None):
        bf = self.bf
        omega = 2.0 * np.pi * f_hz
        ops, q = self.cache.ops(i0, nonparallel=False)
        bc = self.bcop(i0, ops, q)
        b, by, _ = self.spanwise(i0, beta, n_az)
        e = bf.edge
        if alpha0 is None or q0 is None:
            r = lst.find_second_mode(ops, bc, b, omega, e["u_e"][i0],
                                     e["T_e"][i0], e["delta99"][i0],
                                     c_guess=c_guess, alpha_guess=alpha0,
                                     q_guess=q0, beta_y=by)
            return r
        return alpha0, q0, {}

    # ------------------------------------------------------------- the march
    def march(self, f_hz, beta=0.0, n_az=None, i0=0, i1=None, alpha0=None,
              q0=None, c_guess=0.92, mode_check=True):
        bf, g, opt = self.bf, self.bf.grid, self.opt
        i1 = bf.nx - 1 if i1 is None else i1
        omega = 2.0 * np.pi * f_hz
        res = PSEResult()
        res.f_hz, res.beta, res.n_az = f_hz, beta, n_az
        found = self.initial_mode(f_hz, beta, n_az, i0, c_guess, alpha0, q0)
        if found is None:
            res.ok = False
            res.message = "no discrete mode found at the initial station"
            return res.finalize()
        alpha, qh, dg = found
        ops, q = self.cache.ops(i0, nonparallel=False)
        ue, Te = bf.edge["u_e"][i0], bf.edge["T_e"][i0]
        w2 = self._w2(q, ue, Te)
        qh = qh / np.sqrt(np.abs(self._inner(qh, qh, w2)))
        amp = self.energy(qh, q)
        res.append(x=bf.x[i0], alpha=alpha, sigma=-alpha.imag, N=0.0,
                   N_alpha=0.0, amplitude=amp, c_phase=omega / alpha.real / ue)
        res.diag.append(dg)
        if opt.store_modes:
            res.modes.append(qh.copy())

        hist_q, hist_a, hist_x = [qh], [alpha], [bf.x[i0]]
        N = 0.0
        Na = 0.0
        for i in range(i0 + 1, i1 + 1):
            dx = bf.x[i] - bf.x[i - 1]
            ops, q = self.cache.ops(i, nonparallel=opt.nonparallel)
            bc = self.bcop(i, ops, q)
            ue, Te = bf.edge["u_e"][i], bf.edge["T_e"][i]
            w2 = self._w2(q, ue, Te)
            b, by, bx = self.spanwise(i, beta, n_az)
            order = min(opt.order, len(hist_q))
            xs = hist_x[-order:] + [bf.x[i]]
            cb = bdf_coefficients(xs, order)
            a = alpha
            qn = hist_q[-1]
            scaling = None
            s_stab = 0.0
            if opt.stabilize > 0.0:
                s_stab = opt.stabilize * max(0.5 / max(abs(alpha.real), 1e-12)
                                             - dx, 0.0)
            elif dx < opt.min_step_factor / max(abs(alpha.real), 1e-12):
                res.message = ("dx=%g < 1/|alpha_r| at x=%g: residual "
                               "ellipticity not suppressed (set "
                               "pse.stabilize)" % (dx, bf.x[i]))
            for it in range(opt.alpha_iter):
                progress.poke()
                dadx = (a - hist_a[-1]) / dx
                L0, L1, L2, M0, M1 = ops.coefficients(
                    a, b, omega, nonparallel=opt.nonparallel, dalpha_dx=dadx,
                    dbeta_dx=bx, beta_y=by, mixed=opt.mixed_derivatives)
                L = op.assemble(L0, L1, L2, g)
                M = op.assemble(M0, M1, None, g)
                if s_stab > 0.0:
                    M = M + s_stab * L
                fb = None
                if opt.far == "asymptotic":
                    fb = bc.far_block(*ops.farfield_coefficients(a, b[-1], omega))
                A = bc.apply(L + (cb[-1] / 1.0) * M, far_block=fb)
                past = 0.0
                for k in range(order):
                    past = past + cb[k] * hist_q[-order + k]
                rhs = -bc.apply_rhs(M @ past)
                if scaling is None:
                    scaling = lst.Scaling(self.model, g, q, ue, Te).calibrate(A)
                As = scaling.apply(A)
                bs = scaling.r * rhs
                sol = (make_solver(As, self.comm, g.ny, self.model.nv).solve(bs)
                       if self.comm is not None
                       else backend.solve(As, bs))
                qn = scaling.unscale_vector(sol)
                dqdx = cb[-1] * qn + past
                da = -1j * self._inner(qn, dqdx, w2) / self._inner(qn, qn, w2)
                a = a + da
                if abs(da) < opt.alpha_tol * max(abs(a), 1.0):
                    break
            amp_new = self.energy(qn, q)
            sigma = -a.imag + np.log(max(amp_new, 1e-300)
                                     / max(res.amplitude[-1], 1e-300)) / dx
            N += sigma * dx
            Na += -0.5 * (a.imag + alpha.imag) * dx
            alpha = a
            hist_q.append(qn)
            hist_a.append(a)
            hist_x.append(bf.x[i])
            if len(hist_q) > 3:
                hist_q.pop(0)
                hist_a.pop(0)
                hist_x.pop(0)
            res.append(x=bf.x[i], alpha=a, sigma=sigma, N=N, N_alpha=Na,
                       amplitude=amp_new, c_phase=omega / a.real / ue)
            dg = lst.mode_diagnostics(self.model, qn, g, q, ue, Te,
                                      bf.edge["delta99"][i])
            res.diag.append(dg)
            if opt.store_modes:
                res.modes.append(qn.copy())
            if mode_check and (dg["far_fraction"] > 1e-2 or dg["p_wall"] < 0.2):
                res.ok = False
                res.message = ("mode lost at x=%g (far_fraction=%.2e, "
                               "p_wall=%.2f)" % (bf.x[i], dg["far_fraction"],
                                                 dg["p_wall"]))
                break
            if opt.verbose:
                print("x %.4f alpha real %.4f imag %.5f sigma %.4f N %.3f"
                      % (bf.x[i], a.real, a.imag, sigma, N))
        return res.finalize()

    # ------------------------------------------------- local LST tracking
    def lst_march(self, f_hz, beta=0.0, n_az=None, i0=0, i1=None,
                  c_guess=0.92):
        """Quasi-parallel N-factor: track the second mode station by station
        with Newton continuation, N = int -alpha_i dx."""
        bf = self.bf
        i1 = bf.nx - 1 if i1 is None else i1
        omega = 2.0 * np.pi * f_hz
        res = PSEResult()
        res.f_hz, res.beta, res.n_az = f_hz, beta, n_az
        found = self.initial_mode(f_hz, beta, n_az, i0, c_guess)
        if found is None:
            res.ok = False
            res.message = "no discrete mode found at the initial station"
            return res.finalize()
        a, v, d = found
        N = 0.0
        prev = None
        for i in range(i0, i1 + 1):
            if i > i0:
                ops, q = self.cache.ops(i, nonparallel=False)
                bc = self.bcop(i, ops, q)
                b, by, _ = self.spanwise(i, beta, n_az)
                e = bf.edge
                r = lst.find_second_mode(ops, bc, b, omega, e["u_e"][i],
                                         e["T_e"][i], e["delta99"][i],
                                         alpha_guess=a, q_guess=v, beta_y=by)
                if r is None or not r[2].get("converged", False):
                    res.ok = False
                    res.message = "LST tracking lost the mode at x=%g" % bf.x[i]
                    break
                a, v, d = r
                N += -0.5 * (a.imag + prev.imag) * (bf.x[i] - bf.x[i - 1])
            ue = bf.edge["u_e"][i]
            res.append(x=bf.x[i], alpha=a, sigma=-a.imag, N=N, N_alpha=N,
                       amplitude=1.0, c_phase=omega / a.real / ue)
            res.diag.append(d)
            prev = a
        return res.finalize()
