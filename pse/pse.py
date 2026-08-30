"""Linear parabolised stability equations: streamwise marching and N-factors.

    ( L0 + L1 d/dy + L2 d2/dy2 ) qhat + M0 dqhat/dx = 0

is marched in x with a first- or second-order backward difference.  After each
solve the wavenumber is updated from the normalisation

    < qhat , dqhat/dx > = 0

(weighted by reference values of the primitive variables) until alpha
converges; this removes the residual streamwise dependence from the amplitude
function and makes the split between alpha and qhat unique.
"""
import numpy as np
import scipy.sparse.linalg as spla

from . import fluxes as fx
from . import lst
from . import operators as op
from .bcs import BCOperator
from .linalg import make_solver

NV, NS = fx.NV, fx.NS
I_U, I_V, I_W, I_T, I_TV = fx.I_U, fx.I_V, fx.I_W, fx.I_T, fx.I_TV


class PSEOptions:
    def __init__(self, alpha_tol=1e-8, alpha_iter=25, order=2,
                 nonparallel=True, far="asymptotic", min_step_factor=1.0,
                 stabilize=0.0, store_modes=False, verbose=False,
                 species_floor=1e-12, energy_norm="kinetic"):
        self.alpha_tol = alpha_tol
        self.alpha_iter = alpha_iter
        self.order = order                    # 1 or 2 (backward difference)
        self.nonparallel = nonparallel
        self.far = far
        self.min_step_factor = min_step_factor  # dx >= f / |alpha_r|
        self.stabilize = stabilize            # Li-Malik dp/dx damping in [0,1]
        self.store_modes = store_modes
        self.verbose = verbose
        self.species_floor = species_floor
        self.energy_norm = energy_norm        # kinetic | chu | umax


class PSEResult:
    def __init__(self):
        self.x = []
        self.alpha = []
        self.sigma = []
        self.N = []
        self.N_alpha = []
        self.amplitude = []
        self.c_phase = []
        self.diag = []
        self.modes = []
        self.f_hz = None
        self.beta = None
        self.ok = True
        self.message = ""

    def finalize(self):
        for k in ("x", "alpha", "sigma", "N", "N_alpha", "amplitude",
                  "c_phase"):
            setattr(self, k, np.asarray(getattr(self, k)))
        return self

    def to_dict(self):
        d = {k: np.asarray(getattr(self, k)) for k in
             ("x", "alpha", "sigma", "N", "N_alpha", "amplitude", "c_phase")}
        d.update(f_hz=self.f_hz, beta=self.beta, ok=self.ok,
                 message=self.message)
        if self.modes:
            d["modes"] = np.asarray(self.modes)
        return d


def _qref(q, u_e, T_e):
    rho = q[:NS].sum(axis=0).max()
    return np.array([rho] * NS + [u_e] * 3 + [T_e] * 2)


class PSESolver:
    def __init__(self, baseflow, model=None, wallbc=None, options=None,
                 comm=None):
        self.bf = baseflow
        self.comm = comm if (comm is not None and comm.size > 1) else None
        self.model = model or fx.Model()
        self.wallbc = wallbc or __import__("pse.bcs", fromlist=["x"]).WallBC()
        self.opt = options or PSEOptions()

    # ------------------------------------------------------------- utilities
    def _station(self, i):
        q, qx, qy, qz = self.bf.station(i)
        q = np.array(q, dtype=float)
        rho = q[:NS].sum(axis=0)
        q[:NS] = np.maximum(q[:NS], self.opt.species_floor * rho)
        return q, qx, qy, qz

    def _ops(self, i, J_prev=None, dx=None):
        q, qx, qy, qz = self._station(i)
        mx, my = self.bf.metric(i)
        return op.StationOperators(q, qx, qy, qz, self.model, self.bf.grid,
                                   J_prev=J_prev, dx=dx, metric=(mx, my),
                                   comm=self.comm), q

    def _inner(self, a, b, w2):
        g = self.bf.grid
        A = a.reshape(g.ny, NV)
        B = b.reshape(g.ny, NV)
        return np.einsum("jv,jv,v,j->", A.conj(), B, w2, g.w)

    def _energy(self, v, q, u_e):
        g = self.bf.grid
        V = v.reshape(g.ny, NV)
        rho = q[:NS].sum(axis=0)
        if self.opt.energy_norm == "umax":
            return float(np.abs(V[:, I_U]).max())
        E = rho * (np.abs(V[:, I_U]) ** 2 + np.abs(V[:, I_V]) ** 2
                   + np.abs(V[:, I_W]) ** 2)
        return float(np.sqrt(np.abs((E * g.w).sum())))

    # ---------------------------------------------------------------- driver
    def march(self, f_hz, beta=0.0, i0=0, i1=None, alpha0=None, q0=None,
              c_guess=0.92, mode_check=True):
        bf, g, opt = self.bf, self.bf.grid, self.opt
        i1 = bf.nx - 1 if i1 is None else i1
        omega = 2.0 * np.pi * f_hz
        res = PSEResult()
        res.f_hz, res.beta = f_hz, beta

        ops, q = self._ops(i0)
        bcop = BCOperator(g, q, self.wallbc)
        ue, Te, d99 = bf.edge["u_e"][i0], bf.edge["T_e"][i0], \
            bf.edge["delta99"][i0]
        if alpha0 is None:
            found = lst.find_second_mode(ops, bcop, beta, omega, ue, Te, d99,
                                         c_guess=c_guess)
            if found is None:
                res.ok = False
                res.message = "no discrete mode found at the initial station"
                return res.finalize()
            alpha, qh, dg = found
        else:
            alpha, qh, dg = alpha0, q0, {}
            if qh is None:
                alpha, qh, info = lst.newton_alpha(ops, bcop, alpha0, beta,
                                                   omega, u_e=ue, T_e=Te)
                qh = info["scaling"].unscale_vector(qh)

        w2 = 1.0 / _qref(q, ue, Te) ** 2
        qh = qh / np.sqrt(np.abs(self._inner(qh, qh, w2)))
        amp = self._energy(qh, q, ue)
        res.x.append(bf.x[i0]); res.alpha.append(alpha)
        res.sigma.append(-alpha.imag); res.N.append(0.0)
        res.N_alpha.append(0.0); res.amplitude.append(amp)
        res.c_phase.append(omega / alpha.real / ue)
        res.diag.append(dg)
        if opt.store_modes:
            res.modes.append(qh.copy())

        hist_q = [qh]
        hist_a = [alpha]
        J_prev = ops.J
        N = 0.0
        Na = 0.0
        for i in range(i0 + 1, i1 + 1):
            dx = bf.x[i] - bf.x[i - 1]
            ops, q = self._ops(i, J_prev=J_prev, dx=dx)
            bcop = BCOperator(g, q, self.wallbc)
            ue, Te = bf.edge["u_e"][i], bf.edge["T_e"][i]
            w2 = 1.0 / _qref(q, ue, Te) ** 2
            amin = opt.min_step_factor / max(abs(alpha.real), 1e-12)
            if dx < amin and opt.stabilize <= 0.0:
                res.message = ("step %g < 1/|alpha_r| = %g at x=%g; "
                               "PSE ellipticity may pollute the solution"
                               % (dx, amin, bf.x[i]))
            order = opt.order if (len(hist_q) >= 2 and opt.order == 2) else 1
            a = alpha
            qn = hist_q[-1]
            scaling = None
            for it in range(opt.alpha_iter):
                A, M0 = lst.build_operator(ops, bcop, a, beta, omega,
                                           nonparallel=opt.nonparallel,
                                           dalpha_dx=(a - hist_a[-1]) / dx,
                                           far=opt.far)
                Mg = op.assemble(M0, None, None, g)
                if order == 2:
                    A_eff = A + (1.5 / dx) * (bcop.P @ Mg)
                    rhs = bcop.P @ (Mg @ (2.0 * hist_q[-1]
                                          - 0.5 * hist_q[-2]) / dx)
                else:
                    A_eff = A + (1.0 / dx) * (bcop.P @ Mg)
                    rhs = bcop.P @ (Mg @ hist_q[-1] / dx)
                if scaling is None:
                    scaling = lst.Scaling(g, q, ue, Te).calibrate(A_eff)
                As = scaling.apply(A_eff)
                bs = scaling.r * rhs
                sol = (make_solver(As, self.comm, g.ny, NV).solve(bs)
                       if self.comm is not None
                       else spla.spsolve(As.tocsc(), bs))
                qn = scaling.unscale_vector(sol)
                dqdx = ((1.5 * qn - 2.0 * hist_q[-1] + 0.5 * hist_q[-2]) / dx
                        if order == 2 else (qn - hist_q[-1]) / dx)
                num = self._inner(qn, dqdx, w2)
                den = self._inner(qn, qn, w2)
                da = -1j * num / den
                a = a + da
                if abs(da) < opt.alpha_tol * max(abs(a), 1.0):
                    break
            nrm = np.sqrt(np.abs(self._inner(qn, qn, w2)))
            qn = qn / nrm
            amp_new = self._energy(qn, q, ue)
            sigma = -a.imag + np.log(max(amp_new, 1e-300) /
                                     max(res.amplitude[-1], 1e-300)) / dx
            N += sigma * dx
            Na += -a.imag * dx
            alpha = a
            hist_q.append(qn)
            hist_a.append(a)
            if len(hist_q) > 3:
                hist_q.pop(0)
                hist_a.pop(0)
            J_prev = ops.J
            res.x.append(bf.x[i]); res.alpha.append(a)
            res.sigma.append(sigma); res.N.append(N); res.N_alpha.append(Na)
            res.amplitude.append(amp_new)
            res.c_phase.append(omega / a.real / ue)
            dg = lst.mode_diagnostics(qn, g, q, ue, Te, bf.edge["delta99"][i])
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
                print("  x=%8.4f  alpha=%10.4f%+10.5fj  sigma=%+9.4f N=%6.3f"
                      % (bf.x[i], a.real, a.imag, sigma, N))
        return res.finalize()
