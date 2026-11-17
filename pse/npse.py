"""Nonlinear PSE (harmonic balance in t and z), after Krypton's NPSE.

The disturbance is a truncated double Fourier series

    q'(x,y,z,t) = sum_{m,n} qhat_mn(x,y) E_mn(x) exp(i n beta z - i m omega t),
    E_mn = exp( i int alpha_mn dx ),   qhat_{-m,-n} = conj(qhat_mn),

and every mode obeys the linear PSE operator of pse.py forced by the
nonlinear terms:

    (L_mn + c0 M_mn) qhat_mn = -M_mn (past) - NLT_mn .

The nonlinear terms are formed in flux form, so the full AIR-7 TTv physics
(equation of state, energies, transport, kinetics, relaxation) enters them
exactly, not as a quadratic truncation:

    N_X = FFT[ X(qbar + q') - X(qbar) ] / E_mn  -  (linearised X)_mn,
    X in {U, lF, G, H, E}
    NLT_mn = -i omega_m N_U + i alpha_mn N_F + dN_F/dx + D N_G
             + i beta_n N_H + N_E,

with dN_F/dx from the stored history (backward differences) and the
physical field reconstructed on a zero-padded (t, z) grid (de-aliasing).
The mode (0,0) is the mean-flow distortion; its streamwise pressure
gradient is dropped from the x-derivative operator (Krypton's 'Phatx = 0')
and it uses Neumann far-field conditions.  Fundamental modes keep the
normalisation condition; harmonics are phase-locked, alpha_mn = m Re(alpha_10).

Transport properties from an external library (Mutation++ module) enter
through their local first-order expansion, which makes the nonlinear
transport terms second-order accurate in the disturbance amplitude.
"""
import numpy as np
import scipy.sparse.linalg as spla

from . import lst
from . import backend
from . import progress
from . import operators as op
from . import fluxes as fx
from .pse import PSESolver, PSEOptions, bdf_coefficients


class NPSEOptions(PSEOptions):
    def __init__(self, n_harmonics=3, n_spanwise=0, pad=2, nl_iter=12,
                 nl_tol=1e-6, relax=1.0, mean_flow_distortion=True,
                 amplitude_measure="u_max", **kw):
        super().__init__(**kw)
        self.M = int(n_harmonics)
        self.N = int(n_spanwise)
        self.pad = int(pad)
        self.nl_iter = int(nl_iter)
        self.nl_tol = float(nl_tol)
        self.relax = float(relax)
        self.mfd = bool(mean_flow_distortion)
        self.amplitude_measure = amplitude_measure


class NPSESolver(PSESolver):
    def __init__(self, baseflow, model, wallbc=None, options=None, comm=None):
        super().__init__(baseflow, model, wallbc, options or NPSEOptions(),
                         comm)

    # ------------------------------------------------------------- modes
    def mode_set(self):
        M, N = self.opt.M, self.opt.N
        out = []
        for m in range(0, M + 1):
            for n in range(-N, N + 1):
                if m == 0 and n < 0:
                    continue
                if m == 0 and n == 0 and not self.opt.mfd:
                    continue
                out.append((m, n))
        return out

    def _grid_sizes(self):
        M, N, pad = self.opt.M, self.opt.N, self.opt.pad
        nt = max(2, int(2 ** np.ceil(np.log2(max(pad * (2 * M + 1), 2)))))
        nz = 1 if N == 0 else max(2, int(2 ** np.ceil(np.log2(pad * (2 * N + 1)))))
        return nt, nz

    # -------------------------------------------------- physical <-> modal
    def _to_physical(self, coef):
        """coef[(m,n)] -> real field (NV, ny, nt, nz) by inverse DFT,
        filling the conjugate modes."""
        nt, nz = self._grid_sizes()
        key = next(iter(coef))
        nv, ny = coef[key].shape
        S = np.zeros((nv, ny, nt, nz), dtype=complex)
        for (m, n), c in coef.items():
            S[:, :, m % nt, n % nz] += c
            if (m, n) != (0, 0):
                S[:, :, (-m) % nt, (-n) % nz] += np.conj(c)
        # f[k,l] = sum C_mn exp(-2 pi i m k/nt) exp(+2 pi i n l/nz)
        f = np.fft.fft(S, axis=2)
        f = np.fft.ifft(f, axis=3) * nz
        return f.real

    def _to_modal(self, f, modes):
        nt, nz = self._grid_sizes()
        F = np.fft.ifft(f, axis=2)                    # (1/nt) sum e^{+...}
        F = np.fft.fft(F, axis=3) / nz
        return {(m, n): F[:, :, m % nt, n % nz] for (m, n) in modes}

    # ------------------------------------------------------- nonlinear term
    def nonlinear_terms(self, i, state, Jst, D1):
        """Spectral nonlinear flux residuals N_X for every mode.

        state[(m,n)] = dict(q, qx, alpha, beta, E)."""
        bf = self.bf
        g = self.model.gas
        modes = list(state)
        q0, qx0, qy0, qz0 = self.station(i)
        metric = bf.metric(i)
        coef_q, coef_x, coef_y, coef_z = {}, {}, {}, {}
        for mn, st in state.items():
            qh = st["q"].reshape(-1, g.nv).T             # (NV, ny)
            qxh = st["qx"].reshape(-1, g.nv).T
            E = st["E"]
            coef_q[mn] = qh * E
            coef_x[mn] = (1j * st["alpha"] * qh + qxh) * E
            coef_y[mn] = (D1 @ qh.T).T * E
            coef_z[mn] = 1j * st["beta"][None, :] * qh * E
        qp = self._to_physical(coef_q)
        qxp = self._to_physical(coef_x)
        qyp = self._to_physical(coef_y)
        qzp = self._to_physical(coef_z)
        ex = (slice(None), slice(None), None, None)
        mb = metric.broadcast(3)
        g.anchor(q0[:g.ns], q0[g.i_T], q0[g.i_Tv])
        Tmin = min((q0[g.i_T][:, None, None] + qp[g.i_T]).min(),
                   (q0[g.i_Tv][:, None, None] + qp[g.i_Tv]).min())
        if not np.isfinite(Tmin) or Tmin <= 0.0:
            raise FloatingPointError(
                "NPSE disturbance drives T or Tve negative at x=%g: the "
                "amplitude is beyond what the harmonic truncation can carry "
                "(raise n_harmonics/pad or lower the seed amplitude)" % bf.x[i])
        tot = fx.fluxes(self.model, q0[ex] + qp, qx0[ex] + qxp, qy0[ex] + qyp,
                        qz0[ex] + qzp, mb)
        tot = [np.asarray(a) for a in tot] + \
            [np.asarray(fx.conservative(self.model, q0[ex] + qp))]
        ref = fx.fluxes(self.model, q0, qx0, qy0, qz0, metric)
        ref = [np.asarray(a)[ex] for a in ref] + \
            [np.asarray(fx.conservative(self.model, q0))[ex]]
        out = {}
        spec = [self._to_modal(t - r, modes) for t, r in zip(tot, ref)]
        J = Jst
        for mn, st in state.items():
            qh = st["q"].reshape(-1, g.nv)               # (ny, NV)
            qxh = st["qx"].reshape(-1, g.nv)
            qyh = D1 @ qh
            a = st["alpha"]
            b = st["beta"][:, None]
            gx = 1j * a * qh + qxh
            gz = 1j * b * qh
            lin = {}
            for key, fam in (("F", "A"), ("G", "B"), ("H", "C"), ("E", "E")):
                k0 = "Eq" if fam == "E" else fam
                lin[key] = (np.einsum("jab,jb->ja", J[k0], qh)
                            + np.einsum("jab,jb->ja", J[fam + "x"], gx)
                            + np.einsum("jab,jb->ja", J[fam + "y"], qyh)
                            + np.einsum("jab,jb->ja", J[fam + "z"], gz))
            lin["U"] = np.einsum("jab,jb->ja", J["Gam"], qh)
            E = st["E"]
            res = {}
            for k, key in enumerate(("F", "G", "H", "E", "U")):
                res[key] = spec[k][mn].T / E - lin[key]   # (ny, NV)
            out[mn] = res
        return out

    # ------------------------------------------------------------- march
    def march_nonlinear(self, f0_hz, beta0=0.0, n_az0=None, amplitudes=None,
                        i0=0, i1=None, c_guess=0.92, seeds=None):
        """amplitudes: {(m,n): A} initial max|u'|/u_e of the seeded modes,
        default {(1,0): 1e-3}.  Returns a dict of per-station histories."""
        bf, g, opt = self.bf, self.bf.grid, self.opt
        gm = self.model.gas
        i1 = bf.nx - 1 if i1 is None else i1
        amplitudes = amplitudes or {(1, 0): 1e-3}
        w0 = 2.0 * np.pi * f0_hz
        modes = self.mode_set()
        for mn in amplitudes:
            if mn not in modes:
                raise ValueError("seeded mode %s outside the mode set" % (mn,))
        D1 = g.D1
        ny, nv = g.ny, self.model.nv

        def beta_of(i, n):
            b, by, bx = self.spanwise(i, n * beta0,
                                      None if n_az0 is None else n * n_az0)
            return b, by, bx

        # --- initial station: LST eigenfunctions for the seeded modes
        ue0 = bf.edge["u_e"][i0]
        state = {}
        for mn in modes:
            m, n = mn
            b, _, _ = beta_of(i0, n)
            st = dict(q=np.zeros(ny * nv, complex), qx=np.zeros(ny * nv, complex),
                      alpha=0.0 + 0.0j, beta=b, E=1.0 + 0j, theta=0.0 + 0j,
                      seeded=mn in amplitudes, hist_q=[], hist_N=[])
            if mn in amplitudes:
                if seeds and mn in seeds:
                    a, v = seeds[mn]
                else:
                    r = self.initial_mode(m * f0_hz, n * beta0,
                                          None if n_az0 is None else n * n_az0,
                                          i0, c_guess)
                    if r is None:
                        raise RuntimeError("no LST mode for seed %s" % (mn,))
                    a, v, _ = r
                U = v.reshape(ny, nv)[:, gm.i_u]
                v = v / U[np.argmax(np.abs(U))] * amplitudes[mn] * ue0
                st.update(q=v, alpha=a)
            state[mn] = st
        self._lock_alpha(state)

        hist = dict(x=[], alpha={mn: [] for mn in modes},
                    u_max={mn: [] for mn in modes},
                    T_max={mn: [] for mn in modes},
                    p_wall={mn: [] for mn in modes},
                    N={mn: [] for mn in modes}, iters=[], nl_res=[])
        xs = [bf.x[i0]]
        self._record(hist, i0, state)
        for mn in modes:
            state[mn]["hist_q"].append(state[mn]["q"].copy())
        Jprev = None
        for i in range(i0 + 1, i1 + 1):
            dx = bf.x[i] - bf.x[i - 1]
            ops, q = self.cache.ops(i, nonparallel=opt.nonparallel)
            bc_ = self.bcop(i, ops, q)
            bc00 = self.bcop(i, ops, q, far="neumann")
            ue, Te = bf.edge["u_e"][i], bf.edge["T_e"][i]
            dpdq = lst.pressure_gradient_matrix(self.model, q)
            order = min(opt.order, len(xs))
            cb = bdf_coefficients(xs[-order:] + [bf.x[i]], order)
            # advance the phase factors with the previous alpha
            for mn, st in state.items():
                st["theta_old"] = st["theta"]
                st["alpha_old"] = st["alpha"]
                st["theta"] = st["theta"] + st["alpha"] * dx
                st["E"] = np.exp(1j * st["theta"])
                st["beta"], st["by"], st["bx"] = beta_of(i, mn[1])
                st["q_old"] = st["q"].copy()
            NLT = None
            it = 0
            res_nl = np.inf
            for it in range(opt.nl_iter):
                progress.poke()
                Nst = self.nonlinear_terms(i, state, ops.J, D1)
                NLT_new = {}
                for mn, st in state.items():
                    m, n = mn
                    Nm = Nst[mn]
                    dNF = 0.0
                    oN = min(order, len(st["hist_N"]))
                    if oN > 0:
                        cN = bdf_coefficients(xs[-oN:] + [bf.x[i]], oN)
                        past = 0.0
                        for k in range(oN):
                            past = past + cN[k] * st["hist_N"][-oN + k]
                        dNF = cN[-1] * Nm["F"] + past
                    f = (-1j * m * w0 * Nm["U"] + 1j * st["alpha"] * Nm["F"]
                         + dNF + D1 @ Nm["G"] + 1j * st["beta"][:, None] * Nm["H"]
                         + Nm["E"])
                    NLT_new[mn] = f.reshape(-1)
                    st["N_F"] = Nm["F"]
                if NLT is None or opt.relax >= 1.0:
                    NLT = NLT_new
                else:
                    NLT = {k: (1 - opt.relax) * NLT[k] + opt.relax * NLT_new[k]
                           for k in NLT}
                change = 0.0
                for mn, st in state.items():
                    m, n = mn
                    a = st["alpha"]
                    is00 = mn[0] == 0
                    L0, L1, L2, M0, M1 = ops.coefficients(
                        a, st["beta"], m * w0, nonparallel=opt.nonparallel,
                        dalpha_dx=(a - st["alpha_old"]) / dx,
                        dbeta_dx=st["bx"], beta_y=st["by"],
                        mixed=opt.mixed_derivatives)
                    if is00:
                        M0 = M0.copy()
                        M0[:, gm.i_u, :] -= dpdq
                    L = op.assemble(L0, L1, L2, g)
                    Mm = op.assemble(M0, M1, None, g)
                    s_stab = 0.0
                    if opt.stabilize > 0.0 and abs(a.real) > 0:
                        s_stab = opt.stabilize * max(0.5 / abs(a.real) - dx, 0.0)
                        Mm = Mm + s_stab * L
                    bco = bc00 if mn == (0, 0) else bc_
                    fb = None
                    if mn != (0, 0) and opt.far == "asymptotic":
                        fb = bco.far_block(*ops.farfield_coefficients(
                            a, st["beta"][-1], m * w0))
                    A = bco.apply(L + cb[-1] * Mm, far_block=fb)
                    hq = st["hist_q"]
                    past = 0.0
                    for k in range(order):
                        past = past + cb[k] * hq[-order + k]
                    rhs = -bco.apply_rhs(Mm @ past + NLT[mn])
                    sc = lst.Scaling(self.model, g, q, ue, Te).calibrate(A)
                    sol = backend.solve(sc.apply(A), sc.r * rhs)
                    qn = sc.unscale_vector(sol)
                    change = max(change, np.linalg.norm(qn - st["q"]))
                    st["q"] = qn
                    st["qx"] = cb[-1] * qn + past
                    if st["seeded"] and mn[0] > 0:
                        w2 = self._w2(q, ue, Te)
                        da = -1j * self._inner(qn, st["qx"], w2) / \
                            self._inner(qn, qn, w2)
                        st["alpha"] = st["alpha"] + da
                        st["theta"] = st["theta_old"] + 0.5 * (
                            st["alpha"] + st["alpha_old"]) * dx
                        st["E"] = np.exp(1j * st["theta"])
                self._lock_alpha(state, dx=dx)
                scale = max(np.linalg.norm(st["q"]) for st in state.values()
                            if st["seeded"])
                res_nl = change / max(scale, 1e-300)
                if res_nl < opt.nl_tol:
                    break
            for mn, st in state.items():
                st["hist_q"].append(st["q"].copy())
                st["hist_N"].append(st.get("N_F", np.zeros((ny, nv))).copy())
                if len(st["hist_q"]) > 3:
                    st["hist_q"].pop(0)
                if len(st["hist_N"]) > 3:
                    st["hist_N"].pop(0)
            xs.append(bf.x[i])
            if len(xs) > 3:
                xs.pop(0)
            hist["iters"].append(it + 1)
            hist["nl_res"].append(res_nl)
            self._record(hist, i, state)
            if opt.verbose:
                print("x %.4f iterations %d residual %.1e A10 %.3e"
                      % (bf.x[i], it + 1, res_nl,
                         hist["u_max"].get((1, 0), [np.nan])[-1]))
        out = dict(x=np.array(hist["x"]), iters=np.array(hist["iters"]),
                   nl_res=np.array(hist["nl_res"]), modes=modes,
                   f0_hz=f0_hz, beta0=beta0)
        for k in ("alpha", "u_max", "T_max", "p_wall", "N"):
            out[k] = {mn: np.array(v) for mn, v in hist[k].items()}
        return out

    def _lock_alpha(self, state, dx=None):
        ref = state.get((1, 0))
        for (m, n), st in state.items():
            if st["seeded"] and m > 0:
                continue
            a = 0.0 if (m == 0 or ref is None) else m * ref["alpha"].real
            st["alpha"] = a + 0.0j
            if dx is not None and "theta_old" in st:
                st["theta"] = st["theta_old"] + 0.5 * (st["alpha"]
                                                       + st["alpha_old"]) * dx
                st["E"] = np.exp(1j * st["theta"])

    def _record(self, hist, i, state):
        bf = self.bf
        gm = self.model.gas
        ny, nv = bf.grid.ny, self.model.nv
        ue, Te, pe = bf.edge["u_e"][i], bf.edge["T_e"][i], bf.edge["p_e"][i]
        q = self.station(i)[0]
        dpdq = lst.pressure_gradient_matrix(self.model, q)
        hist["x"].append(bf.x[i])
        for mn, st in state.items():
            V = st["q"].reshape(ny, nv) * st["E"]
            hist["alpha"][mn].append(st["alpha"])
            hist["u_max"][mn].append(float(np.abs(V[:, gm.i_u]).max() / ue))
            hist["T_max"][mn].append(float(np.abs(V[:, gm.i_T]).max() / Te))
            hist["p_wall"][mn].append(float(abs(dpdq[0] @ V[0]) / pe))
            hist["N"][mn].append(float(-st["theta"].imag))
