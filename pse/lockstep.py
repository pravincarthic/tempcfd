"""Lockstep LPSE: a batch of modes (frequency, spanwise / azimuthal
wavenumber) marched together, station by station.

Same equations, boundary conditions, scaling, BDF coefficients, alpha
update and stopping rules as PSESolver.march; only the data layout changes:

* the operator of every mode is assembled directly as node-offset blocks
  G[b, j, o] (row node j, column node j+o) from the shared station
  Jacobians, by broadcasting over the batch;
* boundary rows (P projector, wall rows, trace rows, far-field Riccati rows)
  are inserted in the same block form;
* all modes are solved at once by batched block-tridiagonal LU
  (pse.batched), on the CPU with NumPy or on the GPU with CuPy;
* modes converge, stop (mode lost) and are removed individually.

The far-field characteristic eigenproblems (23 x 23, one per mode per alpha
iteration) and the mode diagnostics stay on the host.
"""
import numpy as np

from . import lst
from . import progress
from . import batched as bt
from .bcs import farfield_rows
from .pse import PSEResult, bdf_coefficients


def _xp(name=None):
    from . import backend
    if backend.name() == "gpu":
        return backend._STATE["xp"]
    return np


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class LockstepMarcher:
    def __init__(self, solver, xp=None):
        self.s = solver
        self.xp = xp or _xp()
        bf = solver.bf
        self.band = bt.NodeBand(bf.grid.ny, solver.model.nv)
        self.d1 = self.band.offsets(bf.grid.D1)
        self.d2 = self.band.offsets(bf.grid.D2)

    # ---------------------------------------------------- batched operator
    def _coefficients(self, J, Jy, Jx, a, b, by, bx, omega, dadx):
        """Batched ops.coefficients: arrays (B, ny, nv, nv)."""
        xp = self.xp
        opt = self.s.opt
        ia = (1j * a)[:, None, None, None]
        bb = b[:, :, None, None]
        ib = 1j * bb
        byy = by[:, :, None, None]
        bxx = bx[:, :, None, None]
        F0 = J["A"] + ia * J["Ax"] + ib * J["Az"]
        G0 = J["B"] + ia * J["Bx"] + ib * J["Bz"]
        H0 = J["C"] + ia * J["Cx"] + ib * J["Cz"]
        E0 = J["Eq"] + ia * J["Ex"] + ib * J["Ez"]
        dG0 = Jy["B"] + ia * Jy["Bx"] + 1j * (byy * J["Bz"] + bb * Jy["Bz"])
        om = omega[:, None, None, None]
        L0 = -1j * om * J["Gam"] + ia * F0 + dG0 + ib * H0 + E0
        L1 = ia * J["Ay"] + G0 + Jy["By"] + ib * J["Cy"] + J["Ey"]
        L2 = xp.broadcast_to(J["By"], L1.shape)
        M0 = ia * J["Ax"] + F0 + Jy["Bx"] + ib * J["Cx"] + J["Ex"]
        M1 = xp.broadcast_to(J["Ay"] + J["Bx"], L1.shape) \
            if opt.mixed_derivatives else None
        if opt.nonparallel:
            L0 = L0 + Jx["A"] + ia * Jx["Ax"] \
                + 1j * dadx[:, None, None, None] * J["Ax"] \
                + 1j * (bxx * J["Az"] + bb * Jx["Az"])
            L1 = L1 + Jx["Ay"]
            M0 = M0 + Jx["Ax"]
        return L0, L1, L2, M0, M1

    def _offsets(self, C0, C1, C2):
        """node-offset blocks of  C0 + C1 D1 + C2 D2."""
        xp = self.xp
        B, ny, nv, _ = C0.shape
        G = xp.zeros((B, ny, 9, nv, nv), dtype=complex)
        G[:, :, 4] += C0
        d1 = xp.asarray(self.d1)[None, :, :, None, None]
        if C1 is not None:
            G += d1 * C1[:, :, None]
        if C2 is not None:
            G += xp.asarray(self.d2)[None, :, :, None, None] * C2[:, :, None]
        return G

    def _far_offsets(self, bc, L0f, L1f, L2f, B):
        """Per-mode asymptotic far-field rows as node-offset blocks at the
        last node (host computation, small dense eigenproblems)."""
        ny, nv = self.s.bf.grid.ny, self.s.model.nv
        out = np.zeros((B, 9, nv, nv), dtype=complex)
        act = bc.active
        d1l = self.d1[ny - 1]
        for k in range(B):
            R0, R1, R2 = bc.reduce_far(L0f[k], L1f[k], L2f[k])
            C0, C1 = farfield_rows(R0, R1, R2, col=bc.col[act])
            blk0 = np.zeros((nv, nv), complex)
            blk1 = np.zeros((nv, nv), complex)
            blk0[np.ix_(act, act)] = C0
            blk1[np.ix_(act, act)] = C1
            out[k, 4] += blk0
            for o in range(9):
                if d1l[o] != 0.0:
                    out[k, o] += blk1 * d1l[o]
        return out

    # --------------------------------------------------------------- march
    def march(self, modes, i0=0, i1=None, c_guess=0.92, mode_check=True):
        """modes: list of (f_hz, beta, n_az).  Returns list of PSEResult."""
        s, xp = self.s, self.xp
        bf, g, opt = s.bf, s.bf.grid, s.opt
        ny, nv = g.ny, s.model.nv
        i1 = bf.nx - 1 if i1 is None else i1
        res, state = [], []
        for (f, beta, n_az) in modes:
            r = PSEResult()
            r.f_hz, r.beta, r.n_az = f, beta, n_az
            res.append(r)
            found = s.initial_mode(f, beta, n_az, i0, c_guess)
            if found is None:
                r.ok = False
                r.message = "no discrete mode found at the initial station"
                state.append(None)
                continue
            alpha, qh, dg = found
            ops, q = s.cache.ops(i0, nonparallel=False)
            ue, Te = bf.edge["u_e"][i0], bf.edge["T_e"][i0]
            w2 = s._w2(q, ue, Te)
            qh = qh / np.sqrt(np.abs(s._inner(qh, qh, w2)))
            amp = s.energy(qh, q)
            r.append(x=bf.x[i0], alpha=alpha, sigma=-alpha.imag, N=0.0,
                     N_alpha=0.0, amplitude=amp,
                     c_phase=2 * np.pi * f / alpha.real / ue)
            r.diag.append(dg)
            if opt.store_modes:
                r.modes.append(qh.copy())
            state.append(dict(alpha=alpha, hq=[qh], ha=[alpha], N=0.0,
                              Na=0.0))
        alive = [k for k in range(len(modes)) if state[k] is not None]
        hist_x = [bf.x[i0]]
        for i in range(i0 + 1, i1 + 1):
            if not alive:
                break
            progress.poke()
            self._step(i, modes, alive, state, res, hist_x, mode_check)
            hist_x.append(bf.x[i])
            if len(hist_x) > 3:
                hist_x.pop(0)
            alive = [k for k in alive if res[k].ok]
        return [r.finalize() for r in res]

    def _step(self, i, modes, alive, state, res, hist_x, mode_check):
        s, xp = self.s, self.xp
        bf, g, opt = s.bf, s.bf.grid, s.opt
        ny, nv = g.ny, s.model.nv
        B = len(alive)
        dx = bf.x[i] - bf.x[i - 1]
        ops, q = s.cache.ops(i, nonparallel=opt.nonparallel)
        bc = s.bcop(i, ops, q)
        ue, Te = bf.edge["u_e"][i], bf.edge["T_e"][i]
        w2 = s._w2(q, ue, Te)
        J = {k: xp.asarray(v) for k, v in ops.J.items()}
        Jy = {k: xp.asarray(v) for k, v in ops.dJdy.items()}
        Jx = {k: xp.asarray(v) for k, v in ops.dJdx.items()}
        P = bt.sparse_to_offsets(bc.P, ny, nv)[:, 4]          # node local
        Bw = bt.sparse_to_offsets(bc.B_wall if opt.far == "asymptotic"
                                  else bc.B, ny, nv)
        Pd, Bwd = xp.asarray(P), xp.asarray(Bw)
        sp = [s.spanwise(i, modes[k][1], modes[k][2]) for k in alive]
        b = xp.asarray(np.array([t[0] for t in sp]))
        by = xp.asarray(np.array([t[1] for t in sp]))
        bx = xp.asarray(np.array([t[2] for t in sp]))
        omega = np.array([2 * np.pi * modes[k][0] for k in alive])
        om_d = xp.asarray(omega)
        order = min(opt.order, len(state[alive[0]]["hq"]))
        cb = bdf_coefficients(hist_x[-order:] + [bf.x[i]], order)
        alpha0 = np.array([state[k]["alpha"] for k in alive])
        a = alpha0.copy()
        ha = np.array([state[k]["ha"][-1] for k in alive])
        past = sum(cb[m] * np.stack([state[k]["hq"][-order + m] for k in alive])
                   for m in range(order))
        past_d = xp.asarray(past)
        s_stab = np.zeros(B)
        if opt.stabilize > 0.0:
            s_stab = opt.stabilize * np.maximum(
                0.5 / np.maximum(np.abs(alpha0.real), 1e-12) - dx, 0.0)
        qref = lst.reference_values(s.model, q, ue, Te)
        c = xp.asarray(qref)
        c_full = xp.asarray(np.tile(qref, ny))
        r_row = None
        qn = xp.asarray(np.stack([state[k]["hq"][-1] for k in alive]))
        done = np.zeros(B, bool)
        wq = xp.asarray(np.repeat(g.w, nv) * np.tile(w2, ny))
        for it in range(opt.alpha_iter):
            progress.poke()
            dadx = (a - ha) / dx
            L0, L1, L2, M0, M1 = self._coefficients(
                J, Jy, Jx, xp.asarray(a), b, by, bx, om_d, xp.asarray(dadx))
            GL = self._offsets(L0, L1, L2)
            GM = self._offsets(M0, M1, None)
            if np.any(s_stab > 0):
                GM = GM + xp.asarray(s_stab)[:, None, None, None, None] * GL
            A = xp.einsum("jac,bjocd->bjoad", Pd, GL + cb[-1] * GM) + Bwd[None]
            if opt.far == "asymptotic":
                # parallel operator at y_max with the scalar beta(y_max),
                # exactly as the reference march
                bl = _host(b)[:, -1]
                L0h, L1h, L2h = self._far_parallel(ops, a, bl, omega)
                far = self._far_offsets(bc, L0h, L1h, L2h, B)
                A[:, ny - 1] += xp.asarray(far)
            rhs = -xp.einsum("jac,bjc->bja", Pd,
                             bt.band_matvec(xp, GM, past_d, ny, nv)
                             .reshape(B, ny, nv)).reshape(B, ny * nv)
            Ac = A * c[None, None, None, None, :]
            if r_row is None:
                rm = xp.abs(Ac).max(axis=(2, 4))              # (B, ny, nv)
                rm = xp.where(rm == 0, 1.0, rm)
                r_row = 1.0 / rm
            As = Ac * r_row[:, :, None, :, None]
            Lo, Dg, Up = self.band.to_tridiag(xp, As)
            rb = self.band.vec_to_blocks(xp, (r_row.reshape(B, -1) * rhs))
            x = bt.block_tridiag_solve(xp, Lo, Dg, Up, rb)
            sol = self.band.blocks_to_vec(x) * c_full[None]
            dqdx = cb[-1] * sol + past_d
            num = (sol.conj() * dqdx * wq[None]).sum(axis=1)
            den = (sol.conj() * sol * wq[None]).sum(axis=1)
            da = _host(-1j * num / den)
            upd = ~done
            a = np.where(upd, a + da, a)
            qn = xp.where(xp.asarray(upd)[:, None], sol, qn)
            done |= upd & (np.abs(da) < opt.alpha_tol * np.maximum(np.abs(a), 1.0))
            if done.all():
                break
        qn_h = _host(qn)
        for n, k in enumerate(alive):
            st, r = state[k], res[k]
            ak = complex(a[n])
            qk = qn_h[n]
            if opt.stabilize <= 0.0 and dx < opt.min_step_factor / max(
                    abs(st["alpha"].real), 1e-12):
                r.message = ("dx=%g < 1/|alpha_r| at x=%g: residual "
                             "ellipticity not suppressed (set "
                             "pse.stabilize)" % (dx, bf.x[i]))
            amp_new = s.energy(qk, q)
            sigma = -ak.imag + np.log(max(amp_new, 1e-300)
                                      / max(r.amplitude[-1], 1e-300)) / dx
            st["N"] += sigma * dx
            st["Na"] += -0.5 * (ak.imag + st["alpha"].imag) * dx
            st["alpha"] = ak
            st["hq"].append(qk)
            st["ha"].append(ak)
            if len(st["hq"]) > 3:
                st["hq"].pop(0)
                st["ha"].pop(0)
            r.append(x=bf.x[i], alpha=ak, sigma=sigma, N=st["N"],
                     N_alpha=st["Na"], amplitude=amp_new,
                     c_phase=omega[n] / ak.real / ue)
            dg = lst.mode_diagnostics(s.model, qk, g, q, ue, Te,
                                      bf.edge["delta99"][i])
            r.diag.append(dg)
            if opt.store_modes:
                r.modes.append(qk.copy())
            if mode_check and (dg["far_fraction"] > 1e-2 or dg["p_wall"] < 0.2):
                r.ok = False
                r.message = ("mode lost at x=%g (far_fraction=%.2e, "
                             "p_wall=%.2f)" % (bf.x[i], dg["far_fraction"],
                                               dg["p_wall"]))

    def _far_parallel(self, ops, a, bl, omega):
        """ops.farfield_coefficients for every mode (host)."""
        B = len(a)
        out = [ops.farfield_coefficients(a[k], bl[k], omega[k])
               for k in range(B)]
        return ([o[0] for o in out], [o[1] for o in out], [o[2] for o in out])
