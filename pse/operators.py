"""Linearised stability operators.

Disturbance ansatz (one Fourier mode)

    q'(x,y,z,t) = qhat(x,y) exp( i int alpha dx + i beta z - i omega t )

with beta(y) = n / r(y) for an azimuthal mode n of an axisymmetric body.
Linearising R = dU/dt + d(lF)/dx + dG/dy + dH/dz + E and parabolising
(dropping d2/dx2 of the amplitude, keeping the mixed d2/dxdy terms as
Krypton does) gives

    ( L0 + L1 D + L2 D^2 ) qhat + ( M0 + M1 D ) dqhat/dx = f

where, with F0 = A + i alpha Ax + i beta Az (and likewise G0, H0, E0),

    L0 = -i omega Gam + i alpha F0 + dF0/dx + dG0/dy + i beta H0 + E0
    L1 = i alpha Ay + dAy/dx + G0 + dBy/dy + i beta Cy + Ey
    L2 = By
    M0 = i alpha Ax + F0 + dAx/dx + dBx/dy + i beta Cx + Ex
    M1 = Ay + Bx

The x-derivatives of the Jacobians (non-parallel terms) come from the
neighbouring stations.  Dropping them and M gives the quasi-parallel LST
operator, quadratic in alpha.
"""
import numpy as np
import scipy.sparse as sps

from . import jacobians as jac


class StationOperators:
    def __init__(self, model, grid, q, qx, qy, qz, metric, J=None,
                 dJdx=None, comm=None):
        self.model = model
        self.grid = grid
        self.q = q
        self.metric = metric
        self.nv = model.nv
        if J is None:
            model.gas.anchor(q[:model.ns], q[model.gas.i_T], q[model.gas.i_Tv])
            J = jac.flux_jacobians(model, q, qx, qy, qz, metric, comm=comm)
        self.J = J
        self.dJdy = {k: np.einsum("jk,kmn->jmn", grid.D1, v)
                     for k, v in J.items()}
        self.dJdx = dJdx if dJdx is not None else \
            {k: np.zeros_like(v) for k, v in J.items()}

    # ------------------------------------------------------------ helpers
    def beta_profile(self, beta, n_az=None):
        """Spanwise wavenumber at each node and its y-derivative."""
        m = self.metric
        ny = self.grid.ny
        if n_az is not None and m.axisymmetric:
            b = n_az / m.r
            return b, self.grid.D1 @ b
        return np.full(ny, float(beta)), np.zeros(ny)

    def coefficients(self, alpha, beta, omega, nonparallel=True,
                     dalpha_dx=0.0, dbeta_dx=None, beta_y=None,
                     mixed=True):
        J, Jy, Jx = self.J, self.dJdy, self.dJdx
        ny = self.grid.ny
        b = np.broadcast_to(np.asarray(beta, dtype=float), (ny,))[:, None, None]
        by = (np.zeros((ny, 1, 1)) if beta_y is None
              else np.asarray(beta_y, float).reshape(ny, 1, 1))
        bx = (np.zeros((ny, 1, 1)) if dbeta_dx is None
              else np.asarray(dbeta_dx, float).reshape(ny, 1, 1))
        ia = 1j * alpha
        ib = 1j * b
        F0 = J["A"] + ia * J["Ax"] + ib * J["Az"]
        G0 = J["B"] + ia * J["Bx"] + ib * J["Bz"]
        H0 = J["C"] + ia * J["Cx"] + ib * J["Cz"]
        E0 = J["Eq"] + ia * J["Ex"] + ib * J["Ez"]
        dG0 = Jy["B"] + ia * Jy["Bx"] + 1j * (by * J["Bz"] + b * Jy["Bz"])
        L0 = -1j * omega * J["Gam"] + ia * F0 + dG0 + ib * H0 + E0
        L1 = ia * J["Ay"] + G0 + Jy["By"] + ib * J["Cy"] + J["Ey"]
        L2 = J["By"].copy()
        M0 = ia * J["Ax"] + F0 + Jy["Bx"] + ib * J["Cx"] + J["Ex"]
        M1 = (J["Ay"] + J["Bx"]) if mixed else None
        if nonparallel:
            L0 = L0 + Jx["A"] + ia * Jx["Ax"] + 1j * dalpha_dx * J["Ax"] \
                + 1j * (bx * J["Az"] + b * Jx["Az"])
            L1 = L1 + Jx["Ay"]
            M0 = M0 + Jx["Ax"]
        return L0, L1, L2, M0, M1

    def lst_pencil(self, beta, omega, beta_y=None):
        """Quasi-parallel operator as A0 + alpha A1 + alpha^2 A2."""
        J, Jy = self.J, self.dJdy
        ny = self.grid.ny
        D1, D2 = self.grid.D1, self.grid.D2
        b = np.broadcast_to(np.asarray(beta, dtype=float), (ny,))[:, None, None]
        by = (np.zeros((ny, 1, 1)) if beta_y is None
              else np.asarray(beta_y, float).reshape(ny, 1, 1))
        ib = 1j * b
        P00 = (-1j * omega * J["Gam"] + Jy["B"]
               + 1j * (by * J["Bz"] + b * Jy["Bz"])
               + ib * (J["C"] + ib * J["Cz"]) + J["Eq"] + ib * J["Ez"])
        P01 = (1j * J["A"] - b * J["Az"] + 1j * Jy["Bx"] - b * J["Cx"]
               + 1j * J["Ex"])
        P02 = -J["Ax"]
        P10 = J["B"] + ib * J["Bz"] + Jy["By"] + ib * J["Cy"] + J["Ey"]
        P11 = 1j * (J["Ay"] + J["Bx"])
        A0 = _assemble(P00, P10, J["By"], D1, D2)
        A1 = _assemble(P01, P11, None, D1, D2)
        A2 = _assemble(P02, None, None, D1, D2)
        return A0, A1, A2

    def farfield_coefficients(self, alpha, beta, omega):
        L0, L1, L2, _, _ = self.coefficients(alpha, beta, omega,
                                             nonparallel=False, mixed=False)
        return L0[-1], L1[-1], L2[-1]

    def temporal_pencil(self, alpha, beta):
        """A_a qhat = i omega Gam qhat (quasi-parallel, fixed alpha, beta)."""
        L0, L1, L2, _, _ = self.coefficients(alpha, beta, 0.0,
                                             nonparallel=False, mixed=False)
        D1, D2 = self.grid.D1, self.grid.D2
        return (_assemble(L0, L1, L2, D1, D2),
                _assemble(self.J["Gam"].astype(complex), None, None, D1, D2))


def _assemble(L0, L1, L2, D1, D2, sparse=True):
    """(ny*NV) matrix of  L0 + L1 d/dy + L2 d2/dy2  (node-major blocks)."""
    ny = D1.shape[0]
    nv = (L0 if L0 is not None else (L1 if L1 is not None else L2)).shape[1]
    pat = np.zeros((ny, ny), dtype=bool)
    if L0 is not None:
        pat |= np.eye(ny, dtype=bool)
    if L1 is not None:
        pat |= D1 != 0.0
    if L2 is not None:
        pat |= D2 != 0.0
    jj, kk = np.nonzero(pat)
    blocks = np.zeros((jj.size, nv, nv), dtype=complex)
    if L0 is not None:
        d = jj == kk
        blocks[d] += L0[jj[d]]
    if L1 is not None:
        blocks += D1[jj, kk][:, None, None] * L1[jj]
    if L2 is not None:
        blocks += D2[jj, kk][:, None, None] * L2[jj]
    r = (jj[:, None, None] * nv + np.arange(nv)[None, :, None])
    c = (kk[:, None, None] * nv + np.arange(nv)[None, None, :])
    r = np.broadcast_to(r, blocks.shape).ravel()
    c = np.broadcast_to(c, blocks.shape).ravel()
    n = ny * nv
    M = sps.coo_matrix((blocks.ravel(), (r, c)), shape=(n, n)).tocsr()
    return M if sparse else M.toarray()


def assemble(L0, L1, L2, grid, sparse=True):
    return _assemble(L0, L1, L2, grid.D1, grid.D2, sparse=sparse)
