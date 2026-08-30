"""Assembly of the linearised stability operators.

Disturbance ansatz
    q'(x,y,z,t) = qhat(x,y) exp( i \\int alpha dx + i beta z - i omega t )

Linearising R = dU/dt + dF/dx + dG/dy + dH/dz - S about the base state and
parabolising (dropping d2/dx2 and d2/dxdy of the amplitude) gives

    ( L0 + L1 d/dy + L2 d2/dy2 ) qhat + M0 dqhat/dx = 0

with the coefficient matrices built below from the flux Jacobians.  Setting
nonparallel=False and M0=0 recovers the quasi-parallel LST operator, which is
quadratic in alpha.
"""
import numpy as np
import scipy.sparse as sps
from . import fluxes as fx
from . import jacobians as jac

NV = fx.NV
NS = fx.NS
I_U, I_V, I_W, I_T, I_TV = fx.I_U, fx.I_V, fx.I_W, fx.I_T, fx.I_TV


class StationOperators:
    """Jacobians and their derivatives at one streamwise station."""

    def __init__(self, q, qx, qy, qz, model, grid, J_prev=None, dx=None,
                 metric=None, comm=None):
        self.grid = grid
        self.q = q
        ny = grid.ny
        if metric is None:
            metric = (np.zeros(ny), np.zeros(ny))
        self.mx = np.asarray(metric[0]).reshape(ny, 1, 1)
        self.my = np.asarray(metric[1]).reshape(ny, 1, 1)
        self.J = jac.flux_jacobians(q, qx, qy, qz, model, comm=comm)
        self.dJdy = jac.dcoef_dy(self.J, grid.D1)
        if J_prev is not None and dx is not None and dx > 0.0:
            self.dJdx = {k: (self.J[k] - J_prev[k]) / dx for k in self.J}
        else:
            self.dJdx = {k: np.zeros_like(v) for k, v in self.J.items()}

    # ---------------------------------------------------------------- pieces
    def coefficients(self, alpha, beta, omega, nonparallel=True, dalpha_dx=0.0):
        J, Jy, Jx = self.J, self.dJdy, self.dJdx
        ia, ib = 1j * alpha, 1j * beta
        L0 = (-1j * omega * J["Gam"] - J["E"]
              + ia * J["A"] - alpha ** 2 * J["Ax"] - alpha * beta * J["Az"]
              + ib * J["C"] - alpha * beta * J["Cx"] - beta ** 2 * J["Cz"]
              + Jy["B"] + ia * Jy["Bx"] + ib * Jy["Bz"])
        L1 = (J["B"] + ia * J["Ay"] + ia * J["Bx"] + ib * J["Cy"] + Jy["By"])
        L2 = J["By"].copy()
        M0 = J["A"] + ia * J["Ax"] + ib * J["Az"] + ib * J["Cx"] + Jy["Bx"]
        if nonparallel:
            L0 = L0 + Jx["A"] + ia * Jx["Ax"] + ib * Jx["Az"] \
                + 1j * dalpha_dx * J["Ax"]
            L1 = L1 + Jx["Ay"]
            M0 = M0 + Jx["Ax"]
        # axisymmetric divergence metric:  R += (r_x/r) F + (r_y/r) G
        mx, my = self.mx, self.my
        if np.any(mx) or np.any(my):
            L0 = L0 + mx * (J["A"] + ia * J["Ax"] + ib * J["Az"]) \
                + my * (J["B"] + ia * J["Bx"] + ib * J["Bz"])
            L1 = L1 + mx * J["Ay"] + my * J["By"]
            M0 = M0 + mx * J["Ax"] + my * J["Bx"]
        return L0, L1, L2, M0

    def lst_pencil(self, beta, omega):
        """Quasi-parallel operator split as L^0 + alpha L^1 + alpha^2 L^2."""
        J, Jy = self.J, self.dJdy
        D1, D2 = self.grid.D1, self.grid.D2
        ib = 1j * beta
        P0 = (-1j * omega * J["Gam"] - J["E"] + ib * J["C"]
              - beta ** 2 * J["Cz"] + Jy["B"] + ib * Jy["Bz"])
        P0d1 = J["B"] + ib * J["Cy"] + Jy["By"]
        P1 = 1j * J["A"] + 1j * Jy["Bx"] - beta * J["Az"] - beta * J["Cx"]
        P1d1 = 1j * J["Ay"] + 1j * J["Bx"]
        mx, my = self.mx, self.my
        if np.any(mx) or np.any(my):
            P0 = P0 + mx * (J["A"] + ib * J["Az"]) + my * (J["B"] + ib * J["Bz"])
            P0d1 = P0d1 + mx * J["Ay"] + my * J["By"]
            P1 = P1 + 1j * (mx * J["Ax"] + my * J["Bx"])
        P2 = -J["Ax"]
        A0 = _assemble(P0, P0d1, J["By"], D1, D2)
        A1 = _assemble(P1, P1d1, None, D1, D2)
        A2 = _assemble(P2, None, None, D1, D2)
        return A0, A1, A2


    def farfield_coefficients(self, alpha, beta, omega):
        """(L0, L1, L2) of the constant-coefficient operator at y_max."""
        L0, L1, L2, _ = self.coefficients(alpha, beta, omega,
                                          nonparallel=False)
        return L0[-1], L1[-1], L2[-1]

    def temporal_pencil(self, alpha, beta):
        """Quasi-parallel operator at fixed (alpha, beta), linear in omega:
        A_a qhat = i omega Gam_g qhat."""
        J, Jy = self.J, self.dJdy
        D1, D2 = self.grid.D1, self.grid.D2
        ia, ib = 1j * alpha, 1j * beta
        P0 = (-J["E"] + ia * J["A"] - alpha ** 2 * J["Ax"]
              - alpha * beta * J["Az"] + ib * J["C"] - alpha * beta * J["Cx"]
              - beta ** 2 * J["Cz"] + Jy["B"] + ia * Jy["Bx"] + ib * Jy["Bz"])
        P0d1 = (J["B"] + ia * J["Ay"] + ia * J["Bx"] + ib * J["Cy"] + Jy["By"])
        mx, my = self.mx, self.my
        if np.any(mx) or np.any(my):
            P0 = P0 + mx * (J["A"] + ia * J["Ax"] + ib * J["Az"]) \
                + my * (J["B"] + ia * J["Bx"] + ib * J["Bz"])
            P0d1 = P0d1 + mx * J["Ay"] + my * J["By"]
        A_a = _assemble(P0, P0d1, J["By"], D1, D2)
        Gam = _assemble(J["Gam"], None, None, D1, D2)
        return A_a, Gam


def _assemble(L0, L1, L2, D1, D2, sparse=True):
    """Build the (ny*NV) matrix  L0 + L1 d/dy + L2 d2/dy2."""
    ny = D1.shape[0]
    pat = np.zeros((ny, ny), dtype=bool)
    if L0 is not None:
        pat |= np.eye(ny, dtype=bool)
    if L1 is not None:
        pat |= D1 != 0.0
    if L2 is not None:
        pat |= D2 != 0.0
    jj, kk = np.nonzero(pat)
    blocks = np.zeros((jj.size, NV, NV), dtype=complex)
    if L0 is not None:
        d = jj == kk
        blocks[d] += L0[jj[d]]
    if L1 is not None:
        blocks += D1[jj, kk][:, None, None] * L1[jj]
    if L2 is not None:
        blocks += D2[jj, kk][:, None, None] * L2[jj]
    r = (jj[:, None, None] * NV + np.arange(NV)[None, :, None])
    c = (kk[:, None, None] * NV + np.arange(NV)[None, None, :])
    r = np.broadcast_to(r, blocks.shape).ravel()
    c = np.broadcast_to(c, blocks.shape).ravel()
    n = ny * NV
    M = sps.coo_matrix((blocks.ravel(), (r, c)), shape=(n, n)).tocsr()
    return M if sparse else M.toarray()


def assemble(L0, L1, L2, grid, sparse=True):
    return _assemble(L0, L1, L2, grid.D1, grid.D2, sparse=sparse)
