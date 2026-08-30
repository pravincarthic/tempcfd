"""Wall-normal grids and differentiation matrices."""
import numpy as np


def fornberg(z, x, m):
    """Finite-difference weights (Fornberg 1988) up to order m at point z."""
    n = len(x) - 1
    c = np.zeros((n + 1, m + 1))
    c1, c4 = 1.0, x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


def fd_matrices(y, order=4):
    """Non-uniform finite-difference D1, D2 with `order+1`-point stencils."""
    n = len(y)
    w = order + 1
    D1 = np.zeros((n, n))
    D2 = np.zeros((n, n))
    half = w // 2
    for j in range(n):
        s = min(max(j - half, 0), n - w)
        idx = np.arange(s, s + w)
        c = fornberg(y[j], y[idx], 2)
        D1[j, idx] = c[:, 1]
        D2[j, idx] = c[:, 2]
    return D1, D2


def chebyshev(n):
    """Chebyshev-Gauss-Lobatto nodes on [-1,1] (descending) and D matrix."""
    x = np.cos(np.pi * np.arange(n) / (n - 1))
    c = np.ones(n)
    c[0] = c[-1] = 2.0
    c = c * (-1.0) ** np.arange(n)
    X = np.tile(x, (n, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n))
    D -= np.diag(D.sum(axis=1))
    return x, D


class Grid:
    """Wall-normal grid with an algebraic wall-clustering map.

    y(eta) = a*eta/(b - eta),  eta in [0,1],  a = y_i*y_max/(y_max - 2*y_i),
    b = 1 + a/y_max, so that half the points lie below y_i.
    """

    def __init__(self, ny, y_max, y_half, kind="fd4", fd_order=4):
        self.ny, self.y_max, self.y_half, self.kind = ny, y_max, y_half, kind
        if y_half >= 0.5 * y_max:
            raise ValueError("y_half must be < y_max/2")
        a = y_half * y_max / (y_max - 2.0 * y_half)
        b = 1.0 + a / y_max
        if kind == "chebyshev":
            xc, Dc = chebyshev(ny)               # xc descends 1 -> -1
            eta = 0.5 * (1.0 - xc)               # ascends 0 -> 1
            Deta = -2.0 * Dc
            self.y = a * eta / (b - eta)
            detady = (b - eta) ** 2 / (a * b)
            self.D1 = detady[:, None] * Deta
            self.D2 = self.D1 @ self.D1
            self.bandwidth = ny
        else:
            eta = np.linspace(0.0, 1.0, ny)
            self.y = a * eta / (b - eta)
            self.D1, self.D2 = fd_matrices(self.y, order=fd_order)
            self.bandwidth = fd_order // 2
        self.w = self._weights()

    def _weights(self):
        y = self.y
        w = np.zeros_like(y)
        w[0] = 0.5 * (y[1] - y[0])
        w[-1] = 0.5 * (y[-1] - y[-2])
        w[1:-1] = 0.5 * (y[2:] - y[:-2])
        return w

    def integrate(self, f, axis=-1):
        return np.tensordot(f, self.w, axes=([axis], [0]))
