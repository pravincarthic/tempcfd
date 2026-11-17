"""Batched block-banded linear algebra for the lockstep PSE march.

Every PSE system couples a wall-normal node j only to nodes j-4..j+4 (FD4
stencils, one-sided at the boundaries; wall, trace and far-field rows stay
inside the same stencil).  Grouping m = 4 consecutive nodes into a
super-block of size S = 4 NV makes the matrix block TRIDIAGONAL in
super-blocks.  It is solved by banded LU with partial pivoting at
super-block granularity (LAPACK gbtrf in blocked form):

    step k :  W = [ D_k  U_k  0    | r_k     ]      (2S x 3S+1)
                  [ L_k+1 D_k+1 U_k+1 | r_k+1 ]
              panel LU with row pivoting over the 2S x S panel   (CUDA kernel)
              T_top <- L11^-1 P T_top                             (batched solve)
              T_bot <- T_bot - L21 T_top                          (batched GEMM)

then a block back-substitution.  Partial pivoting over the full 2S rows
keeps it as stable as the banded LU in LAPACK; only the pivot order differs
from SuperLU, so solutions agree to rounding.  Work per solve is
O(n_blocks S^3) instead of O((ny NV)^3) for the dense LU, and all of it is
batched over the modes marched in lockstep.  The GEMMs go to cuBLAS (FP64
tensor cores on A100); the panel factorisation is one CUDA thread block per
batch member.

All functions take the array module `xp` (numpy or cupy).
"""
import numpy as np

M_NODES = 4

_PANEL_SRC = r"""
#include <cupy/complex.cuh>
extern "C" __global__
void panel_lu(complex<double>* W, int* perm, const int R, const int C)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    complex<double>* A = W + (size_t)b * R * C;
    int* P = perm + (size_t)b * R;
    __shared__ double smax[256];
    __shared__ int sidx[256];
    for (int r = tid; r < R; r += nt) P[r] = r;
    __syncthreads();
    for (int j = 0; j < C; ++j) {
        double best = -1.0; int bi = R;
        for (int r = j + tid; r < R; r += nt) {
            double v = abs(A[(size_t)r * C + j]);
            if (v > best || (v == best && r < bi)) { best = v; bi = r; }
        }
        smax[tid] = best; sidx[tid] = bi;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) {
                double v = smax[tid + s]; int i = sidx[tid + s];
                if (v > smax[tid] || (v == smax[tid] && i < sidx[tid])) {
                    smax[tid] = v; sidx[tid] = i;
                }
            }
            __syncthreads();
        }
        const int p = sidx[0];
        __syncthreads();
        if (p != j) {
            for (int c = tid; c < C; c += nt) {
                complex<double> t = A[(size_t)j * C + c];
                A[(size_t)j * C + c] = A[(size_t)p * C + c];
                A[(size_t)p * C + c] = t;
            }
            if (tid == 0) { int t = P[j]; P[j] = P[p]; P[p] = t; }
        }
        __syncthreads();
        const complex<double> piv = A[(size_t)j * C + j];
        for (int r = j + 1 + tid; r < R; r += nt) A[(size_t)r * C + j] /= piv;
        __syncthreads();
        const int w = C - j - 1;
        const int tot = (R - j - 1) * w;
        for (int e = tid; e < tot; e += nt) {
            const int r = j + 1 + e / w;
            const int c = j + 1 + e % w;
            A[(size_t)r * C + c] -= A[(size_t)r * C + j] * A[(size_t)j * C + c];
        }
        __syncthreads();
    }
}
"""

_KERNEL = {"fn": None, "checked": False, "ok": False}


def _is_cupy(xp):
    return xp.__name__.startswith("cupy")


def panel_lu_reference(xp, W):
    """Row-pivoted LU of the first C columns of W (B, R, C), in place over
    the panel only.  Returns (W, perm) with perm the composed row order."""
    B, R, C = W.shape
    ar = xp.arange(B)
    perm = xp.broadcast_to(xp.arange(R), (B, R)).copy()
    for j in range(C):
        p = j + xp.argmax(xp.abs(W[:, j:, j]), axis=1)
        rj = W[ar, j].copy()
        W[ar, j] = W[ar, p]
        W[ar, p] = rj
        pj = perm[ar, j].copy()
        perm[ar, j] = perm[ar, p]
        perm[ar, p] = pj
        W[:, j + 1:, j] = W[:, j + 1:, j] / W[:, j, j][:, None]
        W[:, j + 1:, j + 1:] -= W[:, j + 1:, j, None] * W[:, j, None, j + 1:]
    return W, perm


def _panel_lu_cuda(xp, W):
    k = _KERNEL
    if k["fn"] is None:
        k["fn"] = xp.RawKernel(_PANEL_SRC, "panel_lu")
    B, R, C = W.shape
    W = xp.ascontiguousarray(W, dtype=xp.complex128)
    perm = xp.empty((B, R), dtype=xp.int32)
    k["fn"]((B,), (256,), (W, perm, np.int32(R), np.int32(C)))
    return W, perm.astype(xp.int64)


def panel_lu(xp, W):
    if not _is_cupy(xp):
        return panel_lu_reference(xp, W)
    k = _KERNEL
    if not k["checked"]:
        # verify the CUDA kernel once against the array-op reference
        rng = np.random.default_rng(7)
        T = rng.normal(size=(3, W.shape[1], W.shape[2])) + \
            1j * rng.normal(size=(3, W.shape[1], W.shape[2]))
        a, pa = panel_lu_reference(xp, xp.asarray(T))
        try:
            b_, pb = _panel_lu_cuda(xp, xp.asarray(T))
            k["ok"] = bool(xp.allclose(a, b_, rtol=1e-12, atol=1e-12)) and \
                bool((pa == pb).all())
        except Exception:
            k["ok"] = False
        k["checked"] = True
        if not k["ok"]:
            import warnings
            warnings.warn("CUDA panel kernel disagreed with the reference; "
                          "using array ops", RuntimeWarning)
    return _panel_lu_cuda(xp, W) if k["ok"] else panel_lu_reference(xp, W)


def _unit_lower(xp, A):
    L = xp.tril(A, -1)
    n = A.shape[-1]
    return L + xp.eye(n, dtype=A.dtype)


def block_tridiag_solve(xp, Lo, Dg, Up, r):
    """Solve batched block-tridiagonal systems.

    Lo, Dg, Up : (B, nb, S, S)  sub-, main and super-diagonal blocks
                 (Lo[:, 0] and Up[:, -1] are ignored)
    r          : (B, nb, S)
    returns x  : (B, nb, S)
    """
    B, nb, S, _ = Dg.shape
    dt = xp.complex128
    U11 = xp.zeros((B, nb, S, S), dtype=dt)
    T1 = xp.zeros((B, nb, S, S), dtype=dt)
    T2 = xp.zeros((B, nb, S, S), dtype=dt)
    rr = xp.zeros((B, nb, S), dtype=dt)
    D = Dg[:, 0].astype(dt)
    U = Up[:, 0].astype(dt) if nb > 1 else None
    F = xp.zeros((B, S, S), dtype=dt)
    rk = r[:, 0].astype(dt)
    for k in range(nb - 1):
        top = xp.concatenate([D, U, F, rk[:, :, None]], axis=2)
        Unext = Up[:, k + 1] if k + 1 < nb - 1 else xp.zeros_like(D)
        bot = xp.concatenate([Lo[:, k + 1], Dg[:, k + 1], Unext,
                              r[:, k + 1][:, :, None]], axis=2).astype(dt)
        Wf = xp.concatenate([top, bot], axis=1)            # (B, 2S, 3S+1)
        Pn, perm = panel_lu(xp, xp.ascontiguousarray(Wf[:, :, :S]))
        T = xp.take_along_axis(Wf[:, :, S:], perm[:, :, None], axis=1)
        L11 = _unit_lower(xp, Pn[:, :S])
        L21 = Pn[:, S:]
        Ttop = xp.linalg.solve(L11, T[:, :S])
        Tbot = T[:, S:] - xp.matmul(L21, Ttop)              # GEMM
        U11[:, k] = xp.triu(Pn[:, :S])
        T1[:, k] = Ttop[:, :, :S]
        T2[:, k] = Ttop[:, :, S:2 * S]
        rr[:, k] = Ttop[:, :, 2 * S]
        D = Tbot[:, :, :S]
        U = Tbot[:, :, S:2 * S]
        rk = Tbot[:, :, 2 * S]
    x = xp.zeros((B, nb, S), dtype=dt)
    x[:, nb - 1] = xp.linalg.solve(D, rk[:, :, None])[:, :, 0]
    for k in range(nb - 2, -1, -1):
        rhs = rr[:, k] - xp.matmul(T1[:, k], x[:, k + 1][:, :, None])[:, :, 0]
        if k + 2 < nb:
            rhs = rhs - xp.matmul(T2[:, k], x[:, k + 2][:, :, None])[:, :, 0]
        x[:, k] = xp.linalg.solve(U11[:, k], rhs[:, :, None])[:, :, 0]
    return x


class NodeBand:
    """Index maps between node-offset blocks G[:, j, o] (row node j,
    column node j+o, o = -4..4) and the super-block tridiagonal form."""

    OFF = 4

    def __init__(self, ny, nv, m=M_NODES):
        self.ny, self.nv, self.m = ny, nv, m
        self.nb = -(-ny // m)
        self.nyp = self.nb * m
        self.S = m * nv
        js, os_, Ks, ds, ps, qs = [], [], [], [], [], []
        for j in range(ny):
            for o in range(-self.OFF, self.OFF + 1):
                k = j + o
                if k < 0 or k >= ny:
                    continue
                K, p = divmod(j, m)
                K2, qq = divmod(k, m)
                d = K2 - K + 1
                if d < 0 or d > 2:
                    raise ValueError("stencil wider than a super-block")
                js.append(j); os_.append(o + self.OFF)
                Ks.append(K); ds.append(d); ps.append(p); qs.append(qq)
        self.idx = [np.array(a) for a in (js, os_, Ks, ds, ps, qs)]
        self.pad = [j for j in range(ny, self.nyp)]

    def offsets(self, D):
        """(ny, 9) values D[j, j+o] for a differentiation matrix."""
        out = np.zeros((self.ny, 2 * self.OFF + 1))
        for j in range(self.ny):
            for o in range(-self.OFF, self.OFF + 1):
                k = j + o
                if 0 <= k < self.ny:
                    out[j, o + self.OFF] = D[j, k]
        nz = np.abs(D) > 0
        width = max(abs(k - j) for j, k in zip(*np.nonzero(nz)))
        if width > self.OFF:
            raise ValueError("operator bandwidth %d exceeds %d nodes; the "
                             "batched solver needs an FD grid" % (width, self.OFF))
        return out

    def to_tridiag(self, xp, G):
        """G (B, ny, 9, nv, nv) -> Lo, Dg, Up (B, nb, S, S)."""
        B = G.shape[0]
        m, nv, nb = self.m, self.nv, self.nb
        T = xp.zeros((B, nb, 3, m, m, nv, nv), dtype=G.dtype)
        j, o, K, d, p, q = (xp.asarray(a) for a in self.idx)
        T[:, K, d, p, q] = G[:, j, o]
        if self.pad:
            I = xp.eye(nv, dtype=G.dtype)
            for jj in self.pad:
                Kp, pp = divmod(jj, m)
                T[:, Kp, 1, pp, pp] = I
        T = T.transpose(0, 1, 2, 3, 5, 4, 6).reshape(B, nb, 3, m * nv, m * nv)
        return T[:, :, 0], T[:, :, 1], T[:, :, 2]

    def vec_to_blocks(self, xp, v):
        """(B, ny*nv) -> (B, nb, S) with zero padding."""
        B = v.shape[0]
        out = xp.zeros((B, self.nyp * self.nv), dtype=v.dtype)
        out[:, :self.ny * self.nv] = v
        return out.reshape(B, self.nb, self.S)

    def blocks_to_vec(self, x):
        B = x.shape[0]
        return x.reshape(B, self.nyp * self.nv)[:, :self.ny * self.nv]


def sparse_to_offsets(A, ny, nv, off=4):
    """Scipy sparse (ny*nv)^2 -> dense node-offset array (ny, 9, nv, nv)."""
    A = A.tocoo()
    out = np.zeros((ny, 2 * off + 1, nv, nv), dtype=complex)
    j, a = np.divmod(A.row, nv)
    k, b = np.divmod(A.col, nv)
    o = k - j
    if np.any(np.abs(o) > off):
        raise ValueError("matrix wider than the node band")
    np.add.at(out, (j, o + off, a, b), A.data)
    return out


def band_matvec(xp, G, x, ny, nv, off=4):
    """y = A x for node-offset blocks G (B, ny, 9, nv, nv), x (B, ny*nv)."""
    B = G.shape[0]
    X = xp.zeros((B, ny + 2 * off, nv), dtype=xp.result_type(G, x))
    X[:, off:off + ny] = x.reshape(B, ny, nv)
    y = xp.zeros((B, ny, nv), dtype=X.dtype)
    for o in range(2 * off + 1):
        y += xp.einsum("bjac,bjc->bja", G[:, :, o], X[:, o:o + ny])
    return y.reshape(B, ny * nv)
