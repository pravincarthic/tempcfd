#!/usr/bin/env python3
"""Plot growth rates, N-factor curves and the envelope from a results file."""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))


def load(path):
    if path.endswith((".h5", ".hdf5")):
        import h5py
        out = []
        with h5py.File(path, "r") as f:
            for k in sorted(f):
                g = f[k]
                d = {kk: np.array(g[kk]) for kk in g}
                d.update({kk: g.attrs[kk] for kk in g.attrs})
                out.append((int(k.split("_")[1]), d, float(g.attrs["seconds"])))
        return out
    z = np.load(path, allow_pickle=True)
    keys = sorted({k.split("/")[0] for k in z.files if "/" in k})
    out = []
    for k in keys:
        d = {kk.split("/", 1)[1]: z[kk]
             for kk in z.files if kk.startswith(k + "/")}
        out.append((int(k.split("_")[1]), d, float(d.get("seconds", 0.0))))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("-o", "--out", default="pse_summary.png")
    a = ap.parse_args()
    recs = load(a.results)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    xs, Ns = [], []
    for _, r, _ in recs:
        x = np.atleast_1d(r["x"])
        if x.size < 2:
            continue
        ax[0].plot(x, -np.imag(r["alpha"]), lw=0.8)
        ax[1].plot(x, r["N"], lw=0.8)
        xs.append(x)
        Ns.append(np.asarray(r["N"]))
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel(r"$-\alpha_i$ [1/m]")
    ax[0].set_title("spatial growth rate"); ax[0].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("N")
    ax[1].set_title("N-factor")
    if xs:
        grid = np.linspace(min(x[0] for x in xs), max(x[-1] for x in xs), 400)
        env = np.full_like(grid, -np.inf)
        for x, N in zip(xs, Ns):
            env = np.maximum(env, np.interp(grid, x, N, left=-np.inf,
                                            right=-np.inf))
        env[~np.isfinite(env)] = 0.0
        ax[2].plot(grid, env, "k", lw=2)
    ax[2].set_xlabel("x [m]"); ax[2].set_ylabel("N envelope")
    ax[2].set_title("envelope")
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
