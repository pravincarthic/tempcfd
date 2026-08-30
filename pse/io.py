"""Result output: HDF5 when h5py is present, compressed npz otherwise."""
import json
import numpy as np

try:
    import h5py
    HAVE_H5 = True
except Exception:                                     # pragma: no cover
    HAVE_H5 = False


def save_modes(path, records, meta=None):
    """records: list of (index, result_dict, seconds)."""
    meta = meta or {}
    if HAVE_H5 and path.endswith((".h5", ".hdf5")):
        with h5py.File(path, "w") as f:
            f.attrs["meta"] = json.dumps(meta, default=str)
            for idx, r, dt in records:
                g = f.create_group("mode_%05d" % idx)
                g.attrs["seconds"] = dt
                for k, v in r.items():
                    if isinstance(v, np.ndarray):
                        g.create_dataset(k, data=v, compression="gzip",
                                         compression_opts=4)
                    else:
                        g.attrs[k] = v if v is not None else "None"
        return path
    out = {"meta": json.dumps(meta, default=str)}
    for idx, r, dt in records:
        for k, v in r.items():
            out["mode_%05d/%s" % (idx, k)] = np.asarray(
                v if v is not None else np.nan)
        out["mode_%05d/seconds" % idx] = np.asarray(dt)
    np.savez_compressed(path if path.endswith(".npz") else path + ".npz", **out)
    return path


def envelope(records, x_grid=None):
    """N-factor envelope over all modes on a common x grid."""
    xs = []
    for _, r, _ in records:
        if len(np.atleast_1d(r["x"])) > 1:
            xs.append(np.asarray(r["x"]))
    if not xs:
        return None, None
    if x_grid is None:
        lo = min(x[0] for x in xs)
        hi = max(x[-1] for x in xs)
        x_grid = np.linspace(lo, hi, 400)
    env = np.full_like(x_grid, -np.inf)
    best = np.full(x_grid.shape, np.nan)
    for _, r, _ in records:
        x = np.asarray(r["x"])
        if x.size < 2:
            continue
        N = np.asarray(r["N"], dtype=float)
        Ni = np.interp(x_grid, x, N, left=-np.inf, right=-np.inf)
        m = Ni > env
        env[m] = Ni[m]
        best[m] = r["f_hz"]
    env[~np.isfinite(env)] = 0.0
    return x_grid, (env, best)


def write_envelope_csv(path, x, env, fbest, meta=None):
    with open(path, "w") as f:
        if meta:
            for k, v in meta.items():
                f.write("# %s = %s\n" % (k, v))
        f.write("x_m,N_envelope,f_most_amplified_Hz\n")
        for i in range(len(x)):
            f.write("%.8e,%.6f,%.6e\n" % (x[i], env[i], fbest[i]))
    return path
