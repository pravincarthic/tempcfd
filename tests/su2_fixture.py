"""Write a small SU2-NEMO case (native .su2 mesh + restart) from a BaseFlow,
in SU2's own variable layout, so the reader can be tested without SU2.

The body is a sharp cone (AXISYMMETRIC= YES: x axial, y radial); the mesh
is structured in (wall arclength, wall-normal distance).
"""
import os
import numpy as np

from pse.su2 import write_su2_mesh, write_csv, write_binary


def build_case(case, bf, ns_cells=60, nn_cells=70, binary=False,
               temperatures=True, fname=None, perturb=0.0, seed=0):
    gas = bf.gas
    os.makedirs(case, exist_ok=True)
    th = bf.meta.get("theta_c", 0.0)
    ct, st = np.cos(th), np.sin(th)
    s = np.linspace(bf.x[0], bf.x[-1], ns_cells + 1)
    t = np.linspace(0.0, 1.0, nn_cells + 1)
    n = bf.grid.y_max * t ** 2.0
    S, N = np.meshgrid(s, n, indexing="ij")
    X = S * ct - N * st
    R = S * st + N * ct
    pts = np.column_stack([X.ravel(), R.ravel()])

    def pid(i, j):
        return i * (nn_cells + 1) + j
    quads = [(pid(i, j), pid(i + 1, j), pid(i + 1, j + 1), pid(i, j + 1))
             for i in range(ns_cells) for j in range(nn_cells)]
    markers = {
        "wall": [(pid(i, 0), pid(i + 1, 0)) for i in range(ns_cells)],
        "farfield": [(pid(i, nn_cells), pid(i + 1, nn_cells))
                     for i in range(ns_cells)],
        "inlet": [(pid(0, j), pid(0, j + 1)) for j in range(nn_cells)],
        "outlet": [(pid(ns_cells, j), pid(ns_cells, j + 1))
                   for j in range(nn_cells)],
    }
    write_su2_mesh(os.path.join(case, "mesh.su2"), pts, quads, markers)

    Q = np.zeros((gas.nv, pts.shape[0]))
    for i, si in enumerate(s):
        k = int(np.clip(np.searchsorted(bf.x, si) - 1, 0, bf.nx - 2))
        w = (si - bf.x[k]) / (bf.x[k + 1] - bf.x[k])
        prof = (1 - w) * bf.q[k] + w * bf.q[k + 1]
        for v in range(gas.nv):
            Q[v, i * (nn_cells + 1):(i + 1) * (nn_cells + 1)] = \
                np.interp(n, bf.grid.y, prof[v])
    if perturb:
        rng = np.random.default_rng(seed)
        Q[gas.i_u] *= 1.0 + perturb * rng.standard_normal(Q.shape[1])
    rs = Q[:gas.ns]
    rho = rs.sum(axis=0)
    u, v = Q[gas.i_u], Q[gas.i_v]
    Vx = u * ct - v * st
    Vr = u * st + v * ct
    T, Tv = Q[gas.i_T], Q[gas.i_Tv]
    rE, rEve = gas.energies(rs, T, Tv)
    f = {"x": pts[:, 0], "y": pts[:, 1]}
    for k in range(gas.ns):
        f["Density_%d" % k] = rs[k]
    f["Momentum_x"] = rho * Vx
    f["Momentum_y"] = rho * Vr
    f["Energy"] = rE + 0.5 * rho * (Vx ** 2 + Vr ** 2)
    f["Energy_ve"] = rEve
    if temperatures:
        f["Pressure"] = gas.pressure(rs, T, Tv)
        f["Temperature_tr"] = T
        f["Temperature_ve"] = Tv
        f["Velocity_x"] = Vx
        f["Velocity_y"] = Vr
    name = fname or ("restart_flow.dat" if binary else "restart_flow.csv")
    path = os.path.join(case, name)
    (write_binary if binary else write_csv)(path, f)
    return os.path.join(case, "mesh.su2"), path
