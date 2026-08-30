#!/usr/bin/env python3
"""Strong-scaling harness.

Run under mpirun/srun with a fixed mode count and report the parallel
efficiency of the mode-level scheduler, the per-rank load balance and the
throughput.  With --group-size > 1 it also exercises the wall-normal
(Schur-complement) level.
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("PSE_THREADS", "1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

import numpy as np                                                 # noqa: E402
from pse import config as C                                        # noqa: E402
from pse.driver import build_baseflow, make_solver                 # noqa: E402
from pse.parallel import Parallel                                  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config/mach10_130kft.json")
    ap.add_argument("--modes", type=int, default=None)
    ap.add_argument("--stations", type=int, default=61)
    ap.add_argument("--ny", type=int, default=121)
    ap.add_argument("--group-size", type=int, default=1)
    a = ap.parse_args()

    par = Parallel(group_size=a.group_size)
    n_modes = a.modes or par.n_groups
    cfg = C.load(a.config, overrides={
        "baseflow": {"n_stations": a.stations, "ny": a.ny},
        "modes": {"n_frequencies": n_modes},
        "parallel": {"group_size": a.group_size}})
    t_setup = time.time()
    bf = build_baseflow(cfg, verbose=False)
    solver = make_solver(cfg, bf)
    modes = C.mode_list(cfg)
    order = C.schedule_order(modes, cfg["modes"].get("schedule", "center_out"))
    t_setup = time.time() - t_setup
    par.world.Barrier()

    t0 = time.time()

    def work(kk):
        f, b = modes[order[kk]]
        try:
            return solver.march(f, b).to_dict()
        except Exception:
            return {"x": np.zeros(0)}

    local = par.map_modes(len(modes), work)
    t_run = time.time() - t0
    busy = sum(dt for _, _, dt in local)

    stats = (par.world.gather((par.rank, len(local), busy, t_run), root=0)
             if par.size > 1 else [(0, len(local), busy, t_run)])
    if par.rank == 0:
        stats = [s for s in stats if s is not None]
        lead = [s for s in stats if s[0] % par.group_size == 0]
        busies = np.array([s[2] for s in lead])
        counts = np.array([s[1] for s in lead])
        wall = max(s[3] for s in stats)
        total = busies.sum()
        print("ranks               : %d (groups=%d, group_size=%d)"
              % (par.size, par.n_groups, par.group_size))
        print("modes               : %d" % len(modes))
        print("setup time          : %.2f s" % t_setup)
        print("wall time (sweep)   : %.2f s" % wall)
        print("aggregate mode time : %.2f s" % total)
        print("modes/group         : min %d  max %d  mean %.2f"
              % (counts.min(), counts.max(), counts.mean()))
        print("busy fraction       : min %.3f  max %.3f"
              % ((busies / wall).min(), (busies / wall).max()))
        print("scheduler efficiency: %.3f" % (total / (wall * len(lead))))
        print("throughput          : %.3f modes/s (%.5f per rank)"
              % (len(modes) / wall, len(modes) / wall / par.size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
