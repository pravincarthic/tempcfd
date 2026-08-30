"""End-to-end driver: base flow -> mode sweep -> N-factor envelope."""
import argparse
import os
import sys
import time
import numpy as np

from . import config as cfgmod
from . import fluxes as fx
from . import io as pio
from . import baseflow as bfm
from . import species as sp
from .atmosphere import freestream
from .bcs import WallBC
from .gasdynamics import taylor_maccoll, oblique_shock
from .parallel import Parallel
from .pse import PSESolver, PSEOptions


def edge_conditions(cfg):
    c = cfg["case"]
    fs = freestream(c["mach"], z_ft=c["altitude_ft"],
                    Y=cfg["baseflow"]["freestream_Y"])
    ang = np.radians(c["half_angle_deg"])
    if c["geometry"] == "cone":
        s = taylor_maccoll(c["mach"], ang)
        Me, Tr, pr = s["M_cone"], s["T_ratio"], s["p_ratio"]
    elif c["geometry"] == "wedge":
        _, Me, pr, _, Tr = oblique_shock(c["mach"], ang)
    else:
        Me, Tr, pr = c["mach"], 1.0, 1.0
    return fs, dict(M_e=Me, T_e=fs["T"] * Tr, p_e=fs["p"] * pr)


def build_baseflow(cfg, verbose=True):
    b = cfg["baseflow"]
    x = np.linspace(b["x_start"], b["x_end"], int(b["n_stations"]))
    if b["source"] == "openfoam":
        bf = bfm.foam_baseflow(
            b["foam_case"], x, ny=int(b["ny"]), y_max=b["y_max"],
            y_half=b["y_half"], wall_patches=tuple(b["wall_patches"]),
            plane=b["plane"], axis=tuple(b["axis"]), time=b["foam_time"],
            field_map=b["field_map"], smooth=int(b["smooth_passes"]),
            y_max_factor=b["y_max_factor"], y_half_factor=b["y_half_factor"],
            quiet=not verbose)
    else:
        fs, e = edge_conditions(cfg)
        bf = bfm.similarity_baseflow(
            x, e["M_e"], e["T_e"], e["p_e"], b["freestream_Y"],
            Tw=b["wall_temperature"], adiabatic=bool(b["adiabatic_wall"]),
            cone=(cfg["case"]["geometry"] == "cone"),
            theta_c=np.radians(cfg["case"]["half_angle_deg"]),
            ny=int(b["ny"]), y_max=b["y_max"], y_half=b["y_half"])
    return bf


def make_solver(cfg, bf, comm=None):
    p = cfg["physics"]
    model = fx.Model(viscous=p["viscous"], reacting=p["reacting"],
                     thermal_noneq=p["thermal_nonequilibrium"],
                     Le=p["lewis"], Sc=p["schmidt"],
                     low_T_blend=p["low_T_blend"], diffusion=p["diffusion"])
    w = cfg["wall_bc"]
    wbc = WallBC(thermal=w["thermal"], vibrational=w["vibrational"],
                 catalytic=w["catalytic"])
    o = cfg["pse"]
    opts = PSEOptions(alpha_tol=o["alpha_tol"], alpha_iter=int(o["alpha_iter"]),
                      order=int(o["order"]), nonparallel=o["nonparallel"],
                      far=o["far_field"], min_step_factor=o["min_step_factor"],
                      stabilize=o["stabilize"], store_modes=o["store_modes"],
                      species_floor=cfg["physics"]["species_floor"],
                      energy_norm=o["energy_norm"])
    return PSESolver(bf, model, wbc, opts, comm=comm)


def run(cfg, verbose=True):
    par = Parallel(group_size=int(cfg["parallel"]["group_size"]))
    os.environ.setdefault("OMP_NUM_THREADS",
                          str(cfg["parallel"]["threads"]))
    t0 = time.time()
    bf = build_baseflow(cfg, verbose=verbose and par.rank == 0)
    par.log("[pse] base flow: %d stations x %d wall-normal points, "
            "delta99(end)=%.4e m, Re_x(end)=%.3e"
            % (bf.nx, bf.ny, bf.edge["delta99"][-1], bf.edge["Re_x"][-1]))
    par.log("[pse] parallel: %s" % par.summary())

    solver = make_solver(cfg, bf, comm=par.group if par.group_size > 1 else None)
    modes = cfgmod.mode_list(cfg)
    order = cfgmod.schedule_order(modes, cfg["modes"].get("schedule",
                                                          "center_out"))
    par.log("[pse] %d modes over %d ranks (%d groups of %d)"
            % (len(modes), par.size, par.n_groups, par.group_size))
    if len(modes) < 3 * par.n_groups:
        par.log("[pse] note: only %.1f modes per group; expect scheduling "
                "tail imbalance.  Use more frequencies/beta or a larger "
                "parallel.group_size."
                % (len(modes) / max(par.n_groups, 1)))

    i0 = int(cfg["pse"]["i_start"])
    check = bool(cfg["pse"]["mode_check"])
    cg = float(cfg["modes"]["c_guess"])

    def work(kk):
        k = order[kk]
        f, beta = modes[k]
        try:
            r = solver.march(f, beta, i0=i0, c_guess=cg, mode_check=check)
        except Exception as exc:                      # keep the sweep alive
            from .pse import PSEResult
            r = PSEResult()
            r.f_hz, r.beta, r.ok = f, beta, False
            r.message = "exception: %r" % (exc,)
            r.finalize()
        return r.to_dict()

    def progress(idx, dt):
        if par.rank == 0:
            print("[pse] mode %d/%d  f=%.1f kHz  %.1f s"
                  % (idx + 1, len(modes), modes[order[idx]][0] / 1e3, dt),
                  flush=True)

    local = par.map_modes(len(modes), work, progress=progress)
    local = [(order[k], r, dt) for k, r, dt in local]
    merged = par.gather_results(local)

    if par.rank == 0:
        out = cfg["output"]
        os.makedirs(out["directory"], exist_ok=True)
        path = os.path.join(out["directory"], out["file"])
        meta = dict(config=cfg, wall_time=time.time() - t0,
                    ranks=par.size, n_modes=len(modes),
                    species=sp.NAMES)
        pio.save_modes(path, merged, meta=meta)
        xg, res = pio.envelope(merged)
        if xg is not None:
            env, fb = res
            pio.write_envelope_csv(
                os.path.join(out["directory"], out["envelope_csv"]),
                xg, env, fb, meta=dict(case=cfg["case"]["name"]))
            k = int(np.argmax(env))
            par.log("[pse] N_max = %.3f at x = %.4f m (f = %.1f kHz)"
                    % (env[k], xg[k], (fb[k] or 0.0) / 1e3))
        if out["save_baseflow"]:
            np.savez_compressed(
                os.path.join(out["directory"], "baseflow.npz"),
                x=bf.x, y=bf.grid.y, q=bf.q,
                **{k: v for k, v in bf.edge.items()})
        par.log("[pse] done in %.1f s -> %s" % (time.time() - t0,
                                                out["directory"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pse", description="PSE second-Mack-mode solver "
                                "(11-species Park two-temperature air)")
    ap.add_argument("config", nargs="?", help="JSON configuration")
    ap.add_argument("-o", "--output", help="override output directory")
    ap.add_argument("--foam-case", help="override the OpenFOAM case directory")
    ap.add_argument("--nf", type=int, help="override the number of frequencies")
    ap.add_argument("--group-size", type=int, help="ranks per mode")
    ap.add_argument("--dump-config", help="write the resolved config and exit")
    a = ap.parse_args(argv)
    over = {}
    if a.output:
        over.setdefault("output", {})["directory"] = a.output
    if a.foam_case:
        over.setdefault("baseflow", {}).update(source="openfoam",
                                               foam_case=a.foam_case)
    if a.nf:
        over.setdefault("modes", {})["n_frequencies"] = a.nf
    if a.group_size:
        over.setdefault("parallel", {})["group_size"] = a.group_size
    cfg = cfgmod.load(a.config, overrides=over)
    if a.dump_config:
        cfgmod.dump(cfg, a.dump_config)
        return 0
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
