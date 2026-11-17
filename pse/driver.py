"""End-to-end driver: gas model, base flow (SU2-NEMO or similarity),
LST / LPSE sweeps over (frequency, spanwise or azimuthal wavenumber) with
dynamic MPI scheduling, or one NPSE run."""
import argparse
import os
import sys
import time
import numpy as np

from . import config as cfgmod
from . import fluxes as fx
from . import io as pio
from . import baseflow as bfm
from .atmosphere import freestream
from .bcs import WallBC
from .gas import make_gas
from .gasdynamics import taylor_maccoll, oblique_shock
from .parallel import Parallel
from .pse import PSESolver, PSEOptions, PSEResult
from .npse import NPSESolver, NPSEOptions
from . import backend


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


def build_gas(cfg):
    g = cfg["gas"]
    gas = make_gas(g["library"], gas_model=g["gas_model"],
                   transport=g["transport"], frozen=g["frozen"],
                   data_dir=g["mpp_data_dir"],
                   transport_fallback=g["transport_fallback"])
    if hasattr(gas, "gy_partner_bug"):
        gas.gy_partner_bug = bool(g["gy_partner_bug"])
    return gas


def build_baseflow(cfg, gas, verbose=True):
    b = cfg["baseflow"]
    x = np.linspace(b["x_start"], b["x_end"], int(b["n_stations"]))
    if b["source"] == "su2":
        from .su2 import su2_baseflow
        s = b["su2"]
        if not s["solution"]:
            raise ValueError("baseflow.su2.solution is not set")
        return su2_baseflow(
            gas, s["solution"], x, mesh=s["mesh"],
            wall_markers=tuple(s["wall_markers"]), ny=int(b["ny"]),
            y_max=b["y_max"], y_half=b["y_half"],
            y_max_factor=b["y_max_factor"], y_half_factor=b["y_half_factor"],
            plane=s["plane"], species_order=cfg["gas"]["species_order"],
            average=s["average"], smooth=int(b["smooth_passes"]),
            curve_smoothing=s["curve_smoothing"], grid_kind=b["grid"],
            quiet=not verbose, s0=s["s0"], station_axis=s["station_axis"])
    fs, e = edge_conditions(cfg)
    return bfm.similarity_baseflow(
        gas, x, e["M_e"], e["T_e"], e["p_e"], b["freestream_Y"],
        Tw=b["wall_temperature"], adiabatic=bool(b["adiabatic_wall"]),
        cone=(cfg["case"]["geometry"] == "cone"),
        theta_c=np.radians(cfg["case"]["half_angle_deg"]),
        ny=int(b["ny"]), y_max=b["y_max"], y_half=b["y_half"],
        grid_kind=b["grid"])


def _model(cfg, gas):
    p = cfg["physics"]
    return fx.Model(gas, viscous=p["viscous"], diffusion=p["diffusion"],
                    thermal_noneq=p["thermal_nonequilibrium"])


def _wall(cfg):
    w = cfg["wall_bc"]
    return WallBC(thermal=w["thermal"], vibrational=w["vibrational"],
                  catalytic=w["catalytic"])


def _pse_kw(cfg):
    o = cfg["pse"]
    p = cfg["physics"]
    return dict(alpha_tol=o["alpha_tol"], alpha_iter=int(o["alpha_iter"]),
                order=int(o["order"]), nonparallel=o["nonparallel"],
                far=o["far_field"], stabilize=o["stabilize"],
                mixed_derivatives=o["mixed_derivatives"],
                min_step_factor=o["min_step_factor"],
                store_modes=o["store_modes"], energy_norm=o["energy_norm"],
                species_floor=p["species_floor"],
                trace_threshold=p["trace_threshold"])


def make_solver(cfg, bf, comm=None):
    return PSESolver(bf, _model(cfg, bf.gas), _wall(cfg),
                     PSEOptions(**_pse_kw(cfg)), comm=comm)


def make_npse(cfg, bf):
    n = cfg["npse"]
    opts = NPSEOptions(n_harmonics=n["n_harmonics"], n_spanwise=n["n_spanwise"],
                       pad=n["pad"], nl_iter=n["nl_iter"], nl_tol=n["nl_tol"],
                       relax=n["relax"],
                       mean_flow_distortion=n["mean_flow_distortion"],
                       **_pse_kw(cfg))
    return NPSESolver(bf, _model(cfg, bf.gas), _wall(cfg), opts)


def run(cfg, verbose=True):
    par = Parallel(group_size=int(cfg["parallel"]["group_size"]))
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg["parallel"]["threads"]))
    t0 = time.time()
    pc = cfg["parallel"]
    be = backend.configure(pc.get("backend", "cpu"), local_rank=par.local_rank)
    par.log("pse backend %s on %d ranks, %d per node"
            % (be["name"], par.size, par.node.size if par.size > 1 else 1))
    gas = build_gas(cfg)
    bf = build_baseflow(cfg, gas, verbose=verbose and par.rank == 0)
    kind = cfg["analysis"]["type"].lower()
    par.log("pse gas %s %s transport %s species %s"
            % (gas.library, gas.gas_model, gas.transport_model,
               " ".join(gas.names)))
    par.log("pse base flow %d stations by %d wall normal points, delta99 "
            "at the last station %.4e m, Re_x %.3e"
            % (bf.nx, bf.ny, bf.edge["delta99"][-1], bf.edge["Re_x"][-1]))
    out = cfg["output"]
    if par.rank == 0:
        os.makedirs(out["directory"], exist_ok=True)

    if kind == "npse":
        res = None
        if par.rank == 0:
            n = cfg["npse"]
            ns = make_npse(cfg, bf)
            res = ns.march_nonlinear(
                n["f0_hz"], beta0=n["beta0"], n_az0=n["n_az0"],
                amplitudes=cfgmod.parse_amplitudes(n["amplitudes"]),
                i0=int(cfg["pse"]["i_start"]), c_guess=cfg["modes"]["c_guess"])
            path = os.path.join(out["directory"], "npse.npz")
            pio.save_npse(path, res, meta=dict(config=cfg))
            par.log("pse npse done in %.1f s, results in %s"
                    % (time.time() - t0, path))
        _save_baseflow(cfg, bf, par)
        return 0

    solver = make_solver(cfg, bf, comm=par.group if par.group_size > 1 else None)
    if pc.get("share_jacobians", True):
        tj = time.time()
        solver.share_jacobians(par)
        par.log("pse base flow Jacobians shared in %.1f s" % (time.time() - tj))
    modes = cfgmod.mode_list(cfg)
    order = cfgmod.schedule_order(modes, cfg["modes"].get("schedule",
                                                          "center_out"))
    par.log("pse %s sweep of %d modes over %d ranks in %d groups of %d"
            % (kind, len(modes), par.size, par.n_groups, par.group_size))
    i0 = int(cfg["pse"]["i_start"])
    check = bool(cfg["pse"]["mode_check"])
    cg = float(cfg["modes"]["c_guess"])

    def work(kk):
        f, beta, n_az = modes[order[kk]]
        try:
            if kind == "lst":
                r = solver.lst_march(f, beta, n_az=n_az, i0=i0, c_guess=cg)
            else:
                r = solver.march(f, beta, n_az=n_az, i0=i0, c_guess=cg,
                                 mode_check=check)
        except Exception as exc:                      # keep the sweep alive
            r = PSEResult()
            r.f_hz, r.beta, r.n_az, r.ok = f, beta, n_az, False
            r.message = "exception %r" % (exc,)
            r.finalize()
        return r.to_dict()

    def progress(idx, dt):
        if par.rank == 0 and verbose:
            print("pse mode %d of %d f %.1f kHz %.1f s"
                  % (idx + 1, len(modes), modes[order[idx]][0] / 1e3, dt),
                  flush=True)

    nbatch = int(pc.get("batch", 0) or 0)
    if kind == "lpse" and pc.get("lockstep", False) and nbatch > 0:
        from .lockstep import LockstepMarcher
        lm = LockstepMarcher(solver)
        chunks = [order[k:k + nbatch] for k in range(0, len(order), nbatch)]
        par.log("pse lockstep batches of %d modes, %d batches"
                % (nbatch, len(chunks)))

        def work_batch(kb):
            ids = chunks[kb]
            rs = lm.march([modes[k] for k in ids], i0=i0, c_guess=cg,
                          mode_check=check)
            return [r.to_dict() for r in rs]

        loc = par.map_modes(len(chunks), work_batch)
        local = []
        for kb, rs, dt in loc:
            for k, r in zip(chunks[kb], rs):
                local.append((k, r, dt / len(rs)))
    else:
        local = par.map_modes(len(modes), work, progress=progress)
        local = [(order[k], r, dt) for k, r, dt in local]
    merged = par.gather_results(local)
    if par.rank == 0:
        path = os.path.join(out["directory"], out["file"])
        meta = dict(config=cfg, wall_time=time.time() - t0, ranks=par.size,
                    n_modes=len(modes), species=gas.names, analysis=kind)
        pio.save_modes(path, merged, meta=meta)
        xg, res = pio.envelope(merged)
        if xg is not None:
            env, fb = res
            pio.write_envelope_csv(
                os.path.join(out["directory"], out["envelope_csv"]), xg, env,
                fb, meta=dict(case=cfg["case"]["name"], analysis=kind))
            k = int(np.argmax(env))
            par.log("pse N max %.3f at x %.4f m for f %.1f kHz"
                    % (env[k], xg[k], (fb[k] or 0.0) / 1e3))
        par.log("pse done in %.1f s, results in %s"
                % (time.time() - t0, out["directory"]))
    _save_baseflow(cfg, bf, par)
    return 0


def _save_baseflow(cfg, bf, par):
    out = cfg["output"]
    if par.rank == 0 and out["save_baseflow"]:
        np.savez_compressed(os.path.join(out["directory"], "baseflow.npz"),
                            x=bf.x, y=bf.grid.y, q=bf.q,
                            species=np.array(bf.gas.names),
                            **{k: v for k, v in bf.edge.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pse-nemo",
        description="LST, LPSE and NPSE for SU2-NEMO AIR-7 TTv boundary layers")
    ap.add_argument("config", nargs="?", help="JSON configuration")
    ap.add_argument("-o", "--output", help="override the output directory")
    ap.add_argument("--su2-solution", help="SU2 restart file or glob")
    ap.add_argument("--su2-mesh", help="SU2 mesh file")
    ap.add_argument("--analysis", choices=("lst", "lpse", "npse"))
    ap.add_argument("--library", choices=("su2tclib", "mutationpp"))
    ap.add_argument("--nf", type=int, help="number of frequencies")
    ap.add_argument("--group-size", type=int, help="ranks per mode")
    ap.add_argument("--dump-config", help="write the resolved config and exit")
    a = ap.parse_args(argv)
    over = {}
    if a.output:
        over.setdefault("output", {})["directory"] = a.output
    if a.su2_solution or a.su2_mesh:
        b = over.setdefault("baseflow", {})
        b["source"] = "su2"
        s = b.setdefault("su2", {})
        if a.su2_solution:
            s["solution"] = a.su2_solution
        if a.su2_mesh:
            s["mesh"] = a.su2_mesh
    if a.analysis:
        over.setdefault("analysis", {})["type"] = a.analysis
    if a.library:
        over.setdefault("gas", {})["library"] = a.library
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
