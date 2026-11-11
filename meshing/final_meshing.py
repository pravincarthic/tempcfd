import gmsh
import math
import os
import sys
import json
import re
import numpy as np

# Empirically measured on gmsh 4.15.2: a tetrahedral fill contains about this
# many tets per h^3 of volume, where h is the local target size. Measured by
# meshing a 100^3 box at h=50, 25 and 10 (4955 tets at h=10). Used by the
# pre-flight element budget estimate.
TETS_PER_CUBED_SIZE = 5.0


def load_config(config_file):
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return None
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Config loaded: {config_file}\n")
        return config
    except json.JSONDecodeError as e:
        print(f"Config parse error: {e}")
        return None
    except Exception as e:
        print(f"Config read error: {e}")
        return None


def export_step(path):
    if not path:
        return
    try:
        gmsh.write(path)
        print(f"Geometry exported: {path} ({os.path.getsize(path) / 1024:.1f} KB)")
    except Exception as e:
        print(f"Geometry export failed: {e}")


def probe_field_options(field_type, candidates):
    """
    Return the subset of candidate option names the installed gmsh accepts on
    this field type.

    Option names are not stable across gmsh versions, and an assignment to an
    unknown option raises on some builds and only warns on others, which means
    a field can silently keep its defaults while the script reports success.
    Probes a throwaway field, then removes it.
    """
    try:
        terminal = gmsh.option.getNumber("General.Terminal")
    except Exception:
        terminal = 1
    gmsh.option.setNumber("General.Terminal", 0)
    probe = gmsh.model.mesh.field.add(field_type)
    valid = []
    try:
        for name in candidates:
            ok = False
            try:
                gmsh.model.mesh.field.setNumber(probe, name, 1.0)
                ok = True
            except Exception:
                try:
                    gmsh.model.mesh.field.setNumbers(probe, name, [1.0])
                    ok = True
                except Exception:
                    ok = False
            if ok:
                valid.append(name)
    finally:
        gmsh.model.mesh.field.remove(probe)
        gmsh.option.setNumber("General.Terminal", terminal)
    return valid


# ---------------------------------------------------------------------------
# Smart body segmentation: nose / hump / rear, detected from geometry rather
# than typed in as coordinates.
#
# Method: probe-mesh the bare body cheaply (a coarse, uniform surface mesh,
# generated on a throwaway model so it never touches the real meshing state),
# bin the resulting wall points by station along the body's long axis, and
# take the local max radial extent per station as R(x). A blunt nose rises
# from ~0, a hump is the local max of R(x), a rear taper falls away from it.
# The split stations are where R(x) crosses a fraction of the peak -- this is
# a shoulder detector, not a curvature/derivative one, because it is far less
# sensitive to the mesh noise a coarse probe mesh introduces (verified: raw
# per-station max is visibly jagged at 150 stations on a 2% probe size, a
# moving-average derivative of that is jagged enough to false-trigger, a
# level crossing on the smoothed curve is not).
# ---------------------------------------------------------------------------

def profile_body_radius(step_file, dilation, angle_deg, rotation_by_com,
                         body_bbox, n_stations, probe_size_fraction):
    """
    Build an R(x) profile of the body by coarse-probe-meshing its own bare
    surface (no far-field box, no boolean cut yet) on a throwaway model.

    Runs in its own gmsh.model so the probe mesh and its coarse size options
    never leak into the real meshing pass that follows. Re-imports and
    re-applies dilation/rotation exactly as the main flow will, so the
    profile is taken in the same frame the real geometry ends up in.

    Returns (centers, r_raw, r_smooth) in the body's local X, or None if the
    body could not be imported for profiling (caller falls back to fixed
    fractions).
    """
    current = gmsh.model.getCurrent()
    gmsh.model.add("_profile_probe")
    try:
        shapes = gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
        if dilation != 1.0:
            gmsh.model.occ.dilate(shapes, 0, 0, 0, dilation, dilation, dilation)
            gmsh.model.occ.synchronize()
        body_tags = [t for d, t in shapes if d == 3]
        if not body_tags:
            return None
        body_tag = body_tags[0]
        if angle_deg != 0.0:
            origin = gmsh.model.occ.getCenterOfMass(3, body_tag) \
                if rotation_by_com else (0.0, 0.0, 0.0)
            gmsh.model.occ.rotate([(3, body_tag)], origin[0], origin[1],
                                  origin[2], 0, 1, 0,
                                  angle_deg * math.pi / 180.0)
            gmsh.model.occ.synchronize()

        body_length = body_bbox[3] - body_bbox[0]
        if body_length <= 0:
            return None
        probe_size = max(body_length * probe_size_fraction, 1e-3)

        terminal = gmsh.option.getNumber("General.Terminal")
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", probe_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", probe_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("General.Terminal", terminal)

        node_tags, coords, _ = gmsh.model.mesh.getNodes(-1, -1, includeBoundary=True)
        if len(node_tags) == 0:
            return None
        xyz = np.asarray(coords, dtype=float).reshape(-1, 3)

        x_min, x_max = body_bbox[0], body_bbox[3]
        y0 = 0.5 * (body_bbox[1] + body_bbox[4])
        z0 = 0.5 * (body_bbox[2] + body_bbox[5])
        r = np.sqrt((xyz[:, 1] - y0) ** 2 + (xyz[:, 2] - z0) ** 2)

        edges = np.linspace(x_min, x_max, n_stations + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.searchsorted(edges, xyz[:, 0], side='right') - 1,
                      0, n_stations - 1)

        r_profile = np.full(n_stations, np.nan)
        for i in range(n_stations):
            mask = idx == i
            if mask.any():
                r_profile[i] = r[mask].max()

        valid = ~np.isnan(r_profile)
        if valid.sum() < 2:
            return None
        if valid.sum() < n_stations:
            r_profile = np.interp(centers, centers[valid], r_profile[valid])

        return centers, r_profile
    except Exception as e:
        print(f"  Body profiling failed: {e}")
        return None
    finally:
        gmsh.model.remove()
        if current:
            gmsh.model.setCurrent(current)


def smooth_profile(r_profile, window):
    window = max(1, int(window))
    if window <= 1 or window > len(r_profile):
        return r_profile.copy()
    kernel = np.ones(window) / window
    return np.convolve(r_profile, kernel, mode='same')


def detect_segment_splits(centers, r_profile, body_bbox, seg_params):
    """
    Turn an R(x) profile into (x_split_1, x_split_2): the nose/hump and
    hump/rear station boundaries.

    Finds the peak of the smoothed profile, then walks outward from it to the
    first station on each side where R(x) crosses shoulder_fraction of the
    peak. Falls back to fixed length fractions of the body when the profile
    is degenerate (near flat, monotonic, or a shoulder too narrow to trust) --
    a slender body with no distinct hump does not have a real shoulder to
    find, and forcing one would just be noise.

    Returns (x_split_1, x_split_2, method) where method is "detected" or
    "fallback".
    """
    bx_min, bx_max = body_bbox[0], body_bbox[3]
    body_length = bx_max - bx_min
    fallback_nose = seg_params.get("fallback_nose_fraction", 0.12)
    fallback_rear = seg_params.get("fallback_rear_fraction", 0.28)
    min_seg_frac = seg_params.get("min_segment_length_fraction", 0.05)

    def fallback(reason):
        x1 = bx_min + fallback_nose * body_length
        x2 = bx_max - fallback_rear * body_length
        if x2 <= x1:
            x2 = x1 + min_seg_frac * body_length
        print(f"  Segmentation fallback ({reason}): using fixed fractions "
              f"nose={fallback_nose} rear={fallback_rear}")
        return x1, x2, "fallback"

    if centers is None or r_profile is None or len(centers) < 5:
        return fallback("no usable profile")

    window = seg_params.get("smoothing_window", 5)
    r_smooth = smooth_profile(r_profile, window)

    peak_i = int(np.argmax(r_smooth))
    peak_r = float(r_smooth[peak_i])
    r_start, r_end = float(r_smooth[0]), float(r_smooth[-1])
    baseline = max(min(r_start, r_end), 1e-9)

    # A real hump needs the peak to stand meaningfully above both ends, not
    # just be the tail end of a monotonically flaring body (a pure cone would
    # have its "peak" at the last station, which is not a hump).
    prominence = seg_params.get("min_peak_prominence_ratio", 1.15)
    if peak_r < baseline * prominence:
        return fallback("no distinct peak above the end radii")
    if peak_i == 0 or peak_i == len(r_smooth) - 1:
        return fallback("peak sits at a body extremity, not a hump")

    shoulder_frac = seg_params.get("shoulder_fraction", 0.6)
    threshold = shoulder_frac * peak_r

    left = np.where(r_smooth[:peak_i + 1] >= threshold)[0]
    x_split_1 = float(centers[left[0]]) if left.size else float(centers[0])

    right = np.where(r_smooth[peak_i:] < threshold)[0]
    x_split_2 = float(centers[peak_i + right[0]]) if right.size else float(centers[-1])

    min_len = min_seg_frac * body_length
    if (x_split_1 - bx_min) < min_len or (x_split_2 - x_split_1) < min_len \
            or (bx_max - x_split_2) < min_len:
        return fallback("a detected segment is shorter than "
                        f"min_segment_length_fraction={min_seg_frac}")

    print(f"  Segmentation detected: peak R={peak_r:.1f}mm at x={centers[peak_i]:.0f}mm, "
          f"shoulder={shoulder_frac} of peak")
    return x_split_1, x_split_2, "detected"


def slice_body_into_segments(step_file, dilation, angle_deg, rotation_by_com,
                             body_bbox, x_split_1, x_split_2, pad_mm,
                             mass_tolerance, isolate=True):
    """
    Cut the body solid into 3 chunks (nose, hump, rear) at the two X splits,
    each as its own OCC solid, via boolean intersect against big boxes.

    This is real geometric slicing, not a soft sizing box: it splits the
    underlying BREP faces at the cut planes, which is what lets a single
    large blended surface (the usual case on a BWB) end up as separate face
    entities per section -- the boundary layer extrusion further down needs
    that, since one gmsh.model.geo.extrudeBoundaryLayer call uses one layer
    recipe for its whole input face list.

    Each chunk is built from its own fresh re-import (re-dilated, re-rotated)
    rather than 3 copies of one in-model body, mirroring the reimport pattern
    the far-field boolean cut already uses for its own strategy retries --
    verified more robust than sharing one body across sequential consuming
    booleans.

    Verifies mass conservation (sum of chunk masses against the whole body)
    and returns None, with a printed reason, if it does not reconcile within
    mass_tolerance -- the caller falls back to meshing the body unsplit.

    isolate controls where the chunk volumes end up, and this matters a lot:
    gmsh's OCC/GEO entities live inside whichever named model created them,
    and gmsh.model.remove() on that model destroys them, not just the
    bookkeeping -- verified directly (a "Unknown OpenCASCADE entity" error on
    the very next boolean once the scratch model was torn down). So:
      isolate=True   builds the chunks in a throwaway scratch model and tears
                     it down before returning, on purpose. Only the mass-check
                     pass/fail and the split diagnostics are meant to survive
                     the call; the tags are not usable afterwards. This is
                     what the one validation-only call at startup wants: it
                     decides IF slicing is viable without leaving stray
                     volumes lying around in the real model that step 4 is
                     about to build the far-field box into.
      isolate=False  builds the chunks directly in whatever model is current,
                     no scratch model, so the returned tags stay alive in it.
                     This is what the actual boolean-cut step needs, since it
                     uses the chunk volumes as cut tools in that live model.

    Returns a dict {"nose": [tags], "hump": [tags], "rear": [tags]}, or None
    on failure.
    """
    if isolate:
        current = gmsh.model.getCurrent()
        gmsh.model.add("_slice_body")
    try:
        return _slice_body_ops(step_file, dilation, angle_deg, rotation_by_com,
                               body_bbox, x_split_1, x_split_2, pad_mm,
                               mass_tolerance)
    finally:
        if isolate:
            gmsh.model.remove()
            if current:
                gmsh.model.setCurrent(current)


def _slice_body_ops(step_file, dilation, angle_deg, rotation_by_com,
                    body_bbox, x_split_1, x_split_2, pad_mm, mass_tolerance):
    """
    The actual slicing booleans, run in whatever model is current at the
    time -- never creates or removes a model itself. Factored out of
    slice_body_into_segments so that function's isolate flag can choose
    between a throwaway scratch model (validation only) and the live model
    (when the chunks need to survive for reuse). See that function's
    docstring for why the distinction matters.
    """
    bx_min, by_min, bz_min, bx_max, by_max, bz_max = body_bbox
    y0, y1 = by_min - pad_mm, by_max + pad_mm
    z0, z1 = bz_min - pad_mm, bz_max + pad_mm
    x_lo, x_hi = bx_min - pad_mm, bx_max + pad_mm

    segments = [("nose", x_lo, x_split_1),
                ("hump", x_split_1, x_split_2),
                ("rear", x_split_2, x_hi)]

    # The body is imported ONCE and deep-copied per section (occ.copy),
    # rather than re-imported from the STEP file for the mass reference
    # and again for each section: fewer file reads, and each section gets
    # its own independent shape instead of OCC re-handing the same one.
    chunk_tags = {}
    base = None
    try:
        shapes = gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
        if dilation != 1.0:
            gmsh.model.occ.dilate(shapes, 0, 0, 0, dilation, dilation, dilation)
            gmsh.model.occ.synchronize()
        base = [t for d, t in shapes if d == 3][0]
        if angle_deg != 0.0:
            origin = gmsh.model.occ.getCenterOfMass(3, base) \
                if rotation_by_com else (0.0, 0.0, 0.0)
            gmsh.model.occ.rotate([(3, base)], origin[0], origin[1], origin[2],
                                  0, 1, 0, angle_deg * math.pi / 180.0)
            gmsh.model.occ.synchronize()
        ref_mass = gmsh.model.occ.getMass(3, base)
        total_mass = 0.0
        for name, xa, xb in segments:
            btag = gmsh.model.occ.copy([(3, base)])[0][1]
            gmsh.model.occ.synchronize()

            box = gmsh.model.occ.addBox(xa, y0, z0, xb - xa, y1 - y0, z1 - z0)
            gmsh.model.occ.synchronize()
            res, _ = gmsh.model.occ.intersect([(3, btag)], [(3, box)],
                                              removeObject=True, removeTool=True)
            gmsh.model.occ.synchronize()
            tags = [t for d, t in res if d == 3]
            if not tags:
                print(f"  Slice '{name}' produced no volume, x=[{xa:.0f},{xb:.0f}]")
                _cleanup_chunks(chunk_tags, base)
                return None
            mass = sum(gmsh.model.occ.getMass(3, t) for t in tags)
            total_mass += mass
            chunk_tags[name] = tags
            print(f"  Slice '{name}': x=[{xa:.0f},{xb:.0f}] -> "
                  f"{len(tags)} volume(s), mass {mass:.4e}")

        gmsh.model.occ.remove([(3, base)], recursive=True)
        gmsh.model.occ.synchronize()
        base = None
        frac_diff = abs(total_mass - ref_mass) / max(ref_mass, 1e-30)
        print(f"  Mass check: chunks {total_mass:.4e} vs whole body "
              f"{ref_mass:.4e}, diff {100 * frac_diff:.4f}%")
        if frac_diff > mass_tolerance:
            print(f"  Slicing rejected: mass mismatch exceeds tolerance "
                  f"{100 * mass_tolerance:.4f}%. Likely an OCC seam coincided "
                  "with a cut plane. Falling back to an unsliced body.")
            _cleanup_chunks(chunk_tags)
            return None
        return chunk_tags
    except Exception as e:
        print(f"  Slicing failed: {e}")
        _cleanup_chunks(chunk_tags, base)
        return None


def _cleanup_chunks(chunk_tags, base=None):
    """
    Remove any chunk volumes already created before a rejected or failed
    slice, so a caller running with isolate=False (chunks built directly in
    the live model) is not left with stray leftover solids contaminating
    later entity counts (e.g. the far-field cut's own volume bookkeeping).
    Best-effort: a throwaway scratch model (isolate=True) gets torn down by
    its own caller regardless, so a failure there is harmless either way.
    """
    for tags in chunk_tags.values():
        try:
            gmsh.model.occ.remove([(3, t) for t in tags], recursive=True)
        except Exception:
            pass
    if base is not None:
        try:
            gmsh.model.occ.remove([(3, base)], recursive=True)
        except Exception:
            pass
    if chunk_tags or base is not None:
        try:
            gmsh.model.occ.synchronize()
        except Exception:
            pass


def merge_section_params(defaults, override):
    """Shallow merge: override's keys win, nested dicts merge one level deep."""
    out = dict(defaults)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def compute_first_layer_size(target_yplus, freestream, station_mm=None,
                             verbose=True):
    """
    First prism cell height from a target y+, using a flat-plate turbulent
    skin-friction correlation, in place of guessing the Size parameter.

    Cf = 0.026 / Re_x^(1/7)          (Schlichting-type turbulent flat-plate)
    tau_w = 0.5 * Cf * rho_inf * U^2
    u_tau = sqrt(tau_w / rho_w)
    y1 = target_yplus * mu_w / (rho_w * u_tau)

    freestream needs velocity_mps, density_kgm3, dynamic_viscosity_pas and
    reference_length_mm. Optional:
      yplus_station_mm        running length x at which y+ is to be exact
                              (default reference_length_mm). Cf falls with
                              x, so y+ is higher upstream of this station and
                              lower downstream.
      wall_density_kgm3,      near-wall gas properties. y+ is defined with
      wall_dynamic_viscosity_pas  wall values; at hypersonic speed with a cold
                              wall they differ a lot from freestream. Default
                              to the freestream values when not given.
    station_mm overrides yplus_station_mm. Returns the first layer height in
    mm, or None if a required value is missing or non-positive.
    """
    try:
        u = float(freestream["velocity_mps"])
        rho = float(freestream["density_kgm3"])
        mu = float(freestream["dynamic_viscosity_pas"])
        l_ref = float(freestream["reference_length_mm"])
        x_mm = float(station_mm if station_mm is not None else
                     freestream.get("yplus_station_mm", l_ref))
        rho_w = float(freestream.get("wall_density_kgm3", rho))
        mu_w = float(freestream.get("wall_dynamic_viscosity_pas", mu))
    except (KeyError, TypeError, ValueError):
        return None
    if min(u, rho, mu, l_ref, x_mm, rho_w, mu_w) <= 0:
        return None

    re_x = rho * u * (x_mm / 1000.0) / mu
    cf = 0.026 / (re_x ** (1.0 / 7.0))
    tau_w = 0.5 * cf * rho * u ** 2
    u_tau = math.sqrt(max(tau_w, 1e-30) / rho_w)
    y1_mm = target_yplus * mu_w / (rho_w * u_tau) * 1000.0
    if verbose:
        print(f"    y+ sizing at x={x_mm:.0f}mm: Re_x={re_x:.3e} Cf={cf:.5f} "
              f"u_tau={u_tau:.3f}m/s -> first layer {y1_mm:.5f}mm for "
              f"y+={target_yplus}")
    return y1_mm


# ---------------------------------------------------------------------------
# Compressible, conservative first-cell sizing (y+ <= target everywhere).
#
# At Mach 10 y+ has to be defined with wall gas properties, and the wall
# temperature is not known before the solve (cold isothermal skin up to
# adiabatic, ~19x freestream here). Both move the answer, in opposite
# directions for density and viscosity, so the first cell is taken as the
# SMALLEST of:
#   - the incompressible freestream estimate (compute_first_layer_size),
#   - the compressible estimate, Eckert reference temperature
#       T*/Te = 1 + 0.032 M^2 + 0.58 (Tw/Te - 1),
#     Cf from the same 1/7-power law at Re*_x (rho*, mu* at T*),
#     y1 = y+ mu_w / sqrt(rho_w tau_w), rho_w = rho_e Te/Tw, mu_w Sutherland,
#     evaluated at every candidate wall temperature: wall_temperature_K if
#     given, else freestream T, 300 K (if inside the range) and T_aw,
# each at the most demanding running length (Cf, so tau_w, is highest near
# the leading edge): yplus_min_running_length_mm (default 10 mm) and the
# configured yplus_station_mm. Turbulent Cf is used throughout, which is
# above laminar Cf, so a laminar or transitional region is also covered.
# The result can only be equal to or finer than the old single estimate.
# ---------------------------------------------------------------------------

AIR_GAS_CONSTANT = 287.05


def sutherland_viscosity(T):
    return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)


def freestream_state(fs):
    """Freestream T, M, gamma and the Sutherland scale that reproduces the
    configured viscosity. T comes from temperature_K, else from inverting
    Sutherland for dynamic_viscosity_pas."""
    try:
        u = float(fs["velocity_mps"])
        rho = float(fs["density_kgm3"])
        mu = float(fs["dynamic_viscosity_pas"])
    except (KeyError, TypeError, ValueError):
        return None
    gamma = float(fs.get("gamma", 1.4))
    if fs.get("temperature_K"):
        T = float(fs["temperature_K"])
    else:
        lo, hi = 20.0, 5000.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if sutherland_viscosity(mid) < mu:
                lo = mid
            else:
                hi = mid
        T = 0.5 * (lo + hi)
    mach = float(fs.get("mach", u / math.sqrt(gamma * AIR_GAS_CONSTANT * T)))
    recovery = float(fs.get("recovery_factor", 0.89))
    t_aw = T * (1.0 + recovery * 0.5 * (gamma - 1.0) * mach ** 2)
    return {"u": u, "rho": rho, "mu": mu, "T": T, "M": mach, "gamma": gamma,
            "T_aw": t_aw, "mu_scale": mu / sutherland_viscosity(T)}


def wall_temperature_candidates(fs, st):
    if fs.get("wall_temperature_K"):
        return [float(fs["wall_temperature_K"])]
    temps = [st["T"], st["T_aw"]]
    if st["T"] < 300.0 < st["T_aw"]:
        temps.append(300.0)
    return sorted(temps)


def compressible_first_cell(target_yplus, fs, station_mm, t_wall, st=None):
    st = st or freestream_state(fs)
    if st is None or station_mm <= 0:
        return None
    Te, M = st["T"], st["M"]
    t_ref = Te * (1.0 + 0.032 * M ** 2 + 0.58 * (t_wall / Te - 1.0))
    rho_ref = st["rho"] * Te / t_ref
    mu_ref = sutherland_viscosity(t_ref) * st["mu_scale"]
    re_ref = rho_ref * st["u"] * (station_mm / 1000.0) / mu_ref
    cf = 0.026 / re_ref ** (1.0 / 7.0)
    tau_w = 0.5 * cf * rho_ref * st["u"] ** 2
    rho_w = st["rho"] * Te / t_wall
    mu_w = sutherland_viscosity(t_wall) * st["mu_scale"]
    return target_yplus * mu_w / math.sqrt(rho_w * max(tau_w, 1e-30)) * 1000.0


def conservative_first_cell(target_yplus, fs, verbose=True):
    """
    Smallest first cell over the incompressible and compressible estimates,
    all candidate wall temperatures and the critical stations (see the block
    comment above). Returns (y1_mm, details) or (None, None).
    """
    st = freestream_state(fs)
    l_ref = fs.get("reference_length_mm")
    if st is None or not l_ref:
        return None, None
    stations = [float(fs.get("yplus_min_running_length_mm", 10.0))]
    if fs.get("yplus_station_mm"):
        stations.append(float(fs["yplus_station_mm"]))
    rows = []
    for s in stations:
        y_inc = compute_first_layer_size(target_yplus, fs, station_mm=s, verbose=False)
        if y_inc:
            rows.append(("incompressible, freestream properties", s, None, y_inc))
        for tw in wall_temperature_candidates(fs, st):
            y_c = compressible_first_cell(target_yplus, fs, s, tw, st)
            if y_c:
                rows.append(("compressible, Eckert T*", s, tw, y_c))
    if not rows:
        return None, None
    best = min(rows, key=lambda r: r[3])
    if verbose:
        print(f"  Freestream: T={st['T']:.1f}K M={st['M']:.2f} "
              f"T_aw={st['T_aw']:.0f}K")
        for label, s, tw, y in rows:
            tws = f" Tw={tw:.0f}K" if tw else ""
            mark = "  (used)" if (label, s, tw, y) == best else ""
            print(f"    {label}{tws} at x={s:.0f}mm: {y:.5f}mm{mark}")
    return best[3], {"rows": rows, "state": st}


def yplus_unit_length(fs, stations_mm):
    """Height in mm of y+ = 1 at each running length, the minimum over every
    model and wall temperature above. Vectorised by interpolating a log
    grid, since it is smooth and monotonic in running length."""
    st = freestream_state(fs)
    s = np.maximum(np.asarray(stations_mm, float),
                   float(fs.get("yplus_min_running_length_mm", 10.0)))
    if st is None or s.size == 0:
        return None
    grid = np.geomspace(s.min(), max(s.max(), s.min() * 1.0001), 64)
    vals = []
    for g in grid:
        cands = [compute_first_layer_size(1.0, fs, station_mm=g, verbose=False)]
        cands += [compressible_first_cell(1.0, fs, g, tw, st)
                  for tw in wall_temperature_candidates(fs, st)]
        vals.append(min(c for c in cands if c))
    return np.interp(np.log(s), np.log(grid), np.asarray(vals))


def first_cell_height(bl_params, fs, refine_state=None, verbose=False):
    """The first cell the stack will use: conservative y+ sizing when
    target_yplus is set, else Size; times any y+ correction the refinement
    loop has applied (which can only shrink it)."""
    y1 = None
    if bl_params.get("target_yplus") and fs:
        y1, _ = conservative_first_cell(bl_params["target_yplus"], fs, verbose)
    if y1 is None:
        y1 = float(bl_params.get("Size", 0.05))
    if refine_state:
        y1 *= min(1.0, float(refine_state.get("first_cell_factor", 1.0)))
    return y1


def plan_yplus_stack(first_cell, bl_params):
    """
    Layer count and growth ratio for a y+ driven first cell.

    The first cell comes from y+, so NbLayers and Ratio can no longer both be
    fixed if the stack is also meant to reach a physical Thickness. When
    Thickness > 0, Ratio is kept as the largest growth allowed, the layer
    count is the smallest that reaches Thickness at that ratio (capped by
    max_layers), and the ratio is then eased down so the stack ends exactly
    at Thickness. When Thickness is 0, NbLayers and Ratio are used as given.

    Returns (n_layers, ratio, total_mm).
    """
    r_max = float(bl_params.get("Ratio", 1.1))
    thick = float(bl_params.get("Thickness", 0.0) or 0.0)
    n_cfg = int(bl_params.get("NbLayers", 36))
    if thick <= 0 or thick <= first_cell:
        total = first_cell * (n_cfg if r_max == 1.0 else
                              (r_max ** n_cfg - 1.0) / (r_max - 1.0))
        return n_cfg, r_max, total
    if r_max <= 1.0:
        n = int(math.ceil(thick / first_cell))
    else:
        n = int(math.ceil(math.log(1.0 + thick * (r_max - 1.0) / first_cell)
                          / math.log(r_max)))
    n_cap = int(bl_params.get("max_layers", 200))
    if n > n_cap:
        print(f"  Warning: reaching Thickness={thick}mm from a {first_cell:.5f}mm "
              f"first cell at Ratio<={r_max} needs {n} layers, above "
              f"max_layers={n_cap}; using {n_cap} with a larger ratio.")
        n = n_cap
    n = max(n, 1)
    _, r = _solve_ratios(first_cell, n, np.array([thick]))
    return n, float(r[0]), thick


def setup_box_field(label, box, size_in, size_out, thickness):
    """Create a Box sizing field. box is (xmin, xmax, ymin, ymax, zmin, zmax)."""
    xmin, xmax, ymin, ymax, zmin, zmax = box
    f = gmsh.model.mesh.field.add("Box")
    for key, val in (("XMin", xmin), ("XMax", xmax), ("YMin", ymin),
                     ("YMax", ymax), ("ZMin", zmin), ("ZMax", zmax),
                     ("VIn", size_in), ("VOut", size_out)):
        gmsh.model.mesh.field.setNumber(f, key, val)
    if thickness > 0:
        gmsh.model.mesh.field.setNumber(f, "Thickness", thickness)
    print(f"{label} box field {f}: X=[{xmin:.0f},{xmax:.0f}] "
          f"Y=[{ymin:.0f},{ymax:.0f}] Z=[{zmin:.0f},{zmax:.0f}] size_in={size_in}mm")
    return f


def sample_wall_points(wall_faces, probe_size):
    """
    Points on the real (trimmed) wall, from a throwaway coarse surface mesh.

    Used to find how wide and tall the body really is over a given X range,
    so a refinement box hugs that part of the body instead of spanning the
    whole body bounding box. Sampling a face's parameter range instead is
    not usable: it covers the whole untrimmed surface, which on this BWB put
    the nose box out to the wing tip. The probe mesh is cleared afterwards
    and the size options restored, so meshing proper is unaffected.
    """
    keys = ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax", "Mesh.MeshSizeFromCurvature",
            "Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
            "General.Terminal")
    saved = {k: gmsh.option.getNumber(k) for k in keys}
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", probe_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", probe_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        pts = []
        gmsh.model.mesh.generate(2)
        for f in wall_faces:
            _, c, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=True)
            if len(c):
                pts.append(np.asarray(c, dtype=float).reshape(-1, 3))
        return np.concatenate(pts) if pts else np.zeros((0, 3))
    except Exception as e:
        print(f"  Wall probe mesh failed ({e}); refinement boxes use the "
              "body bounding box")
        return None
    finally:
        gmsh.model.mesh.clear()
        for k, v in saved.items():
            gmsh.option.setNumber(k, v)


def section_box(section_name, sect_params, x_range, body_bbox, half_model,
                sym_coord, size_far, wall_points=None):
    """
    Build the (box, size_in, thickness) tuple for one section (nose/hump/rear
    or the nose_refinement_parameters box) over its X range.

    With wall_points the box's Y and Z are the body's own extent over that X
    range plus lateral_pad_mm. Without them it falls back to the whole body
    bounding box, which over a BWB nose is several times wider and taller than
    the nose itself: at a 40 mm box size that one box was ~58M tets.
    """
    if not sect_params.get("enable", True):
        return None
    xa, xb = x_range
    pad = sect_params.get("lateral_pad_mm", 500.0)
    bx_min, by_min, bz_min, bx_max, by_max, bz_max = body_bbox
    if wall_points is not None and len(wall_points):
        sel = wall_points[(wall_points[:, 0] >= xa) & (wall_points[:, 0] <= xb)]
        if half_model and len(sel):
            sel = sel[sel[:, 1] >= sym_coord - 1e-6] if by_max > sym_coord else sel
        if len(sel):
            by_min, bz_min = sel[:, 1].min(), sel[:, 2].min()
            by_max, bz_max = sel[:, 1].max(), sel[:, 2].max()
    box = (xa, xb,
          sym_coord if half_model else by_min - pad, by_max + pad,
          bz_min - pad, bz_max + pad)
    size_in = sect_params.get("size_mm", size_far * 0.1)
    thickness = sect_params.get("transition_thickness_mm", 500.0)
    return (box, size_in, thickness)


def shock_box(shock_params, body_bbox, domain, half_model, sym_coord):
    """
    Box over the expected bow shock region.

    The safety factor is here because arcsin(1/M) is the asymptotic Mach angle
    of an infinitesimal disturbance. A blunt nose throws a detached, curved bow
    shock, near normal at the stagnation point, relaxing towards the Mach angle
    only downstream, so the real envelope is wider everywhere and widest at the
    nose.

    Be aware what this costs. Filling a whole Mach cone at a shock-resolving
    size is not affordable at a large domain size: a 250 m by 63 m by 125 m cone
    at 100 mm comes to roughly 10^10 elements. The budget check below exists
    because of exactly that error. Either keep the domain tight, refine only a
    shell around the expected shock position, or leave this off and use
    solution adaptive refinement in the solver, which is the only approach that
    does not require knowing where the shock is beforehand.
    """
    if not shock_params.get("enable", False):
        return None
    mach = shock_params.get("mach_number", 10.0)
    if mach <= 1.0:
        print(f"Shock box skipped: mach_number={mach} is not supersonic")
        return None

    x_inlet, x_outlet, y_lo, y_hi, z_lo, z_hi = domain
    bx_min, by_min, bz_min, bx_max, by_max, bz_max = body_bbox

    safety = shock_params.get("safety_factor", 2.5)
    pad = shock_params.get("nose_upstream_pad_mm", 2000.0)
    mach_angle = math.asin(1.0 / mach)
    x_nose = bx_min - pad
    radius = math.tan(mach_angle) * (x_outlet - x_nose) * safety

    box = (
        max(shock_params.get("XMin", x_nose), x_inlet),
        min(shock_params.get("XMax", x_outlet), x_outlet),
        max(shock_params.get("YMin", sym_coord if half_model else -radius), y_lo),
        min(shock_params.get("YMax", radius), y_hi),
        max(shock_params.get("ZMin", min(bz_min - radius, -radius)), z_lo),
        min(shock_params.get("ZMax", max(bz_max + radius, radius)), z_hi),
    )
    print(f"Shock box: mach={mach} mach_angle={math.degrees(mach_angle):.2f}deg "
          f"safety={safety} radius={radius:.0f}mm")
    return (box,
            shock_params.get("size_in_mm", 100.0),
            shock_params.get("transition_thickness_mm", 0.0))


def auto_sampling(wall_faces, dist_min, size_params):
    """
    Choose the Distance field Sampling count from the size of the wall.

    Sampling is how many points gmsh places per wall entity when building the
    distance function, so the distance field is only accurate to roughly
    (entity extent / Sampling). A fixed 100 on a 49 m body resolves distance to
    about half a metre, coarser than the whole near-wall band it is meant to
    position. Verified on a 500 mm plate with a 5 mm band: element count kept
    changing up to Sampling=100 and saturated after, i.e. once the sample
    spacing became finer than the band.
    """
    max_extent = 0.0
    for f in wall_faces:
        bb = gmsh.model.getBoundingBox(2, f)
        max_extent = max(max_extent, bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])

    cap = size_params.get("max_sampling", 400)
    floor = size_params.get("min_sampling", 20)
    needed = cap if dist_min <= 0 else int(math.ceil(max_extent / dist_min))
    sampling = max(floor, min(cap, needed))

    print(f"Distance sampling: largest wall extent={max_extent:.0f}mm "
          f"dist_min={dist_min}mm needed={needed} using={sampling}")
    if needed > cap:
        print(f"  Warning: capped at {cap}. The near-wall band is positioned to "
              f"about {max_extent / sampling:.0f}mm accuracy, so a dist_min "
              "below that is not meaningful. Raise max_sampling if needed.")
    return sampling


def setup_size_field(wall_faces, size_params, small_faces=()):
    """
    Near-wall Threshold on the distance to the whole wall.

    Sampling is per parametric direction, so each surface costs Sampling^2
    points. Small faces (cavities, a few mm across) are put in a second
    Distance field sampled at their own scale and the two distances are
    combined with Min: sampling 4300 cavity faces at the body-scale count
    was ~10^8 points and a surface mesh that never finished.
    """
    size_near = size_params.get("size_near", 2.0)
    size_far = size_params.get("size_far", 500.0)
    dist_min = size_params.get("dist_min", 200.0)
    dist_max = size_params.get("dist_max", 20000.0)
    # growth_rate g: the size rises linearly with wall distance past dist_min,
    # h = size_near + g * (d - dist_min), so each cell is about (1 + g) times
    # its neighbour nearer the wall. It sets dist_max itself. Without it the
    # older Sigmoid ramp is kept, which holds the size near size_near for
    # roughly the first third of the DistMin-DistMax span: on the cavity body
    # at size_near 60 that band alone was most of a 117M cell mesh.
    growth = size_params.get("growth_rate")
    if growth:
        dist_max = dist_min + (size_far - size_near) / float(growth)
    sigmoid = 0 if growth else int(size_params.get("sigmoid", 1))

    small = [f for f in wall_faces if f in set(small_faces)]
    big = [f for f in wall_faces if f not in set(small)] or wall_faces
    if len(big) == len(wall_faces):
        small = []
    sampling = auto_sampling(big, dist_min, size_params)

    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", big)
    gmsh.model.mesh.field.setNumber(dist_field, "Sampling", sampling)
    if small:
        ext = 0.0
        for f in small:
            bb = gmsh.model.getBoundingBox(2, f)
            ext = max(ext, bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
        small_sampling = int(min(50, max(5, math.ceil(
            ext / max(min(size_near, dist_min), 1e-9)) + 1)))
        d_small = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d_small, "SurfacesList", small)
        gmsh.model.mesh.field.setNumber(d_small, "Sampling", small_sampling)
        d_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(d_min, "FieldsList", [dist_field, d_small])
        print(f"  {len(small)} small wall face(s) sampled separately at "
              f"{small_sampling}")
        dist_field = d_min

    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", size_near)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", size_far)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", dist_max)
    gmsh.model.mesh.field.setNumber(thresh_field, "Sigmoid", sigmoid)

    print(f"Size field: near={size_near}mm far={size_far}mm "
          f"dist_min={dist_min}mm dist_max={dist_max:.0f}mm "
          f"{'linear, growth ' + str(growth) if growth else ('sigmoid' if sigmoid else 'linear')}\n")
    return thresh_field


def estimate_elements(domain, body_bbox, size_params, boxes, size_min, size_max,
                      n_samples=200000, seed=0):
    """
    Pre-flight element count estimate, before any meshing runs.

    Reimplements the same sizing rules in numpy and Monte Carlo integrates
    TETS_PER_CUBED_SIZE / h(x)^3 over the domain. Wall distance is approximated
    as distance to the body bounding box, which understates the true distance
    and so overstates the count, making the estimate conservative.

    Returns (estimated_elements, domain_volume_m3).
    """
    x_inlet, x_outlet, y_lo, y_hi, z_lo, z_hi = domain
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(x_inlet, x_outlet, n_samples),
        rng.uniform(y_lo, y_hi, n_samples),
        rng.uniform(z_lo, z_hi, n_samples),
    ])

    lo = np.array(body_bbox[:3])
    hi = np.array(body_bbox[3:])
    d = np.linalg.norm(np.maximum(np.maximum(lo - pts, pts - hi), 0.0), axis=1)

    size_near = size_params.get("size_near", 2.0)
    size_far = size_params.get("size_far", 500.0)
    dist_min = size_params.get("dist_min", 200.0)
    dist_max = size_params.get("dist_max", 20000.0)
    if size_params.get("growth_rate"):
        dist_max = dist_min + (size_far - size_near) / float(size_params["growth_rate"])

    t = np.clip((d - dist_min) / max(dist_max - dist_min, 1e-9), 0.0, 1.0)
    h = size_near + (size_far - size_near) * t

    for box, size_in, _thickness in boxes:
        xmin, xmax, ymin, ymax, zmin, zmax = box
        inside = ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
                  (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
                  (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax))
        h = np.where(inside, np.minimum(h, size_in), h)

    h = np.clip(h, size_min, size_max)

    volume_mm3 = (x_outlet - x_inlet) * (y_hi - y_lo) * (z_hi - z_lo)
    density = float(np.mean(TETS_PER_CUBED_SIZE / h ** 3))
    return density * volume_mm3, volume_mm3 / 1e9


def find_ear_triangles(faces, sliver_gamma=0.1):
    """
    Wall triangles whose three nodes all lie on the face's boundary curves
    (no node in the face interior). Along a curved edge -- e.g. the circle
    where a cylinder meets a cone -- three consecutive boundary nodes form a
    sliver that lies almost in the plane of that edge, so its normal points
    along the edge's axis instead of out of the wall. Extruding a prism
    layer from it gives a near-flat prism and skews the averaged extrusion
    normal at its nodes; on the test body every inverted prism came from
    one of these. Returns {face: count}.
    """
    ears = {}
    for f in faces:
        try:
            etags, ntags = gmsh.model.mesh.getElementsByType(2, f)
        except Exception:
            continue
        if len(etags) == 0:
            continue
        interior, _, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=False)
        interior = set(int(n) for n in interior)
        tri = np.asarray(ntags, dtype=np.int64).reshape(-1, 3)
        all_bnd = np.array([not any(int(n) in interior for n in row) for row in tri])
        if not all_bnd.any():
            continue
        # A corner triangle of a planar patch can also have only boundary
        # nodes and be perfectly fine; only count the slivers.
        q = np.asarray(gmsh.model.mesh.getElementQualities(
            np.asarray(etags)[all_bnd].tolist(), "gamma"))
        n_ear = int((q < sliver_gamma).sum())
        if n_ear:
            ears[f] = n_ear
    return ears


def report_surface_mesh(section_wall_faces):
    """
    Triangle count and edge length range on the wall, split by detected
    section (nose/hump/rear/cavity/unsliced), classified by the actual face
    groups the segmentation produced -- exact, not a post-hoc guess from
    triangle centroids.
    """
    print("Wall surface mesh report:")
    for name, faces in section_wall_faces.items():
        count, lo, hi = 0, np.inf, 0.0
        for surf in faces:
            try:
                node_tags, coords, _ = gmsh.model.mesh.getNodes(2, surf, includeBoundary=True)
                elem_types, _, elem_nodes = gmsh.model.mesh.getElements(2, surf)
            except Exception:
                continue
            if node_tags.size == 0:
                continue
            tags = np.asarray(node_tags, dtype=np.int64)
            xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
            lookup = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
            lookup[tags] = np.arange(tags.size)
            for etype, enodes in zip(elem_types, elem_nodes):
                if etype != 2:
                    continue
                tri = np.asarray(enodes, dtype=np.int64).reshape(-1, 3)
                tri = tri[tri.max(axis=1) < lookup.size]
                if tri.size == 0:
                    continue
                idx = lookup[tri]
                tri = tri[~(idx < 0).any(axis=1)]
                if tri.size == 0:
                    continue
                p = xyz[lookup[tri]]
                e = np.stack([
                    np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
                    np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
                    np.linalg.norm(p[:, 2] - p[:, 0], axis=1),
                ], axis=1)
                count += tri.shape[0]
                lo = min(lo, float(e.min()))
                hi = max(hi, float(e.max()))
        if count:
            print(f"  {name}: {count} triangles, edges {lo:.3f} to {hi:.3f}mm")
        else:
            print(f"  {name}: no triangles")
    print()


def report_quality(threshold):
    """
    Worst tet quality and how many cells fall below threshold.

    Slivers are the usual cause of an immediate NaN in the solver, so the claim
    that optimisation removed them needs checking rather than assuming. Netgen
    optimisation is present in the official gmsh wheel and was verified to
    drive illegal tets to zero on a test case, but it is serial, so expect a
    long single threaded tail on a large mesh.
    """
    try:
        etypes, etags, _ = gmsh.model.mesh.getElements(3)
    except Exception as e:
        print(f"Quality check unavailable: {e}\n")
        return True

    worst = 1.0
    bad = 0
    total = 0
    for _etype, tags in zip(etypes, etags):
        if len(tags) == 0:
            continue
        q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSICN"), dtype=float)
        total += q.size
        worst = min(worst, float(q.min()))
        bad += int((q < threshold).sum())

    print(f"Mesh quality: {total} volume elements, worst minSICN={worst:.4f}, "
          f"{bad} below {threshold}")
    if bad:
        print("  Warning: cells below the threshold remain. These are the ones "
              "that produce a NaN on the first iteration. Consider raising "
              "Mesh.OptimizeThreshold, or relaxing the near-wall size so the "
              "gradient into the far field is gentler.")
    if worst <= 0.0:
        print("  Error: at least one inverted or zero volume element.")
        return False
    print()
    return True


def bl_heights(bl_params):
    """
    Cumulative layer heights for the prism stack.

    Driven by first cell size and growth ratio, since those are what the
    physics sets: the first cell resolves the near-wall gradient, the ratio
    controls how fast the profile is allowed to coarsen. Thickness in the
    config is honoured as a cap, because a stack taller than the boundary
    layer buys nothing and collides sooner.
    """
    h0 = bl_params.get("Size", 0.05)
    ratio = bl_params.get("Ratio", 1.1112)
    n = int(bl_params.get("NbLayers", 36))
    cap = bl_params.get("Thickness", 0.0)

    heights, acc = [], 0.0
    for i in range(n):
        acc += h0 * ratio ** i
        if cap > 0 and acc > cap * (1.0 + 1e-9):
            print(f"  Stack capped by Thickness={cap}mm at {len(heights)} "
                  f"layers instead of {n}.")
            break
        heights.append(acc)
    if not heights:
        return []
    print(f"  Prism stack: {len(heights)} layers, first {h0:.5f}mm, ratio "
          f"{ratio:.4f}, total {heights[-1]:.3f}mm")
    return heights


def estimate_local_wall_gap(xyz, nn, search_radius, max_gap):
    """
    Per-node wall-to-wall gap: distance from this wall point to the nearest
    point whose own normal faces roughly opposite this one, independent of
    which face either point belongs to.

    A plain nearest-neighbour search mostly finds the node's own immediate
    mesh neighbours (same surface, near-zero distance), which is not a gap.
    Requiring the candidate's normal to be roughly antiparallel (dot < -0.3)
    keeps only points on a facing, opposite skin. This is what catches a
    single large face that is mostly thick but pinches to near-zero gap in
    one region (e.g. a trailing edge) -- cavity_max_diag_mm and
    cavity_surface_tags only ever look at a whole face's own size, so
    neither can see a local thin spot on an otherwise-large face.

    search_radius should be picked against a known local thickness (e.g. the
    thinnest section you actually expect), not guessed: too small and the
    true opposite wall is missed (gap reported as max_gap, i.e. uncapped),
    too large and unrelated far-away skin gets pulled in (gap reported too
    small, over-capping good geometry).
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    gap = np.full(xyz.shape[0], max_gap)
    pairs = tree.query_ball_point(xyz, r=search_radius)
    for i, cand in enumerate(pairs):
        if not cand:
            continue
        cand = np.asarray(cand)
        cand = cand[cand != i]
        if cand.size == 0:
            continue
        facing = np.einsum('j,ij->i', nn[i], nn[cand]) < -0.3
        if not facing.any():
            continue
        d = np.linalg.norm(xyz[cand[facing]] - xyz[i], axis=1)
        gap[i] = min(gap[i], float(d.min()))
    return gap


def _face_in_box(f, box):
    """True if face f's bounding box overlaps the given (xmin,xmax,ymin,
    ymax,zmin,zmax) box at all -- used to select which wall faces a named
    surface_refinement_boxes region applies to, rather than sizing the raw
    3D volume inside the box."""
    bb = gmsh.model.getBoundingBox(2, f)
    xmin, xmax, ymin, ymax, zmin, zmax = box
    return not (bb[3] < xmin or bb[0] > xmax or bb[4] < ymin or bb[1] > ymax
               or bb[5] < zmin or bb[2] > zmax)


def detect_small_wall_faces(wall_faces, max_diag_mm):
    """
    Auto-detect small-scale wall faces (e.g. mm-scale cavities) by bounding
    box diagonal, so they can get their own size field and boundary layer
    treatment instead of the body-scale defaults.
    """
    small = []
    for f in wall_faces:
        bb = gmsh.model.getBoundingBox(2, f)
        diag = math.sqrt((bb[3] - bb[0]) ** 2 + (bb[4] - bb[1]) ** 2
                         + (bb[5] - bb[2]) ** 2)
        if diag <= max_diag_mm:
            small.append(f)
    return small


def group_curves_into_loops(curves):
    """
    Group a flat set of curve tags into closed loops by shared end points.

    addCurveLoop needs one loop at a time: handed the curves of two disjoint
    loops it reports "Curve loop N is wrong". A body can meet the symmetry
    plane in more than one closed silhouette, so the curves have to be sorted
    into loops before any of them can be used.
    """
    ends = {}
    for c in curves:
        pts = tuple(sorted(t for d, t in gmsh.model.getBoundary(
            [(1, c)], combined=False, oriented=False)))
        ends[c] = pts

    remaining = set(curves)
    loops = []
    while remaining:
        seed = remaining.pop()
        loop = [seed]
        touched = set(ends[seed])
        grew = True
        while grew:
            grew = False
            for c in list(remaining):
                if touched & set(ends[c]):
                    loop.append(c)
                    touched |= set(ends[c])
                    remaining.discard(c)
                    grew = True
        loops.append(loop)
    return loops


def rebuild_symmetry_plane(sym_outer_curves, strip_curve_groups, sym_coord=0.0):
    """
    Rebuild the symmetry plane as a surface with the prism strip(s) cut out.

    Where the body meets the symmetry plane, the prism blocks occupy a strip of
    that plane. The original symmetry face spans the whole cross section, strip
    included, so meshing it and then extruding gives two surfaces over the same
    area and generate(3) is handed a self-intersecting boundary.

    The fix is ordering plus a hole: the symmetry face is removed before the
    surface mesh is built, the layers are extruded, and the plane is then
    rebuilt from the domain outline with the outer edge of the strip as an
    interior boundary. The strip itself stays covered by the extrusion's own
    lateral faces, which carry the symmetry marker.

    strip_curve_groups is a list of curve-tag lists, one per section that was
    extruded separately (nose, hump, rear, cavity...): each group is grouped
    into loops ON ITS OWN, not pooled with the others first. Verified this
    matters: two sections extruded in separate extrudeBoundaryLayer calls do
    not share endpoint tags at the section boundary even where their strips
    are geometrically coincident, since each call builds its own new prism
    geometry from the shared base mesh rather than reusing the neighbour's
    curve entities. Pooling all curves and grouping once ("Curve loop N is
    wrong" on a real run) can misfile a curve from one section's strip into a
    loop being built from another's. Grouping each section's own curves
    independently sidesteps that: a contiguous section's strip is still one
    closed loop on its own, it just is not literally the same GEO entities as
    its neighbour's.

    Returns the new surface tags.
    """
    outer_loops = group_curves_into_loops(sym_outer_curves)
    hole_loops = []
    for group in strip_curve_groups:
        if group:
            hole_loops.extend(group_curves_into_loops(group))
    print(f"  Symmetry plane: {len(outer_loops)} outline loop(s), "
          f"{len(hole_loops)} strip loop(s) to cut out "
          f"(from {len(strip_curve_groups)} section group(s))")

    # The loops are built from the ORIGINAL curves -- the far-field faces'
    # edges and the prism strip's own outer edges -- not from fresh copies.
    # The rebuilt plane has to share those curves (and so their 1D mesh
    # nodes) with its neighbours, or the tet region's boundary is not
    # watertight. An earlier version rebuilt the loops from new straight
    # lines to force exact planarity; the result duplicated every node along
    # the strip and far-field edges, and the tet fill either came back empty
    # or (with HXT) crashed with "point ... was not inserted" and thousands
    # of "missing facets".
    for loop in hole_loops:
        deg = {}
        for c in loop:
            for _, p in gmsh.model.getBoundary([(1, c)], combined=False,
                                               oriented=False):
                deg[p] = deg.get(p, 0) + 1
        open_ends = [p for p, n in deg.items() if n != 2]
        if open_ends:
            raise RuntimeError(
                f"a boundary-layer strip on the symmetry plane does not close "
                f"(open at point(s) {open_ends}). This happens when a section "
                "with its boundary layer disabled sits between sections that "
                "keep theirs, splitting the strip. Enable the boundary layer "
                "on all sections, disable it only on an end section, or use "
                "the full body (enable_symmetry_half_model: false).")

    # The strip's outer edge is an offset along the extrusion's mesh
    # normals, which are only approximately in-plane, so its nodes sit
    # 0.1-0.7 mm off the plane on the test body. Left there, projecting them
    # into the plane surface's parametrization collapses some edges to zero
    # length ("Impossible to recover edge N N", hundreds of them, and the
    # 2D mesher never finished). Snapped in place -- same entities, same
    # node tags, so the strip faces that share them stay conforming.
    moved = 0
    for loop in hole_loops:
        for c in loop:
            for _, p in gmsh.model.getBoundary([(1, c)], combined=False,
                                               oriented=False):
                x, y, z = gmsh.model.getValue(0, p, [])
                if y != sym_coord:
                    gmsh.model.setCoordinates(p, x, sym_coord, z)
            ntags, ncoords, npar = gmsh.model.mesh.getNodes(1, c, includeBoundary=True)
            ncoords = np.asarray(ncoords, dtype=float).reshape(-1, 3)
            for i, n in enumerate(ntags):
                if ncoords[i, 1] != sym_coord:
                    ncoords[i, 1] = sym_coord
                    gmsh.model.mesh.setNode(int(n), ncoords[i].tolist(), [])
                    moved += 1
    print(f"  Snapped {moved} strip-edge node(s) onto the symmetry plane")
    # Every later generate() must keep those snapped nodes: without this,
    # gmsh re-meshes existing surfaces on the next call (verified with a
    # hand-edited triangle) and the strip goes back off-plane. Safe here
    # because the prism layers have already been meshed from their sources;
    # it could NOT be set before that point (the extrusion re-meshes its
    # source surfaces to build its node map, and fails without doing so).
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)

    holes = [gmsh.model.geo.addCurveLoop(l, reorient=True) for l in hole_loops]
    surfaces = []
    for outline in outer_loops:
        loop = gmsh.model.geo.addCurveLoop(outline, reorient=True)
        surfaces.append(gmsh.model.geo.addPlaneSurface([loop] + holes))
    gmsh.model.geo.synchronize()
    print(f"  Rebuilt symmetry surface(s): {surfaces}")
    return surfaces


def build_prism_layers(wall_faces, farfield_faces, symmetry_faces, bl_params,
                       sym_coord, tol, heights=None, view_index=-1):
    """
    Grow prism layers off the wall, then hand back the surfaces that bound the
    remaining tet region.

    Called ONCE for the whole BL-enabled wall (any section with boundary
    layer disabled, plus a cavity if present, is excluded from wall_faces
    beforehand and joins the tet region directly instead -- the same
    treatment the cavity already got). This single-call, single-recipe design
    is the one that survived: a separate extrudeBoundaryLayer call per
    section crashes gmsh the moment a second section's own call reaches the
    curve where it borders a section already extruded ("Could not find
    extruded node ... in surface N"), since that curve's extrusion
    bookkeeping only supports being the source of one such call, and two
    sections always share a boundary curve here by construction. The other
    candidate, a per-node scalar View passed via viewIndex to locally rescale
    the one shared heights[] array, produces geometrically correct surface
    offsets but was found -- checked with an actual generate(3), not just
    the 2D positions -- to make gmsh's "Extruded" 3D mesher realize every
    single prism as degenerate, even at a uniform (no-op) scale of 1.0.
    view_index therefore stays a parameter here for completeness but the
    caller always passes -1; see the step 9 comment in create_mesh for the
    full account and what per-section control remains available instead.

    Pass an empty farfield_faces if this is not the first (and, per the
    above, now only) call in a boundary-layer sequence within one run, or the
    farfield surfaces end up duplicated in the outer surface loop.

    Uses gmsh.model.geo.extrudeBoundaryLayer, which extrudes an already meshed
    surface along its mesh normals. This is NOT the BoundaryLayer size field:
    that field is 2D only and aborts generate(3) with "Only 2D Boundary Layers
    are supported". Verified on gmsh 4.15.2 that this path produces real
    prisms and that they survive the .su2 write as element type 13.

    Must be called after generate(2) and before generate(3), since there has to
    be a wall surface mesh to extrude from.

    Returns (ok, outer_surfaces, bl_volumes, sym_laterals, strip_curves).
    """
    if heights is None:
        heights = bl_heights(bl_params)
    if not heights:
        print("  No layers requested, skipping.")
        return False, [], [], [], []

    ex = gmsh.model.geo.extrudeBoundaryLayer(
        [(2, f) for f in wall_faces], [1] * len(heights), heights,
        bl_params.get("Quads", True), viewIndex=view_index)
    gmsh.model.geo.synchronize()
    # A freshly extruded surface has no mesh yet, and getBoundingBox raises
    # "Empty bounding box" on an unmeshed surface (verified on gmsh 4.15.2).
    # The symmetry-vs-seam check just below needs real coordinates, so mesh
    # these surfaces now; already-meshed entities are left untouched.
    gmsh.model.mesh.generate(2)

    # extrude returns, per input surface: (2, top), (3, volume), then the
    # lateral surfaces. Only the tops bound the outer tet region. Feeding the
    # lateral quads into the surface loop gives
    # "non-manifold quad boundaries not supported yet".
    tops, vols, laterals = [], [], []
    for i, (dim, tag) in enumerate(ex):
        if dim == 3:
            vols.append(tag)
        elif dim == 2:
            if i + 1 < len(ex) and ex[i + 1][0] == 3:
                tops.append(tag)
            else:
                laterals.append(tag)
    # A lateral shared by two neighbouring input faces appears twice in the
    # return and is internal. One appearing once sits on an open edge of this
    # call's own face set. On a half model most of those are the symmetry
    # plane, but if wall_faces excludes some faces that share an edge with
    # this set (e.g. a cavity carved out of one of these faces, so this face
    # has an inner hole boundary, or a neighbouring section extruded in its
    # own call), that hole's rim is *also* an open edge here -- and it is not
    # the symmetry plane. The two cases must be told apart geometrically (is
    # it at y=sym_coord?), not just by "appears once", or the non-symmetry
    # ones get folded into the symmetry marker and the seam is left out of
    # the outer boundary loop -- generate(3) then reports "No tetrahedra in
    # region N" with no error, because that loop was never actually closed.
    seen = {}
    for tag in laterals:
        seen[tag] = seen.get(tag, 0) + 1
    exterior = [tag for tag, n in seen.items() if n == 1]

    print(f"  Extruded {len(wall_faces)} wall face(s) into {len(vols)} prism "
          f"block(s): {len(tops)} tops, {len(seen)} laterals "
          f"({len(exterior)} exterior)")

    # tol (far_field_face_tolerance_mm, ~0.5mm by default) is sized for
    # picking out flat far-field box faces and is too tight here: even the
    # ORIGINAL wall faces, before any extrusion, sit a few mm off Y=sym_coord
    # at the symmetry plane on a body assembled from several fused OCC
    # primitives (measured up to 3.3mm on the synthetic test body, from
    # tessellation of the half-model cut, not from anything the extrusion
    # does). The offset then adds its own small normal-direction Y drift on
    # top. Classifying with the tight tol here dropped real symmetry-plane
    # laterals downstream of the nose into "seam" once that noise crossed
    # 0.5mm, which then left the hole loop for rebuild_symmetry_plane open
    # (unclosed) rather than a simple wrong split, and this only shows up on
    # geometry past a certain length/slenderness, not on every body. Scale
    # the tolerance with the actual layer thickness (the offset's own
    # contribution) and give it a generous floor for the pre-existing
    # tessellation noise, rather than reusing the far-field figure.
    sym_tol = max(tol, 5.0, 2.0 * (heights[-1] if heights else 0.0))
    sym_laterals, seam_laterals = [], []
    for tag in exterior:
        bb = gmsh.model.getBoundingBox(2, tag)
        if symmetry_faces and abs(bb[1] - sym_coord) < sym_tol and abs(bb[4] - sym_coord) < sym_tol:
            sym_laterals.append(tag)
        else:
            seam_laterals.append(tag)
    if sym_laterals:
        print(f"  {len(sym_laterals)} lateral face(s) lie in the symmetry "
              "plane and take the symmetry marker.")
    if seam_laterals:
        print(f"  {len(seam_laterals)} lateral face(s) are an open seam not "
              "on the symmetry plane (a section boundary, or a cavity rim "
              "within this wall set) -- added to the outer boundary, not the "
              "symmetry marker.")
        if os.environ.get("DEBUG_SYM_REBUILD"):
            for tag in seam_laterals:
                bb = gmsh.model.getBoundingBox(2, tag)
                print(f"    DEBUG seam_lateral {tag}: bbox={bb}")
    if os.environ.get("DEBUG_SYM_REBUILD"):
        for tag in wall_faces:
            bb = gmsh.model.getBoundingBox(2, tag)
            print(f"    DEBUG wall_face {tag}: bbox={bb}")
        for tag in sym_laterals:
            bb = gmsh.model.getBoundingBox(2, tag)
            print(f"    DEBUG sym_lateral {tag}: bbox={bb}")

    # Outer edge of the symmetry strip: curves used once across the exterior
    # laterals, not part of the original wall boundary, and actually sitting
    # at Y=sym_coord. This is the hole the rebuilt symmetry plane is cut
    # around. Pulled from BOTH sym_laterals and seam_laterals, not just
    # sym_laterals: a seam lateral is mostly a transverse cap that is NOT on
    # the symmetry plane (which is exactly why it did not qualify as a
    # sym_lateral above, by the same Y check applied to its own whole face),
    # but where that cap closes off a strip that stops short of the wall's
    # other end -- a section with its own boundary layer disabled, not just
    # the two true open ends of the whole wall -- ONE of its own four edges
    # still runs along Y=sym_coord, and that edge is exactly the strip's
    # closing curve there. Excluding seam_laterals entirely, as an earlier
    # version of this did, silently drops that edge and leaves the strip
    # loop open at every boundary where a neighbouring section has no
    # boundary layer, which addCurveLoop reports as "Curve loop N is wrong"
    # once you actually have such a boundary rather than just failing
    # loudly enough to say why. The per-CURVE Y check (not a per-face one)
    # is what keeps this from also pulling in the cap's other three edges,
    # which run away from the symmetry plane and belong to the outer
    # boundary instead.
    strip_curves = []
    if sym_laterals or seam_laterals:
        wall_edges = set(t for d, t in gmsh.model.getBoundary(
            [(2, f) for f in wall_faces], combined=False, oriented=False))
        # From the sym_laterals alone: the offset (outer) edge of the strip,
        # and any true open end (nose tip, tail) where a sym_lateral's own
        # curve is not shared with any neighbour at all. Counted within
        # sym_laterals only -- a curve a sym_lateral shares with a
        # NEIGHBOURING seam_lateral is real and wanted here (see below), so
        # counting it against the combined set would wrongly read as
        # "internal" and drop it.
        counts = {}
        for tag in sym_laterals:
            for d, c in gmsh.model.getBoundary([(2, tag)], combined=False,
                                               oriented=False):
                counts[c] = counts.get(c, 0) + 1
        strip_curves = [c for c, n in counts.items()
                        if n == 1 and c not in wall_edges]
        # From the seam_laterals: whichever of their own edges actually run
        # along Y=sym_coord (checked per curve, not by the seam face's own
        # bbox, which is mostly NOT on the plane -- that is exactly why it
        # did not qualify as a sym_lateral above). A seam cap sits wherever
        # this wall set's boundary layer stops short of the wall's true
        # open ends -- a section with its own boundary layer disabled, not
        # only the two ends of the whole wall -- and one of its four edges
        # is the strip's closing curve there. Skipping seam_laterals
        # entirely, as an earlier version of this did, leaves that edge
        # out and the strip loop open at every such boundary, which
        # addCurveLoop only reports as "Curve loop N is wrong" rather than
        # explaining why.
        seen = set(strip_curves)
        for tag in seam_laterals:
            for d, c in gmsh.model.getBoundary([(2, tag)], combined=False,
                                               oriented=False):
                if c in seen or c in wall_edges:
                    continue
                cbb = gmsh.model.getBoundingBox(1, c)
                if abs(cbb[1] - sym_coord) < sym_tol and abs(cbb[4] - sym_coord) < sym_tol:
                    seen.add(c)
                    strip_curves.append(c)
        print(f"  Strip outer edge: {len(strip_curves)} curve(s)")

    return True, tops + farfield_faces + seam_laterals, vols, sym_laterals, strip_curves




# ---------------------------------------------------------------------------
# Custom prism boundary layer (used for the half model, and on request for
# the full body).
#
# gmsh's own extrudeBoundaryLayer has two limits that make it unusable on a
# half model of the cavity body: one thickness for the whole wall (a 30 mm
# stack cannot enter a 5 mm cavity, so cavities had to be excluded and each
# one ended up under a 30 mm "chimney" of quad side faces), and no way to keep
# the stack in the symmetry plane, which forced the fragile rebuild of the
# plane around the prism strip (rebuild_symmetry_plane).
#
# This path builds the prisms directly from the wall surface mesh instead:
#   1. wall triangles are oriented consistently, pointing into the fluid
#   2. node normals are angle weighted; nodes on the symmetry plane get their
#      normal projected into the plane, so the stack stays in it
#   3. each node gets its own total thickness: the recipe thickness on the
#      main wall, cavity_thickness_mm on cavity faces, blended by a limited
#      thickness gradient in between. The first cell height is kept (y+ is
#      unchanged) and the growth ratio is solved per node, so every node has
#      the same layer count and the prisms stay conforming
#   4. every prism is checked for inversion; thickness is cut locally where
#      one inverts, until none do
#   5. the symmetry plane surface mesh is kept, not rebuilt: its nodes are
#      moved outward by the local stack thickness, decaying to zero over
#      symmetry_morph_length_mm, so its inner edge lands exactly on the top
#      of the prism strip
#   6. the tet region is a volume bounded by the stack top, the moved
#      symmetry plane and the far field, filled by generate(3)
# ---------------------------------------------------------------------------

def _gradient_limit(t, edges, lengths, slope, max_iter=100000):
    """t_i <= t_j + slope * |e_ij| over every mesh edge, in place."""
    i, j = edges[:, 0], edges[:, 1]
    for _ in range(max_iter):
        cand = np.full_like(t, np.inf)
        np.minimum.at(cand, i, t[j] + slope * lengths)
        np.minimum.at(cand, j, t[i] + slope * lengths)
        new = np.minimum(t, cand)
        if np.all(new >= t - 1e-12):
            return t
        t[:] = new
    return t


def _solve_ratios(h0, n, t):
    """
    First cell h and growth ratio r per node so that n layers sum to t.

    Where t >= n * h0 the first cell stays h0 and r >= 1 is solved; below that
    the layers are uniform and thinner than h0.
    """
    h = np.full_like(t, h0)
    r = np.ones_like(t)
    thin = t <= n * h0 * (1.0 + 1e-9)
    h[thin] = t[thin] / n
    grow = ~thin
    if grow.any():
        tt = t[grow]
        lo = np.ones_like(tt)
        hi = np.full_like(tt, 2.0)
        def total(rr):
            return h0 * (rr ** n - 1.0) / np.maximum(rr - 1.0, 1e-300)
        while np.any(total(hi) < tt):
            hi = np.where(total(hi) < tt, hi * 1.5, hi)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = total(mid) < tt
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        r[grow] = 0.5 * (lo + hi)
    return h, r


def _layer_offset(h, r, k):
    """Cumulative height of layer k (1-based) for each node."""
    out = h * k
    g = r > 1.0 + 1e-12
    out[g] = h[g] * (r[g] ** k - 1.0) / (r[g] - 1.0)
    return out


def _prism_dets(b, tp):
    """
    Six corner Jacobian determinants of the prisms between node layers b and
    tp (each m x 3 x 3: triangle, corner, xyz). All positive for a valid,
    correctly oriented prism.
    """
    def det(u, v, w):
        return np.einsum('ij,ij->i', np.cross(u, v), w)
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    t0, t1, t2 = tp[:, 0], tp[:, 1], tp[:, 2]
    return np.stack([
        det(b1 - b0, b2 - b0, t0 - b0),
        det(b2 - b1, b0 - b1, t1 - b1),
        det(b0 - b2, b1 - b2, t2 - b2),
        det(t1 - t0, t2 - t0, t0 - b0),
        det(t2 - t1, t0 - t1, t1 - b1),
        det(t0 - t2, t1 - t2, t2 - b2),
    ], axis=1)


def _prism_gamma(b, tp):
    """
    gmsh's 'gamma' of the prisms between node layers b and tp (m x 3 x 3):
    2/sqrt(3) times the smallest normalised corner Jacobian, i.e. the sine
    of the triangle corner times the cosine of the column tilt there.
    Checked against getElementQualities on 2,800 prisms (ratio 1.0000).
    """
    out = np.full(b.shape[0], np.inf)
    for lo, hi, sign in ((b, tp, 1.0), (tp, b, -1.0)):
        for i in range(3):
            p = lo[:, i]
            u = lo[:, (i + 1) % 3] - p
            v = lo[:, (i + 2) % 3] - p
            w = hi[:, i] - p
            det = sign * np.einsum('ij,ij->i', np.cross(u, v), w)
            den = (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
                   * np.linalg.norm(w, axis=1))
            out = np.minimum(out, det / np.maximum(den, 1e-300))
    return out * (2.0 / math.sqrt(3.0))


def _orient_wall_triangles(tris, ftri, xyz, sym_coord):
    """
    Orient every wall triangle so its normal points into the fluid.

    gmsh keeps each surface's triangles consistent with that surface's own
    parametrization, which can differ from face to face. Faces are first made
    consistent with their neighbours through shared mesh edges, then each
    connected wall piece is flipped as a whole if it points into the body,
    tested with the enclosed volume sum((y - sym) * n_y * A): the missing cap
    of a half body lies in the symmetry plane where (y - sym) is zero, so the
    sum is the half body volume, positive for outward (fluid side) normals.

    Returns (tris, open_edges, component_volumes). open_edges are mesh edges
    used by one triangle only (for a half model, the wall's trace on the
    symmetry plane).
    """
    m = tris.shape[0]
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    tid = np.tile(np.arange(m), 3)
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    fwd = e[:, 0] < e[:, 1]
    order = np.lexsort((hi, lo))
    lo, hi, fwd, tid = lo[order], hi[order], fwd[order], tid[order]
    same_as_next = (lo[:-1] == lo[1:]) & (hi[:-1] == hi[1:])
    starts = np.r_[True, ~same_as_next]
    grp = np.cumsum(starts) - 1
    counts = np.bincount(grp)
    if np.any(counts > 2):
        raise RuntimeError(f"{int((counts > 2).sum())} wall mesh edge(s) are "
                           "shared by more than two triangles (non-manifold "
                           "wall mesh)")
    first = np.nonzero(starts)[0]
    pair = counts[grp[first]] == 2
    a = first[pair]
    b = a + 1
    fa, fb = ftri[tid[a]], ftri[tid[b]]
    need_flip = fwd[a] == fwd[b]        # same direction: one of them is flipped
    inside = fa == fb
    if np.any(need_flip & inside):
        print(f"    Warning: {int((need_flip & inside).sum())} edge(s) with "
              "inconsistent orientation inside one face")
    cross = ~inside
    rel = np.unique(np.stack([fa[cross], fb[cross],
                              need_flip[cross].astype(np.int64)], axis=1), axis=0)
    adj = {}
    for x, y, fl in rel.tolist():
        adj.setdefault(x, []).append((y, fl))
        adj.setdefault(y, []).append((x, fl))
    faces = np.unique(ftri)
    flip, comp = {}, {}
    ncomp = 0
    for f0 in faces.tolist():
        if f0 in flip:
            continue
        flip[f0] = 0
        comp[f0] = ncomp
        stack = [f0]
        while stack:
            f = stack.pop()
            for g, fl in adj.get(f, []):
                want = flip[f] ^ fl
                if g not in flip:
                    flip[g] = want
                    comp[g] = ncomp
                    stack.append(g)
                elif flip[g] != want:
                    print(f"    Warning: faces {f} and {g} cannot be oriented "
                          "consistently (non-orientable wall?)")
        ncomp += 1
    fmax = int(faces.max()) + 1
    flip_arr = np.zeros(fmax, dtype=bool)
    comp_arr = np.zeros(fmax, dtype=np.int64)
    for f, v in flip.items():
        flip_arr[f] = bool(v)
        comp_arr[f] = comp[f]
    tris = tris.copy()
    ft = flip_arr[ftri]
    tris[ft] = tris[ft][:, [0, 2, 1]]

    p = xyz[tris]
    ny_area = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 1]
    ybar = p[:, :, 1].mean(axis=1) - sym_coord
    tri_comp = comp_arr[ftri]
    vols = np.bincount(tri_comp, weights=ybar * ny_area, minlength=ncomp)
    bad = vols[tri_comp] < 0
    tris[bad] = tris[bad][:, [0, 2, 1]]
    vols = np.abs(vols)

    open_edges = np.stack([lo[first[~pair]], hi[first[~pair]]], axis=1)
    return tris, open_edges, vols


def _node_normals(xyz, tris, sym_nodes, iters=60, target=0.3):
    """
    Extrusion direction per wall node.

    Starts from the angle weighted average of the adjacent triangle normals.
    At a corner where those normals spread by more than about 90 degrees
    (a cavity rim meeting a curved wall, a cavity's inner corner), the
    average can end up behind one of the adjacent faces, and the prism built
    on that face then folds whatever the thickness. There the faces that are
    seen worst are weighted up until every one is in front of the direction
    (normal . face normal > 0), which is what a valid prism needs. Nodes on
    the symmetry plane stay in the plane throughout.
    """
    p = xyz[tris]
    tn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    tn /= np.maximum(np.linalg.norm(tn, axis=1), 1e-300)[:, None]
    ang = np.zeros(tris.shape)
    for k in range(3):
        u = p[:, (k + 1) % 3] - p[:, k]
        v = p[:, (k + 2) % 3] - p[:, k]
        cosa = np.einsum('ij,ij->i', u, v) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        ang[:, k] = np.arccos(np.clip(cosa, -1.0, 1.0))
    nw = xyz.shape[0]

    def accumulate(w):
        out = np.zeros((nw, 3))
        for k in range(3):
            np.add.at(out, tris[:, k], tn * w[:, k, None])
        out[sym_nodes, 1] = 0.0
        ln = np.linalg.norm(out, axis=1)
        return out / np.maximum(ln, 1e-300)[:, None], ln

    nn, ln = accumulate(ang)
    if np.any(ln < 1e-12):
        raise RuntimeError(f"{int((ln < 1e-12).sum())} wall node(s) have no "
                           "usable normal (opposing faces cancel)")

    def visibility(nv):
        d = np.stack([np.einsum('ij,ij->i', nv[tris[:, k]], tn)
                      for k in range(3)], axis=1)
        md = np.full(nw, np.inf)
        np.minimum.at(md, tris.ravel(), d.ravel())
        return d, md

    d, md = visibility(nn)
    weak = np.nonzero(md < target)[0]
    if weak.size:
        # Maximise the worst dot product at each weak node by stepping its
        # direction towards its currently worst seen face (subgradient
        # ascent on min over faces of normal . face normal), keeping the best
        # direction found. Only the triangles around weak nodes take part.
        is_weak = np.zeros(nw, dtype=bool)
        is_weak[weak] = True
        tt, kk = np.nonzero(is_weak[tris])
        node = tris[tt, kk]
        fnorm = tn[tt]
        best = nn.copy()
        best_md = md.copy()
        cur = nn.copy()
        order_nodes = np.argsort(node, kind='stable')
        node_s = node[order_nodes]
        fnorm_s = fnorm[order_nodes]
        first = np.r_[True, node_s[1:] != node_s[:-1]]
        seg = np.cumsum(first) - 1
        uniq = node_s[first]
        step = 0.3
        for _ in range(iters * 5):
            dd = np.einsum('ij,ij->i', cur[node_s], fnorm_s)
            # worst face per node: sort by (segment, dot) and take the first
            o = np.lexsort((dd, seg))
            head = o[np.r_[True, seg[o][1:] != seg[o][:-1]]]
            worst_d = dd[head]
            improve = worst_d > best_md[uniq]
            best[uniq[improve]] = cur[uniq[improve]]
            best_md[uniq[improve]] = worst_d[improve]
            if np.all(best_md[uniq] >= target):
                break
            cur[uniq] = cur[uniq] + step * fnorm_s[head]
            cur[sym_nodes, 1] = 0.0
            cur[uniq] /= np.linalg.norm(cur[uniq], axis=1)[:, None]
            step *= 0.985
        nn = best
        d, md = visibility(nn)
    return nn, md


def _dijkstra_nearest(n_nodes, edges, lengths, sources, cutoff):
    """Graph distance to, and index of, the nearest source node, within cutoff."""
    import heapq
    adj_ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    both = np.concatenate([edges, edges[:, ::-1]])
    wl = np.concatenate([lengths, lengths])
    order = np.argsort(both[:, 0], kind='stable')
    both, wl = both[order], wl[order]
    np.add.at(adj_ptr, both[:, 0] + 1, 1)
    adj_ptr = np.cumsum(adj_ptr)
    nbr = both[:, 1].tolist()
    wl = wl.tolist()
    ptr = adj_ptr.tolist()
    dist = [math.inf] * n_nodes
    src = [-1] * n_nodes
    heap = []
    for s in sources.tolist():
        dist[s] = 0.0
        src[s] = s
        heap.append((0.0, s))
    heapq.heapify(heap)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for k in range(ptr[u], ptr[u + 1]):
            v = nbr[k]
            nd = d + wl[k]
            if nd < dist[v] and nd < cutoff:
                dist[v] = nd
                src[v] = src[u]
                heapq.heappush(heap, (nd, v))
    return np.asarray(dist), np.asarray(src, dtype=np.int64)


BL_STATS = {}


def build_custom_prism_layers(wall_faces, symmetry_faces, farfield_faces,
                              cavity_faces, bl_params, sym_coord,
                              fluid_volume_tags, sym_name, fluid_name="fluid",
                              thickness_spots=None, quality_min=0.0,
                              retry_wall=False):
    """
    Build the prism layer from the existing wall surface mesh (see the block
    comment above), replace the pre-layer fluid volume by the prism block and
    a tet region, and move the symmetry plane mesh around the stack.

    Must run after generate(2) (every wall, symmetry and far-field surface
    meshed) and before generate(3). Returns (prism_volume, tet_volume,
    top_surface).

    thickness_spots: optional local thickness factors from the iterative
    refinement loop (see create_mesh_iterative). Raises MeshDefectError, with
    locations, when the stack cannot be made valid.

    quality_min: prism gamma limit. A column whose upper layers fall below
    it (the layer triangles deform where neighbouring normals diverge) has
    its thickness cut locally, like a folding one. A column that is already
    below it in the first layer has a wall triangle too poor for its tilt;
    with retry_wall those are raised as a MeshDefectError of kind
    "prism_wall" (triangles in .wall_tris) before the model is touched, so
    the caller can repair those wall triangles in place and call again.
    """
    heights = bl_heights(bl_params)
    if not heights:
        raise RuntimeError("no boundary layers requested")
    n = len(heights)
    h0 = bl_params.get("Size", 0.05)
    t_main = heights[-1]
    t_cav = min(bl_params.get("cavity_thickness_mm", 0.5), t_main)
    slope = bl_params.get("thickness_gradient", 0.25)
    shrink = bl_params.get("inversion_shrink_factor", 0.6)
    max_fix = int(bl_params.get("max_inversion_passes", 25))

    # 1. Wall mesh
    all_tags, all_xyz, _ = gmsh.model.mesh.getNodes(-1, -1, includeBoundary=True,
                                                     returnParametricCoord=False)
    all_tags = np.asarray(all_tags, dtype=np.int64)
    all_xyz = np.asarray(all_xyz, dtype=float).reshape(-1, 3)
    lookup = np.full(int(all_tags.max()) + 1, -1, dtype=np.int64)
    lookup[all_tags] = np.arange(all_tags.size)

    tri_blocks, face_blocks = [], []
    for f in wall_faces:
        etypes, _, enodes = gmsh.model.mesh.getElements(2, f)
        for et, en in zip(etypes, enodes):
            if et != 2:
                raise RuntimeError(f"wall surface {f} has non-triangle "
                                   f"elements (type {et})")
            blk = np.asarray(en, dtype=np.int64).reshape(-1, 3)
            tri_blocks.append(blk)
            face_blocks.append(np.full(len(blk), f, dtype=np.int64))
    if not tri_blocks:
        raise RuntimeError("the wall has no surface mesh")
    tri_tags = np.concatenate(tri_blocks)
    ftri = np.concatenate(face_blocks)
    wnodes, tris = np.unique(tri_tags, return_inverse=True)
    tris = tris.reshape(-1, 3)
    if np.any(lookup[wnodes] < 0):
        raise RuntimeError("wall triangles reference unknown nodes")
    xyz = all_xyz[lookup[wnodes]]
    nw = wnodes.size
    print(f"  Wall mesh: {tris.shape[0]} triangles, {nw} nodes, "
          f"{len(wall_faces)} faces")

    tris, open_edges, vols = _orient_wall_triangles(tris, ftri, xyz, sym_coord)
    print(f"  Oriented into the fluid: {len(vols)} wall piece(s), enclosed "
          f"volume {vols.sum():.4e} mm3")

    # 2. Node normals
    sym_nodes = np.zeros(nw, dtype=bool)
    if symmetry_faces:
        if len(open_edges):
            sym_nodes[open_edges.ravel()] = True
        off = np.abs(xyz[sym_nodes, 1] - sym_coord)
        if off.size and off.max() > 1e-3 * max(1.0, t_main):
            raise RuntimeError(
                f"the wall mesh has open edges off the symmetry plane (up to "
                f"{off.max():.3f}mm away); the prism stack needs a closed wall "
                "apart from its trace on the symmetry plane")
        print(f"  Symmetry plane trace: {int(sym_nodes.sum())} node(s), "
              "normals kept in the plane")
    elif len(open_edges):
        raise RuntimeError(f"the wall mesh has {len(open_edges)} open edge(s) "
                           "but there is no symmetry plane")
    nn, min_dot = _node_normals(xyz, tris, sym_nodes)
    print(f"  Node normals: worst visibility (min normal . face normal) "
          f"{min_dot.min():.3f}")
    e_all = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e_len = np.linalg.norm(xyz[e_all[:, 0]] - xyz[e_all[:, 1]], axis=1)
    node_h = np.zeros(nw)
    np.maximum.at(node_h, e_all[:, 0], e_len)
    np.maximum.at(node_h, e_all[:, 1], e_len)
    del e_all, e_len
    if min_dot.min() <= 0.0:
        sharp = min_dot <= 0.0
        raise MeshDefectError(
            f"{int(sharp.sum())} wall node(s) have no extrusion "
            "direction that clears every adjacent face (a corner sharper "
            "than the prism stack can turn); coarsen or fillet the geometry "
            "there", points=xyz[sharp], sizes=node_h[sharp], kind="sharp_corner",
            in_bl=False)

    # 3. Thickness per node
    edges = np.unique(np.sort(np.concatenate(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    elen = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    t = np.full(nw, t_main)
    cav = np.zeros(nw, dtype=bool)
    if cavity_faces:
        cset = np.isin(ftri, np.asarray(list(cavity_faces), dtype=np.int64))
        cav[np.unique(tris[cset])] = True
        t[cav] = t_cav

    # Cap thickness wherever the body is locally thin, regardless of which
    # face it's on -- a large face can still pinch to near-zero gap (e.g. a
    # trailing edge), which the face-level cavity checks above cannot see.
    gap_safety = bl_params.get("gap_safety_factor", 0.0)
    if gap_safety > 0:
        search_radius = bl_params.get("gap_search_radius_mm", 3.0 * t_main)
        gap = estimate_local_wall_gap(xyz, nn, search_radius, t_main / gap_safety)
        gap_limit = gap * gap_safety
        n_capped = int((gap_limit < t).sum())
        t = np.minimum(t, gap_limit)
        print(f"  Local-gap cap: {n_capped} node(s) thickness-limited by a "
              f"nearby opposite wall (safety={gap_safety}, search radius="
              f"{search_radius:.1f}mm)")

    if thickness_spots:
        n_sp = apply_thickness_spots(t, xyz, thickness_spots)
        print(f"  Refinement thinning spots: {len(thickness_spots)} spot(s), "
              f"{n_sp} node(s) thinned")

    _gradient_limit(t, edges, elen, slope)
    print(f"  Thickness: {t_main:.3f}mm on the wall, {t_cav:.3f}mm on "
          f"{int(cav.sum())} cavity node(s), gap-capped where thinner, "
          f"gradient {slope}")

    # 4. Symmetry plane mesh: which of its nodes are on the wall trace, and
    # how far each other node is (along the plane mesh) from the trace
    sym_tri_tags = np.zeros((0, 3), dtype=np.int64)
    sym_new_xyz = None
    morph = bl_params.get("symmetry_morph_length_mm", 10.0 * t_main)
    if symmetry_faces:
        blocks = []
        for f in symmetry_faces:
            etypes, _, enodes = gmsh.model.mesh.getElements(2, f)
            for et, en in zip(etypes, enodes):
                if et != 2:
                    raise RuntimeError(f"symmetry surface {f} has non-triangle "
                                       f"elements (type {et})")
                blocks.append(np.asarray(en, dtype=np.int64).reshape(-1, 3))
        sym_tri_tags = np.concatenate(blocks)
        snodes, stris = np.unique(sym_tri_tags, return_inverse=True)
        stris = stris.reshape(-1, 3)
        sxyz = all_xyz[lookup[snodes]]
        wpos = np.clip(np.searchsorted(wnodes, snodes), 0, nw - 1)
        on_wall = wnodes[wpos] == snodes
        trace_s = np.nonzero(on_wall)[0]
        trace_w = wpos[on_wall]
        if trace_s.size != int(sym_nodes.sum()):
            print(f"    Note: {trace_s.size} symmetry nodes on the wall vs "
                  f"{int(sym_nodes.sum())} wall trace nodes")
        sedges = np.unique(np.sort(np.concatenate(
            [stris[:, [0, 1]], stris[:, [1, 2]], stris[:, [2, 0]]]), axis=1), axis=0)
        slen = np.linalg.norm(sxyz[sedges[:, 0]] - sxyz[sedges[:, 1]], axis=1)
        sdist, ssrc = _dijkstra_nearest(snodes.size, sedges, slen, trace_s, morph)
        sreach = ssrc >= 0
        sweight = np.where(sreach, (1.0 - np.minimum(sdist, morph) / morph) ** 2, 0.0)
        # wall node index of each symmetry node's nearest trace node
        s2w = np.full(snodes.size, -1, dtype=np.int64)
        s2w[trace_s] = trace_w
        ssrc_w = np.where(sreach, s2w[np.maximum(ssrc, 0)], -1)

        def area_xz(q):
            u = q[:, 1] - q[:, 0]
            v = q[:, 2] - q[:, 0]
            return u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        area0 = area_xz(sxyz[stris])

        def morph_symmetry(top_disp):
            d = np.zeros((snodes.size, 3))
            d[sreach] = top_disp[ssrc_w[sreach]] * sweight[sreach, None]
            moved = sxyz + d
            moved[:, 1] = sym_coord
            ratio = area_xz(moved[stris]) / np.where(area0 == 0.0, 1e-300, area0)
            return moved, ratio

    # 5. Inversion check, cutting thickness locally where a prism folds or
    # where the moved symmetry plane would fold (at a concave corner of the
    # trace, e.g. inside a cavity cut by the plane, the moved nodes converge)
    min_sym_ratio = bl_params.get("symmetry_min_area_ratio", 0.05)

    # Stack overlap: two facing walls (normals more than ~107 deg apart)
    # whose stacks together fill more than overlap_max_fill of the gap
    # between them. Checked on the final thickness, after the gradient limit
    # and the inversion cuts, so it also covers gap_safety_factor 0 or a
    # gap_search_radius_mm that missed the opposite wall.
    max_fill = bl_params.get("overlap_max_fill", 0.8)
    ov_pairs, ov_d = None, None
    if max_fill and max_fill > 0:
        from scipy.spatial import cKDTree
        r_ov = bl_params.get("overlap_search_radius_mm", 2.0 * t_main / max_fill)
        pr = cKDTree(xyz).query_pairs(r=r_ov, output_type='ndarray')
        if len(pr):
            facing = np.einsum('ij,ij->i', nn[pr[:, 0]], nn[pr[:, 1]]) < -0.3
            ov_pairs = pr[facing]
            ov_d = np.linalg.norm(xyz[ov_pairs[:, 0]] - xyz[ov_pairs[:, 1]], axis=1)
        print(f"  Overlap check: {0 if ov_pairs is None else len(ov_pairs)} "
              f"facing node pair(s) within {r_ov:.2f}mm, stacks may fill "
              f"{max_fill} of each gap")
    passes = 0
    qcuts = np.zeros(nw, dtype=np.int16)
    max_qcuts = int(bl_params.get("max_quality_cuts", 6))
    # Converging normals at a concave crease squeeze the stack top: a 15 mm
    # wall edge can end up 3 mm long there, and the tets standing on it
    # cannot be well shaped. Thinning the column locally undoes the squeeze.
    squeeze_min = float(bl_params.get("top_edge_shrink_min", 0.4)) \
        if quality_min > 0 else 0.0

    def _min_edge(q):
        return np.minimum(np.minimum(
            np.linalg.norm(q[:, 0] - q[:, 1], axis=1),
            np.linalg.norm(q[:, 1] - q[:, 2], axis=1)),
            np.linalg.norm(q[:, 2] - q[:, 0], axis=1))
    wall_min_edge = _min_edge(xyz[tris])
    wall_poor = np.zeros(tris.shape[0], dtype=bool)
    q_left = 0
    while True:
        h, r = _solve_ratios(h0, n, t)
        bad = np.zeros(tris.shape[0], dtype=bool)
        qbad = np.zeros(tris.shape[0], dtype=bool)
        prev = xyz[tris]
        for k in range(1, n + 1):
            cur = (xyz + nn * _layer_offset(h, r, k)[:, None])[tris]
            bad |= np.any(_prism_dets(prev, cur) <= 0.0, axis=1)
            if quality_min > 0:
                low = _prism_gamma(prev, cur) < quality_min
                if k == 1:
                    wall_poor = low
                else:
                    qbad |= low
                if k == n and squeeze_min > 0:
                    qbad |= _min_edge(cur) < squeeze_min * wall_min_edge
            prev = cur
        cut = [np.unique(tris[bad])]
        # poor upper layers: thin those columns, a limited number of times
        qbad &= ~wall_poor & ~bad
        q_left = int(qbad.sum())
        nq = 0
        if q_left:
            qn = np.unique(tris[qbad])
            qn = qn[qcuts[qn] < max_qcuts]
            nq = int(qn.size)
            if nq:
                qcuts[qn] += 1
                cut.append(qn)
        nsym = 0
        if symmetry_faces:
            top_disp = nn * t[:, None]
            top_disp[:, 1] = 0.0
            _, ratio = morph_symmetry(top_disp)
            sbad = ratio < min_sym_ratio
            nsym = int(sbad.sum())
            if nsym:
                src_w = ssrc_w[stris[sbad].ravel()]
                cut.append(np.unique(src_w[src_w >= 0]))
        nov = 0
        if ov_pairs is not None and len(ov_pairs):
            over = (t[ov_pairs[:, 0]] + t[ov_pairs[:, 1]]) > max_fill * ov_d
            nov = int(over.sum())
            if nov:
                cut.append(np.unique(ov_pairs[over]))
        nbad = int(bad.sum())
        if nbad == 0 and nsym == 0 and nov == 0 and nq == 0:
            break
        passes += 1
        if passes > max_fix and nbad == 0 and nsym == 0 and nov == 0:
            break
        if passes > max_fix:
            where = np.unique(np.concatenate(cut))
            raise MeshDefectError(
                f"{nbad} prism column(s), {nsym} symmetry triangle(s) and "
                f"{nov} overlapping node pair(s) still fold after {max_fix} "
                "thickness reductions", points=xyz[where],
                sizes=node_h[where],
                kind="prism_fold", in_bl=True)
        idx = np.unique(np.concatenate(cut))
        t[idx] *= shrink
        _gradient_limit(t, edges, elen, slope)
        print(f"    pass {passes}: {nbad} folding prism column(s), {nsym} "
              f"folding symmetry triangle(s), {nov} overlapping pair(s), "
              f"{q_left} column(s) with poor or squeezed upper layers, "
              f"thickness cut at {idx.size} node(s)")
    if quality_min > 0:
        print(f"  Prism quality: {int(wall_poor.sum())} column(s) below gamma "
              f"{quality_min} from the first layer (wall triangle shape or "
              f"tilt), {q_left} only in upper layers")
        if retry_wall and wall_poor.any():
            wp = np.nonzero(wall_poor)[0]
            p0 = xyz[tris[wp]]
            nrm = np.cross(p0[:, 1] - p0[:, 0], p0[:, 2] - p0[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
            cosv = np.einsum('ijk,ik->ij', nn[tris[wp]], nrm).min(axis=1)
            err = MeshDefectError(
                f"{wp.size} wall triangle(s) too poor for their column tilt",
                points=p0.mean(axis=1), sizes=node_h[tris[wp]].max(axis=1),
                kind="prism_wall", in_bl=False)
            err.wall_tris = wnodes[tris[wp]]
            err.cos_tilt = np.clip(cosv, 0.05, 1.0)
            raise err
    h, r = _solve_ratios(h0, n, t)
    # First cell per wall node, for the y+ check. _solve_ratios keeps h0
    # wherever the stack is thick enough and goes uniform and THINNER below
    # that, so thinning (gap cap, inversion cuts, refinement spots) can only
    # lower y+, never raise it.
    BL_STATS.clear()
    BL_STATS.update({"xyz": xyz.copy(), "h_first": h.copy(), "h0": h0})
    print(f"  Prism stack: {n} layers, first cell {h.min():.5f} to "
          f"{h.max():.5f}mm, ratio {r.min():.4f} to {r.max():.4f}, total "
          f"{t.min():.3f} to {t.max():.3f}mm")

    # Layer node coordinates; layer 0 is the wall itself
    layer_xyz = [xyz + nn * _layer_offset(h, r, k)[:, None] for k in range(1, n + 1)]
    if symmetry_faces:
        for lx in layer_xyz:
            lx[sym_nodes, 1] = sym_coord
        top_disp = layer_xyz[-1] - xyz
        top_disp[:, 1] = 0.0
        sym_new_xyz, ratio = morph_symmetry(top_disp)
        print(f"  Symmetry plane: {stris.shape[0]} triangles, "
              f"{int((sweight > 0).sum())} node(s) moved over {morph:.1f}mm, "
              f"smallest area ratio {ratio.min():.3f}")

    # 6. Replace the fluid volume and the symmetry faces by discrete entities
    old_groups = []
    for dim in (2, 3):
        for d_, tg in gmsh.model.getPhysicalGroups(dim):
            nm = gmsh.model.getPhysicalName(d_, tg)
            if (dim == 3) or (dim == 2 and nm == sym_name):
                old_groups.append((d_, tg))
    if old_groups:
        gmsh.model.removePhysicalGroups(old_groups)
    gmsh.model.removeEntities([(3, v) for v in fluid_volume_tags])
    if symmetry_faces:
        gmsh.model.removeEntities([(2, f) for f in symmetry_faces])

    top = gmsh.model.addDiscreteEntity(2)
    strip = gmsh.model.addDiscreteEntity(2) if symmetry_faces else None
    sym_surf = gmsh.model.addDiscreteEntity(2) if symmetry_faces else None
    prism_vol = gmsh.model.addDiscreteEntity(3)

    next_node = int(gmsh.model.mesh.getMaxNodeTag()) + 1
    layer_tags = [wnodes]
    for k in range(1, n + 1):
        tags_k = np.arange(next_node, next_node + nw, dtype=np.int64)
        next_node += nw
        layer_tags.append(tags_k)
        ent = (2, top) if k == n else (3, prism_vol)
        gmsh.model.mesh.addNodes(ent[0], ent[1], tags_k.tolist(),
                                 layer_xyz[k - 1].ravel().tolist())

    top_conn = layer_tags[n][tris]
    gmsh.model.mesh.addElementsByType(top, 2, [], top_conn.ravel().tolist())
    for k in range(1, n + 1):
        lo_t = layer_tags[k - 1][tris]
        hi_t = layer_tags[k][tris]
        conn = np.concatenate([lo_t, hi_t], axis=1)
        gmsh.model.mesh.addElementsByType(prism_vol, 6, [], conn.ravel().tolist())

    if symmetry_faces:
        # strip quads along the trace, one per open wall edge per layer
        oe = open_edges
        quads = []
        for k in range(1, n + 1):
            a0, b0 = layer_tags[k - 1][oe[:, 0]], layer_tags[k - 1][oe[:, 1]]
            a1, b1 = layer_tags[k][oe[:, 0]], layer_tags[k][oe[:, 1]]
            quads.append(np.stack([a0, b0, b1, a1], axis=1))
        gmsh.model.mesh.addElementsByType(strip, 3, [],
                                          np.concatenate(quads).ravel().tolist())

        # symmetry triangles: trace nodes -> top layer nodes; free interior
        # nodes (they belonged to the removed faces) get new tags; nodes on
        # curves (shared with the far field) keep theirs
        cur_tags = np.asarray(gmsh.model.mesh.getNodes(
            -1, -1, includeBoundary=True, returnParametricCoord=False)[0],
            dtype=np.int64)
        still = np.zeros(lookup.size, dtype=bool)
        still[cur_tags[cur_tags < still.size]] = True
        new_tag = snodes.copy()
        new_tag[trace_s] = layer_tags[n][trace_w]
        free = (~on_wall) & (~still[snodes])
        kept = (~on_wall) & still[snodes]
        free_idx = np.nonzero(free)[0]
        new_tag[free_idx] = np.arange(next_node, next_node + free_idx.size)
        next_node += free_idx.size
        gmsh.model.mesh.addNodes(2, sym_surf, new_tag[free_idx].tolist(),
                                 sym_new_xyz[free_idx].ravel().tolist())
        # Nodes that survived the face removal sit on curves. Those shared
        # with the far field must not move (the far field is metres away,
        # so they never should); the others are curves inside the plane,
        # e.g. where a cavity cross-section face met the main symmetry face,
        # and take their moved position.
        kidx = np.nonzero(kept)[0]
        if kidx.size:
            dk = np.linalg.norm(sym_new_xyz[kidx] - sxyz[kidx], axis=1)
            moving = kidx[dk > 0.0]
            if moving.size:
                far_tags = [np.asarray(gmsh.model.mesh.getNodes(
                    2, f, includeBoundary=True, returnParametricCoord=False)[0],
                    dtype=np.int64) for f in farfield_faces]
                if far_tags and np.isin(snodes[moving],
                                        np.concatenate(far_tags)).any():
                    raise RuntimeError("the symmetry morph reaches the "
                                       "far-field boundary; reduce "
                                       "symmetry_morph_length_mm")
                for i in moving.tolist():
                    gmsh.model.mesh.setNode(int(snodes[i]),
                                            sym_new_xyz[i].tolist(), [])
        gmsh.model.mesh.addElementsByType(sym_surf, 2, [],
                                          new_tag[stris].ravel().tolist())

    bounding = [top] + ([sym_surf] if symmetry_faces else []) + list(farfield_faces)
    loop = gmsh.model.geo.addSurfaceLoop(bounding)
    tet_vol = gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()

    if symmetry_faces:
        g = gmsh.model.addPhysicalGroup(2, [sym_surf, strip])
        gmsh.model.setPhysicalName(2, g, sym_name)
    g = gmsh.model.addPhysicalGroup(3, [prism_vol, tet_vol])
    gmsh.model.setPhysicalName(3, g, fluid_name)
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
    print(f"  Prism block: {tris.shape[0] * n} prisms in volume {prism_vol}, "
          f"tet region volume {tet_vol}")
    return prism_vol, tet_vol, top

def report_quality_mixed(threshold):
    """
    Quality report that separates prisms from tets.

    An isotropic metric is the wrong test for a viscous prism: a 1000:1 aspect
    ratio cell is meant to be sliver shaped and scores near zero on minSICN by
    construction. Aborting on that rejects every valid boundary layer mesh.
    What actually matters for a prism is that its volume is positive, so the
    threshold is applied to tets only and prisms are checked for inversion.
    """
    try:
        etypes, etags, _ = gmsh.model.mesh.getElements(3)
    except Exception as e:
        print(f"Quality check unavailable: {e}\n")
        return True

    stats = {}
    inverted = 0
    inverted_xyz = []
    for etype in sorted(set(int(t) for t in etypes)):
        tags, ntags = gmsh.model.mesh.getElementsByType(etype)
        if len(tags) == 0:
            continue
        q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSICN"),
                       dtype=float)
        name = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}.get(etype, str(etype))
        prev = stats.get(name, [0, 1.0, 0])
        bad = int((q < threshold).sum()) if name == "tet" else 0
        stats[name] = [prev[0] + q.size, min(prev[1], float(q.min())),
                       prev[2] + bad]
        bad_idx = np.nonzero(q <= 0.0)[0]
        inverted += bad_idx.size
        if bad_idx.size:
            npe = len(ntags) // len(tags)
            conn = np.asarray(ntags).reshape(-1, npe)[bad_idx]
            for row in conn[:2000]:
                pts = [gmsh.model.mesh.getNode(int(n))[0] for n in row]
                inverted_xyz.append((name, np.mean(pts, axis=0)))
            if os.environ.get("DEBUG_INVERTED"):
                for row in conn[:3]:
                    print(f"    DEBUG {name} nodes {list(map(int, row))}")
                    for n in row:
                        c, _, d, t = gmsh.model.mesh.getNode(int(n))
                        print(f"      node {int(n)} dim{d} ent{t} {np.round(c, 4).tolist()}")

    print("Mesh quality by element type:")
    for name, (count, worst, bad) in sorted(stats.items()):
        line = f"  {name}: {count} cells, worst minSICN={worst:.4f}"
        if name == "tet":
            line += f", {bad} below {threshold}"
        else:
            line += "  (threshold not applied: anisotropic by design)"
        print(line)

    if inverted_xyz:
        xyz = np.array([p for _, p in inverted_xyz])
        # Group by rounded X station so a cluster at one junction reads as one
        # line rather than dozens.
        step = max(1.0, 0.002 * float(np.ptp(xyz[:, 0]) + 1.0) * 50)
        keys = np.round(xyz[:, 0] / step) * step
        print("  Inverted element locations (centroid, grouped by X):")
        for k in sorted(set(keys.tolist()))[:12]:
            sel = xyz[keys == k]
            kinds = sorted(set(n for (n, _), kk in zip(inverted_xyz, keys) if kk == k))
            r = np.hypot(sel[:, 1], sel[:, 2])
            print(f"    x~{k:.0f}: {len(sel)} {'/'.join(kinds)}, "
                  f"x=[{sel[:, 0].min():.1f},{sel[:, 0].max():.1f}] "
                  f"y=[{sel[:, 1].min():.1f},{sel[:, 1].max():.1f}] "
                  f"z=[{sel[:, 2].min():.1f},{sel[:, 2].max():.1f}] "
                  f"r(yz)=[{r.min():.1f},{r.max():.1f}]")
    if inverted:
        print(f"  Error: {inverted} inverted or zero volume element(s). On a "
              "prism mesh this usually means the layers collided, at a sharp "
              "trailing edge or in a concave junction. Reduce NbLayers or "
              "Thickness, or coarsen the wall mesh there.")
        return False

    tet_bad = stats.get("tet", [0, 1.0, 0])[2]
    if tet_bad:
        print("  Warning: poor tets remain. These are the ones that produce a "
              "NaN on the first iteration.")
    print()
    return True


# ---------------------------------------------------------------------------
# Iterative local refinement.
#
# create_mesh_iterative() runs create_mesh() repeatedly. Each pass checks the
# mesh against the criteria in auto_refinement_parameters and, where a check
# fails, turns the failing locations into local fixes for the next pass:
#
#   size spots  Ball size fields centred on the bad cells, at refine_factor
#               times the local cell size, decaying back to size_far at
#               spot_growth_rate (so they grade like the main wall field).
#               They refine the wall surface and the tets around it.
#   BL spots    a local thickness factor on the custom prism stack, for
#               defects in or on the prism layer (folded or sheared prisms,
#               stacks overlapping, poor tets sitting on the stack top).
#
# Checks, in pipeline order:
#   2D   degenerate wall triangles (gamma below min_wall_triangle_gamma);
#        fixed in place by re-meshing the surface (max_surface_passes) since
#        this is cheap and nothing depends on the surface mesh yet.
#   BL   stacks overlapping across a thin gap (custom builder, fixed in
#        place by thinning, like the existing inversion check), and the
#        tet region boundary (stack top plus symmetry plane) intersecting
#        itself, which is what makes the 3D fill fail.
#   3D   inverted cells, degenerate cells (tet / pyramid minSICN, prism
#        gamma, which measures shear but not aspect ratio), face
#        non-orthogonality and, optionally, neighbour volume ratio.
#
# Spots are kept across passes and written to spots_file, so a converged set
# can be reused (resume_from_spots_file) without repeating the iterations. A
# spot that keeps failing is tightened each pass until max_spot_level or
# min_spot_size_mm, then marked stuck; when every remaining defect sits in a
# stuck spot the loop stops.
# ---------------------------------------------------------------------------

REFINE_DEFAULTS = {
    "enable": False,
    "max_iterations": 5,
    "max_surface_passes": 3,
    "criteria": {
        "max_inverted": 0,
        "min_tet_sicn": 0.02,
        "min_pyramid_sicn": 0.01,
        "min_prism_gamma": 0.15,
        "max_non_orthogonality_deg": 80.0,
        "max_volume_ratio": 0.0,
        "max_yplus": 1.0,
        "max_prism_aspect_ratio": 0.0,
        "max_bl_top_jump": 0.0,
        "max_tet_aspect_ratio": 0.0,
        "min_wall_triangle_gamma": 0.1,
        "check_boundary_intersections": True,
        "allowed_bad_cells": 0,
    },
    "refine_factor": 0.5,
    "spot_radius_factor": 3.0,
    "spot_growth_rate": None,
    "min_spot_size_mm": 1.0,
    "max_spot_level": 4,
    "cluster_factor": 2.0,
    "max_new_spots_per_iteration": 150,
    "max_total_spots": 400,
    "bl_thickness_factor": 0.6,
    "min_bl_thickness_factor": 0.05,
    "spots_file": "",
    "resume_from_spots_file": False,
    "export_if_not_converged": False,
    # Local repair: defects are fixed in place, on and next to the bad
    # cells, in the mesh that is already in memory. global_remesh_fallback
    # true hands whatever is left to the older whole-mesh loop above.
    "local_repair": {
        "enable": True,
        "max_iterations": 8,
        "wall_rings": 2,
        "max_wall_passes": 4,
        "wall_sine_margin": 1.5,
        "short_edge_ratio": 0.15,
        "short_edge_reach": 8.0,
        "max_forced_sine": 0.5,
        "cavity_rings": 1,
        "refine_factor": 0.6,
        "spot_radius_factor": 1.5,
        "spot_growth_rate": 0.3,
        "max_spot_level": 4,
        "cluster_factor": 2.0,
        "max_new_spots_per_iteration": 5000,
        "max_total_spots": 50000,
        "global_remesh_fallback": False,
    },
}


class MeshDefectError(RuntimeError):
    """A meshing step failed at known locations, which local refinement or a
    locally thinner boundary layer may fix on the next pass."""

    def __init__(self, message, points=None, sizes=None, kind="defect", in_bl=True):
        super().__init__(message)
        pts = np.zeros((0, 3)) if points is None else np.asarray(points, float)
        self.points = pts.reshape(-1, 3)
        self.sizes = (np.full(len(self.points), np.nan) if sizes is None
                      else np.asarray(sizes, float).ravel())
        self.kind = kind
        self.in_bl = in_bl


def refinement_settings(config):
    ar = dict(REFINE_DEFAULTS)
    ar.update(config.get("auto_refinement_parameters", {}))
    crit = dict(REFINE_DEFAULTS["criteria"])
    crit.update(config.get("auto_refinement_parameters", {}).get("criteria", {}))
    ar["criteria"] = crit
    lr = dict(REFINE_DEFAULTS["local_repair"])
    lr.update(config.get("auto_refinement_parameters", {}).get("local_repair", {}))
    lr.setdefault("min_spot_size_mm", ar["min_spot_size_mm"])
    ar["local_repair"] = lr
    if not ar.get("spot_growth_rate"):
        ar["spot_growth_rate"] = config.get("size_field_parameters", {}).get(
            "growth_rate") or 0.2
    return ar


def make_defect(kind, points, sizes, in_bl, target=None, tags=None, etype=None):
    """One defect group: kind, (N,3) points, (N,) local cell size, in_bl,
    and optionally (N,) explicit target sizes (e.g. the wall spacing an
    aspect ratio limit needs) used instead of refine_factor x size."""
    pts = np.asarray(points, float).reshape(-1, 3)
    h = np.asarray(sizes, float).ravel()
    if h.size != len(pts):
        h = np.full(len(pts), np.nan)
    flags = np.broadcast_to(np.asarray(in_bl, bool), (len(pts),)).copy()
    tgt = np.full(len(pts), np.inf) if target is None else \
        np.broadcast_to(np.asarray(target, float), (len(pts),)).copy()
    out = {"kind": kind, "points": pts, "h": h, "in_bl": flags, "target": tgt}
    if tags is not None:
        out["tags"] = np.asarray(tags, dtype=np.int64).ravel()
        out["etype"] = np.broadcast_to(np.asarray(etype, dtype=np.int8),
                                       (len(pts),)).copy()
    return out


def cluster_defect_points(points, h, cluster_factor, target=None):
    """
    Bin defect points into clusters so a band of hundreds of bad cells
    becomes a handful of spots. Bins scale with the local cell size
    (log2-quantised), so a cluster never spans more than a few cells.
    Returns a list of dicts sorted by defect count, largest first.
    """
    if len(points) == 0:
        return []
    h = np.maximum(np.asarray(h, float), 1e-6)
    q = np.round(np.log2(h)).astype(np.int64)
    bs = cluster_factor * np.power(2.0, q)
    keys = np.column_stack([q, np.floor(points / bs[:, None]).astype(np.int64)])
    uk, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    centers = np.zeros((len(uk), 3))
    np.add.at(centers, inv, points)
    centers /= cnt[:, None]
    hmin = np.full(len(uk), np.inf)
    np.minimum.at(hmin, inv, h)
    tmin = np.full(len(uk), np.inf)
    if target is not None:
        np.minimum.at(tmin, inv, np.asarray(target, float))
    out = [{"x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
            "h": float(hh), "target": float(tt), "count": int(n)}
           for c, hh, tt, n in zip(centers, hmin, tmin, cnt)]
    out.sort(key=lambda d: -d["count"])
    return out


def merge_spots(spots, cands, ar, kind="size", budget=None, stage="volume"):
    """
    Merge clustered defect candidates into the persistent spot list.

    Candidates inside an existing spot tighten it (smaller size or thinner
    stack, level + 1), but only while that helps: a spot whose defect count
    did not drop since it was last tightened is marked stuck and left alone.
    That is what stops the loop refining a sharp crease forever, where a
    finer mesh only makes more of the same poor cells. Counts are kept per
    stage (surface / volume), since the two checks count different things.
    Other candidates become new spots, up to budget. Returns (changed spots,
    number new, number tightened, number stuck).
    """
    changed, n_new, n_tight, n_stuck = [], 0, 0, 0
    budget = len(cands) if budget is None else budget
    key = "last_count_" + stage
    centers = np.array([[s["x"], s["y"], s["z"]] for s in spots]) if spots \
        else np.zeros((0, 3))
    radii = np.array([s["radius"] for s in spots]) if spots else np.zeros(0)
    hits, fresh = {}, []
    for c in cands:
        p = np.array([c["x"], c["y"], c["z"]])
        radius = ar["spot_radius_factor"] * c["h"]
        hit = -1
        if len(centers):
            d = np.linalg.norm(centers - p, axis=1)
            inside = np.nonzero(d < np.maximum(radii, 0.5 * radius))[0]
            if inside.size:
                hit = int(inside[np.argmin(d[inside])])
        if hit < 0:
            fresh.append(c)
            continue
        cnt, h, tg = hits.get(hit, (0, np.inf, np.inf))
        hits[hit] = (cnt + c["count"], min(h, c["h"]), min(tg, c.get("target", np.inf)))

    for k, (cnt, h, tg) in hits.items():
        s = spots[k]
        last = s.get(key)
        if (s.get("stuck") or s["level"] >= ar["max_spot_level"]
                or (last is not None and cnt >= last)):
            s["stuck"] = True
            n_stuck += 1
            continue
        if kind == "size":
            new = max(ar["min_spot_size_mm"],
                      min(s["size"] * ar["refine_factor"], ar["refine_factor"] * h, tg))
            if new >= s["size"] * (1.0 - 1e-6):
                s["stuck"] = True
                n_stuck += 1
                continue
            s["size"] = new
        else:
            new = max(ar["min_bl_thickness_factor"],
                      s["factor"] * ar["bl_thickness_factor"])
            if new >= s["factor"] * (1.0 - 1e-6):
                s["stuck"] = True
                n_stuck += 1
                continue
            s["factor"] = new
        s["level"] += 1
        s["radius"] = max(s["radius"], ar["spot_radius_factor"] * h)
        s[key] = cnt
        changed.append(s)
        n_tight += 1

    for c in fresh[:max(0, budget)]:
        s = {"x": c["x"], "y": c["y"], "z": c["z"],
             "radius": ar["spot_radius_factor"] * c["h"],
             "level": 1, "stuck": False, key: c["count"]}
        if kind == "size":
            tg = c.get("target", np.inf)
            s["size"] = max(ar["min_spot_size_mm"],
                            tg if np.isfinite(tg) else ar["refine_factor"] * c["h"])
        else:
            s["factor"] = max(ar["min_bl_thickness_factor"], ar["bl_thickness_factor"])
        spots.append(s)
        changed.append(s)
        n_new += 1
    return changed, n_new, n_tight, n_stuck


def register_defects(state, defects, ar, use_bl_spots=True, stage="volume"):
    """
    Turn defect groups into new or tightened spots in state. Returns the
    number of spots changed (0 means nothing more can be done locally).
    """
    pts = [d["points"] for d in defects if len(d["points"])]
    if not pts:
        return 0
    pts = np.concatenate(pts)
    h = np.concatenate([d["h"] for d in defects if len(d["points"])])
    in_bl = np.concatenate([d["in_bl"] for d in defects if len(d["points"])])
    tgt = np.concatenate([d.get("target", np.full(len(d["points"]), np.inf))
                          for d in defects if len(d["points"])])
    if np.isnan(h).any():
        fill = np.nanmedian(h) if np.isfinite(h).any() else \
            10.0 * ar["min_spot_size_mm"]
        h = np.where(np.isnan(h), fill, h)

    room = max(0, ar["max_total_spots"] - len(state["size_spots"]))
    budget = min(ar["max_new_spots_per_iteration"], room)
    cands = cluster_defect_points(pts, h, ar["cluster_factor"], tgt)
    ch, n_new, n_tight, n_stuck = merge_spots(state["size_spots"], cands, ar,
                                              "size", budget, stage)
    print(f"  Size spots: {n_new} new, {n_tight} tightened, {n_stuck} stuck "
          f"({len(state['size_spots'])} total)")
    if room == 0:
        print(f"  Note: max_total_spots ({ar['max_total_spots']}) reached, "
              "only existing spots can be tightened")
    total = len(ch)
    if use_bl_spots and in_bl.any():
        bc = cluster_defect_points(pts[in_bl], h[in_bl], ar["cluster_factor"])
        room_b = max(0, ar["max_total_spots"] - len(state["bl_spots"]))
        chb, nb_new, nb_tight, nb_stuck = merge_spots(
            state["bl_spots"], bc, ar, "bl",
            min(ar["max_new_spots_per_iteration"], room_b), stage)
        print(f"  Boundary layer thinning spots: {nb_new} new, {nb_tight} "
              f"tightened, {nb_stuck} stuck ({len(state['bl_spots'])} total)")
        total += len(chb)
    return total


def add_spot_fields(spots, size_far, growth):
    """One Ball field per size spot. Returns the field ids."""
    ids = []
    for s in spots:
        f = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(f, "VIn", s["size"])
        gmsh.model.mesh.field.setNumber(f, "VOut", size_far)
        gmsh.model.mesh.field.setNumber(f, "Radius", s["radius"])
        gmsh.model.mesh.field.setNumber(f, "Thickness",
                                        max(size_far - s["size"], 0.0) / growth)
        gmsh.model.mesh.field.setNumber(f, "XCenter", s["x"])
        gmsh.model.mesh.field.setNumber(f, "YCenter", s["y"])
        gmsh.model.mesh.field.setNumber(f, "ZCenter", s["z"])
        ids.append(f)
    return ids


def apply_thickness_spots(t, xyz, spots):
    """Scale the prism stack thickness t in place inside each BL spot (the
    smallest factor wins where spots overlap). Returns nodes affected."""
    if not spots:
        return 0
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    fac = np.ones(len(t))
    for s in spots:
        idx = tree.query_ball_point([s["x"], s["y"], s["z"]], r=s["radius"])
        if idx:
            idx = np.asarray(idx)
            fac[idx] = np.minimum(fac[idx], s["factor"])
    t *= fac
    return int((fac < 1.0).sum())


def wall_triangle_defects(faces, gamma_min):
    """Wall triangles with gamma below gamma_min: (points, sizes)."""
    pts, hs = [], []
    for f in faces:
        try:
            tags, _ = gmsh.model.mesh.getElementsByType(2, f)
        except Exception:
            continue
        if len(tags) == 0:
            continue
        q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "gamma"))
        bad = np.nonzero(q < gamma_min)[0]
        if bad.size == 0:
            continue
        bary = np.asarray(gmsh.model.mesh.getBarycenters(2, f, False, True)).reshape(-1, 3)
        pts.append(bary[bad])
        hs.append(np.asarray(gmsh.model.mesh.getElementQualities(
            np.asarray(tags)[bad].tolist(), "maxEdge")))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(pts), np.concatenate(hs)


def _node_lookup():
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    lookup = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    lookup[tags] = np.arange(tags.size)
    return lookup, xyz


def _segments_hit_triangles(p, q, a, b, c, eps=1e-9):
    """Vectorised segment pq vs triangle abc test (Moller-Trumbore), strict
    interior on both, so shared edges and touching corners do not count."""
    d = q - p
    e1 = b - a
    e2 = c - a
    pv = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, pv)
    ok = np.abs(det) > eps * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) \
        * np.linalg.norm(d, axis=1) + 1e-300
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = p - a
    u = np.einsum('ij,ij->i', tv, pv) * inv
    qv = np.cross(tv, e1)
    v = np.einsum('ij,ij->i', d, qv) * inv
    s = np.einsum('ij,ij->i', e2, qv) * inv
    tol = 1e-7
    return ok & (u > tol) & (v > tol) & (u + v < 1 - tol) & (s > tol) & (s < 1 - tol)


def boundary_intersections(surfaces, chunk=200000):
    """
    Triangles of the given surfaces that intersect another, non-adjacent
    triangle of the same set. This is the 'segment and facet intersect'
    failure of the 3D fill, found before the fill is attempted, and with
    locations, which Delaunay does not report at all.

    Candidate pairs come from a KD-tree on centroids with a per-triangle
    radius of twice its own circumradius bound, so each intersecting pair is
    found from the larger triangle's side. Returns (points, sizes).
    """
    from scipy.spatial import cKDTree
    blocks = []
    for s in surfaces:
        try:
            _, nt = gmsh.model.mesh.getElementsByType(2, s)
        except Exception:
            continue
        if len(nt):
            blocks.append(np.asarray(nt, dtype=np.int64).reshape(-1, 3))
    if not blocks:
        return np.zeros((0, 3)), np.zeros(0)
    tri = np.concatenate(blocks)
    lookup, xyz = _node_lookup()
    P = xyz[lookup[tri]]
    cen = P.mean(axis=1)
    R = np.linalg.norm(P - cen[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(cen)
    hit = np.zeros(len(tri), dtype=bool)
    for lo in range(0, len(tri), chunk):
        hi = min(lo + chunk, len(tri))
        lists = tree.query_ball_point(cen[lo:hi], r=2.0 * R[lo:hi] + 1e-9)
        lens = np.fromiter((len(l) for l in lists), dtype=np.int64, count=hi - lo)
        if lens.sum() == 0:
            continue
        i = np.repeat(np.arange(lo, hi), lens)
        j = np.fromiter((x for l in lists for x in l), dtype=np.int64, count=int(lens.sum()))
        keep = j != i
        i, j = i[keep], j[keep]
        # drop pairs that share a node: neighbours touch by construction
        shared = np.zeros(len(i), dtype=bool)
        for a in range(3):
            for b in range(3):
                shared |= tri[i, a] == tri[j, b]
        i, j = i[~shared], j[~shared]
        if len(i) == 0:
            continue
        A, B, C = P[j, 0], P[j, 1], P[j, 2]
        x = np.zeros(len(i), dtype=bool)
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            x |= _segments_hit_triangles(P[i, e0], P[i, e1], A, B, C)
        hit[i[x]] = True
        hit[j[x]] = True
    idx = np.nonzero(hit)[0]
    if idx.size == 0:
        return np.zeros((0, 3)), np.zeros(0)
    edges = np.stack([np.linalg.norm(P[idx, 0] - P[idx, 1], axis=1),
                      np.linalg.norm(P[idx, 1] - P[idx, 2], axis=1),
                      np.linalg.norm(P[idx, 2] - P[idx, 0], axis=1)], axis=1)
    return cen[idx], edges.max(axis=1)


_COORD_RE = re.compile(r'\(\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*,\s*'
                       r'([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*,\s*'
                       r'([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*\)')


def parse_message_points(messages):
    """(x, y, z) locations quoted in gmsh error messages about intersecting
    or unrecoverable boundary facets."""
    pts = []
    for msg in messages:
        low = msg.lower()
        if not any(k in low for k in ("intersect", "overlap", "recover", "invalid")):
            continue
        for g in _COORD_RE.findall(msg):
            pts.append([float(v) for v in g])
    return np.asarray(pts, dtype=float).reshape(-1, 3)


def _pair_faces(fn):
    """Indices (a, b) of rows of fn (F x k node tags) that are the same face."""
    s = np.sort(fn, axis=1)
    primes = (73856093, 19349663, 83492791, 2654435761)
    key = s[:, 0].astype(np.int64) * np.int64(primes[0])
    for c in range(1, s.shape[1]):
        key ^= s[:, c].astype(np.int64) * np.int64(primes[c])
    order = np.argsort(key, kind='stable')
    ks = key[order]
    cand = np.nonzero(ks[1:] == ks[:-1])[0]
    a, b = order[cand], order[cand + 1]
    same = np.all(s[a] == s[b], axis=1)
    return a[same], b[same]


def collect_volume_defects(crit, interface_nodes=None, pair_chunk=2000000):
    """
    Check the 3D mesh against the refinement criteria.

    Per cell: inverted (minSICN <= 0, any type), degenerate tets and pyramids
    (minSICN), sheared prisms (gamma, which is aspect-ratio independent: a
    right prism scores 0.82 at any aspect ratio, 0.58 at 45 degrees of shear,
    0.14 at 80). Per interior face: non-orthogonality, the angle between the
    face normal and the line joining the two cell centroids, and optionally
    the neighbour volume ratio.

    Memory: the face pass holds every cell face once, roughly 100 bytes per
    cell. Returns (defect groups, summary dict).
    """
    names = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}
    lookup, xyz = _node_lookup()
    iface = None
    if interface_nodes is not None and len(interface_nodes):
        iface = np.zeros(lookup.size, dtype=bool)
        nodes = np.asarray(interface_nodes, dtype=np.int64)
        iface[nodes[nodes < lookup.size]] = True

    etypes = [int(t) for t in gmsh.model.mesh.getElementTypes(3)]
    defects = []
    summary = {"cells": 0, "inverted": 0, "degenerate": 0, "non_orthogonal_faces": 0,
               "volume_ratio_faces": 0, "aspect_ratio": 0, "worst": {}}
    cent_blocks, vol_blocks, tag_blocks, type_blocks = [], [], [], []
    tri_faces, tri_owner, quad_faces, quad_owner = [], [], [], []
    offset = 0
    need_vol = crit.get("max_volume_ratio", 0) > 0

    for et in etypes:
        tags, ntags = gmsh.model.mesh.getElementsByType(et)
        if len(tags) == 0:
            continue
        tags = np.asarray(tags, dtype=np.int64)
        m = tags.size
        name = names.get(et, str(et))
        conn = np.asarray(ntags, dtype=np.int64).reshape(m, -1)
        bary = np.asarray(gmsh.model.mesh.getBarycenters(et, -1, False, True)).reshape(-1, 3)
        sicn = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSICN"))
        inv = sicn <= 0.0
        if name == "tet":
            deg = (sicn < crit["min_tet_sicn"]) & ~inv
            worst = float(sicn.min())
        elif name == "pyramid":
            deg = (sicn < crit["min_pyramid_sicn"]) & ~inv
            worst = float(sicn.min())
        elif name in ("prism", "hex"):
            gam = np.asarray(gmsh.model.mesh.getElementQualities(tags, "gamma"))
            deg = (gam < crit["min_prism_gamma"]) & ~inv
            worst = float(gam.min())
        else:
            deg = np.zeros(m, dtype=bool)
            worst = float(sicn.min())
        summary["worst"][name] = {"metric": "gamma" if name in ("prism", "hex")
                                  else "minSICN", "value": worst,
                                  "min_sicn": float(sicn.min()), "count": int(m)}
        # inverted tets touching the prism stack top count as boundary layer defects
        if name == "tet":
            touches = iface[conn].any(axis=1) if iface is not None \
                else np.zeros(m, dtype=bool)
        else:
            touches = np.ones(m, dtype=bool)
        summary["cells"] += m
        summary["inverted"] += int(inv.sum())
        summary["degenerate"] += int(deg.sum())
        for label, mask in (("inverted", inv), ("degenerate_" + name, deg)):
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                continue
            h = np.asarray(gmsh.model.mesh.getElementQualities(
                tags[idx].tolist(), "maxEdge"))
            # Only validity failures thin the stack. A low-gamma prism is a
            # poor wall triangle or fanned normals, which thinning does not
            # change (gamma is the same at any height); a poor tet is a
            # sizing problem. Both get size spots only.
            flag = touches[idx] if label == "inverted" else False
            defects.append(make_defect(label, bary[idx], h, flag,
                                       tags=tags[idx], etype=et))

        # Aspect ratio. Prism: largest base edge over column height, per
        # cell, so the first layer (thinnest) is the one that binds. Tet:
        # maxEdge / (2 sqrt(6) inradius), 1 for a regular tet.
        ar_p = crit.get("max_prism_aspect_ratio", 0) or 0
        if name == "prism" and ar_p > 0:
            worst_ar, bad_ar = 0.0, []
            for lo in range(0, m, 1000000):
                c = conn[lo:lo + 1000000]
                P = xyz[lookup[c]]
                base = P[:, :3]
                lat = np.max(np.stack([np.linalg.norm(base[:, 0] - base[:, 1], axis=1),
                                       np.linalg.norm(base[:, 1] - base[:, 2], axis=1),
                                       np.linalg.norm(base[:, 2] - base[:, 0], axis=1)],
                                      axis=1), axis=1)
                hgt = np.linalg.norm(P[:, 3:] - base, axis=2).mean(axis=1)
                a = lat / np.maximum(hgt, 1e-300)
                worst_ar = max(worst_ar, float(a.max()))
                sel = np.nonzero(a > ar_p)[0]
                if sel.size:
                    bad_ar.append((lo + sel, lat[sel], hgt[sel]))
            summary["worst_prism_aspect_ratio"] = worst_ar
            if bad_ar:
                idx = np.concatenate([b[0] for b in bad_ar])
                lat = np.concatenate([b[1] for b in bad_ar])
                hgt = np.concatenate([b[2] for b in bad_ar])
                summary["aspect_ratio"] += int(idx.size)
                defects.append(make_defect("prism_aspect_ratio", bary[idx], lat,
                                           False, target=0.85 * ar_p * hgt,
                                           tags=tags[idx], etype=et))
        ar_t = crit.get("max_tet_aspect_ratio", 0) or 0
        if name == "tet" and ar_t > 0:
            r_in = np.asarray(gmsh.model.mesh.getElementQualities(tags, "innerRadius"))
            e_max = np.asarray(gmsh.model.mesh.getElementQualities(tags, "maxEdge"))
            a = e_max / np.maximum(2.0 * math.sqrt(6.0) * r_in, 1e-300)
            summary["worst_tet_aspect_ratio"] = float(a.max())
            sel = np.nonzero(a > ar_t)[0]
            if sel.size:
                summary["aspect_ratio"] += int(sel.size)
                defects.append(make_defect("tet_aspect_ratio", bary[sel],
                                           e_max[sel], False,
                                           tags=tags[sel], etype=et))
            del r_in, e_max, a

        cent_blocks.append(bary)
        tag_blocks.append(tags)
        type_blocks.append(np.full(m, et, dtype=np.int8))
        if need_vol:
            vol_blocks.append(np.abs(np.asarray(
                gmsh.model.mesh.getElementQualities(tags, "volume"))))
        del conn, sicn
        for ftype, store, owners in ((3, tri_faces, tri_owner), (4, quad_faces, quad_owner)):
            try:
                fn = gmsh.model.mesh.getElementFaceNodes(et, ftype, -1, True)
            except Exception:
                continue
            if len(fn) == 0:
                continue
            fn = lookup[np.asarray(fn, dtype=np.int64)].reshape(-1, ftype)
            if xyz.shape[0] < 2 ** 31 - 1:
                fn = fn.astype(np.int32)
            per = fn.shape[0] // m
            store.append(fn)
            owners.append((offset + np.repeat(np.arange(m, dtype=np.int64), per)))
        offset += m

    if not cent_blocks:
        return defects, summary
    cent = np.concatenate(cent_blocks)
    tag_all = np.concatenate(tag_blocks)
    type_all = np.concatenate(type_blocks)
    vol = np.concatenate(vol_blocks) if need_vol else None
    del cent_blocks, vol_blocks

    max_angle = crit.get("max_non_orthogonality_deg", 0) or 0
    max_ratio = crit.get("max_volume_ratio", 0) or 0
    worst_angle, worst_ratio = 0.0, 1.0
    bad_no, bad_vr = [], []
    for store, owners in ((tri_faces, tri_owner), (quad_faces, quad_owner)):
        if not store:
            continue
        fn = np.concatenate(store)
        own = np.concatenate(owners)
        store.clear()
        owners.clear()
        a, b = _pair_faces(fn)
        for lo in range(0, len(a), pair_chunk):
            aa, bb = a[lo:lo + pair_chunk], b[lo:lo + pair_chunk]
            e1, e2 = own[aa], own[bb]
            p = xyz[fn[aa]]
            if fn.shape[1] == 3:
                n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            else:
                n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
            d = cent[e2] - cent[e1]
            cosv = np.abs(np.einsum('ij,ij->i', n, d)) / np.maximum(
                np.linalg.norm(n, axis=1) * np.linalg.norm(d, axis=1), 1e-300)
            ang = np.degrees(np.arccos(np.clip(cosv, 0.0, 1.0)))
            if ang.size:
                worst_angle = max(worst_angle, float(ang.max()))
            if max_angle > 0:
                sel = ang > max_angle
                if sel.any():
                    bad_no.append(np.concatenate([e1[sel], e2[sel]]))
                    summary["non_orthogonal_faces"] += int(sel.sum())
            if vol is not None:
                v1, v2 = vol[e1], vol[e2]
                r = np.maximum(v1, v2) / np.maximum(np.minimum(v1, v2), 1e-300)
                if r.size:
                    worst_ratio = max(worst_ratio, float(r.max()))
                sel = r > max_ratio
                if sel.any():
                    bad_vr.append(np.concatenate([e1[sel], e2[sel]]))
                    summary["volume_ratio_faces"] += int(sel.sum())
        del fn, own, a, b
    summary["worst_non_orthogonality_deg"] = worst_angle
    if vol is not None:
        summary["worst_volume_ratio"] = worst_ratio

    for label, blocks in (("non_orthogonal", bad_no), ("volume_ratio", bad_vr)):
        if not blocks:
            continue
        idx = np.unique(np.concatenate(blocks))
        h = np.asarray(gmsh.model.mesh.getElementQualities(
            tag_all[idx].tolist(), "maxEdge"))
        defects.append(make_defect(label, cent[idx], h, False,
                                   tags=tag_all[idx], etype=type_all[idx]))
    return defects, summary


# ---------------------------------------------------------------------------
# Local repair. Defects are fixed where they are: the wall triangles on and
# next to a poor one are re-triangulated in place, and the tets on and next
# to a poor one are removed and that cavity alone is filled again. Nothing
# else in the mesh is touched, so the cell count only changes inside those
# patches. The CAD geometry is never modified; new wall nodes are projected
# onto the CAD face or curve they belong to.
# ---------------------------------------------------------------------------

def _tri_gamma_pts(p0, p1, p2):
    """gamma (2 x inradius / circumradius) of one triangle; 1 if equilateral."""
    a = math.dist(p1, p2)
    b = math.dist(p2, p0)
    c = math.dist(p0, p1)
    n = np.cross(p1 - p0, p2 - p0)
    area2 = float(np.dot(n, n))          # (2A)^2
    den = a * b * c * (a + b + c)
    return 4.0 * area2 / den if den > 0.0 else 0.0


def _tri_quality_pts(p0, p1, p2, gamma_min, sine_min):
    """
    Wall triangle quality relative to its limits: below 1 is a defect.
    gamma is the triangle shape measure; the corner sine is what a prism
    extruded from the triangle inherits (gmsh's prism gamma is 2/sqrt(3) x
    the worst corner sine x the cosine of the column tilt), so a needle with
    one short edge can pass gamma and still give poor prisms on every layer.
    """
    a = math.dist(p1, p2)
    b = math.dist(p2, p0)
    c = math.dist(p0, p1)
    n = np.cross(p1 - p0, p2 - p0)
    area2 = float(np.dot(n, n))          # (2A)^2
    den = a * b * c * (a + b + c)
    if den <= 0.0:
        return 0.0
    q = np.inf
    if gamma_min > 0:
        q = 4.0 * area2 / den / gamma_min
    if sine_min > 0:
        e = sorted((a, b, c))
        q = min(q, math.sqrt(area2) / (e[1] * e[2]) / sine_min)
    return q


def _tri_gamma_array(P):
    """gamma, longest edge and worst corner sine of triangles P (m, 3, 3)."""
    a = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)
    b = np.linalg.norm(P[:, 2] - P[:, 0], axis=1)
    c = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area2 = np.einsum('ij,ij->i', n, n)
    den = a * b * c * (a + b + c)
    e = np.sort(np.stack([a, b, c], axis=1), axis=1)
    sine = np.sqrt(area2) / np.maximum(e[:, 1] * e[:, 2], 1e-300)
    return np.where(den > 0, 4.0 * area2 / np.maximum(den, 1e-300), 0.0), \
        e[:, 2], sine


class _WallPatch:
    """
    A set of wall triangles being re-triangulated in place. Vertices on the
    rim of the patch never move and rim edges are never changed, so the
    patch stays conforming with the untouched mesh around it. An edge between
    two CAD faces is a feature: it can be split (the new node goes on the CAD
    curve) but not flipped, and a node on it only slides along it.
    """

    def __init__(self, tri_tags, tri_nodes, tri_face, xyz_of, nfaces_of,
                 next_node, min_edge, max_edge, gamma_min, sine_min,
                 cos_dev=0.9):
        self.v = {}                       # node tag -> xyz
        self.t = {}                       # tri id -> (a, b, c, face)
        self.v2t = {}
        self.nfaces = dict(nfaces_of)     # CAD faces around each node
        self.owner = {}                   # new node -> face entity
        self.removed = {}                 # original tri tag -> face
        self.next_node = int(next_node)
        self.next_tri = -1
        self.min_edge = float(min_edge)
        self.max_edge = float(max_edge) if max_edge and max_edge > 0 else np.inf
        self.cos_dev = cos_dev
        self.gamma_min = float(gamma_min)
        self.sine_min = float(sine_min)
        self._curves = {}
        self.counts = {"collapse": 0, "flip": 0, "split": 0}
        for tg, nd, f in zip(tri_tags, tri_nodes, tri_face):
            a, b, c = (int(x) for x in nd)
            for x in (a, b, c):
                if x not in self.v:
                    self.v[x] = np.asarray(xyz_of[x], dtype=float)
            self._add(int(tg), a, b, c, int(f))
        self.original = set(self.t)
        # rim: an edge with a single patch triangle
        self.frozen = set()
        for tid, (a, b, c, f) in self.t.items():
            for u, w in ((a, b), (b, c), (c, a)):
                if len(self.v2t[u] & self.v2t[w]) == 1:
                    self.frozen.add(u)
                    self.frozen.add(w)

    # -- structure ---------------------------------------------------------
    def _add(self, tid, a, b, c, f):
        self.t[tid] = (a, b, c, f)
        for x in (a, b, c):
            self.v2t.setdefault(x, set()).add(tid)

    def _del(self, tid):
        a, b, c, f = self.t.pop(tid)
        for x in (a, b, c):
            self.v2t[x].discard(tid)
        if tid in self.original:
            self.removed[tid] = f

    def _new(self, a, b, c, f):
        tid = self.next_tri
        self.next_tri -= 1
        self._add(tid, a, b, c, f)
        return tid

    def edge_tris(self, u, w):
        return self.v2t[u] & self.v2t[w]

    def gamma(self, tid):
        """Quality relative to the limits (see _tri_quality_pts)."""
        a, b, c, _ = self.t[tid]
        return _tri_quality_pts(self.v[a], self.v[b], self.v[c],
                                self.gamma_min, self.sine_min)

    def _gamma_of(self, a, b, c):
        return _tri_quality_pts(self.v[a], self.v[b], self.v[c],
                                self.gamma_min, self.sine_min)

    def _normal_of(self, a, b, c):
        n = np.cross(self.v[b] - self.v[a], self.v[c] - self.v[a])
        l = math.sqrt(float(np.dot(n, n)))
        return n / l if l > 0 else n

    def edges_len(self, tid):
        a, b, c, _ = self.t[tid]
        return [(math.dist(self.v[a], self.v[b]), a, b),
                (math.dist(self.v[b], self.v[c]), b, c),
                (math.dist(self.v[c], self.v[a]), c, a)]

    def is_feature(self, u, w):
        ts = list(self.edge_tris(u, w))
        return len(ts) == 2 and self.t[ts[0]][3] != self.t[ts[1]][3]

    # -- CAD projection ----------------------------------------------------
    def _face_curves(self, f):
        if f not in self._curves:
            self._curves[f] = set(abs(t) for d, t in gmsh.model.getBoundary(
                [(2, f)], combined=False, oriented=False))
        return self._curves[f]

    def _project(self, p, f1, f2, scale):
        """p moved onto CAD face f1 (or the curve between f1 and f2)."""
        best = None
        try:
            if f1 == f2:
                q, _ = gmsh.model.getClosestPoint(2, f1, p.tolist())
                best = np.asarray(q, dtype=float)
            else:
                for c in self._face_curves(f1) & self._face_curves(f2):
                    q, _ = gmsh.model.getClosestPoint(1, c, p.tolist())
                    q = np.asarray(q, dtype=float)
                    if best is None or math.dist(q, p) < math.dist(best, p):
                        best = q
        except Exception:
            best = None
        if best is None or not np.all(np.isfinite(best)) or \
                math.dist(best, p) > 0.5 * scale:
            return p
        return best

    # -- operations --------------------------------------------------------
    def flip(self, u, w, force=False):
        """Flip edge uw if that raises the worse gamma of its two triangles."""
        ts = list(self.edge_tris(u, w))
        if len(ts) != 2:
            return False
        t1, t2 = ts
        if self.t[t1][3] != self.t[t2][3]:
            return False
        f = self.t[t1][3]
        a1 = [x for x in self.t[t1][:3] if x != u and x != w]
        a2 = [x for x in self.t[t2][:3] if x != u and x != w]
        if len(a1) != 1 or len(a2) != 1:
            return False
        p, q = a1[0], a2[0]
        if p == q or self.edge_tris(p, q):
            return False
        if math.dist(self.v[p], self.v[q]) > self.max_edge:
            return False
        # keep the orientation of t1: (u, w, p) in its own cyclic order
        tri = self.t[t1][:3]
        i = tri.index(u)
        if tri[(i + 1) % 3] != w:
            u, w = w, u
        n_old1 = self._normal_of(*self.t[t1][:3])
        n_old2 = self._normal_of(*self.t[t2][:3])
        new1, new2 = (u, q, p), (q, w, p)
        g_old = min(self.gamma(t1), self.gamma(t2))
        g_new = min(self._gamma_of(*new1), self._gamma_of(*new2))
        if not force and g_new <= 1.05 * g_old + 1e-12:
            return False
        for nt in (new1, new2):
            nn_ = self._normal_of(*nt)
            if np.dot(nn_, n_old1) < self.cos_dev or np.dot(nn_, n_old2) < self.cos_dev:
                return False
        self._del(t1)
        self._del(t2)
        self._new(new1[0], new1[1], new1[2], f)
        self._new(new2[0], new2[1], new2[2], f)
        self.counts["flip"] += 1
        return True

    def split(self, u, w):
        """Split edge uw at its midpoint, projected onto the CAD."""
        ts = list(self.edge_tris(u, w))
        if len(ts) != 2:
            return None
        length = math.dist(self.v[u], self.v[w])
        if 0.5 * length < self.min_edge:
            return None
        f1, f2 = self.t[ts[0]][3], self.t[ts[1]][3]
        mid = self._project(0.5 * (self.v[u] + self.v[w]), f1, f2, length)
        m = self.next_node
        self.v[m] = mid
        # refuse a split that would fold or flatten a triangle
        plan = []
        for tid in ts:
            a, b, c, f = self.t[tid]
            n_old = self._normal_of(a, b, c)
            for old, keep in ((u, w), (w, u)):
                tri = tuple(m if x == old else x for x in (a, b, c))
                if np.dot(self._normal_of(*tri), n_old) < 0.5:
                    del self.v[m]
                    return None
                plan.append((tri, f))
        self.next_node += 1
        self.owner[m] = f1
        self.nfaces[m] = 1 if f1 == f2 else 2
        for tid in ts:
            self._del(tid)
        for tri, f in plan:
            self._new(tri[0], tri[1], tri[2], f)
        self.counts["split"] += 1
        return m

    def collapse(self, u, w, need_gain=True):
        """Remove node u by merging it into w, if that is allowed and helps
        (need_gain False: allowed as long as no triangle ends up a defect)."""
        if u in self.frozen or self.nfaces.get(u, 1) >= 3:
            return False
        ts = self.edge_tris(u, w)
        if len(ts) != 2:
            return False
        if self.nfaces.get(u, 1) == 2:
            if not self.is_feature(u, w):
                return False
        around = self.v2t[u]
        # link condition: u and w share exactly the two opposite nodes
        nu = set(x for tid in around for x in self.t[tid][:3]) - {u}
        nw = set(x for tid in self.v2t[w] for x in self.t[tid][:3]) - {w}
        if len(nu & nw) != 2:
            return False
        g_old = min(self.gamma(tid) for tid in around)
        plan = []
        for tid in around:
            if tid in ts:
                continue
            a, b, c, f = self.t[tid]
            tri = tuple(w if x == u else x for x in (a, b, c))
            g = self._gamma_of(*tri)
            if g <= g_old and (need_gain or g < 1.0):
                return False
            if np.dot(self._normal_of(*tri), self._normal_of(a, b, c)) < self.cos_dev:
                return False
            plan.append((tri, f))
        if self.nfaces.get(u, 1) == 2:
            # sliding along a curve: the two feature edges at u must be
            # nearly in line, or the curve would be cut short
            others = [x for x in nu if x != w and self.is_feature(u, x)]
            if len(others) != 1:
                return False
            d1 = self.v[w] - self.v[u]
            d2 = self.v[u] - self.v[others[0]]
            c_ = np.dot(d1, d2) / max(np.linalg.norm(d1) * np.linalg.norm(d2), 1e-300)
            if c_ < 0.98:
                return False
        for tid in list(around):
            self._del(tid)
        for tri, f in plan:
            self._new(tri[0], tri[1], tri[2], f)
        self.counts["collapse"] += 1
        return True

    def collapse_short(self, ratio):
        """
        Collapse edges much shorter than the mesh around them (shorter than
        ratio x the longest edge at either end node). The stack top carries
        the same edge, and a tet standing on a 3 mm edge in a 200 mm region
        cannot have a reasonable aspect ratio.
        """
        if ratio <= 0:
            return 0
        vmax = {}
        edges = set()
        for a, b, c, _ in self.t.values():
            for u, w in ((a, b), (b, c), (c, a)):
                l = math.dist(self.v[u], self.v[w])
                if l > vmax.get(u, 0.0):
                    vmax[u] = l
                if l > vmax.get(w, 0.0):
                    vmax[w] = l
                edges.add((min(u, w), max(u, w), l))
        n = 0
        for u, w, l in sorted(edges, key=lambda e: e[2]):
            if l >= ratio * min(vmax[u], vmax[w]):
                continue
            if u not in self.v2t or w not in self.v2t or not self.edge_tris(u, w):
                continue
            if self.collapse(u, w, need_gain=False) or self.collapse(w, u, need_gain=False):
                n += 1
        return n

    # -- driver ------------------------------------------------------------
    def repair(self, sweeps=8, short_ratio=0.0):
        gamma_min = 1.0
        for _ in range(sweeps):
            changed = self.collapse_short(short_ratio)
            # over-long edges first
            if np.isfinite(self.max_edge):
                for tid in list(self.t):
                    if tid not in self.t:
                        continue
                    l, a, b = max(self.edges_len(tid))
                    if l > self.max_edge and self.split(a, b) is not None:
                        changed += 1
            bad = [tid for tid in list(self.t) if self.gamma(tid) < gamma_min]
            for tid in bad:
                if tid not in self.t or self.gamma(tid) >= gamma_min:
                    continue
                el = sorted(self.edges_len(tid))
                (l0, a0, b0), (l2, a2, b2) = el[0], el[2]
                done = False
                if l0 < 0.5 * l2:                      # needle: drop the short edge
                    done = self.collapse(a0, b0) or self.collapse(b0, a0)
                if not done:                            # cap: flip the long edge
                    done = self.flip(a2, b2)
                if not done:
                    m = self.split(a2, b2)
                    if m is not None:
                        done = True
                        for t2 in list(self.v2t[m]):
                            if t2 in self.t:
                                x = [y for y in self.t[t2][:3] if y != m]
                                self.flip(x[0], x[1])
                changed += int(done)
            if not changed:
                break
        return sum(1 for tid in self.t if self.gamma(tid) < gamma_min)


def local_wall_repair(wall_faces, gamma_min, sine_min, max_edge, min_edge,
                      rings=2, max_passes=4, forced=None, forced_sine=0.0,
                      short_ratio=0.0):
    """
    Fix poor wall triangles where they are, instead of re-meshing the wall
    with a finer size around them.

    Wall triangles with gamma below gamma_min, or an edge longer than
    max_edge (the prism aspect ratio limit times the first cell), are
    collected with `rings` rings of neighbours into patches, and each patch
    is re-triangulated in place by edge collapses, flips and splits. Only
    those triangles change. sine_min is the smallest corner sine allowed
    (see _tri_quality_pts). forced: (k, 3) node tags of wall triangles the
    prism builder found too poor for their column tilt; those are held to
    forced_sine instead. Returns (bad left, triangles before, after).
    """
    forced_keys = None
    prev_short = None
    if forced is not None and len(forced):
        fk = np.sort(np.asarray(forced, dtype=np.int64), axis=1)
        forced_keys = set(map(tuple, fk.tolist()))
    n_before = None
    left = 0
    for p in range(max_passes):
        tags_b, conn_b, face_b = [], [], []
        for f in wall_faces:
            try:
                tg, nt = gmsh.model.mesh.getElementsByType(2, f)
            except Exception:
                continue
            if len(tg) == 0:
                continue
            tags_b.append(np.asarray(tg, dtype=np.int64))
            conn_b.append(np.asarray(nt, dtype=np.int64).reshape(-1, 3))
            face_b.append(np.full(len(tg), f, dtype=np.int64))
        if not tags_b:
            return 0, 0, 0
        tags = np.concatenate(tags_b)
        conn = np.concatenate(conn_b)
        face = np.concatenate(face_b)
        if n_before is None:
            n_before = len(tags)
        lookup, xyz = _node_lookup()
        gam, emax, sine = _tri_gamma_array(xyz[lookup[conn]])
        bad = gam < gamma_min
        thin = (sine < sine_min) & ~bad
        s_use = sine_min
        if forced_keys:
            fmask = np.fromiter((tuple(r) in forced_keys for r in
                                 np.sort(conn, axis=1).tolist()), dtype=bool,
                                count=len(conn))
            thin |= fmask & (sine < forced_sine) & ~bad
            s_use = max(sine_min, forced_sine)
            forced_keys = None            # the first pass only
        long_ = emax > max_edge if max_edge and max_edge > 0 else np.zeros(len(tags), bool)
        # edges much shorter than the mesh around them
        short = np.zeros(len(tags), dtype=bool)
        if short_ratio > 0:
            P = xyz[lookup[conn]]
            el = np.stack([np.linalg.norm(P[:, i] - P[:, (i + 1) % 3], axis=1)
                           for i in range(3)], axis=1)
            vmax = np.zeros(lookup.size)
            for i in range(3):
                np.maximum.at(vmax, conn[:, i], el[:, i])
                np.maximum.at(vmax, conn[:, (i + 1) % 3], el[:, i])
            # an edge across a narrow face joins two CAD curves and cannot
            # be collapsed without changing the shape: only edges with an
            # end inside a face, or lying along a CAD curve, count
            nfc = np.zeros(lookup.size, dtype=np.int32)
            pr = np.unique(np.column_stack([conn.ravel(), np.repeat(face, 3)]), axis=0)
            np.add.at(nfc, pr[:, 0], 1)
            ea = np.concatenate([conn[:, [0, 1]], conn[:, [1, 2]], conn[:, [2, 0]]])
            ea.sort(axis=1)
            key = ea[:, 0] * np.int64(lookup.size) + ea[:, 1]
            order = np.argsort(key, kind='stable')
            ks = key[order]
            eq = np.nonzero(ks[1:] == ks[:-1])[0]
            fe = np.tile(face, 3)
            feat = np.zeros(len(key), dtype=bool)
            d_ = fe[order[eq]] != fe[order[eq + 1]]
            feat[order[eq[d_]]] = True
            feat[order[eq[d_] + 1]] = True
            feat = feat.reshape(3, -1).T
            for i in range(3):
                u, w = conn[:, i], conn[:, (i + 1) % 3]
                can = (nfc[u] == 1) | (nfc[w] == 1) | feat[:, i]
                short |= can & (el[:, i] < short_ratio * np.minimum(vmax[u], vmax[w]))
            del nfc, pr, ea, key, order, ks, eq, fe, feat
            if p > 0 and prev_short is not None and int(short.sum()) >= prev_short:
                short[:] = False          # what is left cannot be collapsed
            prev_short = int(short.sum()) if short.any() else prev_short
            del P, el, vmax
        left = int(bad.sum() + thin.sum())
        sel = bad | thin | long_ | short
        print(f"  Wall check {p + 1}: {int(bad.sum())} triangle(s) below gamma "
              f"{gamma_min}, {int(thin.sum())} with a corner too sharp or flat "
              f"for the prisms, {int(short.sum())} on an isolated short edge"
              + (f", {int(long_.sum())} with an edge over {max_edge:.1f}mm"
                 if max_edge and max_edge > 0 else ""))
        if not sel.any():
            break
        # patch: the flagged triangles and their neighbours
        node_in = np.zeros(lookup.size, dtype=bool)
        mask = sel.copy()
        for _ in range(max(1, int(rings))):
            node_in[:] = False
            node_in[conn[mask].ravel()] = True
            mask = node_in[conn].any(axis=1)
        idx = np.nonzero(mask)[0]
        pn = np.unique(conn[idx])
        # CAD faces around each patch node, counted on the whole wall
        touch = np.zeros(lookup.size, dtype=bool)
        touch[pn] = True
        rows = np.nonzero(touch[conn].any(axis=1))[0]
        pairs = np.unique(np.column_stack([conn[rows].ravel(),
                                           np.repeat(face[rows], 3)]), axis=0)
        pairs = pairs[touch[pairs[:, 0]]]
        un, cnt = np.unique(pairs[:, 0], return_counts=True)
        nfaces_of = dict(zip(un.tolist(), cnt.tolist()))
        xyz_of = {int(n): xyz[lookup[n]] for n in pn}
        patch = _WallPatch(tags[idx], conn[idx], face[idx], xyz_of, nfaces_of,
                           int(gmsh.model.mesh.getMaxNodeTag()) + 1,
                           min_edge, max_edge, gamma_min, s_use)
        patch.repair(short_ratio=short_ratio)
        c = patch.counts
        if not (c["collapse"] or c["flip"] or c["split"]):
            print("    nothing could be changed there")
            break
        # write back, face by face
        by_face = {}
        for tid, f in patch.removed.items():
            by_face.setdefault(f, [[], []])[0].append(tid)
        for tid, (a, b, c_, f) in patch.t.items():
            if tid < 0:
                by_face.setdefault(f, [[], []])[1].extend((a, b, c_))
        new_nodes = {}
        for m, f in patch.owner.items():
            if patch.v2t.get(m):
                new_nodes.setdefault(f, []).append(m)
        for f, ms in new_nodes.items():
            gmsh.model.mesh.addNodes(2, f, ms, np.concatenate(
                [patch.v[m] for m in ms]).tolist())
        for f, (gone, added) in by_face.items():
            if gone:
                gmsh.model.mesh.removeElements(2, f, gone)
            if added:
                gmsh.model.mesh.addElementsByType(f, 2, [], added)
        print(f"    {len(idx)} triangle(s) in the patches: {c['collapse']} "
              f"collapse(s), {c['flip']} flip(s), {c['split']} split(s), "
              f"now {len(patch.t)}")
    n_after, left = 0, 0
    lookup, xyz = _node_lookup()
    for f in wall_faces:
        nt = np.asarray(gmsh.model.mesh.getElementsByType(2, f)[1], dtype=np.int64)
        if nt.size == 0:
            continue
        gam, _, sine = _tri_gamma_array(xyz[lookup[nt.reshape(-1, 3)]])
        n_after += len(gam)
        left += int(((gam < gamma_min) | (sine < sine_min)).sum())
    return left, n_before or 0, n_after

def _tet_volume_tags(state):
    return [v for v in state.get("tet_volumes", [])
            if (3, v) in set(gmsh.model.getEntities(3))]


def _cavity_shells(tri):
    """Split boundary triangles (m x 3 node tags) into edge-connected
    shells. Returns (list of index arrays, number of non-manifold edges)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m = len(tri)
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(m), 3)
    key = e[:, 0] * (int(e.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind='stable')
    ks, ow = key[order], owner[order]
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    _, cnt = np.unique(ks, return_counts=True)
    g = coo_matrix((np.ones(len(same), dtype=np.int8), (ow[same], ow[same + 1])),
                   shape=(m, m))
    n, lab = connected_components(g, directed=False)
    return [np.nonzero(lab == k)[0] for k in range(n)], int((cnt != 2).sum())


def _undo_cavities(comps):
    """Drop the new fill of these cavities and put their original tets back
    (under fresh element tags)."""
    vols = [c["vol"] for c in comps if c.get("vol") is not None]
    for v in vols:
        try:
            gmsh.model.mesh.removeElements(3, v)
        except Exception:
            pass
    try:
        gmsh.model.geo.remove([(3, v) for v in vols])
        gmsh.model.geo.synchronize()
    except Exception:
        pass
    for c in comps:
        for dt in ([(3, c["vol"])] if c.get("vol") is not None else []) + \
                [(2, s_) for s_ in c.get("surfs", [])]:
            try:
                gmsh.model.removeEntities([dt])
            except Exception:
                pass
        c["vol"], c["surfs"] = None, []
        fresh = []
        for v, tg, cn in c["backup"]:
            t0 = int(gmsh.model.mesh.getMaxElementTag()) + 1
            tg = np.arange(t0, t0 + len(tg), dtype=np.int64)
            gmsh.model.mesh.addElementsByType(v, 4, tg.tolist(), cn.ravel().tolist())
            fresh.append((v, tg, cn))
        c["backup"] = fresh


def _merge_cavities(comps):
    """
    Move the new tets (and nodes) of accepted cavities into the tet volume
    their old tets came from, and delete the temporary cavity volumes and
    surfaces. One tet region is all that ever stays in the model: a cavity
    volume left behind would be re-meshed by gmsh the moment a later cavity
    emptied it.
    """
    data = []
    for c in comps:
        nt, nx, _ = gmsh.model.mesh.getNodes(3, c["vol"], includeBoundary=False)
        cn = gmsh.model.mesh.getElementsByType(4, c["vol"])[1]
        data.append((np.asarray(nt, dtype=np.int64).copy(),
                     np.asarray(nx, dtype=float).copy(),
                     np.asarray(cn, dtype=np.int64).copy()))
        gmsh.model.mesh.removeElements(3, c["vol"])
    try:
        gmsh.model.geo.remove([(3, c["vol"]) for c in comps])
        gmsh.model.geo.synchronize()
    except Exception:
        pass
    n = 0
    for c, (nt, nx, cn) in zip(comps, data):
        for dt in [(3, c["vol"])] + [(2, s_) for s_ in c["surfs"]]:
            try:
                gmsh.model.removeEntities([dt])
            except Exception:
                pass
        home = c["backup"][0][0]
        if nt.size:
            gmsh.model.mesh.addNodes(3, home, nt.tolist(), nx.tolist())
        gmsh.model.mesh.addElementsByType(home, 4, [], cn.tolist())
        n += cn.size // 4
        c["vol"], c["surfs"] = None, []
    return n


def _fill_cavities(comps, ctx):
    """
    Remove the tets of the given cavities and fill them again. On failure
    the original tets are put back and the set is halved and retried, so one
    cavity that cannot be filled does not stop the others. Returns
    (filled cavities, failed cavities); a filled one has its new volume tag
    in ["vol"].
    """
    for c in comps:
        for v, tg, cn in c["backup"]:
            gmsh.model.mesh.removeElements(3, v, tg.tolist())
    for c in comps:
        loops, c["surfs"] = [], []
        for tri in c["shells"]:
            sc = gmsh.model.addDiscreteEntity(2)
            gmsh.model.mesh.addElementsByType(sc, 2, [], tri.ravel().tolist())
            c["surfs"].append(sc)
            loops.append(gmsh.model.geo.addSurfaceLoop([sc]))
        c["vol"] = ctx["next_vol"]
        ctx["next_vol"] += 1
        gmsh.model.geo.addVolume(loops, c["vol"])
    gmsh.model.geo.synchronize()
    ok, err = True, ""
    gmsh.logger.start()
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        ok, err = False, str(e)
    try:
        msgs = [m for m in gmsh.logger.get() if m.startswith("Error")]
        gmsh.logger.stop()
    except Exception:
        msgs = []
    counts = []
    for c in comps:
        try:
            counts.append(len(gmsh.model.mesh.getElementsByType(4, c["vol"])[0]))
        except Exception:
            counts.append(0)
    if ok and all(counts):
        return list(comps), []
    _undo_cavities(comps)
    if len(comps) == 1:
        comps[0]["error"] = "; ".join(msgs[:2]) or err or "cavity came back empty"
        return [], list(comps)
    half = len(comps) // 2
    f1, x1 = _fill_cavities(comps[:half], ctx)
    f2, x2 = _fill_cavities(comps[half:], ctx)
    return f1 + f2, x1 + x2


def local_tet_repair(defects, state, lr, background, size_far, fluid_name,
                     prism_volumes, threads, crit):
    """
    Re-mesh only the tets on and next to the defective ones.

    The poor tets are clustered into spots (same bookkeeping as the size
    spots: a spot that was tightened and did not improve is marked stuck).
    For every spot that can still be refined, the tets inside its radius,
    the poor tets themselves and `cavity_rings` rings of neighbours are
    removed. Each connected cavity becomes its own small volume, bounded by
    the faces it shared with the untouched mesh (and by the prism stack top
    or far field where it reaches them), and is filled again at the spot
    size. The boundary of a cavity is never altered, so the rest of the mesh,
    the prism layer and the wall stay exactly as they were. A new fill is
    kept only if it has fewer poor tets than the cavity had; otherwise the
    original tets go back, so a pass can never make a cavity worse.

    Returns the number of cavities re-meshed (0: nothing more can be done).
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pts, hs, tgs = [], [], []
    for d in defects:
        if "etype" not in d or len(d["points"]) == 0:
            continue
        m = d["etype"] == 4
        if m.any():
            pts.append(d["points"][m])
            hs.append(d["h"][m])
            tgs.append(d["tags"][m])
    if not pts:
        print("  No tet among the defects")
        return 0
    pts = np.concatenate(pts)
    bad_tags, first = np.unique(np.concatenate(tgs), return_index=True)
    pts = pts[first]
    # Local size from the tet's own edges. A poor tet standing on a short
    # edge of the prism stack top (a narrow face of the body) needs
    # neighbours a few times that edge, not a fraction of its longest one.
    e_max = np.asarray(gmsh.model.mesh.getElementQualities(bad_tags.tolist(), "maxEdge"))
    e_min = np.asarray(gmsh.model.mesh.getElementQualities(bad_tags.tolist(), "minEdge"))
    hs = np.minimum(e_max, float(lr["short_edge_reach"]) * e_min)
    tgt = np.maximum(lr["refine_factor"] * hs, lr["min_spot_size_mm"])

    spots = state.setdefault("tet_spots", [])
    room = max(0, lr["max_total_spots"] - len(spots))
    cands = cluster_defect_points(pts, hs, lr["cluster_factor"], tgt)
    # A fresh stage key per pass: a spot is retried at a smaller size until
    # max_spot_level, whatever its defect count did (a rejected fill leaves
    # the count unchanged, and the next size may still work).
    state["tet_pass"] = state.get("tet_pass", 0) + 1
    changed, n_new, n_tight, n_stuck = merge_spots(
        spots, cands, lr, "size", min(lr["max_new_spots_per_iteration"], room),
        f"tet{state['tet_pass']}")
    print(f"  Tet spots: {n_new} new, {n_tight} tightened, {n_stuck} stuck "
          f"({len(spots)} total)")
    if not changed:
        return 0

    # every tet of the tet region(s)
    vols = _tet_volume_tags(state)
    tag_b, conn_b, vol_b = [], [], []
    for v in vols:
        tg, nt = gmsh.model.mesh.getElementsByType(4, v)
        if len(tg) == 0:
            continue
        tag_b.append(np.asarray(tg, dtype=np.int64))
        conn_b.append(np.asarray(nt, dtype=np.int64).reshape(-1, 4))
        vol_b.append(np.full(len(tg), v, dtype=np.int64))
    if not tag_b:
        return 0
    tags = np.concatenate(tag_b)
    conn = np.concatenate(conn_b)
    tvol = np.concatenate(vol_b)
    del tag_b, conn_b, vol_b
    lookup, xyz = _node_lookup()
    cent = xyz[lookup[conn]].mean(axis=1)

    sel = np.zeros(len(tags), dtype=bool)
    tree = cKDTree(cent)
    cpts = np.array([[s["x"], s["y"], s["z"]] for s in changed])
    crad = np.array([s["radius"] for s in changed])
    for c, r in zip(cpts, crad):
        sel[tree.query_ball_point(c, r)] = True
    del tree
    # the poor tets near a changed spot (a stuck spot keeps its tets)
    order = np.argsort(tags)
    pos = order[np.clip(np.searchsorted(tags[order], bad_tags), 0, len(tags) - 1)]
    pos = pos[tags[pos] == bad_tags]
    if len(pos):
        dd, ii = cKDTree(cpts).query(cent[pos])
        sel[pos[dd <= 2.0 * crad[ii]]] = True
    node_in = np.zeros(lookup.size, dtype=bool)

    def face_list(cc):
        return np.concatenate([cc[:, [0, 2, 1]], cc[:, [0, 1, 3]],
                               cc[:, [0, 3, 2]], cc[:, [1, 2, 3]]])

    def boundary(cc):
        fl = face_list(cc)
        _, finv, fcnt = np.unique(np.sort(fl, axis=1), axis=0,
                                  return_inverse=True, return_counts=True)
        return fl, fcnt[finv.ravel()] == 1

    for _ in range(max(0, int(lr["cavity_rings"]))):
        node_in[:] = False
        node_in[conn[sel].ravel()] = True
        sel = node_in[conn].any(axis=1)
    for _ in range(4):
        # close pinches: a tet with all four nodes on the cavity joins it,
        # and so do the tets around an edge where the cavity boundary is
        # not a clean two-sided surface
        node_in[:] = False
        node_in[conn[sel].ravel()] = True
        sel |= node_in[conn].all(axis=1)
        cc = conn[sel]
        fl, on_b = boundary(cc)
        bt = fl[on_b]
        e = np.sort(np.concatenate([bt[:, [0, 1]], bt[:, [1, 2]], bt[:, [2, 0]]]), axis=1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        nm = ue[cnt != 2]
        if len(nm) == 0:
            break
        node_in[:] = False
        node_in[nm.ravel()] = True
        sel |= node_in[conn].any(axis=1)
    idx = np.nonzero(sel)[0]
    if idx.size == 0:
        return 0
    cc = conn[idx]

    # connected cavities (through shared nodes)
    un, inv = np.unique(cc, return_inverse=True)
    inv = inv.reshape(-1, 4)
    rows = np.concatenate([inv[:, 0], inv[:, 1], inv[:, 2]])
    cols = np.concatenate([inv[:, 1], inv[:, 2], inv[:, 3]])
    g = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                   shape=(len(un), len(un)))
    ncomp, lab = connected_components(g, directed=False)
    comp = lab[inv[:, 0]]
    fl, on_b = boundary(cc)
    fcomp = np.tile(comp, 4)[on_b]
    fl = fl[on_b]

    comps = []
    for k in range(ncomp):
        m = comp == k
        tri = fl[fcomp == k]
        if not m.any() or len(tri) == 0:
            continue
        shells, n_nm = _cavity_shells(tri)
        # the outer shell first, any enclosed island after it as a hole
        diag = [np.ptp(xyz[lookup[tri[sh].ravel()]], axis=0).sum() for sh in shells]
        shells = [tri[shells[i]] for i in np.argsort(diag)[::-1]]
        backup = []
        for v in np.unique(tvol[idx][m]):
            mv = m & (tvol[idx] == v)
            backup.append((int(v), tags[idx][mv].copy(), cc[mv].copy()))
        comps.append({"shells": shells, "backup": backup,
                      "n_bad": int(np.isin(tags[idx][m], bad_tags).sum()),
                      "center": xyz[lookup[cc[m].ravel()]].mean(axis=0)})
    del tags, conn, tvol, cent, cc, fl

    opt_keys = ("Mesh.Algorithm3D", "Mesh.Optimize", "Mesh.OptimizeNetgen",
                "Mesh.MeshOnlyEmpty", "General.NumThreads", "General.Terminal")
    saved = {k: gmsh.option.getNumber(k) for k in opt_keys}
    # element tags must stay put from here on: gmsh otherwise renumbers
    # every element after each generate()
    gmsh.option.setNumber("Mesh.Renumber", 0)
    balls = add_spot_fields(changed, size_far, lr["spot_growth_rate"])
    bg = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(bg, "FieldsList", [background] + balls)
    gmsh.model.mesh.field.setAsBackgroundMesh(bg)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
    gmsh.option.setNumber("General.NumThreads", max(1, min(int(threads), 4)))
    gmsh.option.setNumber("General.Terminal", 0)
    ctx = {"next_vol": max([t for d_, t in gmsh.model.getEntities(3)] +
                           [int(gmsh.model.occ.getMaxTag(3)),
                            int(gmsh.model.geo.getMaxTag(3))]) + 1}
    try:
        filled, failed = _fill_cavities(comps, ctx)
    finally:
        for k, val in saved.items():
            gmsh.option.setNumber(k, val)
        gmsh.model.mesh.field.setAsBackgroundMesh(background)
        for f in [bg] + balls:
            gmsh.model.mesh.field.remove(f)

    # keep a new fill only where it is better than what it replaced
    kept, rejected = [], []
    ar_t = crit.get("max_tet_aspect_ratio", 0) or 0
    n_old = n_new_tets = 0
    for c in filled:
        tg = gmsh.model.mesh.getElementsByType(4, c["vol"])[0]
        q = np.asarray(gmsh.model.mesh.getElementQualities(tg, "minSICN"))
        poor = q < crit["min_tet_sicn"]
        if ar_t > 0:
            r_in = np.asarray(gmsh.model.mesh.getElementQualities(tg, "innerRadius"))
            e_mx = np.asarray(gmsh.model.mesh.getElementQualities(tg, "maxEdge"))
            poor |= e_mx / np.maximum(2.0 * math.sqrt(6.0) * r_in, 1e-300) > ar_t
        n_poor = int(poor.sum())
        if n_poor == 0 or n_poor < c["n_bad"]:
            kept.append(c)
            n_old += sum(len(t_) for v, t_, cn in c["backup"])
            n_new_tets += len(tg)
        else:
            rejected.append(c)
    if rejected:
        _undo_cavities(rejected)
        print(f"  {len(rejected)} new fill(s) were no better than the tets "
              "they replaced and were undone")
    if failed:
        print(f"  {len(failed)} cavit{'y' if len(failed) == 1 else 'ies'} could "
              f"not be filled and kept the original tets "
              f"(first: {failed[0].get('error', '')[:160]})")
    if kept:
        _merge_cavities(kept)
        print(f"  {len(kept)} cavit{'y' if len(kept) == 1 else 'ies'} re-meshed: "
              f"{n_old} tets removed, {n_new_tets} created "
              f"({n_new_tets - n_old:+d})")
    # something was tried: go round again (smaller size next time) until
    # every spot is at its last level
    return len(kept) + len(rejected) + len(failed)


def print_defect_groups(defects):
    names = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}
    for d in defects:
        if len(d["points"]) == 0:
            continue
        if "etype" in d:
            parts = ", ".join(f"{int((d['etype'] == t).sum())} {names.get(int(t), t)}"
                              for t in np.unique(d["etype"]))
        else:
            parts = f"{len(d['points'])}"
        p = d["points"]
        print(f"    {d['kind']}: {parts}, x=[{p[:, 0].min():.0f},{p[:, 0].max():.0f}]")


def print_defect_summary(summary, crit):
    print("Refinement criteria check:")
    for name, w in sorted(summary.get("worst", {}).items()):
        lim = {"tet": crit["min_tet_sicn"], "pyramid": crit["min_pyramid_sicn"],
               "prism": crit["min_prism_gamma"], "hex": crit["min_prism_gamma"]}.get(name)
        print(f"  {name}: {w['count']} cells, worst {w['metric']}={w['value']:.4f}"
              + (f" (limit {lim})" if lim is not None else ""))
    print(f"  inverted cells: {summary.get('inverted', 0)} "
          f"(allowed {crit['max_inverted']})")
    print(f"  degenerate cells: {summary.get('degenerate', 0)}")
    if crit.get("max_non_orthogonality_deg"):
        print(f"  non-orthogonality: worst {summary.get('worst_non_orthogonality_deg', 0):.1f} deg, "
              f"{summary.get('non_orthogonal_faces', 0)} face(s) above "
              f"{crit['max_non_orthogonality_deg']} deg")
    if crit.get("max_prism_aspect_ratio"):
        print(f"  prism aspect ratio: worst {summary.get('worst_prism_aspect_ratio', 0):.0f} "
              f"(limit {crit['max_prism_aspect_ratio']})")
    if crit.get("max_tet_aspect_ratio"):
        print(f"  tet aspect ratio: worst {summary.get('worst_tet_aspect_ratio', 0):.2f} "
              f"(limit {crit['max_tet_aspect_ratio']})")
    if crit.get("max_prism_aspect_ratio") or crit.get("max_tet_aspect_ratio"):
        print(f"  cells over the aspect ratio limits: {summary.get('aspect_ratio', 0)}")
    if "worst_yplus" in summary:
        print(f"  y+ (first cell, worst model / wall temperature): max "
              f"{summary['worst_yplus']:.3f} (limit {crit['max_yplus']}), "
              f"{summary.get('yplus_violations', 0)} wall node(s) above")
    if crit.get("max_volume_ratio"):
        print(f"  volume ratio: worst {summary.get('worst_volume_ratio', 1):.1f}, "
              f"{summary.get('volume_ratio_faces', 0)} face(s) above "
              f"{crit['max_volume_ratio']}")


def criteria_met(summary, crit):
    if summary.get("inverted", 0) > crit["max_inverted"]:
        return False
    if summary.get("yplus_violations", 0) > 0:
        return False
    bad = (summary.get("degenerate", 0) + summary.get("non_orthogonal_faces", 0)
           + summary.get("volume_ratio_faces", 0) + summary.get("aspect_ratio", 0))
    return bad <= crit["allowed_bad_cells"]


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def create_mesh_iterative(config_file='gmsh_config.json', dry_run=False):
    """
    Run create_mesh until the auto_refinement_parameters criteria are met,
    adding local size spots and boundary layer thinning spots between passes.
    Falls through to a single create_mesh call when the block is absent or
    disabled.
    """
    import copy
    config = load_config(config_file)
    if config is None:
        return False
    ar = refinement_settings(config)
    if not ar.get("enable", False):
        return create_mesh(config_file, dry_run, config=config)

    cfg_dir = os.path.dirname(os.path.abspath(config_file))

    def resolve(path):
        if not path:
            return path
        path = os.path.expanduser(path)
        return path if os.path.isabs(path) else os.path.join(cfg_dir, path)

    output_path = resolve(config.get("output_file", "mesh.msh"))
    stem = os.path.splitext(output_path)[0]
    spots_file = resolve(ar.get("spots_file")) or stem + "_refinement_spots.json"

    state = {"size_spots": [], "bl_spots": [], "history": []}
    if ar.get("resume_from_spots_file") and os.path.exists(spots_file):
        with open(spots_file) as fh:
            saved = json.load(fh)
        state["size_spots"] = saved.get("size_spots", [])
        state["bl_spots"] = saved.get("bl_spots", [])
        state["first_cell_factor"] = min(1.0, float(saved.get("first_cell_factor", 1.0)))
        for s in state["size_spots"] + state["bl_spots"]:
            s["stuck"] = False
        print(f"Resumed {len(state['size_spots'])} size spot(s) and "
              f"{len(state['bl_spots'])} boundary layer spot(s) from {spots_file}\n")

    # Every pass repeats import and boolean cut unless they are cached; the
    # cut does not depend on any sizing, so cache it for the loop.
    geo = config.setdefault("geometry_parameters", {})
    if not geo.get("cut_cache_file"):
        geo["cut_cache_file"] = stem + "_auto_cut_cache.brep"
        print(f"Auto refinement: caching the boolean cut in {geo['cut_cache_file']}\n")

    max_iter = int(ar["max_iterations"])
    ok = False
    for it in range(1, max_iter + 1):
        state["iteration"] = it
        state["final"] = it == max_iter
        state["report"] = {}
        print("=" * 72)
        print(f"Auto refinement pass {it} of {max_iter}: "
              f"{len(state['size_spots'])} size spot(s), "
              f"{len(state['bl_spots'])} boundary layer spot(s)")
        print("=" * 72)
        ok = create_mesh(config_file, dry_run, config=copy.deepcopy(config),
                         refine_state=state)
        rep = state.get("report", {})
        state["history"].append(_json_safe({"iteration": it, "ok": ok, **rep}))
        try:
            with open(spots_file, "w") as fh:
                json.dump(_json_safe({"size_spots": state["size_spots"],
                                      "bl_spots": state["bl_spots"],
                                      "first_cell_factor": state.get("first_cell_factor", 1.0),
                                      "history": state["history"]}), fh, indent=1)
            print(f"Refinement spots saved: {spots_file}")
        except Exception as e:
            print(f"Could not write the spots file: {e}")
        status = rep.get("status")
        if status != "needs_refinement":
            break
        print(f"Pass {it}: {rep.get('reason', '')}, re-meshing with the new spots\n")

    print("\nAuto refinement history:")
    for h in state["history"]:
        print(f"  pass {h['iteration']}: {h.get('status', 'stopped')}"
              f"{' (' + h['reason'] + ')' if h.get('reason') else ''}")
    final = state["history"][-1].get("status") if state["history"] else None
    if final == "needs_refinement":
        print(f"Criteria not met after {max_iter} pass(es). Raise max_iterations, "
              "or resume from the spots file with resume_from_spots_file.")
        return False
    return ok


def create_mesh(config_file='gmsh_config.json', dry_run=False, config=None,
                refine_state=None):
    """
    One meshing pass. config: an already loaded config dict (else read from
    config_file). refine_state: set by create_mesh_iterative; carries the
    local refinement spots in, and the pass outcome out in
    refine_state["report"]["status"] (converged / needs_refinement / stuck /
    not_converged / dry_run / failed).
    """
    if config is None:
        config = load_config(config_file)
    if config is None:
        return False
    refining = refine_state is not None
    BL_STATS.clear()
    ar = refinement_settings(config) if refining else None
    crit = ar["criteria"] if refining else None
    lr = ar["local_repair"] if refining else None
    final_pass = bool(refine_state.get("final")) if refining else True

    def report(status, reason="", **extra):
        if refining:
            refine_state["report"] = {"status": status, "reason": reason, **extra}

    def refine_and_stop(defects, reason):
        """Register defects as spots; finalize and tell the driver to rerun.
        Returns the value create_mesh should return."""
        n_pts = sum(len(d["points"]) for d in defects)
        print(f"Auto refinement: {reason}, {n_pts} defect location(s)")
        changed = register_defects(refine_state, defects, ar,
                                   use_bl_spots=use_custom_bl_flag[0], stage="bl")
        if changed and not final_pass:
            report("needs_refinement", reason)
        elif changed:
            report("not_converged", reason + ", last pass")
        else:
            report("stuck", reason + ", no spot could be refined further")
        gmsh.finalize()
        return False

    use_custom_bl_flag = [False]

    dry_run = dry_run or config.get('utility_parameters', {}).get('dry_run', False)
    if dry_run:
        print("DRY RUN: will build the surface mesh and prism layers (if "
              "enabled) to get real counts, then stop before the 3D tet fill "
              "and export. No .msh or .su2 written.\n")

    # Relative paths in the config are taken relative to the config file, so
    # a folder holding the script, config and brep works on any machine.
    cfg_dir = os.path.dirname(os.path.abspath(config_file))

    def resolve(path):
        if not path:
            return path
        path = os.path.expanduser(path)
        return path if os.path.isabs(path) else os.path.join(cfg_dir, path)

    step_file = resolve(config.get('step_file'))
    output_path = resolve(config.get('output_file'))
    export_geometry_path = resolve(config.get('export_geometry_path'))
    geo_cfg = config.get('geometry_parameters', {})
    if geo_cfg.get('cut_cache_file'):
        geo_cfg['cut_cache_file'] = resolve(geo_cfg['cut_cache_file'])
    out_dir = os.path.dirname(output_path) if output_path else ""
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    gmsh_params = config.get('gmsh_parameters', {})
    size_params = config.get('size_field_parameters', {})
    geom_params = config.get('geometry_parameters', {})
    util_params = config.get('utility_parameters', {})
    bl_defaults = config.get('boundary_layer_parameters', {})
    shock_params = config.get('shock_box_parameters', {})
    seg_params = config.get('body_segmentation_parameters', {})
    marker_names = config.get('su2_marker_names', {})
    freestream_defaults = config.get('freestream_parameters', {})

    wall_name = marker_names.get("wall", "wall")
    sym_name = marker_names.get("symmetry", "symmetry")
    farfield_name = marker_names.get("farfield", "farfield")

    farfield_faces, symmetry_faces = [], []

    # 1. Initialize
    gmsh.initialize()
    print(f"gmsh {gmsh.__version__}")
    print(f"Input: {step_file}\nOutput: {output_path}\n")
    # gmsh's own default (1e-9). This script used to default to 1e-5, and
    # that larger jitter produced sliver "ear" triangles along curved wall
    # edges (all three nodes on one circle, normal pointing along the axis);
    # every inverted prism on the test body was extruded from one. Measured:
    # 14 ears at 1e-5, none at 1e-9, same geometry and sizing.
    gmsh.option.setNumber("Mesh.RandomFactor", gmsh_params.get("Mesh.RandomFactor", 1e-9))
    gmsh.option.setNumber("General.NumThreads", gmsh_params.get("General.NumThreads", 16))
    # Surface meshing single-threaded; the 3D fill keeps General.NumThreads.
    # Parallel 2D meshing is nondeterministic where neighbouring surfaces
    # share a curve and one needs gmsh's "refine all its bounding edges"
    # recovery: measured on identical input, 4 threads left an invalid
    # element on alternate runs (which cascaded into an empty wall surface,
    # empty prism blocks and an empty tet region), 1 thread was clean on
    # every run. The 2D pass is a small fraction of total run time.
    gmsh.option.setNumber("Mesh.MaxNumThreads2D",
                          gmsh_params.get("Mesh.MaxNumThreads2D", 1))
    gmsh.option.setNumber("Mesh.MaxNumThreads1D",
                          gmsh_params.get("Mesh.MaxNumThreads1D", 1))
    gmsh.model.add("BWB_Spaceplane_Mesh")

    # 2. Import geometry
    if not os.path.exists(step_file):
        print(f"File not found: {step_file}")
        gmsh.finalize()
        return False
    try:
        shapes = gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"Geometry import failed: {e}")
        gmsh.finalize()
        return False

    dilation = geom_params.get('uniform_dilation_factor', 1.0)
    if dilation != 1.0:
        try:
            gmsh.model.occ.dilate(shapes, 0, 0, 0, dilation, dilation, dilation)
            gmsh.model.occ.synchronize()
            print(f"Geometry scaled {dilation}x\n")
        except Exception as e:
            print(f"Geometry scaling failed: {e}")
            gmsh.finalize()
            return False

    vols = [item for item in shapes if item[0] == 3]
    if not vols:
        print("No 3D volume found in geometry file")
        gmsh.finalize()
        return False
    body_tag = vols[0][1]

    # 3. Rotate for angle of attack, about Y, the symmetry plane normal, so the
    # body stays symmetric and the half model cut remains valid.
    angle_deg = geom_params.get("rotation_angle_deg", -5.0)
    rotation_by_com = geom_params.get("rotation_by_CoM", False)
    if angle_deg != 0.0:
        origin = gmsh.model.occ.getCenterOfMass(3, body_tag) \
            if rotation_by_com else (0.0, 0.0, 0.0)
        gmsh.model.occ.rotate([(3, body_tag)], origin[0], origin[1], origin[2],
                              0, 1, 0, angle_deg * math.pi / 180.0)
        gmsh.model.occ.synchronize()
        if geom_params.get("test_for_CoM_only", False):
            print(f"CoM after rotation: {gmsh.model.occ.getCenterOfMass(3, body_tag)}")
            gmsh.finalize()
            return True
        if export_geometry_path:
            export_step(export_geometry_path + ".1.step")
        print(f"Rotated {angle_deg} deg about Y\n")

    body_bbox = gmsh.model.getBoundingBox(3, body_tag)
    body_length = body_bbox[3] - body_bbox[0]
    print(f"Body bbox: X=[{body_bbox[0]:.0f},{body_bbox[3]:.0f}] "
          f"Y=[{body_bbox[1]:.0f},{body_bbox[4]:.0f}] "
          f"Z=[{body_bbox[2]:.0f},{body_bbox[5]:.0f}], length={body_length:.0f}mm\n")

    # 3b. Smart body segmentation: detect nose/hump/rear split stations from
    # the geometry's own radius profile, then physically slice the body solid
    # there. This replaces hand-placed refinement boxes with real geometric
    # chunks, which is what lets each section get its own independent size
    # AND its own independent boundary layer recipe (a single blended CAD
    # face cannot be extruded with two different layer counts; three separate
    # faces can).
    # Off unless the config asks for it: with it on, every section gets a
    # refinement box over the body (default size_far / 10), which an older
    # config without this block never had.
    seg_enabled = seg_params.get("enable", False)
    x_split_1 = x_split_2 = None
    seg_method = "disabled"
    section_x_ranges = {"nose": (body_bbox[0], body_bbox[3])}
    chunk_tags = None

    if seg_enabled:
        print("Body segmentation: profiling nose/hump/rear from geometry...")
        profile = profile_body_radius(
            step_file, dilation, angle_deg, rotation_by_com, body_bbox,
            seg_params.get("n_stations", 150),
            seg_params.get("probe_size_fraction", 0.02))
        centers, r_profile = profile if profile else (None, None)
        x_split_1, x_split_2, seg_method = detect_segment_splits(
            centers, r_profile, body_bbox, seg_params)
        print(f"  Splits: nose/hump at x={x_split_1:.0f}mm, hump/rear at "
              f"x={x_split_2:.0f}mm ({seg_method})\n")

        pad_mm = seg_params.get("slice_pad_mm", 1000.0)
        mass_tol = seg_params.get("mass_tolerance", 1e-6)
        # Physical slicing only splits wall faces at the section planes; the
        # size boxes and the single shared boundary layer recipe work from
        # section_x_ranges alone. On a body with thousands of faces (e.g. the
        # cavity model) each slicing boolean takes many minutes, so it can
        # be switched off.
        if seg_params.get("slice_body", False):
            chunk_tags = slice_body_into_segments(
                step_file, dilation, angle_deg, rotation_by_com, body_bbox,
                x_split_1, x_split_2, pad_mm, mass_tol)
        else:
            print("  Physical slicing skipped (slice_body false): sections "
                  "are X ranges only.")

        if chunk_tags is None and seg_params.get("slice_body", False):
            print("  Falling back to an unsliced body: sections still get "
                  "their own size fields and X ranges below, but boundary "
                  "layer extrusion (if enabled) runs once for the whole "
                  "wall rather than per section.\n")
            section_x_ranges = {
                "nose": (body_bbox[0], x_split_1),
                "hump": (x_split_1, x_split_2),
                "rear": (x_split_2, body_bbox[3]),
            }
        else:
            section_x_ranges = {
                "nose": (body_bbox[0], x_split_1),
                "hump": (x_split_1, x_split_2),
                "rear": (x_split_2, body_bbox[3]),
            }
    else:
        print("Body segmentation disabled: meshing as a single unsliced body.\n")

    # 4. Far-field domain, half model when symmetry is enabled
    half_model = geom_params.get("enable_symmetry_half_model", True)
    sym_coord = geom_params.get("symmetry_plane_y_mm", 0.0)
    keep_positive = geom_params.get("symmetry_keep_positive_side", True)

    x0 = geom_params.get("far_field_x0_mm", -50000.0)
    length = geom_params.get("far_field_length_mm", 200000.0)
    r_envelope = geom_params.get("far_field_r_envelope_mm", 100000.0)
    # Optional box extents that override the symmetric r_envelope: lateral
    # extent from the symmetry plane, and the lower and upper Z bounds. A
    # hypersonic body only needs the domain to contain its bow shock, and the
    # body sits above Z=0 here, so a symmetric box wastes most of its volume.
    y_half = geom_params.get("far_field_y_mm", r_envelope)
    z_lo = geom_params.get("far_field_z_min_mm", -r_envelope)
    z_hi = geom_params.get("far_field_z_max_mm", r_envelope)
    dz = z_hi - z_lo
    outside = []
    if x0 >= body_bbox[0] or x0 + length <= body_bbox[3]:
        outside.append("X")
    if y_half <= max(abs(body_bbox[1] - sym_coord), abs(body_bbox[4] - sym_coord)):
        outside.append("Y")
    if z_lo >= body_bbox[2] or z_hi <= body_bbox[5]:
        outside.append("Z")
    if outside:
        print(f"The far-field box does not contain the body in {outside}: "
              f"box X=[{x0:.0f},{x0 + length:.0f}] Y half width {y_half:.0f} "
              f"Z=[{z_lo:.0f},{z_hi:.0f}], body bbox {[round(v) for v in body_bbox]}. "
              "Aborting.")
        gmsh.finalize()
        return False

    if half_model:
        y_origin = sym_coord if keep_positive else sym_coord - y_half
        y_extent = y_half
        print(f"Half model: symmetry plane y={sym_coord}, keeping "
              f"{'positive' if keep_positive else 'negative'} side")
        if not (body_bbox[1] < sym_coord < body_bbox[4]):
            print("  Warning: the symmetry plane does not pass through the body")
        elif abs(abs(body_bbox[1] - sym_coord) - abs(body_bbox[4] - sym_coord)) > 1.0:
            print("  Warning: the body is not centred on the symmetry plane. "
                  "A half model is only valid for a symmetric body.")
    else:
        y_origin = -y_half
        y_extent = 2 * y_half
        print("Half model disabled, meshing the full span")

    if body_length > 0 and max(y_half, -z_lo, z_hi) / body_length > 1.5:
        print(f"  Warning: far field extends {max(y_half, -z_lo, z_hi) / body_length:.1f} body "
              "lengths. At hypersonic speeds disturbances do not travel upstream, "
              "so the domain can be much tighter. Shrinking it saves more cells "
              "than the symmetry cut does.")

    box_tag = gmsh.model.occ.addBox(x0, y_origin, z_lo,
                                    length, y_extent, dz)
    gmsh.model.occ.synchronize()
    if export_geometry_path:
        export_step(export_geometry_path + ".2.step")
    print(f"Far-field domain tag {box_tag}\n")

    # 5. Boolean cut, with verification and automatic escalation.
    #
    # Cutting a half-width box directly by a body that straddles the symmetry
    # plane puts the box's own y=sym boundary exactly where the body's own
    # parametric seam edge runs. OCC cannot split a seam edge, reports
    # BOPAlgo_AlertNotSplittableEdge, and returns the box essentially
    # unmodified: the tool solid is deleted by removeTool and no cavity is ever
    # created. Nothing downstream detects that on its own, so every attempt is
    # verified by volume here and the next strategy is tried automatically.
    #
    # Strategies, in order:
    #   direct     cut the (possibly half) domain box by the body (or its
    #              sliced chunks, if segmentation produced them)
    #   two_stage  cut a full width box by the body first, which is a clean non
    #              coincident boolean, then intersect with the half box so the
    #              symmetry plane passes through an already formed cavity
    #   full_model fall back to the full span domain, which is verified to cut
    #              this class of geometry, at twice the cell count
    #
    # The tool solid(s) are re-imported fresh for every attempt, same as
    # before; when segmentation produced chunk volumes, re-slicing per attempt
    # is wasteful, so instead the already-sliced chunks from the model built
    # above are reused directly by re-running slice_body_into_segments only
    # once and importing its result into whichever fresh model each strategy
    # attempt builds. Simpler in practice: each attempt re-imports the whole
    # body AND re-slices it the same way, since slicing is cheap (a handful of
    # booleans) next to a full far-field cut.
    min_fraction = geom_params.get("min_cut_volume_fraction", 1e-9)
    half_box_volume = length * y_extent * dz
    full_box_volume = length * 2 * y_half * dz

    def cut_volume_check(reference_volume):
        """Return (tags, removed_volume) against the box that strategy built."""
        tags = [t for d, t in gmsh.model.getEntities(3)]
        if not tags:
            return tags, 0.0
        fluid = sum(gmsh.model.occ.getMass(3, t) for t in tags)
        return tags, reference_volume - fluid

    def import_body_or_chunks():
        """
        Re-import the body for this cut attempt, sliced into nose/hump/rear
        if segmentation succeeded, otherwise as one solid. Returns a list of
        (dim, tag) tool tuples and, when sliced, a dict {section: [tags]}.
        """
        if chunk_tags is None:
            shapes = gmsh.model.occ.importShapes(step_file)
            gmsh.model.occ.synchronize()
            if dilation != 1.0:
                gmsh.model.occ.dilate(shapes, 0, 0, 0, dilation, dilation, dilation)
                gmsh.model.occ.synchronize()
            btag = [t for d, t in shapes if d == 3][0]
            if angle_deg != 0.0:
                origin = gmsh.model.occ.getCenterOfMass(3, btag) \
                    if rotation_by_com else (0.0, 0.0, 0.0)
                gmsh.model.occ.rotate([(3, btag)], origin[0], origin[1], origin[2],
                                      0, 1, 0, angle_deg * math.pi / 180.0)
                gmsh.model.occ.synchronize()
            return [(3, btag)], None
        else:
            pad_mm = seg_params.get("slice_pad_mm", 1000.0)
            mass_tol = seg_params.get("mass_tolerance", 1e-6)
            fresh = slice_body_into_segments(
                step_file, dilation, angle_deg, rotation_by_com, body_bbox,
                x_split_1, x_split_2, pad_mm, mass_tol, isolate=False)
            if fresh is None:
                # Should not happen (it succeeded moments ago), but degrade
                # to an unsliced import rather than aborting the whole run.
                shapes = gmsh.model.occ.importShapes(step_file)
                gmsh.model.occ.synchronize()
                if dilation != 1.0:
                    gmsh.model.occ.dilate(shapes, 0, 0, 0, dilation, dilation, dilation)
                    gmsh.model.occ.synchronize()
                btag = [t for d, t in shapes if d == 3][0]
                return [(3, btag)], None
            tools = []
            for name, tags in fresh.items():
                tools.extend((3, t) for t in tags)
            return tools, fresh

    def try_direct(tools):
        gmsh.model.occ.cut([(3, box_tag)], tools,
                           removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

    def try_two_stage(tools):
        full_box = gmsh.model.occ.addBox(x0, -y_half, z_lo,
                                         length, 2 * y_half, dz)
        half_box = gmsh.model.occ.addBox(x0, y_origin, z_lo,
                                         length, y_extent, dz)
        gmsh.model.occ.synchronize()
        cut_res, _ = gmsh.model.occ.cut([(3, full_box)], tools,
                                        removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.intersect(cut_res, [(3, half_box)],
                                 removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

    def try_full_model(tools):
        full_box = gmsh.model.occ.addBox(x0, -y_half, z_lo,
                                         length, 2 * y_half, dz)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.cut([(3, full_box)], tools,
                           removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

    all_strategies = {"direct": try_direct,
                      "two_stage": try_two_stage,
                      "full_model": try_full_model}
    reference = {"direct": half_box_volume if half_model else full_box_volume,
                 "two_stage": half_box_volume,
                 "full_model": full_box_volume}

    forced = geom_params.get("force_cut_strategy")
    if forced:
        if forced not in all_strategies:
            print(f"Unknown force_cut_strategy '{forced}', expected one of "
                  f"{sorted(all_strategies)}, aborting")
            gmsh.finalize()
            return False
        strategies = [(forced, all_strategies[forced])]
        print(f"Cut strategy forced to: {forced}")
    else:
        strategies = [("direct", try_direct)]
        if half_model:
            if geom_params.get("two_stage_half_cut", True):
                strategies.append(("two_stage", try_two_stage))
            if geom_params.get("allow_full_model_fallback", True):
                strategies.append(("full_model", try_full_model))

    fluid_volume_tags = []
    used_strategy = None

    # Optional cache of the cut fluid geometry. The boolean cut is the slow
    # step on a many-face body (over 2 minutes on the cavity model) and does
    # not depend on any meshing setting, so a rerun that only changes sizes
    # or the boundary layer can load it instead. The cache is only valid for
    # the same brep, scaling, rotation and far-field box: delete the file
    # after changing any of those.
    cut_cache = geom_params.get("cut_cache_file", "")
    cache_meta = cut_cache + ".json" if cut_cache else ""
    cache_key = {"step_file": os.path.abspath(step_file), "dilation": dilation,
                 "angle_deg": angle_deg, "rotation_by_com": rotation_by_com,
                 "box": [x0, length, y_half, z_lo, z_hi], "half_model": half_model,
                 "sym_coord": sym_coord, "keep_positive": keep_positive}
    if cut_cache and os.path.exists(cut_cache) and os.path.exists(cache_meta):
        with open(cache_meta) as fh:
            meta = json.load(fh)
        if meta.get("key") != json.loads(json.dumps(cache_key)):
            print(f"Cut cache {cut_cache} was made for different geometry or "
                  "far-field settings, recomputing it")
            os.remove(cache_meta)
    if cut_cache and os.path.exists(cut_cache) and os.path.exists(cache_meta):
        with open(cache_meta) as fh:
            meta = json.load(fh)
        gmsh.clear()
        gmsh.model.add("BWB_Spaceplane_Mesh")
        shapes = gmsh.model.occ.importShapes(cut_cache)
        gmsh.model.occ.synchronize()
        fluid_volume_tags = [t for d, t in shapes if d == 3]
        used_strategy = meta.get("strategy", "direct")
        if used_strategy == "full_model":
            half_model = False
            y_origin = -y_half
            y_extent = 2 * y_half
        strategies = []
        print(f"Cut geometry loaded from cache {cut_cache} (strategy "
              f"'{used_strategy}'), boolean cut skipped")

    for name, attempt in strategies:
        # Each attempt needs a clean slate, since a failed boolean has already
        # consumed the body via removeTool. gmsh.clear() rather than
        # model.remove() so the OCC kernel is reset along with the model
        # (options are kept). Note: this was first suspected of causing
        # the intermittent "Impossible to mesh periodic surface" failure;
        # the measured causes were parallel 2D meshing (see
        # MaxNumThreads2D) and in-memory boolean state (see step 5b).
        gmsh.clear()
        gmsh.model.add("BWB_Spaceplane_Mesh")
        tools, fresh_chunks = import_body_or_chunks()
        if name == "direct":
            box_tag = gmsh.model.occ.addBox(x0, y_origin, z_lo,
                                            length, y_extent, dz)
            gmsh.model.occ.synchronize()

        print(f"Boolean cut, strategy '{name}'...")
        try:
            attempt(tools)
        except Exception as e:
            print(f"  strategy '{name}' raised: {e}")
            continue

        ref = reference[name]
        tags, removed = cut_volume_check(ref)
        frac = removed / max(ref, 1e-30)
        print(f"  volume(s) {tags}, material removed {removed:.4e} mm3 "
              f"({100.0 * frac:.4f}% of the {ref:.4e} mm3 reference box)")
        if tags and frac >= min_fraction:
            fluid_volume_tags = tags
            used_strategy = name
            if name == "full_model":
                half_model = False
                y_origin = -y_half
                y_extent = 2 * y_half
                print("  Fell back to a full span domain. The symmetry plane "
                      "cut could not be made on this geometry, so there is no "
                      "symmetry marker and the cell count is about double.")
            break
        print(f"  strategy '{name}' removed no material, the body was not "
              "subtracted. OCC usually reports BOPAlgo_AlertNotSplittableEdge "
              "here, meaning a seam edge of the body lies in a boundary of the "
              "cutting box.")

    if not fluid_volume_tags:
        print("Every boolean cut strategy failed, aborting.")
        print("  The body is a valid solid (a full span box cuts it), so this "
              "is a coincidence between the body's seam and the domain "
              "boundary. Options: rebuild the brep so no surface seam lies on "
              "the symmetry plane, or set allow_full_model_fallback true.")
        gmsh.finalize()
        return False

    print(f"Boolean cut complete using '{used_strategy}', "
          f"fluid volume(s): {fluid_volume_tags}")
    if cut_cache and not os.path.exists(cache_meta):
        try:
            gmsh.write(cut_cache)
            with open(cache_meta, "w") as fh:
                json.dump({"strategy": used_strategy, "key": cache_key}, fh)
            print(f"Cut geometry cached to {cut_cache}")
        except Exception as e:
            print(f"Could not write the cut cache: {e}")
    if export_geometry_path:
        export_step(export_geometry_path + ".3.step")

    # 5b. Round-trip the finished fluid geometry through BREP and continue
    # from the re-read copy. Measured on the segmented full-body test case,
    # with 2D meshing single-threaded so runs were deterministic: meshing
    # the geometry straight out of the boolean chain left a cone band with
    # invalid elements on every run (and that band then came back empty
    # after the prism extrusion's own 2D pass), while writing that exact
    # state -- same geometry, same fields, same options -- to disk and
    # meshing it in a fresh process was clean on every run. Re-reading the
    # BREP rebuilds the OCC shapes from their stored definition, dropping
    # whatever the chain of imports, copies and booleans left in memory.
    # Nothing holds an entity tag yet at this point (classification, fields
    # and physical groups all come after), so only the volume list needs
    # refreshing.
    if geom_params.get("brep_roundtrip_after_cut", True):
        import tempfile
        n_before = len(fluid_volume_tags)
        mass_before = sum(gmsh.model.occ.getMass(3, t) for t in fluid_volume_tags)
        with tempfile.TemporaryDirectory() as td:
            brep_path = os.path.join(td, "fluid.brep")
            gmsh.write(brep_path)
            gmsh.clear()
            gmsh.model.add("BWB_Spaceplane_Mesh")
            shapes = gmsh.model.occ.importShapes(brep_path)
            gmsh.model.occ.synchronize()
        fluid_volume_tags = [t for d, t in shapes if d == 3]
        mass_after = sum(gmsh.model.occ.getMass(3, t) for t in fluid_volume_tags)
        if (len(fluid_volume_tags) != n_before or
                abs(mass_after - mass_before) > 1e-6 * max(mass_before, 1.0)):
            print(f"BREP round-trip changed the fluid domain ({n_before} -> "
                  f"{len(fluid_volume_tags)} volume(s), volume {mass_before:.6e} "
                  f"-> {mass_after:.6e} mm3). Aborting rather than mesh an "
                  "altered domain; set brep_roundtrip_after_cut to false to "
                  "skip this step.")
            gmsh.finalize()
            return False
        print(f"Fluid geometry re-read from BREP: volume(s) {fluid_volume_tags}")
    print()

    # 6. Classify boundary faces, build physical groups
    all_faces = []
    for vol_tag in fluid_volume_tags:
        bf = gmsh.model.getBoundary([(3, vol_tag)], combined=False, oriented=False)
        all_faces.extend(tag for dim, tag in bf)
    all_faces = list(set(all_faces))

    x_inlet = x0
    x_outlet = x0 + length
    y_lo = y_origin
    y_hi = y_origin + y_extent
    domain = (x_inlet, x_outlet, y_lo, y_hi, z_lo, z_hi)
    tol = geom_params.get("far_field_face_tolerance_mm", 0.5)

    print(f"Domain bounds: X=[{x_inlet:.0f},{x_outlet:.0f}] "
          f"Y=[{y_lo:.0f},{y_hi:.0f}] Z=[{z_lo:.0f},{z_hi:.0f}] tol={tol}mm")
    print(f"Faces on the fluid volume(s): {len(all_faces)}")

    def flush(fmin, fmax, bound):
        return abs(fmin - bound) < tol and abs(fmax - bound) < tol

    verdicts = []
    for face in all_faces:
        fxmin, fymin, fzmin, fxmax, fymax, fzmax = gmsh.model.getBoundingBox(2, face)
        if half_model and flush(fymin, fymax, sym_coord):
            symmetry_faces.append(face)
            why = "symmetry"
        elif flush(fxmin, fxmax, x_inlet):
            farfield_faces.append(face)
            why = "farfield x_inlet"
        elif flush(fxmin, fxmax, x_outlet):
            farfield_faces.append(face)
            why = "farfield x_outlet"
        elif flush(fymin, fymax, y_lo):
            farfield_faces.append(face)
            why = "farfield y_lo"
        elif flush(fymin, fymax, y_hi):
            farfield_faces.append(face)
            why = "farfield y_hi"
        elif flush(fzmin, fzmax, z_lo):
            farfield_faces.append(face)
            why = "farfield z_lo"
        elif flush(fzmin, fzmax, z_hi):
            farfield_faces.append(face)
            why = "farfield z_hi"
        else:
            why = "wall"
        verdicts.append((face, (fxmin, fymin, fzmin, fxmax, fymax, fzmax), why))

    wall_faces = list(set(all_faces) - set(farfield_faces) - set(symmetry_faces))

    # Dump the inventory whenever anything looks wrong, so a failure names the
    # faces and the test that caught them rather than only its own conclusion.
    dump = (not wall_faces) or (half_model and not symmetry_faces) \
        or geom_params.get("dump_face_classification", False)
    if dump:
        print("Face classification inventory:")
        for face, bb, why in sorted(verdicts, key=lambda v: v[0]):
            print(f"  face {face:4d}  X=[{bb[0]:9.1f},{bb[3]:9.1f}]  "
                  f"Y=[{bb[1]:9.1f},{bb[4]:9.1f}]  Z=[{bb[2]:9.1f},{bb[5]:9.1f}]  "
                  f"-> {why}")
        print()

    if not wall_faces:
        print("No wall faces found after classification, aborting.")
        print(f"  {len(all_faces)} faces total, {len(symmetry_faces)} symmetry, "
              f"{len(farfield_faces)} farfield, 0 wall.")
        if len(all_faces) <= 7:
            print("  Only a handful of faces exist, so the domain is a bare box: "
                  "the aircraft cavity is missing despite the volume check above. "
                  "Inspect the .3.step export.")
        else:
            print("  Faces exist but every one matched a domain bound. Check the "
                  "inventory above against the domain bounds, and check "
                  "far_field_face_tolerance_mm is not larger than intended.")
        gmsh.finalize()
        return False
    if half_model and not symmetry_faces:
        print("Warning: half model enabled but no face lies on the symmetry plane. "
              "Check symmetry_plane_y_mm and far_field_face_tolerance_mm.")

    model_surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
    unassigned = [t for t in model_surfaces if t not in set(all_faces)]
    if unassigned:
        wall_faces = list(set(wall_faces + unassigned))
        print(f"Unassigned surfaces folded into {wall_name}: {unassigned}")

    # 6b. Tag each wall face with its section (nose/hump/rear), purely by the
    # X-centroid of its final bounding box. This is deliberately independent
    # of which CAD entity produced the face: after the multi-tool boolean cut,
    # a face's OCC lineage back to a particular chunk is not something gmsh
    # exposes cleanly, but its position is -- and position is exactly what
    # the split stations were defined by in the first place. Verified on a
    # synthetic nose/hump/rear test body: this correctly separates a single
    # cone's lateral surface into two pieces on either side of a split even
    # though the pre-slice CAD only had one face there, and correctly leaves
    # a genuinely single blended surface undivided when it doesn't cross a
    # split. If segmentation is disabled, or slicing fell back, every wall
    # face still gets a section name from the same X-centroid test, driven by
    # section_x_ranges computed above -- this is what lets the per-section
    # size fields and BL settings apply even without a hard geometric cut.
    section_names = list(section_x_ranges.keys())

    def face_section(face_tag):
        bb = gmsh.model.getBoundingBox(2, face_tag)
        cx = 0.5 * (bb[0] + bb[3])
        for name in section_names:
            xa, xb = section_x_ranges[name]
            if xa <= cx <= xb:
                return name
        # off either end due to padding/tolerance: clamp to nearest section
        if section_names:
            first, last = section_names[0], section_names[-1]
            if cx < section_x_ranges[first][0]:
                return first
            return last
        return "body"

    section_wall_faces = {name: [] for name in section_names}
    for f in wall_faces:
        section_wall_faces.setdefault(face_section(f), []).append(f)
    for name, faces in section_wall_faces.items():
        print(f"  Section '{name}': {len(faces)} wall face(s), "
              f"X=[{section_x_ranges.get(name, (0, 0))[0]:.0f},"
              f"{section_x_ranges.get(name, (0, 0))[1]:.0f}]")

    gmsh.model.addPhysicalGroup(2, wall_faces, name=wall_name)
    if symmetry_faces:
        gmsh.model.addPhysicalGroup(2, symmetry_faces, name=sym_name)
    if farfield_faces:
        gmsh.model.addPhysicalGroup(2, farfield_faces, name=farfield_name)
    gmsh.model.addPhysicalGroup(3, fluid_volume_tags, 7, "fluid")
    print(f"Physical groups: {wall_name}={len(wall_faces)} "
          f"{sym_name}={len(symmetry_faces)} {farfield_name}={len(farfield_faces)}\n")

    # 7. Mesh parameters
    size_min = gmsh_params.get("Mesh.MeshSizeMin", 0.2)
    budget_early = util_params.get("max_element_budget", 40e6)
    size_max = gmsh_params.get("Mesh.MeshSizeMax", 1000.0)
    size_far = size_params.get("size_far", 500.0)

    # MeshSizeMax is a hard clamp applied after the background field. Verified
    # by meshing a constant field of 50 under MeshSizeMax of 50, 25 and 10 and
    # watching the element count rise from 100 to 4955. A MeshSizeMax below
    # size_far silently overrides the field, so raise it rather than let the
    # two numbers contradict each other.
    if size_max < size_far:
        print(f"MeshSizeMax {size_max} is below size_far {size_far} and would "
              f"clamp it. Raising MeshSizeMax to {size_far}.")
        size_max = size_far
    gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)

    curvature_nodes = gmsh_params.get("Mesh.MeshSizeFromCurvature", 24)
    if curvature_nodes < 20:
        print(f"Warning: MeshSizeFromCurvature={curvature_nodes} is below 20 nodes "
              "per full circle. Note this option alone cannot fix a faceted nose, "
              "since N nodes around a circle stays N however small the radius.")
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_nodes)

    surface_algorithm = gmsh_params.get("Mesh.Algorithm", 6)
    if surface_algorithm != 6:
        print(f"Warning: Mesh.Algorithm={surface_algorithm}, Frontal-Delaunay (6) "
              "is recommended for the blended surfaces")
    gmsh.option.setNumber("Mesh.Algorithm", surface_algorithm)

    gmsh.option.setNumber("Mesh.Smoothing", gmsh_params.get("Mesh.Smoothing", 5))
    gmsh.option.setNumber("Mesh.ElementOrder", gmsh_params.get("Mesh.ElementOrder", 1))
    gmsh.option.setNumber("General.Terminal", gmsh_params.get("General.Terminal", 1))
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.Optimize", gmsh_params.get("Mesh.Optimize", 1))
    gmsh.option.setNumber("Mesh.OptimizeNetgen", gmsh_params.get("Mesh.OptimizeNetgen", 1))
    gmsh.option.setNumber("Mesh.OptimizeThreshold", gmsh_params.get("Mesh.OptimizeThreshold", 0.3))

    # 8. Size fields: one per detected section (JSON-controlled), replacing
    # the old single nose_box, plus the physics-driven shock_box unchanged.
    # Cavity faces are found first so the main wall distance field can leave
    # them out: Sampling is per parametric direction, so a surface costs
    # Sampling^2 points, and on the cavity body (4300 faces of a few mm)
    # sampling every face at the body-scale count meant ~10^8 points and a
    # surface mesh that never finished. The cavities get their own distance
    # field below, sampled at their own scale.
    # Small faces (diag up to cavity_max_diag_mm) are always found: they are
    # sampled separately in the wall distance field, and the custom boundary
    # layer thins its stack on them. Local cavity refinement (cavity_size)
    # is only added for explicit cavity_surface_tags or auto_detect_cavity.
    small_diag = size_params.get("cavity_max_diag_mm", 100.0)
    small_faces = detect_small_wall_faces(wall_faces, small_diag)
    if small_faces:
        print(f"Found {len(small_faces)} small wall face(s) (diag up to "
              f"{small_diag}mm), treated as cavities by the boundary layer")
    cavity_tags = size_params.get("cavity_surface_tags", [])
    if not cavity_tags and size_params.get("auto_detect_cavity", False):
        cavity_tags = small_faces
        print(f"  auto_detect_cavity: cavity refinement on those faces")
    cavity_tags = [f for f in cavity_tags if f in set(wall_faces)]

    # 7b. Aspect ratio pre-flight. The first cell is fixed by y+ and is never
    # made coarser; a prism aspect ratio limit (wall spacing / layer height)
    # is therefore met by refining the WALL: its size is capped at
    # max_prism_aspect_ratio x first cell. Only the wall surface gets the
    # cap; the volume grades away from it at growth_rate, rather than the
    # whole dist_min band shrinking with it. The stack top is handled too:
    # max_bl_top_jump caps (wall spacing / top layer height), the size jump
    # from the last prism layer into the tets, by extending Thickness (more
    # layers at the same Ratio), again never touching the first cell.
    wall_cap_field = None
    bl_top_thickness = 0.0
    wall_edge_limit = 0.0
    if refining and bl_defaults.get("enable", False):
        h_first = first_cell_height(bl_defaults, freestream_defaults, refine_state)
        ar_max = crit.get("max_prism_aspect_ratio", 0) or 0
        wall_edge_limit = 0.97 * ar_max * h_first
        wall_size = float(size_params.get("size_near", 2.0))
        if ar_max > 0:
            # 0.85: frontal meshing overshoots the target spacing a little
            wall_size = min(wall_size, 0.85 * ar_max * h_first)
        wall_size = max(wall_size, size_min)
        n_pre = plan_yplus_stack(h_first, bl_defaults)[0] \
            if bl_defaults.get("target_yplus") else int(bl_defaults.get("NbLayers", 30))
        jump = crit.get("max_bl_top_jump", 0) or 0
        if jump > 0:
            r_max = float(bl_defaults.get("Ratio", 1.1))
            h_top = wall_size / jump
            if h_top > h_first and r_max > 1.0:
                n_top = int(math.ceil(1.0 + math.log(h_top / h_first) / math.log(r_max)))
                bl_top_thickness = h_first * (r_max ** n_top - 1.0) / (r_max - 1.0)
                n_pre = max(n_pre, n_top)
                if bl_top_thickness > float(bl_defaults.get("Thickness", 0) or 0):
                    print(f"Stack top jump: wall spacing {wall_size:.3f}mm / "
                          f"max_bl_top_jump {jump} needs a top layer of "
                          f"{h_top:.4f}mm, so Thickness is raised to "
                          f"{bl_top_thickness:.3f}mm ({n_top} layers at Ratio "
                          f"<= {r_max})")
        area = sum(gmsh.model.occ.getMass(2, f) for f in wall_faces)
        n_tri_est = 4.0 * area / (math.sqrt(3.0) * wall_size ** 2)
        print(f"Aspect ratio pre-flight: first cell {h_first:.5f}mm, wall "
              f"spacing {wall_size:.3f}mm (aspect {wall_size / h_first:.0f}), "
              f"wall area {area / 1e6:.2f} m2, about {n_tri_est:.3e} wall "
              f"triangles x {n_pre} layers = {n_tri_est * n_pre:.3e} prisms")
        # 317 bytes per cell: measured on the BWB half model (20.5M cells,
        # 6.5 GB); the 3D criteria check adds about 150 more per cell.
        print(f"  Prism block alone needs about {n_tri_est * n_pre * 317 / 1e9:.1f} GB "
              f"of RAM to mesh, plus about {n_tri_est * n_pre * 150 / 1e9:.1f} GB "
              "for the criteria check")
        if n_tri_est * n_pre > budget_early:
            s_ok = math.sqrt(4.0 * area * n_pre / (math.sqrt(3.0) * budget_early))
            print(f"  Aborting: that exceeds max_element_budget "
                  f"({budget_early:.3e}). Within the budget the wall spacing "
                  f"could be {s_ok:.3f}mm, an aspect ratio of "
                  f"{s_ok / h_first:.0f} at this first cell. The first cell "
                  "is not coarsened to fit; raise max_element_budget or "
                  "max_prism_aspect_ratio deliberately.")
            report("failed", "aspect ratio pre-flight exceeds the element budget")
            gmsh.finalize()
            return False
        if wall_size < float(size_params.get("size_near", 2.0)) - 1e-12:
            wall_cap_field = wall_size
        print()

    thresh_field = setup_size_field(wall_faces, size_params, small_faces)
    fields = [thresh_field]
    if wall_cap_field:
        growth = size_params.get("growth_rate") or 0.2
        dist_in = gmsh.model.mesh.field.getNumber(thresh_field, "InField")
        cap = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(cap, "InField", dist_in)
        gmsh.model.mesh.field.setNumber(cap, "SizeMin", wall_cap_field)
        gmsh.model.mesh.field.setNumber(cap, "SizeMax", size_far)
        gmsh.model.mesh.field.setNumber(cap, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(cap, "DistMax",
                                        (size_far - wall_cap_field) / growth)
        fields.append(cap)
        print(f"Wall spacing capped at {wall_cap_field:.3f}mm for the aspect "
              f"ratio limit, grading out at {growth}\n")
    boxes = []

    sections_config = seg_params.get("sections", {})
    wall_points = None
    if seg_enabled or config.get("nose_refinement_parameters", {}).get("enable", False):
        wall_points = sample_wall_points(wall_faces, max(body_length / 200.0, 1.0))
    default_section_params = {
        "enable": seg_enabled, "size_mm": size_far * 0.1,
        "transition_thickness_mm": 500.0, "lateral_pad_mm": 500.0,
    }
    for name, x_range in section_x_ranges.items():
        sect_cfg = merge_section_params(default_section_params,
                                        sections_config.get(name, {}))
        sb = section_box(name, sect_cfg, x_range, body_bbox, half_model,
                         sym_coord, size_far, wall_points)
        if sb:
            boxes.append(sb)
            fields.append(setup_box_field(name.capitalize(), sb[0], sb[1], size_far, sb[2]))

    # Older configs: nose_refinement_parameters, a box over the first
    # length_fraction of the body.
    nose_params = config.get("nose_refinement_parameters", {})
    if nose_params.get("enable", False) and not seg_enabled:
        frac = nose_params.get("length_fraction", 0.12)
        nb = section_box("nose", {"enable": True,
                                  "size_mm": nose_params.get("size_mm", 100.0),
                                  "transition_thickness_mm": nose_params.get(
                                      "transition_thickness_mm", 500.0),
                                  "lateral_pad_mm": nose_params.get("lateral_pad_mm", 500.0)},
                         (body_bbox[0], body_bbox[0] + frac * body_length),
                         body_bbox, half_model, sym_coord, size_far, wall_points)
        boxes.append(nb)
        fields.append(setup_box_field("Nose", nb[0], nb[1], size_far, nb[2]))

    shock = shock_box(shock_params, body_bbox, domain, half_model, sym_coord)
    if shock:
        boxes.append(shock)
        fields.append(setup_box_field("Shock", shock[0], shock[1], size_far, shock[2]))

    # Arbitrary named refinement boxes over a section of surface: an
    # explicit X/Y/Z box with its own target size, independent of the
    # nose/hump/rear segmentation and the shock cone. Used e.g. to resolve a
    # specific instability wavelength or tighten the surface mesh over a
    # known-thin region (a trailing edge) ahead of the boundary-layer pass.
    # Added on 28 Sep 2026: surface_refinement_boxes, a list of boxes with their own size and
    # transition thickness, for any part of the domain. This is a more general
    # version of the nose_refinement_parameters.  
    for rb in config.get("surface_refinement_boxes", []):
        if not rb.get("enable", True):
            continue
        box = (rb["XMin"], rb["XMax"], rb["YMin"], rb["YMax"], rb["ZMin"], rb["ZMax"])
        label = rb.get("name", "refine")
        # Select the actual wall faces whose bounding box falls in this region,
        # instead of sizing the whole 3D volume -- keeps refinement pinned to
        # the surface and lets it decay with wall distance like the rest of
        # the near-wall field, rather than ballooning the tet count in empty
        # fluid volume that happens to sit inside the box.
        faces_here = [f for f in wall_faces if _face_in_box(f, box)]
        if not faces_here:
            print(f"Surface refinement '{label}': no wall faces in "
                  f"X=[{box[0]:.0f},{box[1]:.0f}] Y=[{box[2]:.0f},{box[3]:.0f}] "
                  f"Z=[{box[4]:.0f},{box[5]:.0f}], skipped")
            continue
        size_in = rb.get("size_mm", size_far * 0.1)
        dist_max_local = rb.get("dist_max_mm", 10.0 * size_in)
        sampling = min(200, max(20, int(math.ceil(
            max(box[1] - box[0], box[3] - box[2], box[5] - box[4]) / size_in))))
        d = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d, "SurfacesList", faces_here)
        gmsh.model.mesh.field.setNumber(d, "Sampling", sampling)
        th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(th, "InField", d)
        gmsh.model.mesh.field.setNumber(th, "SizeMin", size_in)
        gmsh.model.mesh.field.setNumber(th, "SizeMax", size_far)
        gmsh.model.mesh.field.setNumber(th, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(th, "DistMax", dist_max_local)
        fields.append(th)
        print(f"Surface refinement '{label}': {len(faces_here)} face(s), "
              f"size_in={size_in}mm, decaying to size_far over "
              f"{dist_max_local:.0f}mm")

    cavity_faces = cavity_tags
    bl_cavity_faces = sorted(set(cavity_faces) | set(small_faces))
    if cavity_tags:
        cav_extent = 0.0
        for f in cavity_tags:
            bb = gmsh.model.getBoundingBox(2, f)
            cav_extent = max(cav_extent, bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
        cav_sampling = int(min(50, max(5, math.ceil(
            cav_extent / max(size_params.get("cavity_size", 1.0), 1e-9)) + 1)))
        cav_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(cav_dist, "SurfacesList", cavity_tags)
        gmsh.model.mesh.field.setNumber(cav_dist, "Sampling", cav_sampling)
        print(f"Cavity distance sampling: largest cavity face extent "
              f"{cav_extent:.2f}mm, using {cav_sampling}")
        cav_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(cav_thresh, "InField", cav_dist)
        gmsh.model.mesh.field.setNumber(cav_thresh, "SizeMin", size_params.get("cavity_size", 1.0))
        gmsh.model.mesh.field.setNumber(cav_thresh, "SizeMax", size_far)
        gmsh.model.mesh.field.setNumber(cav_thresh, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(cav_thresh, "DistMax", size_params.get("cavity_dist_max", 50))
        gmsh.model.mesh.field.setNumber(cav_thresh, "Sigmoid", 1)
        fields.append(cav_thresh)
        print(f"Cavity refinement on {len(cavity_tags)} surfaces")

    spot_growth = ar["spot_growth_rate"] if refining else 0.2
    if refining and refine_state["size_spots"]:
        fields += add_spot_fields(refine_state["size_spots"], size_far, spot_growth)
        print(f"Auto refinement: {len(refine_state['size_spots'])} local size "
              f"spot(s), smallest {min(s['size'] for s in refine_state['size_spots']):.3f}mm")

    if len(fields) > 1:
        background = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(background, "FieldsList", fields)
    else:
        background = thresh_field
    gmsh.model.mesh.field.setAsBackgroundMesh(background)
    print(f"Background field {background}, combining {len(fields)} field(s)\n")

    # 9. Boundary layer: one shared growth recipe (Size/Ratio/NbLayers/Quads
    # from boundary_layer_parameters), extruded in a SINGLE combined pass over
    # every section that wants a boundary layer.
    #
    # Two designs for PER-SECTION thickness were tried and both had to be
    # abandoned, so this needs explaining rather than just asserting "one
    # recipe":
    #
    # 1) A separate extrudeBoundaryLayer call per section -- the more obvious
    #    design. Does not work: gmsh raises "Could not find extruded node ...
    #    in surface N" the moment a second section's own call reaches the
    #    curve where it borders a section already extruded, since that
    #    curve's extrusion bookkeeping only supports being the source of one
    #    such call. Two sections ALWAYS share a boundary curve here, by
    #    construction (adjacent X-range chunks), so this is not an edge case
    #    to route around, it rules the design out entirely.
    #
    # 2) One combined call with a per-node scalar View (gmsh's viewIndex
    #    parameter on extrudeBoundaryLayer) rescaling the shared heights[]
    #    array locally -- this looked like the answer, and its GEOMETRY is
    #    genuinely correct: a node given scale 2.0 measurably lands at
    #    exactly double the offset of a node given scale 0.5. But that
    #    geometric check (comparing 2D top-surface point positions) turned
    #    out not to be the same question as "does the mesh volume it bounds
    #    actually realize". Checked separately, with generate(3) actually
    #    run: EVERY prism comes back degenerate ("Degenerated prism in
    #    extrusion of volume N"), even at a UNIFORM scale of 1.0 -- a value
    #    that should be a no-op and behave identically to not using the view
    #    at all, and does not. Reproduced on a minimal two-face case with
    #    both NbLayers=1 and NbLayers=3, so it is not a multi-layer
    #    interior-node artifact, it is the 3D "Extruded" volume mesher
    #    itself not accounting for view-driven geometry, however many layers
    #    are asked for. That makes viewIndex fine for warping a surface, but
    #    not usable here, where the volume between the surfaces is what
    #    SU2 actually meshes and solves on.
    #
    # With both routes to real per-section THICKNESS closed, what a section
    # actually gets varied per its own config is its near-wall TRIANGULATION
    # via size_mm (a finer surface mesh under the same shared BL recipe
    # places more, better-resolved columns there, which is real control even
    # though the column height itself is shared). A per-section
    # boundary_layer.enable: false is read, but refused in step 9b unless
    # experimental_partial_boundary_layer is set: the stack would end in an
    # abrupt step that does not mesh to usable quality.
    # Size and target_yplus stay configurable per section in the JSON (see
    # section_target_size below) so a future gmsh version that fixes this
    # gmsh-side limitation can pick them back up; today they are read, and
    # a per-section value that disagrees with the shared one is reported as
    # informational only, not applied.
    #
    # Verified separately on gmsh 4.15.2: the BoundaryLayer size FIELD (as
    # opposed to extrudeBoundaryLayer) is 2D only and aborts generate(3) with
    # "Only 2D Boundary Layers are supported", so it was never a candidate
    # either.
    freestream_defaults_present = bool(freestream_defaults)

    def section_target_size(sect_cfg, fallback_size):
        bl_over = sect_cfg.get("boundary_layer", {})
        target_yplus = bl_over.get("target_yplus")
        if target_yplus:
            fs = merge_section_params(freestream_defaults, bl_over.get("freestream", {}))
            y1 = compute_first_layer_size(target_yplus, fs)
            if y1 is not None:
                return y1
            if not freestream_defaults_present:
                print(f"    target_yplus={target_yplus} set but no usable "
                      "freestream_parameters given (need velocity_mps, "
                      "density_kgm3, dynamic_viscosity_pas, "
                      "reference_length_mm) -- keeping Size as configured.")
        return bl_over.get("Size", fallback_size)

    global_bl = dict(bl_defaults)
    any_bl_enabled = global_bl.get("enable", False)
    global_size = global_bl.get("Size", 0.05)

    # target_yplus in boundary_layer_parameters sets the first cell height
    # for the whole wall from freestream_parameters (compute_first_layer_size)
    # and overrides Size. The stack is then planned around it
    # (plan_yplus_stack): Ratio is the largest growth allowed, the layer
    # count follows from Thickness. A y+ target that cannot be evaluated
    # stops the run rather than silently meshing with a different first cell.
    global_target_yplus = global_bl.get("target_yplus")
    if any_bl_enabled and global_target_yplus:
        print(f"Boundary layer y+ sizing, target y+={global_target_yplus} "
              "(smallest of the incompressible and compressible estimates):")
        y1 = first_cell_height(global_bl, freestream_defaults, refine_state,
                               verbose=True) \
            if freestream_state(freestream_defaults) else None
        if refining and refine_state.get("first_cell_factor", 1.0) < 1.0:
            print(f"  y+ correction from an earlier pass: x"
                  f"{refine_state['first_cell_factor']:.3f}")
        if y1 is None:
            print("  target_yplus is set but freestream_parameters is missing "
                  "or invalid (need velocity_mps, density_kgm3, "
                  "dynamic_viscosity_pas, reference_length_mm). Aborting; "
                  "remove target_yplus to use Size directly.")
            gmsh.finalize()
            return False
        if bl_top_thickness:
            global_bl["Thickness"] = max(float(global_bl.get("Thickness", 0) or 0),
                                         bl_top_thickness)
        n_l, ratio, total = plan_yplus_stack(y1, global_bl)
        print(f"  First cell {y1:.5f}mm (overrides Size={global_size}mm), "
              f"{n_l} layers, ratio {ratio:.4f}, total {total:.3f}mm")
        global_size = y1
        global_bl["Size"] = y1
        global_bl["NbLayers"] = n_l
        global_bl["Ratio"] = ratio
    if any_bl_enabled and freestream_defaults:
        # Report what the chosen first cell gives along the body: Cf falls
        # with running length, so y+ is highest near the nose.
        l_ref = freestream_defaults.get("reference_length_mm", 0) or 0
        for frac in (0.0, 0.05, 0.25, 1.0):
            yu = yplus_unit_length(freestream_defaults, [frac * l_ref]) \
                if l_ref and freestream_state(freestream_defaults) else None
            if yu is not None:
                print(f"  y+ of the {global_size:.5f}mm first cell at "
                      f"x={max(frac * l_ref, freestream_defaults.get('yplus_min_running_length_mm', 10.0)):.0f}mm "
                      f"(worst model / wall temperature): {global_size / float(yu[0]):.2f}")
        print()

    for name in section_x_ranges:
        over = sections_config.get(name, {}).get("boundary_layer", {})
        for locked_key in ("NbLayers", "Ratio", "Quads"):
            if locked_key in over and over[locked_key] != global_bl.get(locked_key):
                print(f"  Note: section '{name}' boundary_layer.{locked_key} "
                      f"is ignored -- {locked_key} is one shared value for "
                      "the whole wall in a single extrusion pass (see step 9 "
                      "comment).")
        for size_key in ("Size", "target_yplus"):
            if size_key in over:
                this_size = section_target_size(
                    sections_config.get(name, {}), global_size)
                if abs(this_size - global_size) > 1e-9:
                    print(f"  Note: section '{name}' boundary_layer.{size_key} "
                          f"would give a first cell of {this_size:.4f}mm, but "
                          f"gmsh cannot realize a per-section thickness in "
                          f"this single combined extrusion pass (see step 9 "
                          f"comment) -- using the shared {global_size}mm "
                          "everywhere the boundary layer is enabled.")

    section_bl_enabled = {}
    if any_bl_enabled:
        for name in section_x_ranges:
            sect_cfg = sections_config.get(name, {})
            over = sect_cfg.get("boundary_layer", {})
            section_bl_enabled[name] = over.get("enable", True)

        print("Boundary layer: one shared growth recipe for every enabled "
              "section.")
        print(f"  Shared recipe: Size={global_size:.5f}mm Ratio={float(global_bl.get('Ratio', 1.1112)):.4f} "
              f"NbLayers={global_bl.get('NbLayers')} "
              f"Thickness={global_bl.get('Thickness')}")
        for name in section_x_ranges:
            if section_bl_enabled.get(name, True):
                print(f"  '{name}': boundary layer enabled, shared recipe")
            else:
                print(f"  '{name}': boundary layer disabled, tet region "
                      "touches the wall directly there")
        bl_half = bool(symmetry_faces)
        print()
    else:
        bl_half = False

    # Which extrusion builds the prisms. "custom" is build_custom_prism_layers
    # (per node thickness, prisms inside cavities, symmetry plane kept and
    # moved); "gmsh" is extrudeBoundaryLayer (one thickness for the whole
    # wall, cavities excluded, symmetry plane rebuilt). "auto" takes custom
    # whenever there is a symmetry plane or a cavity, which are exactly the
    # two cases the gmsh path cannot mesh reliably.
    bl_method = str(global_bl.get("method", "auto")).lower()
    if bl_method not in ("auto", "custom", "gmsh"):
        print(f"Unknown boundary_layer_parameters.method '{bl_method}', "
              "expected auto, custom or gmsh")
        gmsh.finalize()
        return False
    if bl_method == "auto":
        bl_method = "custom" if (symmetry_faces or bl_cavity_faces) else "gmsh"
    use_custom_bl = any_bl_enabled and bl_method == "custom"
    use_custom_bl_flag[0] = use_custom_bl
    if refining and refine_state["bl_spots"] and not use_custom_bl:
        print("Note: boundary layer thinning spots only apply to the custom "
              "extrusion; with this method only the size spots are used.")
    if any_bl_enabled:
        print(f"Boundary layer method: {bl_method}")
        if use_custom_bl:
            print(f"  {len(bl_cavity_faces)} cavity face(s) get a "
                  f"{global_bl.get('cavity_thickness_mm', 0.5)}mm stack, tapering to the full stack at "
                  f"thickness_gradient={global_bl.get('thickness_gradient', 0.25)}")
            if bl_half:
                print("  Half model: the symmetry plane mesh is kept and moved "
                      "around the prism strip.")
        elif bl_half:
            print("  Half model: the symmetry plane is removed before meshing "
                  "and rebuilt afterwards with the prism strip cut out.")
        print()

    # 9b. Resolve which wall faces actually get a boundary layer, once, here
    # -- both the pre-layer-volume removal below (step 12) and the extrusion
    # itself (step 12b) need this same partition, and computing it twice
    # risked them disagreeing about whether an extrusion will really happen.
    # The custom extrusion runs into the cavities with a thin stack, so it
    # keeps them in the prism wall; the gmsh extrusion cannot, so it leaves
    # them to the tet region.
    cavity_wall_faces = [] if use_custom_bl else \
        [f for f in cavity_faces if f in wall_faces]
    bl_wall_faces, disabled_section_faces = [], []
    for name, faces in section_wall_faces.items():
        group_faces = [f for f in faces if f not in cavity_wall_faces]
        if not group_faces:
            continue
        if any_bl_enabled and section_bl_enabled.get(name, True):
            bl_wall_faces += group_faces
        else:
            disabled_section_faces += group_faces
    will_extrude_bl = any_bl_enabled and bool(bl_wall_faces)
    bl_half = bl_half and will_extrude_bl

    # Half model + prism layers with the gmsh extrusion: the symmetry plane
    # has to be rebuilt with the prism strip cut out of it, and that step is
    # not reliable. On the test body it either left the tet region empty,
    # crashed HXT, or (with the conforming rebuild below) did not finish
    # meshing the plane within 10 minutes. The custom extrusion (method
    # custom / auto) avoids the rebuild altogether and is the supported half
    # model path; the gmsh one is refused here, before any meshing time is
    # spent.
    # Boundary layer on some sections but not others: the prism stack then
    # ends in an abrupt step (its full thickness, 0.09 mm at the test y+)
    # next to bare wall. Measured on the full-body test case with the hump
    # disabled: 475 degenerate pyramids and 6,210 tets below quality at the
    # two section boundaries with Quads on; 258k tets below quality with
    # Quads off. gmsh cannot taper the stack (see step 9), so stop early.
    if (will_extrude_bl and disabled_section_faces and
            (use_custom_bl or
             not geom_params.get("experimental_partial_boundary_layer", False))):
        off = [n for n in section_x_ranges if not section_bl_enabled.get(n, True)]
        print(f"Boundary layer disabled on section(s) {off} but enabled on "
              "the others: the prism layer would end in an abrupt step at "
              "the section boundary, which does not mesh to usable quality "
              "(degenerate pyramids / sliver tets there).\n"
              "  Enable the boundary layer on all sections, or disable it "
              "for the whole wall (boundary_layer_parameters.enable: false).\n"
              "  Or set geometry_parameters.experimental_partial_boundary_layer: "
              "true to try it anyway.")
        gmsh.finalize()
        return False

    if (bl_half and not use_custom_bl and
            not geom_params.get("experimental_half_model_boundary_layer", False)):
        print("Half model with the gmsh boundary layer extrusion is not "
              "supported: rebuilding the symmetry plane around the prism "
              "strip is not reliable (can hang or leave the tet region "
              "empty).\n"
              "  Use boundary_layer_parameters.method: custom (or auto), "
              "which keeps the symmetry plane and moves it instead.\n"
              "  Or set geometry_parameters.experimental_half_model_boundary_layer: "
              "true to try the gmsh path anyway.")
        gmsh.finalize()
        return False

    # 10. Pre-flight element budget
    estimated, volume_m3 = estimate_elements(domain, body_bbox, size_params,
                                             boxes, size_min, size_max)
    budget = util_params.get("max_element_budget", 40e6)
    print(f"Element budget estimate: domain {volume_m3:.0f} m3, "
          f"about {estimated:.3e} tets (budget {budget:.3e})")
    print("  Upper bound only: wall distance is approximated by the body "
          "bounding box, which for a slender or thin body counts much of the "
          "bounding box interior as near-wall. Measured about 8x high on a "
          "49 m slender test body, so calibrate max_element_budget from the "
          "actual over estimate ratio this script prints after meshing.")
    if estimated > budget:
        print("  Aborting before meshing. This would not finish, or would "
              "exhaust memory.")
        h_wall = max(size_params.get("size_near", 2.0), size_min)
        print(f"  Near-wall size is {h_wall}mm (larger of size_near and "
              f"Mesh.MeshSizeMin) out to dist_min={size_params.get('dist_min', 200.0)}mm, "
              "which dominates the count. Raise size_near and "
              "Mesh.MeshSizeMin, reduce dist_min, disable the shock box or "
              "shrink the far field; or raise max_element_budget deliberately.")
        gmsh.finalize()
        return False
    print()

    # 11. 3D algorithm choice, driven by the estimate rather than a guess
    # Netgen Frontal (4) is never chosen automatically, and is overridden
    # when prism layers are active: its "blockfill local h" background grid
    # is sized by domain extent over the thinnest boundary feature, and a
    # prism stack's top surface a fraction of a mm off the wall makes that
    # grid explode. Measured on the full-body test case: 4 hung past the
    # timeout (14M+ grid points, then OOM-killed), while Delaunay (1) and
    # HXT (10) both filled the identical boundary in seconds, including at
    # a 0.013 mm y+-driven first cell. On a half model it instead reported
    # "intersecting elements" and left the tet region empty.
    # auto = HXT (10): parallel, and handled the prism-layer boundary in
    # testing. Delaunay (1) produced the same mesh but single-threaded
    # (8.75M tets took 193 s on the full-body test case).
    # Exception: when part of the wall has no prism layer (a section with it
    # disabled, or a cavity), the quad side faces at the edge of the prism
    # region become part of the tet region's boundary, and HXT rejects any
    # non-triangle boundary element ("only supports triangles"). Delaunay
    # bridges them with pyramids.
    quad_boundary = (will_extrude_bl and global_bl.get("Quads", True)
                     and bool(disabled_section_faces or cavity_wall_faces))
    algorithm_3d = gmsh_params.get("Mesh.Algorithm3D", "auto")
    if isinstance(algorithm_3d, str) and algorithm_3d.lower() == "auto":
        algorithm_3d = 1 if quad_boundary else 10
        why = ("quad prism side faces border the tet region" if quad_boundary
               else "parallel")
        print(f"Algorithm3D auto: chosen {algorithm_3d} "
              f"({'Delaunay' if algorithm_3d == 1 else 'HXT'}, {why}), "
              f"estimate {estimated:.3e}\n")
    elif algorithm_3d == 10 and quad_boundary:
        print("Algorithm3D=10 (HXT) requested, but quad prism side faces "
              "border the tet region here (a section or cavity without a "
              "boundary layer) and HXT only accepts triangles. Using 1 "
              "(Delaunay) instead.\n")
        algorithm_3d = 1
    elif algorithm_3d == 4 and will_extrude_bl:
        print("Algorithm3D=4 (Netgen Frontal) requested, but it cannot fill "
              "around a prism boundary layer (hangs or leaves the tet region "
              "empty). Using 1 (Delaunay) instead.\n")
        algorithm_3d = 1
    gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_3d)

    # 12. Generate, 2D first so section resolution can be judged cheaply
    surface_only = util_params.get("surface_mesh_only", False)

    # The pre-layer volume is replaced by the prism blocks plus the tet region.
    # Left in the model it is meshed a second time alongside them, doubling
    # elements and memory, so it goes before any meshing happens. On a half
    # model the symmetry face goes too: it must not be meshed over the strip
    # the layers are about to occupy. Both are rebuilt after the extrusion.
    sym_outer_curves = []
    if will_extrude_bl and not surface_only and not use_custom_bl:
        if bl_half:
            sym_outer_curves = [
                t for d, t in gmsh.model.getBoundary(
                    [(2, f) for f in symmetry_faces], combined=False,
                    oriented=False)]
            wall_edges = set(t for d, t in gmsh.model.getBoundary(
                [(2, f) for f in wall_faces], combined=False, oriented=False))
            sym_outer_curves = [c for c in sym_outer_curves
                                if c not in wall_edges]
        # The old symmetry group has to be identified before its entities go:
        # once they are removed it stops appearing in getPhysicalGroups, yet it
        # keeps its name reserved, and the rebuilt plane then cannot be given
        # that name. It ends up in the .su2 as PhysicalSurface<n> instead.
        old_sym_group = None
        if bl_half:
            for dim, tag in gmsh.model.getPhysicalGroups(2):
                if gmsh.model.getPhysicalName(2, tag) == sym_name:
                    old_sym_group = (dim, tag)
                    break
        try:
            gmsh.model.removeEntities([(3, t) for t in fluid_volume_tags])
            if bl_half:
                if old_sym_group:
                    gmsh.model.removePhysicalGroups([old_sym_group])
                gmsh.model.removeEntities([(2, f) for f in symmetry_faces])
                print(f"Removed the symmetry face(s) {symmetry_faces} before "
                      f"meshing, outline kept as {len(sym_outer_curves)} "
                      "curve(s)")
            print(f"Removed the pre-layer fluid volume(s) {fluid_volume_tags}, "
                  "superseded by the layers\n")
        except Exception as e:
            print(f"Could not clear the pre-layer entities: {e}")
            print("  Disabling the boundary layer rather than meshing the "
                  "domain twice.\n")
            will_extrude_bl = bl_half = False

    print("Generating 2D surface mesh...")
    try:
        gmsh.model.mesh.generate(2)
        print("Surface mesh complete\n")
        report_surface_mesh(section_wall_faces)
        # Reported, not fixed: the wall mesh cannot be edited in place,
        # because the prism extrusion re-meshes its source surfaces on the
        # next generate() (and a manually edited triangle was verified to
        # be discarded there). The settings above are chosen to avoid them.
        ears = find_ear_triangles(wall_faces)
        if ears:
            print(f"  Warning: {sum(ears.values())} sliver 'ear' triangle(s) on "
                  f"the wall {dict(ears)} -- triangles with all three nodes on "
                  "one curved edge. Prisms extruded from them can invert. "
                  "Refining that section's size_mm usually clears them.")
        else:
            print("  Sliver 'ear' triangles on the wall: none")
    except Exception as e:
        print(f"Surface mesh generation failed: {e}")
        report("failed", f"surface mesh: {e}")
        gmsh.finalize()
        return False

    # Degenerate wall triangles are fixed in place: re-mesh the surface with
    # extra size spots on them. Cheap, and nothing depends on the surface
    # mesh yet (the prism stack is built from it next).
    # Local wall repair: poor or over-long wall triangles are re-triangulated
    # in place (collapse / flip / split on those triangles and their
    # neighbours only). Needs the custom extrusion, which builds the prisms
    # from the wall mesh exactly as it stands; the gmsh extrusion re-meshes
    # its source surfaces and would discard the edits.
    local_wall = (refining and lr["enable"] and use_custom_bl and will_extrude_bl
                  and not surface_only)
    if local_wall:
        print("Local wall repair (in place, no surface re-mesh):")
        # smallest corner sine a wall triangle may have for its prisms to
        # reach min_prism_gamma (prism gamma = 2/sqrt(3) x sine x cos tilt)
        wall_sine = (lr["wall_sine_margin"] * (math.sqrt(3.0) / 2.0)
                     * (crit.get("min_prism_gamma", 0) or 0.0))
        left, n0, n1 = local_wall_repair(
            bl_wall_faces, crit.get("min_wall_triangle_gamma", 0) or 0.0,
            wall_sine, wall_edge_limit, lr["min_spot_size_mm"],
            lr["wall_rings"], lr["max_wall_passes"],
            short_ratio=float(lr["short_edge_ratio"]))
        print(f"  Wall triangles {n0} -> {n1} ({n1 - n0:+d}), "
              f"{left} still below the gamma limit\n")
    if (refining and crit.get("min_wall_triangle_gamma", 0) > 0
            and (not local_wall or lr["global_remesh_fallback"])):
        remeshed = False
        for sp in range(int(ar["max_surface_passes"]) + 1):
            pts, hs = wall_triangle_defects(wall_faces, crit["min_wall_triangle_gamma"])
            if len(pts) == 0:
                print(f"Surface check: no wall triangle below gamma "
                      f"{crit['min_wall_triangle_gamma']}\n")
                break
            print(f"Surface check: {len(pts)} wall triangle(s) below gamma "
                  f"{crit['min_wall_triangle_gamma']}")
            if sp == int(ar["max_surface_passes"]):
                print("  max_surface_passes reached, continuing with them\n")
                break
            room = max(0, ar["max_total_spots"] - len(refine_state["size_spots"]))
            cands = cluster_defect_points(pts, hs, ar["cluster_factor"])
            changed, n_new, n_tight, n_stuck = merge_spots(
                refine_state["size_spots"], cands, ar, "size",
                min(ar["max_new_spots_per_iteration"], room), "surface")
            print(f"  Size spots: {n_new} new, {n_tight} tightened, {n_stuck} stuck")
            if not changed:
                print("  Nothing left to refine there, continuing\n")
                break
            new_ids = add_spot_fields(changed, size_far, spot_growth)
            background_new = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(background_new, "FieldsList",
                                             [background] + new_ids)
            background = background_new
            gmsh.model.mesh.field.setAsBackgroundMesh(background)
            print(f"  Re-meshing the surface, pass {sp + 1}")
            remeshed = True
            try:
                gmsh.model.mesh.clear()
                gmsh.model.mesh.generate(2)
            except Exception as e:
                print(f"Surface mesh generation failed: {e}")
                report("failed", f"surface mesh: {e}")
                gmsh.finalize()
                return False
        if remeshed:
            report_surface_mesh(section_wall_faces)

    if surface_only:
        # Never export a surface-only mesh as .su2: the solver rejects a mesh
        # with no volume elements, and a half-valid .su2 is worse than none.
        print("surface_mesh_only is set, writing the surface mesh only.")
        surf_path = output_path.replace('.msh', '_surface.msh')
        try:
            gmsh.option.setNumber("Mesh.MshFileVersion",
                                  gmsh_params.get("Mesh.MshFileVersion", 2.2))
            gmsh.write(surf_path)
            print(f"Surface mesh written: {surf_path}")
            print("Inspect the sections, then set surface_mesh_only to false.\n")
        except Exception as e:
            print(f"Surface mesh write failed: {e}\n")
        gmsh.finalize()
        return True

    # 12b. Prism boundary layer, between the 2D and 3D passes.
    #
    # The order is the whole trick: extrudeBoundaryLayer works off the existing
    # surface mesh, so it cannot run before generate(2), and the tet fill has
    # to see the extruded outer surface, so it cannot run after generate(3).
    #
    # ONE call, over every section with boundary layer enabled combined (see
    # step 9 for why this is one call rather than one per section). A
    # section with boundary layer disabled, and the cavity if present, are
    # excluded from that call's face list and join the tet region directly.
    # cavity_wall_faces / bl_wall_faces / disabled_section_faces /
    # will_extrude_bl all come from step 9b so this agrees with the removal
    # gate above about whether an extrusion is really going to happen.
    bl_volumes, outer_accum = [], []
    interface_surfaces = []
    if cavity_wall_faces:
        print(f"Cavity ({len(cavity_wall_faces)} face(s)): excluded from the "
              "boundary layer, tet region touches the wall directly there.")
        outer_accum += cavity_wall_faces
    outer_accum += disabled_section_faces

    def tri_count(faces):
        n = 0
        for f in faces:
            try:
                et, eg, _ = gmsh.model.mesh.getElements(2, f)
                n += sum(len(g) for t, g in zip(et, eg) if t == 2)
            except Exception:
                pass
        return n

    if will_extrude_bl:
        heights = bl_heights(global_bl)
        n_tri = tri_count(bl_wall_faces)
        total_prism = n_tri * len(heights)
        print(f"Prism budget: {n_tri} wall triangles x {len(heights)} layers "
              f"= {total_prism:.3e} prisms (before per-node scaling, which "
              "changes height, not count)")

        if total_prism > budget:
            print(f"  Aborting: total prisms ({total_prism:.3e}) exceed "
                  f"max_element_budget ({budget:.3e}). Coarsen the wall mesh, "
                  "cut NbLayers, or raise the budget deliberately.")
            gmsh.finalize()
            return False

        projected_low = total_prism + estimated / 8.0
        projected_high = total_prism + estimated
        print(f"Projected total with boundary layer: prisms {total_prism:.3e} "
              f"+ tet fill {estimated:.3e} (pre-flight estimate)")
        print(f"  Realistic range {projected_low:.3e} to {projected_high:.3e}, "
              "given the estimator's own up-to-8x slender-body bias.")
        if projected_high > budget:
            print(f"  Warning: the high end exceeds max_element_budget "
                  f"({budget:.3e}). This mesh may still exhaust memory or fail "
                  "to finish even though the prism check above passed.")

    if will_extrude_bl and use_custom_bl:
        print("Building prism boundary layer (custom extrusion)...")
        q_min = (crit.get("min_prism_gamma", 0) or 0.0) \
            if (refining and lr["enable"]) else 0.0
        n_retry = int(lr["max_wall_passes"]) if local_wall else 0
        attempt = 0
        while True:
            try:
                prism_vol, tet_vol, top_surface = build_custom_prism_layers(
                    bl_wall_faces, symmetry_faces, farfield_faces, bl_cavity_faces,
                    global_bl, sym_coord, fluid_volume_tags, sym_name,
                    thickness_spots=refine_state["bl_spots"] if refining else None,
                    quality_min=q_min, retry_wall=attempt < n_retry)
                break
            except MeshDefectError as e:
                if e.kind == "prism_wall":
                    # wall triangles too poor for the tilt of their column:
                    # re-triangulate just those, in place, and build again
                    attempt += 1
                    need = min(float(lr["max_forced_sine"]),
                               1.3 * (math.sqrt(3.0) / 2.0) * q_min
                               / float(e.cos_tilt.min()))
                    print(f"  {len(e.wall_tris)} wall triangle(s) too poor for "
                          f"their column tilt, local wall repair "
                          f"{attempt} of {n_retry}:")
                    left, n0, n1 = local_wall_repair(
                        bl_wall_faces, crit.get("min_wall_triangle_gamma", 0) or 0.0,
                        wall_sine, wall_edge_limit, lr["min_spot_size_mm"],
                        lr["wall_rings"], 2, forced=e.wall_tris, forced_sine=need)
                    print(f"  Wall triangles {n0} -> {n1} ({n1 - n0:+d}); "
                          "rebuilding the prism stack")
                    if n1 == n0 and left:
                        attempt = n_retry
                    continue
                print(f"Boundary layer extrusion failed: {e}")
                if refining and len(e.points):
                    return refine_and_stop(
                        [make_defect(e.kind, e.points, e.sizes, e.in_bl)],
                        f"prism layer {e.kind}")
                report("failed", f"prism layer: {e}")
                gmsh.finalize()
                return False
            except Exception as e:
                print(f"Boundary layer extrusion failed: {e}")
                report("failed", f"prism layer: {e}")
                gmsh.finalize()
                return False
        bl_volumes = [prism_vol]
        interface_surfaces = [top_surface]
        fluid_volume_tags = [prism_vol, tet_vol]
        print(f"  Fluid is now the prism block plus the tet region\n")
    elif will_extrude_bl:
        print("Extruding prism boundary layer...")
        try:
            # No viewIndex here: verified separately (see step 9 comment)
            # that gmsh's per-node view scaling produces a fully degenerate
            # 3D mesh at generate(3), so every enabled section shares this
            # one recipe's actual heights, not just its Ratio/NbLayers.
            ok, outer, bl_volumes, sym_laterals, strip_curves = build_prism_layers(
                bl_wall_faces, farfield_faces, symmetry_faces, global_bl,
                sym_coord, tol, heights=heights, view_index=-1)
            if not ok:
                print("  No layers produced, the wall joins the tet region "
                      "directly instead.")
                outer_accum += bl_wall_faces + farfield_faces
            else:
                outer_accum += outer
                interface_surfaces = list(outer)
        except Exception as e:
            print(f"Boundary layer extrusion failed: {e}")
            gmsh.finalize()
            return False

        if bl_volumes:
            try:
                if bl_half:
                    if not strip_curves:
                        raise RuntimeError(
                            "no strip edge found on the symmetry plane, so the "
                            "plane cannot be rebuilt around the layers")
                    new_sym = rebuild_symmetry_plane(sym_outer_curves,
                                                     [strip_curves],
                                                     sym_coord=sym_coord)
                    # the rebuilt plane still needs its own surface mesh
                    gmsh.model.mesh.generate(2)
                    # A middle section with its boundary layer disabled
                    # while both neighbours keep theirs splits the strip
                    # into separate, disjoint holes in the symmetry plane;
                    # that topology is not reliably meshable by the planar
                    # cut this function does and can come back with a
                    # "surface" that has zero elements rather than an
                    # outright error, which would otherwise surface much
                    # later as a confusing 3D-fill failure. Caught here,
                    # against the actual mesh, not guessed from loop counts.
                    for s in new_sym:
                        et, eg, _ = gmsh.model.mesh.getElements(2, s)
                        if not any(len(g) for g in eg):
                            raise RuntimeError(
                                f"rebuilt symmetry surface {s} meshed with "
                                "zero elements -- likely more than one "
                                "disjoint boundary-layer strip on the "
                                "symmetry plane (a section in the MIDDLE of "
                                "the body with its boundary layer disabled "
                                "while sections on both sides keep theirs). "
                                "Keep the boundary layer enabled on all "
                                "sections, only disable it on a section at "
                                "one end of the body, or set "
                                "enable_symmetry_half_model to false to mesh "
                                "the full body instead.")
                    outer_accum = outer_accum + new_sym

                loop = gmsh.model.geo.addSurfaceLoop(outer_accum)
                outer_volume = gmsh.model.geo.addVolume([loop])
                gmsh.model.geo.synchronize()

                # After the synchronize, not before: a group created earlier
                # comes back unnamed and SU2 gets a marker called
                # PhysicalSurface<n> instead of the configured name.
                if bl_half:
                    gtag = gmsh.model.addPhysicalGroup(2, new_sym + sym_laterals)
                    gmsh.model.setPhysicalName(2, gtag, sym_name)
                    print(f"  Symmetry marker '{sym_name}': {len(new_sym)} "
                          f"rebuilt surface(s) plus {len(sym_laterals)} "
                          "strip face(s)")
            except Exception as e:
                print(f"Could not close the outer volume: {e}")
                gmsh.finalize()
                return False

            fluid_volume_tags = bl_volumes + [outer_volume]
            # gmsh reads removePhysicalGroups([]) as "remove every group", which
            # silently takes the wall and farfield markers with it and writes a
            # .su2 with no NMARK section, so guard the empty case.
            stale = gmsh.model.getPhysicalGroups(3)
            if stale:
                gmsh.model.removePhysicalGroups(stale)
            gmsh.model.addPhysicalGroup(3, fluid_volume_tags, -1, "fluid")
            print(f"  Fluid is now {len(bl_volumes)} prism block(s) plus the "
                  f"tet region\n")
    elif any_bl_enabled:
        print("  No wall faces had boundary layer enabled after exclusions "
              "(or the pre-layer removal above failed and disabled it), "
              "skipping extrusion.\n")

    if dry_run:
        # extrudeBoundaryLayer only builds the geometric entities; the prism
        # elements themselves are not realized until generate(3), which is the
        # same call that fills the tets. So there is no "real" prism count to
        # read back here that is cheaper than just running the tet fill. What
        # dry-run buys instead: the surface mesh is real (not estimated), the
        # prism budget above is an exact count from that real mesh (not the
        # estimate the pre-flight check uses), and on a half model the
        # symmetry plane rebuild has actually run and would have raised by now
        # if the layers do not fit the geometry. Only the tet fill stays an
        # estimate.
        try:
            et, eg, _ = gmsh.model.mesh.getElements(2)
            n_surf = sum(len(g) for g in eg)
        except Exception:
            n_surf = 0
        print(f"DRY RUN stopping here. Real surface mesh: {n_surf} elements.")
        if any_bl_enabled:
            print("  Prism counts above are exact, from this real surface "
                  "mesh, not the pre-flight estimate.")
        print("  Tet fill is still only the pre-flight estimate above -- that "
              "part has not run. Remove dry_run (or --dry-run) to generate "
              "it.\n")
        report("dry_run")
        gmsh.finalize()
        return True

    # Tet region boundary self-intersection, before the (expensive) fill:
    # the stack top crossing itself or the moved symmetry plane is what
    # makes the fill fail with "a segment and a facet intersect", and only
    # HXT names a location for it (on raw stdout, not the gmsh logger).
    if refining and crit.get("check_boundary_intersections", True) and interface_surfaces:
        check_surfs = list(interface_surfaces)
        if use_custom_bl and symmetry_faces:
            check_surfs += [t for d, t in gmsh.model.getEntities(2)
                            if gmsh.model.getType(2, t) == "Discrete surface"
                            and t not in check_surfs]
        print(f"Boundary intersection check on {len(check_surfs)} surface(s)...")
        ipts, ih = boundary_intersections(check_surfs)
        if len(ipts):
            print(f"  {len(ipts)} intersecting boundary triangle(s), "
                  f"x=[{ipts[:, 0].min():.0f},{ipts[:, 0].max():.0f}]")
            if not final_pass:
                return refine_and_stop([make_defect("boundary_overlap", ipts, ih, True)],
                                       "tet region boundary intersects itself")
            print("  Last pass: trying the 3D fill anyway")
        else:
            print("  none\n")

    print("Generating 3D mesh...")
    gmsh.logger.start()
    try:
        gmsh.model.mesh.generate(3)
        print("Mesh generation complete\n")
    except Exception as e:
        print(f"3D mesh generation failed: {e}")
        if refining:
            pts = parse_message_points(gmsh.logger.get() + [str(e)])
            gmsh.logger.stop()
            if len(pts):
                return refine_and_stop([make_defect("fill_failure", pts, None, True)],
                                       "3D fill failed at reported location(s)")
            report("failed", f"3D fill: {e}")
        if bl_volumes:
            print("  With prism layers active, the usual cause is layers "
                  "colliding where the body is thin or concave: the sharp "
                  "trailing edge first. Reduce NbLayers or Thickness, or "
                  "coarsen the wall mesh there.")
        gmsh.finalize()
        return False

    try:
        etypes, etags, _ = gmsh.model.mesh.getElements(3)
        n_elem = sum(len(t) for t in etags)
        names = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}
        breakdown = {names.get(t, t): len(g) for t, g in zip(etypes, etags)}
        print(f"Element types: {breakdown}")
        n_node = len(gmsh.model.mesh.getNodes()[0])
        print(f"Volume elements={n_elem} nodes={n_node}")
        print(f"  estimate was {estimated:.3e}, actual over estimate "
              f"{n_elem / max(estimated, 1.0):.2f}\n")
    except Exception:
        pass

    # generate(3) can come back without raising even when one of its input
    # surfaces failed underneath it -- gmsh logs an Error to its own message
    # stream for a surface it could not mesh (e.g. "Impossible to mesh
    # periodic surface N", seen on a full, uncut body of revolution: a
    # surface like that stays fully periodic when there is no symmetry cut
    # to break it open, unlike a half model) but that does not always
    # propagate as a Python exception, and the run then "completes" with
    # whichever volumes happened to mesh and a silent hole where the failed
    # surface's volume should be -- on a boundary-layer run this reads as a
    # mesh that is ALL prism and NO tet at all, since the bulk tet-filled
    # region is exactly what went missing. Checked here directly, against
    # the actual volumes this run is supposed to have filled
    # (fluid_volume_tags, tracked through both the no-layer and the
    # boundary-layer path), rather than inferred from the element-type
    # breakdown above, which cannot tell "zero tets because there aren't
    # meant to be any" apart from "zero tets because the fill silently
    # failed".
    empty_volumes = []
    for vtag in fluid_volume_tags:
        try:
            vet, veg, _ = gmsh.model.mesh.getElements(3, vtag)
        except Exception as e:
            print(f"  Could not check volume {vtag}'s element count: {e}")
            continue
        if not any(len(g) for g in veg):
            empty_volumes.append(vtag)
    if empty_volumes:
        print(f"  Error: volume(s) {empty_volumes} came back with zero "
              "elements after generate(3) -- part of the domain silently "
              "failed to mesh (gmsh logged the failure to its own message "
              "stream rather than raising here; scroll up for an "
              "'Impossible to mesh' or similar Error line around the "
              "surface that caused it). Aborting rather than export a "
              "mesh with a hole in it.")
        report("failed", f"empty volume(s) {empty_volumes} after the 3D fill")
        gmsh.finalize()
        return False

    try:
        gmsh.logger.stop()
    except Exception:
        pass

    converged = None
    if refining:
        iface_nodes = []
        for s in interface_surfaces:
            try:
                iface_nodes.append(np.asarray(gmsh.model.mesh.getNodes(
                    2, s, includeBoundary=True)[0], dtype=np.int64))
            except Exception:
                pass
        iface_nodes = np.concatenate(iface_nodes) if iface_nodes else None
        tet_state = refine_state.setdefault("local", {})
        tet_state["tet_volumes"] = [v for v in fluid_volume_tags if v not in bl_volumes]
        tet_state["tet_spots"] = []
        local_it = 0
        cells_first = None
        while True:
            print("Checking refinement criteria (inversion, degenerate cells, "
                  "non-orthogonality)...")
            defects, dsum = collect_volume_defects(crit, iface_nodes)
            # y+ of the mesh as built: each wall node's actual first cell
            # against the height of y+ = 1 at its running length from the
            # leading edge (flow along +x), worst model and wall temperature.
            yplus_fix = None
            if (crit.get("max_yplus") and BL_STATS.get("h_first") is not None
                    and freestream_state(freestream_defaults)
                    and freestream_defaults.get("reference_length_mm")):
                bx = BL_STATS["xyz"]
                s_run = bx[:, 0] - bx[:, 0].min()
                yu = yplus_unit_length(freestream_defaults, s_run)
                yp = BL_STATS["h_first"] / yu
                dsum["worst_yplus"] = float(yp.max())
                dsum["yplus_violations"] = int((yp > crit["max_yplus"]).sum())
                if dsum["yplus_violations"]:
                    yplus_fix = 0.95 * crit["max_yplus"] / float(yp.max())
            print_defect_summary(dsum, crit)
            print_defect_groups(defects)
            converged = criteria_met(dsum, crit)
            if cells_first is None:
                cells_first = dsum.get("cells", 0)
            if converged or not lr["enable"] or local_it >= int(lr["max_iterations"]):
                break
            local_it += 1
            print(f"Local repair {local_it} of {int(lr['max_iterations'])} "
                  "(only the cells on and next to the defects):")
            n_cav = local_tet_repair(
                defects, tet_state, lr, background, size_far, "fluid",
                bl_volumes, gmsh_params.get("General.NumThreads", 16), crit)
            if n_cav == 0:
                print("  Nothing more can be repaired locally\n")
                break
            del defects
            print()
        if local_it:
            fluid_volume_tags = list(bl_volumes) + tet_state["tet_volumes"]
            print(f"Local repair: {local_it} pass(es), cells {cells_first} -> "
                  f"{dsum.get('cells', 0)} ({dsum.get('cells', 0) - cells_first:+d})\n")
        refine_state["metrics"] = dsum
        if converged:
            print("  All criteria met\n")
            report("converged", "criteria met", metrics=dsum)
        else:
            n_elem_now = dsum.get("cells", 0)
            over_budget = n_elem_now > budget
            if over_budget:
                print(f"  Element count {n_elem_now:.3e} exceeds "
                      f"max_element_budget {budget:.3e}: no further refinement")
            allow_global = (not lr["enable"]) or lr["global_remesh_fallback"]
            if not allow_global:
                print("  Not handing the rest to a whole-mesh re-mesh "
                      "(local_repair.global_remesh_fallback is false)")
            if not final_pass and not over_budget and allow_global:
                n_changed = register_defects(refine_state, defects, ar,
                                             use_bl_spots=use_custom_bl)
                if yplus_fix:
                    # The only global change the loop makes, and it can only
                    # make the first cell smaller.
                    refine_state["first_cell_factor"] = \
                        refine_state.get("first_cell_factor", 1.0) * yplus_fix
                    print(f"  y+ above {crit['max_yplus']}: first cell scaled by "
                          f"{yplus_fix:.3f} for the next pass")
                    n_changed += 1
                if n_changed:
                    report("needs_refinement", "criteria not met", metrics=dsum)
                    gmsh.finalize()
                    return False
                print("  Every remaining defect sits in a spot that cannot be "
                      "refined further (stuck)")
            report("not_converged" if final_pass or over_budget else "stuck",
                   "criteria not met", metrics=dsum)
            if not ar.get("export_if_not_converged", False):
                print("Criteria not met and export_if_not_converged is false, "
                      "not exporting.")
                gmsh.finalize()
                return False
            print("Criteria not met, exporting anyway (export_if_not_converged).\n")

    quality_ok = report_quality_mixed(gmsh_params.get("quality_abort_threshold", 0.01))
    if refining and not converged and ar.get("export_if_not_converged", False):
        quality_ok = True
    if not quality_ok:
        if util_params.get("abort_on_bad_quality", True):
            print("Aborting before export on mesh quality.")
            gmsh.finalize()
            return False

    # 13. Export
    try:
        gmsh.option.setNumber("Mesh.MshFileVersion",
                              gmsh_params.get("Mesh.MshFileVersion", 2.2))
        gmsh.write(output_path)
        print(f"Mesh exported: {output_path} "
              f"({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")

        if util_params.get("enable_su2_export", False):
            factor = util_params.get("su2_dilation_factor", 1e-3)
            # One call, not a Python loop over every node in the mesh.
            gmsh.model.mesh.affineTransform([factor, 0, 0, 0,
                                             0, factor, 0, 0,
                                             0, 0, factor, 0])
            su2_path = output_path.replace('.msh', '.su2')
            gmsh.write(su2_path)
            print(f"SU2 mesh exported, nodes scaled by {factor}: {su2_path}")
            print("SU2 config markers:")
            print(f"  MARKER_HEATFLUX= ( {wall_name}, 0.0 )")
            print(f"    or MARKER_ISOTHERMAL= ( {wall_name}, T_wall ) for a fixed wall temp")
            sym_present = any(gmsh.model.getPhysicalName(2, t) == sym_name
                              for _, t in gmsh.model.getPhysicalGroups(2))
            if sym_present:
                print(f"  MARKER_SYM= ( {sym_name} )")
            else:
                print("  (full body: no symmetry marker, omit MARKER_SYM)")
            print(f"  MARKER_FAR= ( {farfield_name} )")
            print("  There is no MARKER_WALL or MARKER_FARFIELD keyword in SU2.")
        print()
    except Exception as e:
        print(f"Export failed: {e}\n")
        gmsh.finalize()
        return False

    gmsh.finalize()
    print("Mesh generation successful")
    return True


if __name__ == "__main__":
    # Line buffered, so progress shows up when the log is redirected to a file.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    if args:
        cfg = args[0]
    else:
        # No argument (e.g. run from an IDE): use gmsh_config.json in the
        # working directory, else the config shipped next to this script.
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = next((c for c in ('gmsh_config.json',
                                os.path.join(here, 'gmsh_config.json'),
                                os.path.join(here, 'BWB_v4_cavity_half_BL_config.json'))
                    if os.path.exists(c)), 'gmsh_config.json')
    # create_mesh_iterative runs a single create_mesh pass unless the config
    # has auto_refinement_parameters.enable true.
    ok = create_mesh_iterative(cfg, dry_run='--dry-run' in flags)
    if not ok:
        print("Meshing stopped. The reason is the last message printed above.")
    # Under a debugger a non-zero SystemExit is reported as an exception and
    # hides that message, so only set the exit code when run normally.
    if sys.gettrace() is None:
        sys.exit(0 if ok else 1)
