import gmsh
import math
import os
import sys
import json
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


def build_custom_prism_layers(wall_faces, symmetry_faces, farfield_faces,
                              cavity_faces, bl_params, sym_coord,
                              fluid_volume_tags, sym_name, fluid_name="fluid"):
    """
    Build the prism layer from the existing wall surface mesh (see the block
    comment above), replace the pre-layer fluid volume by the prism block and
    a tet region, and move the symmetry plane mesh around the stack.

    Must run after generate(2) (every wall, symmetry and far-field surface
    meshed) and before generate(3). Returns (prism_volume, tet_volume).
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
    if min_dot.min() <= 0.0:
        raise RuntimeError(
            f"{int((min_dot <= 0).sum())} wall node(s) have no extrusion "
            "direction that clears every adjacent face (a corner sharper "
            "than the prism stack can turn); coarsen or fillet the geometry "
            "there")

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
    _gradient_limit(t, edges, elen, slope)
    print(f"  Thickness: {t_main:.3f}mm on the wall, {t_cav:.3f}mm on "
          f"{int(cav.sum())} cavity node(s), gradient {slope}")

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
    passes = 0
    while True:
        h, r = _solve_ratios(h0, n, t)
        bad = np.zeros(tris.shape[0], dtype=bool)
        prev = xyz[tris]
        for k in range(1, n + 1):
            cur = (xyz + nn * _layer_offset(h, r, k)[:, None])[tris]
            bad |= np.any(_prism_dets(prev, cur) <= 0.0, axis=1)
            prev = cur
        cut = [np.unique(tris[bad])]
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
        nbad = int(bad.sum())
        if nbad == 0 and nsym == 0:
            break
        passes += 1
        if passes > max_fix:
            raise RuntimeError(f"{nbad} prism column(s) and {nsym} symmetry "
                               f"triangle(s) still fold after {max_fix} "
                               "thickness reductions")
        idx = np.unique(np.concatenate(cut))
        t[idx] *= shrink
        _gradient_limit(t, edges, elen, slope)
        print(f"    pass {passes}: {nbad} folding prism column(s), {nsym} "
              f"folding symmetry triangle(s), thickness cut at {idx.size} "
              "node(s)")
    h, r = _solve_ratios(h0, n, t)
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
    return prism_vol, tet_vol

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


def create_mesh(config_file='gmsh_config.json', dry_run=False):
    config = load_config(config_file)
    if config is None:
        return False

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

    thresh_field = setup_size_field(wall_faces, size_params, small_faces)
    fields = [thresh_field]
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
        print(f"Boundary layer y+ sizing, target y+={global_target_yplus}:")
        y1 = compute_first_layer_size(global_target_yplus, freestream_defaults)
        if y1 is None:
            print("  target_yplus is set but freestream_parameters is missing "
                  "or invalid (need velocity_mps, density_kgm3, "
                  "dynamic_viscosity_pas, reference_length_mm). Aborting; "
                  "remove target_yplus to use Size directly.")
            gmsh.finalize()
            return False
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
        for frac in (0.05, 0.25, 1.0):
            y_unit = compute_first_layer_size(1.0, freestream_defaults,
                                              station_mm=frac * l_ref,
                                              verbose=False) if l_ref else None
            if y_unit:
                print(f"  y+ of the {global_size:.5f}mm first cell at "
                      f"x={frac * l_ref:.0f}mm: {global_size / y_unit:.2f}")
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
        gmsh.finalize()
        return False

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
        try:
            prism_vol, tet_vol = build_custom_prism_layers(
                bl_wall_faces, symmetry_faces, farfield_faces, bl_cavity_faces,
                global_bl, sym_coord, fluid_volume_tags, sym_name)
        except Exception as e:
            print(f"Boundary layer extrusion failed: {e}")
            gmsh.finalize()
            return False
        bl_volumes = [prism_vol]
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
        gmsh.finalize()
        return True

    print("Generating 3D mesh...")
    try:
        gmsh.model.mesh.generate(3)
        print("Mesh generation complete\n")
    except Exception as e:
        print(f"3D mesh generation failed: {e}")
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
        gmsh.finalize()
        return False

    if not report_quality_mixed(gmsh_params.get("quality_abort_threshold", 0.01)):
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
    ok = create_mesh(cfg, dry_run='--dry-run' in flags)
    if not ok:
        print("Meshing stopped. The reason is the last message printed above.")
    # Under a debugger a non-zero SystemExit is reported as an exception and
    # hides that message, so only set the exit code when run normally.
    if sys.gettrace() is None:
        sys.exit(0 if ok else 1)
