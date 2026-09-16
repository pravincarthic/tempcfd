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
    # Probing is done by attempting assignments, so gmsh logs an Error for
    # every name that turns out not to exist and a Warning for every scalar
    # option tried as a list. Those lines are the probe working as intended,
    # not failures, so the terminal is silenced for the duration rather than
    # leaving alarming output in the log.
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


def setup_size_field(wall_faces, size_params):
    size_near = size_params.get("size_near", 2.0)
    size_far = size_params.get("size_far", 500.0)
    dist_min = size_params.get("dist_min", 200.0)
    dist_max = size_params.get("dist_max", 20000.0)

    sampling = auto_sampling(wall_faces, dist_min, size_params)

    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", wall_faces)
    gmsh.model.mesh.field.setNumber(dist_field, "Sampling", sampling)

    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", size_near)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", size_far)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", dist_max)
    gmsh.model.mesh.field.setNumber(thresh_field, "Sigmoid", 1)

    print(f"Size field: near={size_near}mm far={size_far}mm "
          f"dist_min={dist_min}mm dist_max={dist_max}mm\n")
    return thresh_field


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


def nose_box(nose_params, body_bbox, half_model, sym_coord):
    """
    Absolute size cap over the nose region, independent of curvature.

    Curvature sizing gives size = 2*pi*R/N, so it only ever places N nodes
    around a full circle however small R is, and on a locally near flat blended
    surface between two tight edges it asks for nothing. That is what leaves a
    chiselled stagnation region, which is the worst possible place for it.
    Verified that curvature sizing still applies alongside a background field,
    the minimum of the two being taken, so this box can only refine further.
    """
    if not nose_params.get("enable", True):
        return None
    bx_min, by_min, bz_min, bx_max, by_max, bz_max = body_bbox
    body_length = bx_max - bx_min
    if body_length <= 0:
        return None

    frac = nose_params.get("length_fraction", 0.12)
    pad = nose_params.get("lateral_pad_mm", 500.0)
    x_end = nose_params.get("XMax", bx_min + frac * body_length)
    box = (bx_min - pad, x_end,
           sym_coord if half_model else by_min - pad, by_max + pad,
           bz_min - pad, bz_max + pad)
    return (box,
            nose_params.get("size_mm", 5.0),
            nose_params.get("transition_thickness_mm", 1000.0))


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


def report_surface_mesh(wall_faces, body_bbox, nose_params):
    """
    Triangle count and edge length range on the wall, split into nose region
    and rest of body, classified per triangle by centroid.

    Per surface classification does not work here: a BWB is often one large
    blended surface whose bounding box starts at the nose, which would mark
    every triangle on the aircraft as nose region. Vectorised, because a Python
    loop over millions of wall triangles defeats the point of a quick check.
    """
    bx_min, _, _, bx_max, _, _ = body_bbox
    frac = nose_params.get("length_fraction", 0.12)
    x_nose_end = nose_params.get("XMax", bx_min + frac * (bx_max - bx_min))

    stats = {"nose": [0, np.inf, 0.0], "body": [0, np.inf, 0.0]}

    for surf in wall_faces:
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

            is_nose = p[:, :, 0].mean(axis=1) <= x_nose_end
            for key, mask in (("nose", is_nose), ("body", ~is_nose)):
                if not mask.any():
                    continue
                sub = e[mask]
                s = stats[key]
                s[0] += int(mask.sum())
                s[1] = min(s[1], float(sub.min()))
                s[2] = max(s[2], float(sub.max()))

    print("Wall surface mesh report:")
    for key, label in (("nose", f"nose region (x <= {x_nose_end:.0f}mm)"),
                       ("body", "rest of body")):
        count, lo, hi = stats[key]
        if count:
            print(f"  {label}: {count} triangles, edges {lo:.3f} to {hi:.3f}mm")
        else:
            print(f"  {label}: no triangles")
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
        if cap > 0 and acc > cap:
            print(f"  Stack capped by Thickness={cap}mm at {len(heights)} "
                  f"layers instead of {n}.")
            break
        heights.append(acc)
    if not heights:
        return []
    print(f"  Prism stack: {len(heights)} layers, first {h0}mm, ratio {ratio}, "
          f"total {heights[-1]:.3f}mm")
    return heights



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


def rebuild_symmetry_plane(sym_outer_curves, strip_outer_curves):
    """
    Rebuild the symmetry plane as a surface with the prism strip cut out.

    Where the body meets the symmetry plane, the prism blocks occupy a strip of
    that plane. The original symmetry face spans the whole cross section, strip
    included, so meshing it and then extruding gives two surfaces over the same
    area and generate(3) is handed a self-intersecting boundary.

    The fix is ordering plus a hole: the symmetry face is removed before the
    surface mesh is built, the layers are extruded, and the plane is then
    rebuilt from the domain outline with the outer edge of the strip as an
    interior boundary. The strip itself stays covered by the extrusion's own
    lateral faces, which carry the symmetry marker.

    Returns the new surface tags.
    """
    outer_loops = group_curves_into_loops(sym_outer_curves)
    hole_loops = group_curves_into_loops(strip_outer_curves)
    print(f"  Symmetry plane: {len(outer_loops)} outline loop(s), "
          f"{len(hole_loops)} strip loop(s) to cut out")

    holes = [gmsh.model.geo.addCurveLoop(l, reorient=True) for l in hole_loops]
    surfaces = []
    for outline in outer_loops:
        loop = gmsh.model.geo.addCurveLoop(outline, reorient=True)
        surfaces.append(gmsh.model.geo.addPlaneSurface([loop] + holes))
    gmsh.model.geo.synchronize()
    print(f"  Rebuilt symmetry surface(s): {surfaces}")
    return surfaces


def build_prism_layers(wall_faces, farfield_faces, symmetry_faces, bl_params,
                       sym_coord, tol, heights=None):
    """
    Grow prism layers off the wall, then hand back the surfaces that bound the
    remaining tet region.

    Uses gmsh.model.geo.extrudeBoundaryLayer, which extrudes an already meshed
    surface along its mesh normals. This is NOT the BoundaryLayer size field:
    that field is 2D only and aborts generate(3) with "Only 2D Boundary Layers
    are supported". Verified on gmsh 4.15.2 that this path produces real
    prisms and that they survive the .su2 write as element type 13.

    Must be called after generate(2) and before generate(3), since there has to
    be a wall surface mesh to extrude from.

    Returns (ok, outer_surfaces, bl_volumes, sym_laterals).
    """
    if heights is None:
        heights = bl_heights(bl_params)
    if not heights:
        print("  No layers requested, skipping.")
        return False, [], [], []

    ex = gmsh.model.geo.extrudeBoundaryLayer(
        [(2, f) for f in wall_faces], [1] * len(heights), heights,
        bl_params.get("Quads", True))
    gmsh.model.geo.synchronize()

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

    # A lateral shared by two neighbouring blocks appears twice in the return
    # and is internal. One appearing once sits on the open edge of the wall
    # patch set, which on a half model is the symmetry plane. This is a
    # topological test: the laterals have no mesh yet, so their coordinates
    # cannot be queried here.
    seen = {}
    for tag in laterals:
        seen[tag] = seen.get(tag, 0) + 1
    exterior = [tag for tag, n in seen.items() if n == 1]

    print(f"  Extruded {len(wall_faces)} wall face(s) into {len(vols)} prism "
          f"block(s): {len(tops)} tops, {len(seen)} laterals "
          f"({len(exterior)} exterior)")

    sym_laterals = exterior if symmetry_faces else []
    if sym_laterals:
        print(f"  {len(sym_laterals)} lateral face(s) lie in the symmetry "
              "plane and take the symmetry marker.")

    # Outer edge of the symmetry strip: curves used once across the exterior
    # laterals and not part of the original wall boundary. This is the hole the
    # rebuilt symmetry plane is cut around.
    strip_curves = []
    if sym_laterals:
        wall_edges = set(t for d, t in gmsh.model.getBoundary(
            [(2, f) for f in wall_faces], combined=False, oriented=False))
        counts = {}
        for tag in sym_laterals:
            for d, c in gmsh.model.getBoundary([(2, tag)], combined=False,
                                               oriented=False):
                counts[c] = counts.get(c, 0) + 1
        strip_curves = [c for c, n in counts.items()
                        if n == 1 and c not in wall_edges]
        print(f"  Strip outer edge: {len(strip_curves)} curve(s)")

    return True, tops + farfield_faces, vols, sym_laterals, strip_curves


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
    for etype, tags in zip(etypes, etags):
        if len(tags) == 0:
            continue
        q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSICN"),
                       dtype=float)
        name = {4: "tet", 5: "hex", 6: "prism", 7: "pyramid"}.get(etype, str(etype))
        prev = stats.get(name, [0, 1.0, 0])
        bad = int((q < threshold).sum()) if name == "tet" else 0
        stats[name] = [prev[0] + q.size, min(prev[1], float(q.min())),
                       prev[2] + bad]
        inverted += int((q <= 0.0).sum())

    print("Mesh quality by element type:")
    for name, (count, worst, bad) in sorted(stats.items()):
        line = f"  {name}: {count} cells, worst minSICN={worst:.4f}"
        if name == "tet":
            line += f", {bad} below {threshold}"
        else:
            line += "  (threshold not applied: anisotropic by design)"
        print(line)

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


def create_mesh(config_file='gmsh_config.json'):
    config = load_config(config_file)
    if config is None:
        return False

    step_file = config.get('step_file')
    output_path = config.get('output_file')
    export_geometry_path = config.get('export_geometry_path')

    gmsh_params = config.get('gmsh_parameters', {})
    size_params = config.get('size_field_parameters', {})
    geom_params = config.get('geometry_parameters', {})
    util_params = config.get('utility_parameters', {})
    bl_params = config.get('boundary_layer_parameters', {})
    shock_params = config.get('shock_box_parameters', {})
    nose_params = config.get('nose_refinement_parameters', {})
    marker_names = config.get('su2_marker_names', {})

    wall_name = marker_names.get("wall", "wall")
    sym_name = marker_names.get("symmetry", "symmetry")
    farfield_name = marker_names.get("farfield", "farfield")

    farfield_faces, symmetry_faces = [], []

    # 1. Initialize
    gmsh.initialize()
    print(f"gmsh {gmsh.__version__}")
    print(f"Input: {step_file}\nOutput: {output_path}\n")
    gmsh.option.setNumber("Mesh.RandomFactor", gmsh_params.get("Mesh.RandomFactor", 1e-5))
    gmsh.option.setNumber("General.NumThreads", gmsh_params.get("General.NumThreads", 16))
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
    if angle_deg != 0.0:
        if geom_params.get("rotation_by_CoM", False):
            origin = gmsh.model.occ.getCenterOfMass(3, body_tag)
        else:
            origin = (0.0, 0.0, 0.0)
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

    # 4. Far-field domain, half model when symmetry is enabled
    half_model = geom_params.get("enable_symmetry_half_model", True)
    sym_coord = geom_params.get("symmetry_plane_y_mm", 0.0)
    keep_positive = geom_params.get("symmetry_keep_positive_side", True)

    x0 = geom_params.get("far_field_x0_mm", -50000.0)
    length = geom_params.get("far_field_length_mm", 200000.0)
    r_envelope = geom_params.get("far_field_r_envelope_mm", 100000.0)

    if half_model:
        y_origin = sym_coord if keep_positive else sym_coord - r_envelope
        y_extent = r_envelope
        print(f"Half model: symmetry plane y={sym_coord}, keeping "
              f"{'positive' if keep_positive else 'negative'} side")
        if not (body_bbox[1] < sym_coord < body_bbox[4]):
            print("  Warning: the symmetry plane does not pass through the body")
        elif abs(abs(body_bbox[1] - sym_coord) - abs(body_bbox[4] - sym_coord)) > 1.0:
            print("  Warning: the body is not centred on the symmetry plane. "
                  "A half model is only valid for a symmetric body.")
    else:
        y_origin = -r_envelope
        y_extent = 2 * r_envelope
        print("Half model disabled, meshing the full span")

    if body_length > 0 and r_envelope / body_length > 1.5:
        print(f"  Warning: far field radius is {r_envelope / body_length:.1f} body "
              "lengths. At hypersonic speeds disturbances do not travel upstream, "
              "so the domain can be much tighter. Shrinking it saves more cells "
              "than the symmetry cut does.")

    box_tag = gmsh.model.occ.addBox(x0, y_origin, -r_envelope,
                                    length, y_extent, 2 * r_envelope)
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
    #   direct     cut the (possibly half) domain box by the body
    #   two_stage  cut a full width box by the body first, which is a clean non
    #              coincident boolean, then intersect with the half box so the
    #              symmetry plane passes through an already formed cavity
    #   full_model fall back to the full span domain, which is verified to cut
    #              this class of geometry, at twice the cell count
    min_fraction = geom_params.get("min_cut_volume_fraction", 1e-9)
    half_box_volume = length * y_extent * 2 * r_envelope
    full_box_volume = length * 2 * r_envelope * 2 * r_envelope

    def cut_volume_check(reference_volume):
        """Return (tags, removed_volume) against the box that strategy built."""
        tags = [t for d, t in gmsh.model.getEntities(3)]
        if not tags:
            return tags, 0.0
        fluid = sum(gmsh.model.occ.getMass(3, t) for t in tags)
        return tags, reference_volume - fluid

    def try_direct():
        gmsh.model.occ.cut([(3, box_tag)], [(3, body_tag)],
                           removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

    def try_two_stage():
        full_box = gmsh.model.occ.addBox(x0, -r_envelope, -r_envelope,
                                         length, 2 * r_envelope, 2 * r_envelope)
        half_box = gmsh.model.occ.addBox(x0, y_origin, -r_envelope,
                                         length, y_extent, 2 * r_envelope)
        gmsh.model.occ.synchronize()
        cut_res, _ = gmsh.model.occ.cut([(3, full_box)], [(3, body_tag)],
                                        removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.intersect(cut_res, [(3, half_box)],
                                 removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

    def try_full_model():
        full_box = gmsh.model.occ.addBox(x0, -r_envelope, -r_envelope,
                                         length, 2 * r_envelope, 2 * r_envelope)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.cut([(3, full_box)], [(3, body_tag)],
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
    for name, attempt in strategies:
        # Each attempt needs a clean slate, since a failed boolean has already
        # consumed the body via removeTool.
        gmsh.model.remove()
        gmsh.model.add("BWB_Spaceplane_Mesh")
        shapes = gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
        if dilation != 1.0:
            gmsh.model.occ.dilate(shapes, 0, 0, 0, dilation, dilation, dilation)
            gmsh.model.occ.synchronize()
        body_tag = [t for d, t in shapes if d == 3][0]
        if angle_deg != 0.0:
            if geom_params.get("rotation_by_CoM", False):
                origin = gmsh.model.occ.getCenterOfMass(3, body_tag)
            else:
                origin = (0.0, 0.0, 0.0)
            gmsh.model.occ.rotate([(3, body_tag)], origin[0], origin[1], origin[2],
                                  0, 1, 0, angle_deg * math.pi / 180.0)
            gmsh.model.occ.synchronize()
        if name == "direct":
            box_tag = gmsh.model.occ.addBox(x0, y_origin, -r_envelope,
                                            length, y_extent, 2 * r_envelope)
            gmsh.model.occ.synchronize()

        print(f"Boolean cut, strategy '{name}'...")
        try:
            attempt()
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
                y_origin = -r_envelope
                y_extent = 2 * r_envelope
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
    if export_geometry_path:
        export_step(export_geometry_path + ".3.step")
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
    z_lo, z_hi = -r_envelope, r_envelope
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

    # 8. Size fields
    thresh_field = setup_size_field(wall_faces, size_params)
    fields = [thresh_field]
    boxes = []

    nose = nose_box(nose_params, body_bbox, half_model, sym_coord)
    if nose:
        boxes.append(nose)
        fields.append(setup_box_field("Nose", nose[0], nose[1], size_far, nose[2]))

    shock = shock_box(shock_params, body_bbox, domain, half_model, sym_coord)
    if shock:
        boxes.append(shock)
        fields.append(setup_box_field("Shock", shock[0], shock[1], size_far, shock[2]))

    cavity_tags = size_params.get("cavity_surface_tags", [])
    if cavity_tags:
        cav_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(cav_dist, "SurfacesList", cavity_tags)
        gmsh.model.mesh.field.setNumber(cav_dist, "Sampling", 200)
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

    # 9. Boundary layer
    # Verified on gmsh 4.15.2: the BoundaryLayer field is 2D only. Attaching it
    # to a closed 3D body and calling generate(3) aborts with
    #   Only 2D Boundary Layers are supported (curve N is adjacent to 2 surfaces)
    # so no version of this script ever produced prism layers, and no choice of
    # option names will. SurfacesList is also not a valid option on that field
    # type and raises on assignment.
    #
    # Consequence: this mesh is all tetrahedra, which is not adequate on its own
    # for a Mach 10 wall heat flux. The realistic paths are to grow prisms with
    # an external inflation tool on this surface mesh, to mesh in a
    # boundary-layer-capable mesher, or to accept a wall-resolved isotropic
    # near-wall size and the cell count that implies.
    bl_enabled = bl_params.get("enable", False)
    bl_half = bl_enabled and bool(symmetry_faces)
    if bl_enabled:
        print("Boundary layer: prism layers will be extruded after the surface "
              "mesh, using gmsh.model.geo.extrudeBoundaryLayer.")
        print("  Note this is not the BoundaryLayer size field, which is 2D "
              "only and aborts generate(3).")
        if bl_half:
            print("  Half model: the symmetry plane is removed before meshing "
                  "and rebuilt afterwards with the prism strip cut out.")
        print()

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
        print("  Reduce or disable the shock box, shrink the far field, or raise "
              "size_near, size_in_mm and max_element_budget deliberately.")
        gmsh.finalize()
        return False
    print()

    # 11. 3D algorithm choice, driven by the estimate rather than a guess
    algorithm_3d = gmsh_params.get("Mesh.Algorithm3D", "auto")
    if isinstance(algorithm_3d, str) and algorithm_3d.lower() == "auto":
        threshold = gmsh_params.get("hxt_element_threshold", 20e6)
        algorithm_3d = 10 if estimated >= threshold else 4
        print(f"Algorithm3D auto: estimate {estimated:.3e} vs threshold "
              f"{threshold:.3e}, chosen {algorithm_3d}\n")
    gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_3d)

    # 12. Generate, 2D first so nose resolution can be judged cheaply
    surface_only = util_params.get("surface_mesh_only", False)

    # The pre-layer volume is replaced by the prism blocks plus the tet region.
    # Left in the model it is meshed a second time alongside them, doubling
    # elements and memory, so it goes before any meshing happens. On a half
    # model the symmetry face goes too: it must not be meshed over the strip
    # the layers are about to occupy. Both are rebuilt after the extrusion.
    sym_outer_curves = []
    if bl_enabled and not surface_only:
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
            bl_enabled = bl_half = False

    print("Generating 2D surface mesh...")
    try:
        gmsh.model.mesh.generate(2)
        print("Surface mesh complete\n")
        report_surface_mesh(wall_faces, body_bbox, nose_params)
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
            print("Inspect the nose, then set surface_mesh_only to false.\n")
        except Exception as e:
            print(f"Surface mesh write failed: {e}\n")
        gmsh.finalize()
        return True

    # 12b. Prism boundary layer, between the 2D and 3D passes.
    #
    # The order is the whole trick: extrudeBoundaryLayer works off the existing
    # surface mesh, so it cannot run before generate(2), and the tet fill has
    # to see the extruded outer surface, so it cannot run after generate(3).
    bl_volumes, sym_laterals = [], []
    if bl_enabled:
        # Prism count is wall triangles times layers, known exactly now that
        # the surface mesh exists. This is the number that decides whether a
        # viscous mesh fits, and it dwarfs the tet estimate made before meshing.
        n_wall_tri = 0
        for f in wall_faces:
            try:
                et, eg, _ = gmsh.model.mesh.getElements(2, f)
                n_wall_tri += sum(len(g) for t, g in zip(et, eg) if t == 2)
            except Exception:
                pass
        heights_preview = bl_heights(bl_params)
        n_layers = len(heights_preview)
        n_prism = n_wall_tri * n_layers
        print(f"Prism budget: {n_wall_tri} wall triangles x {n_layers} layers "
              f"= {n_prism:.3e} prisms, before any tet")
        if n_prism > budget:
            print(f"  Aborting: that alone exceeds max_element_budget "
                  f"({budget:.3e}). Coarsen the wall mesh, cut NbLayers, or "
                  "raise the budget deliberately.")
            gmsh.finalize()
            return False

        print("Extruding prism boundary layer...")
        try:
            ok, outer, bl_volumes, sym_laterals, strip_curves = \
                build_prism_layers(wall_faces, farfield_faces, symmetry_faces,
                                   bl_params, sym_coord, tol,
                                   heights=heights_preview)
        except Exception as e:
            print(f"Boundary layer extrusion failed: {e}")
            gmsh.finalize()
            return False

        if ok:
            try:
                if bl_half:
                    if not strip_curves:
                        raise RuntimeError(
                            "no strip edge found on the symmetry plane, so the "
                            "plane cannot be rebuilt around the layers")
                    new_sym = rebuild_symmetry_plane(sym_outer_curves,
                                                     strip_curves)
                    # the rebuilt plane still needs its own surface mesh
                    gmsh.model.mesh.generate(2)
                    outer = outer + new_sym

                loop = gmsh.model.geo.addSurfaceLoop(outer)
                outer_volume = gmsh.model.geo.addVolume([loop])
                gmsh.model.geo.synchronize()

                # After the synchronize, not before: a group created earlier
                # comes back unnamed and SU2 gets a marker called
                # PhysicalSurface<n> instead of the configured name.
                if bl_half:
                    gtag = gmsh.model.addPhysicalGroup(2, new_sym + sym_laterals)
                    gmsh.model.setPhysicalName(2, gtag, sym_name)
                    print(f"  Symmetry marker '{sym_name}': {len(new_sym)} "
                          f"rebuilt surface(s) plus {len(sym_laterals)} strip "
                          "face(s)")
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
            print(f"  MARKER_SYM= ( {sym_name} )")
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
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'gmsh_config.json'
    sys.exit(0 if create_mesh(cfg) else 1)
