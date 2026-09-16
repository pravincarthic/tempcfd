"""
snap_pse_probes_to_su2.py

Snap PSE stations onto real wall nodes of an SU2 native mesh and emit a
CUSTOM_OUTPUTS block for SU2 v8 (Harrier).

Updated for the current gmsh workflow (final_meshing.py + gmsh_config.json):

  - marker names come from su2_marker_names, so the wall marker defaults to
    "wall", not the old "Solid_Walls"
  - the body is rotated by rotation_angle_deg about Y before the cut, so the
    mesh is in the flow frame while the PSE stations are defined along the
    body axis. Nodes are rotated back into the body frame for snapping and
    probes are written back out in mesh (flow) frame coordinates
  - the domain may be a symmetry half model cut at symmetry_plane_y_mm, with
    either side kept. Spanwise stations on the missing side are mirrored onto
    the retained side instead of being dropped
  - the SU2 export scales nodes by su2_dilation_factor after a
    uniform_dilation_factor on the geometry, so station units are converted
    with the same net factor

  - --mack2 adds a tight streamwise cluster of probes around each primary
    station, spaced to the LOCAL second Mack-mode wavelength rather than a
    fixed distance. A single point per station gives an N-factor
    cross-check against the PSE output; it cannot recover a wavelength or
    phase speed from the CFD signal itself, and the second-mode wavelength
    varies by roughly two orders of magnitude from nose to tail on this
    body (millimetres near the nose, tens of millimetres aft), so no
    single fixed spacing covers both ends. The cluster spacing is worked
    out per station from a laminar flat-plate boundary-layer estimate at
    the given flight condition, reusing this repo's pse.atmosphere for the
    freestream state.

Usage:
    python snap_pse_probes_to_su2.py mesh.su2 --config gmsh_config.json
    python snap_pse_probes_to_su2.py mesh.su2 wall --aoa -5 --half-side +
    python snap_pse_probes_to_su2.py mesh.su2 --config gmsh_config.json \
        --mack2 --mach 10 --altitude-ft 130000 --wall-temp 1000
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------
# Target stations, body frame, metres along the body axis from the nose
# (from OpenFOAM system/probesPSE)
# ----------------------------------------------------------------------
X_STATIONS = [
    0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00,
    8.00, 10.00, 12.00, 14.00, 17.00, 20.00, 23.00, 26.00, 29.00, 32.00,
    35.00, 38.00, 41.00, 44.00, 47.00,
]

SPANWISE_STATIONS = [
    (20.0, -12.0), (20.0, -8.0), (20.0, -4.0), (20.0, 0.0),
    (20.0, 4.0), (20.0, 8.0), (20.0, 12.0),
    (35.0, -12.0), (35.0, -8.0), (35.0, -4.0), (35.0, 0.0),
    (35.0, 4.0), (35.0, 8.0), (35.0, 12.0),
]

FIELDS = ["PRESSURE", "TEMPERATURE", "DENSITY"]

# VTK element type -> node count, for surface elements
SURF_TYPE_NODES = {3: 2, 5: 3, 9: 4}

# The mesh config is the normal way to run this, so it is the default rather
# than something to remember on the command line. Forgetting it silently
# measured stations along the mesh axis instead of the body axis.
DEFAULT_CONFIG = "gmsh_config.json"


def resolve_config(arg):
    """Locate the config, searching the working directory then the script dir."""
    if arg is None or str(arg).lower() == "none":
        return None
    if os.path.exists(arg):
        return arg
    if arg == DEFAULT_CONFIG:
        beside = os.path.join(os.path.dirname(os.path.abspath(__file__)), arg)
        if os.path.exists(beside):
            return beside
        print("No %s found in the working directory or beside the script. "
              "Running without it: pass --config explicitly, or 'none' to "
              "silence this." % DEFAULT_CONFIG)
        return None
    raise FileNotFoundError(arg)


# ----------------------------------------------------------------------
# Second Mack-mode local wavelength, for probe clustering
# ----------------------------------------------------------------------
def edge_conditions(mach, altitude_ft):
    """
    Freestream state at the flight condition, reused from this repo's own
    atmosphere model (pse.atmosphere) rather than reimplemented here, so the
    numbers match what the PSE solver itself uses. Falls back to a request
    for explicit freestream values if the pse package is not importable from
    wherever this script is run (e.g. copied out to a bare post-processing
    box without the rest of the repo).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), ".."))
        from pse import atmosphere as atmo
        fs = atmo.freestream(mach, z_ft=altitude_ft)
        return dict(Ue=fs["U"], rhoe=fs["rho"], mue=fs["mu"], Te=fs["T"],
                   pe=fs["p"])
    except Exception as e:
        raise RuntimeError(
            "Could not get freestream conditions from pse.atmosphere (%s). "
            "--mack2 needs the flight condition to size the probe cluster. "
            "Run this from inside the repo, or pass --Uinf/--rhoinf/--muinf/"
            "--Tinf/--pinf directly." % e)


def sutherland_mu(T):
    """Sutherland's law for air viscosity, Pa.s, T in Kelvin."""
    return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)


def local_wavelength(x_m, edge, wall_temp_K, recovery=0.85, gamma=1.4,
                     gas_R=287.0):
    """
    Local second Mack-mode wavelength at station x, from a laminar
    compressible flat-plate boundary-layer estimate (Eckert reference
    temperature method).

    The second mode is a trapped acoustic wave with wavelength close to
    twice the local boundary-layer thickness and frequency f ~ 0.8 Ue / (2
    delta) (Mack 1984). This is an order-of-magnitude sizing tool for probe
    spacing, not a stability calculation -- the real growth rate and
    wavenumber come from the PSE solve itself. A shock-heated boundary layer
    downstream of the bow shock will be thinner than this predicts, so
    treat delta here as an upper bound and the wavelength as a not-finer-
    than spacing.

    Returns (delta_m, wavelength_m, f2_Hz).
    """
    Ue, rhoe, mue, Te = edge["Ue"], edge["rhoe"], edge["mue"], edge["Te"]
    mach = Ue / np.sqrt(gamma * gas_R * Te)

    Tr = Te * (1.0 + recovery * (gamma - 1.0) / 2.0 * mach ** 2)
    Tstar = Te + 0.5 * (wall_temp_K - Te) + 0.22 * (Tr - Te)
    mustar = sutherland_mu(Tstar)
    rhostar = edge["pe"] / (gas_R * Tstar)
    Cstar = (rhostar * mustar) / (rhoe * mue)

    Rex = rhoe * Ue * x_m / mue
    delta = 5.0 * x_m * np.sqrt(Cstar) / np.sqrt(max(Rex, 1.0))
    wavelength = 2.0 * delta
    f2 = 0.8 * Ue / wavelength
    return delta, wavelength, f2


def cluster_offsets(n_per_wavelength):
    """
    Symmetric integer offsets in units of dx = wavelength / n_per_wavelength,
    covering just under one wavelength around the primary station (which
    stays as offset 0, so it is unchanged when --mack2 is off).
    """
    half = n_per_wavelength // 2
    return [k for k in range(-half, half + 1) if k != 0]


# ----------------------------------------------------------------------
# Mesh workflow configuration
# ----------------------------------------------------------------------
def load_mesh_config(path):
    """
    Pull the mesh frame description out of the gmsh config.

    Returns a dict with marker, aoa_deg, rotation origin flag, unit scale,
    symmetry plane and retained side. Everything the snapping needs to undo
    what final_meshing.py did to the geometry.
    """
    with open(path, "r") as f:
        cfg = json.load(f)

    geom = cfg.get("geometry_parameters", {})
    util = cfg.get("utility_parameters", {})
    markers = cfg.get("su2_marker_names", {})

    # Geometry is dilated on import, then the whole mesh is scaled again on
    # SU2 export. Station coordinates are in the original geometry units, so
    # the net factor is the product of the two.
    net_scale = (geom.get("uniform_dilation_factor", 1.0)
                 * util.get("su2_dilation_factor", 1.0))

    return {
        "marker": markers.get("wall", "wall"),
        "aoa_deg": geom.get("rotation_angle_deg", 0.0),
        "rotation_by_com": geom.get("rotation_by_CoM", False),
        "scale": net_scale,
        "sym_y": geom.get("symmetry_plane_y_mm", 0.0)
                 * util.get("su2_dilation_factor", 1.0),
        "half_model": geom.get("enable_symmetry_half_model", False),
        "keep_positive": geom.get("symmetry_keep_positive_side", True),
    }


def rotation_y(angle_deg):
    """Rotation matrix about Y, the symmetry plane normal, as gmsh applies it."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


# ----------------------------------------------------------------------
# Mesh parsing
# ----------------------------------------------------------------------
def _value_after_equals(line):
    return line.split("=", 1)[1].strip()


def read_su2_surface(mesh_filename, wall_marker_name):
    """Return (all_vertices, faces) where faces is a list of node-index lists."""
    vertices = None
    faces = []
    seen_markers = []
    marker_found = False

    with open(mesh_filename, "r") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("NDIME"):
                ndime = int(_value_after_equals(stripped))
                if ndime != 3:
                    raise ValueError("Mesh is %dD; this script expects 3D" % ndime)
                continue

            if stripped.startswith("NPOIN"):
                # NPOIN can carry extra tokens, e.g. "NPOIN= 1000 800"
                n_points = int(_value_after_equals(stripped).split()[0])
                print("Global mesh points: %d. Loading coordinates." % n_points)
                pts = np.empty((n_points, 3), dtype=np.float64)
                for i in range(n_points):
                    parts = f.readline().split()
                    if len(parts) < 3:
                        raise ValueError(
                            "Truncated point line at index %d" % i)
                    # Trailing point index, if present, is ignored
                    pts[i] = (float(parts[0]), float(parts[1]), float(parts[2]))
                vertices = pts
                continue

            if stripped.startswith("MARKER_TAG"):
                tag = _value_after_equals(stripped)
                seen_markers.append(tag)
                is_target = tag == wall_marker_name
                # MARKER_ELEMS must be the next non-blank line
                elems_line = f.readline()
                while elems_line.strip() == "":
                    elems_line = f.readline()
                n_elems = int(_value_after_equals(elems_line).split()[0])

                if not is_target:
                    for _ in range(n_elems):
                        f.readline()
                    continue

                marker_found = True
                print("Marker %s: %d surface elements." % (tag, n_elems))
                for i in range(n_elems):
                    parts = f.readline().split()
                    etype = int(parts[0])
                    if etype not in SURF_TYPE_NODES:
                        raise ValueError(
                            "Unsupported surface element type %d on marker %s"
                            % (etype, tag))
                    nn = SURF_TYPE_NODES[etype]
                    if len(parts) < nn + 1:
                        raise ValueError(
                            "Truncated element line %d on marker %s" % (i, tag))
                    # Slice by node count so a trailing element index is dropped
                    faces.append([int(p) for p in parts[1:nn + 1]])

    if vertices is None:
        raise ValueError("No NPOIN block found in %s" % mesh_filename)
    if not marker_found:
        raise ValueError(
            "Marker %s not found in %s. Markers present: %s"
            % (wall_marker_name, mesh_filename, ", ".join(seen_markers)))
    if not faces:
        raise ValueError("Marker %s has no elements" % wall_marker_name)

    return vertices, faces


# ----------------------------------------------------------------------
# Surface classification
# ----------------------------------------------------------------------
def nodal_normals(vertices, faces):
    """Area-weighted nodal normals over the marker surface.

    Returns (node_ids, coords, normals) for nodes used by the marker only.
    """
    used = sorted({n for face in faces for n in face})
    remap = {g: i for i, g in enumerate(used)}
    coords = vertices[used]
    normals = np.zeros_like(coords)

    for face in faces:
        if len(face) < 3:
            continue
        local = [remap[n] for n in face]
        # Fan-triangulate; cross product magnitude gives 2x triangle area
        p0 = coords[local[0]]
        for k in range(1, len(local) - 1):
            p1 = coords[local[k]]
            p2 = coords[local[k + 1]]
            n = np.cross(p1 - p0, p2 - p0)
            for li in (local[0], local[k], local[k + 1]):
                normals[li] += n

    mag = np.linalg.norm(normals, axis=1)
    good = mag > 0
    normals[good] /= mag[good][:, None]
    return np.array(used), coords, normals


def orient_outward(coords, normals):
    """
    Make the nodal normals point away from the body.

    The wall marker comes from a boolean cut, so its faces are oriented into
    the fluid on some builds and into the solid on others. The windward and
    leeward split is a sign test on the normal, so a flipped marker silently
    swaps the two sets. Decided by the sign of the mean outward flux about the
    surface centroid, which is robust for a closed or near closed wall.
    """
    centre = coords.mean(axis=0)
    flux = float(np.sum((coords - centre) * normals))
    if flux < 0.0:
        print("Wall normals point inward, flipping to outward.")
        return -normals
    return normals


def split_surfaces(coords, normals):
    """Split into windward (normal points down) and leeward (points up).

    Done in the body frame, so the split follows the airframe's own lower and
    upper surfaces rather than the flow frame vertical.
    """
    nz = normals[:, 2]
    wind = nz < -1e-6
    leew = nz > 1e-6
    ambiguous = int(np.count_nonzero(~(wind | leew)))
    if ambiguous:
        print("Note: %d nodes have near-horizontal normals "
              "(side or trailing edge); excluded from both surfaces."
              % ambiguous)
    return wind, leew


def report_geometry(coords, sym_y, half_model_cfg):
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    print("Wall bounding box, body frame:")
    print("  x: %.4f to %.4f" % (lo[0], hi[0]))
    print("  y: %.4f to %.4f" % (lo[1], hi[1]))
    print("  z: %.4f to %.4f" % (lo[2], hi[2]))

    if max(X_STATIONS) > hi[0] or min(X_STATIONS) < lo[0]:
        print("WARNING: requested x stations fall outside the wall x range. "
              "Check mesh units and the nose origin.")

    span = hi[1] - lo[1]
    tol = max(1e-3, 1e-3 * max(span, 1.0))
    on_positive = lo[1] > sym_y - tol
    on_negative = hi[1] < sym_y + tol
    half_model = on_positive or on_negative
    keep_positive = on_positive

    if half_model:
        print("Half model detected: wall lies on the %s side of y = %.4f. "
              "Spanwise stations on the other side are mirrored."
              % ("positive" if keep_positive else "negative", sym_y))
    elif half_model_cfg:
        print("Note: the config asks for a half model but the wall spans both "
              "sides of the symmetry plane. The cut fell back to a full span "
              "domain, so every spanwise station is used as given.")

    return lo, hi, half_model, keep_positive


# ----------------------------------------------------------------------
# Probe construction
# ----------------------------------------------------------------------
def snap(tree, subset_coords, subset_normals, target_xy, offset):
    dist, idx = tree.query(target_xy)
    pt = subset_coords[idx]
    if offset != 0.0:
        # Step off the wall along the outward normal, into the fluid. A probe
        # sitting exactly on a boundary face is ambiguous: SU2 interpolates it
        # from whichever cell claims the point, so it can return the wall
        # boundary state rather than the first cell state. Normals are oriented
        # outward before this, so a positive offset always moves into the flow.
        pt = pt + offset * subset_normals[idx]
    return pt, dist


def mirror_target(x, y, sym_y, keep_positive):
    """Reflect a spanwise station onto the retained side of the symmetry plane."""
    dy = y - sym_y
    if keep_positive and dy < 0.0:
        return x, sym_y - dy, True
    if not keep_positive and dy > 0.0:
        return x, sym_y - dy, True
    return x, y, False


def build_probe_definitions(coords, normals, offset, tol,
                            half_model, keep_positive, sym_y,
                            mack2=None):
    """
    mack2, when not None, is a dict with edge conditions and clustering
    parameters (wall_temp_K, recovery, n_per_wavelength, min_dx). Each
    primary streamwise station then gets extra probes on either side of it,
    spaced to the local second-mode wavelength at that x, so the cluster can
    resolve the wave in the CFD signal instead of only sampling one point on
    it. Spanwise stations are left as single points: they check spanwise
    mode structure at a fixed x, not the streamwise wave.
    """
    wind_mask, leew_mask = split_surfaces(coords, normals)
    if not wind_mask.any() or not leew_mask.any():
        raise ValueError("Failed to identify both windward and leeward nodes")

    wind_c, wind_n = coords[wind_mask], normals[wind_mask]
    leew_c, leew_n = coords[leew_mask], normals[leew_mask]
    print("Windward nodes: %d. Leeward nodes: %d."
          % (len(wind_c), len(leew_c)))

    tree_wind = cKDTree(wind_c[:, :2])
    tree_leew = cKDTree(leew_c[:, :2])

    probes = []
    rejected = []
    mirrored = 0

    def add(tag, tree, c, n, target):
        pt, dist = snap(tree, c, n, target, offset)
        if dist > tol:
            rejected.append((tag, target, dist))
            return
        probes.append((tag, pt, dist))

    def add_station(tag, tree, c, n, x, centreline):
        n_before = len(probes)
        add(tag, tree, c, n, [x, centreline])
        if mack2 is None:
            return
        delta, wavelength, f2 = local_wavelength(
            x, mack2["edge"], mack2["wall_temp_K"], mack2["recovery"])
        dx = max(wavelength / mack2["n_per_wavelength"], mack2["min_dx"])

        # If the surface mesh is coarser than dx, distinct cluster targets
        # snap to the same node and the cluster silently collapses to
        # duplicate probes -- no error, just wasted output entries that all
        # read the same signal. Checked against the actual snapped points,
        # not the mesh size, since that is what determines the outcome.
        seen_pts = ([probes[n_before][1]] if len(probes) > n_before else [])
        collapsed = 0
        for k in cluster_offsets(mack2["n_per_wavelength"]):
            xc = x + k * dx
            if xc <= 0:
                continue
            n_pre = len(probes)
            add("%s_C%+d" % (tag, k), tree, c, n, [xc, centreline])
            if len(probes) > n_pre:
                pt = probes[-1][1]
                if any(np.allclose(pt, p, atol=1e-9) for p in seen_pts):
                    collapsed += 1
                seen_pts.append(pt)

        n_added = len(probes) - n_before
        print("  %-10s x=%6.2fm  delta~%.2fmm  lambda~%.2fmm  f2~%.1fkHz  "
              "cluster dx=%.2fmm (%d points requested)"
              % (tag, x, delta * 1e3, wavelength * 1e3, f2 / 1e3, dx * 1e3,
                 len(cluster_offsets(mack2["n_per_wavelength"])) + 1))
        if collapsed:
            print("    Warning: %d of %d cluster point(s) landed on a node "
                  "already used by another point in this cluster -- the "
                  "surface mesh here is coarser than the requested %.2fmm "
                  "spacing. Those probes read the same signal and add "
                  "nothing; wall_tangential resolution needs to be finer "
                  "than the second-mode wavelength for the cluster to work."
                  % (collapsed, n_added - 1, dx * 1e3))

    centreline = sym_y if half_model else 0.0
    if mack2 is not None:
        print("Second Mack-mode cluster: laminar boundary-layer estimate, "
              "shock heating not included, so treat these as upper bounds "
              "on delta/lambda and a not-finer-than spacing:")
    for i, x in enumerate(X_STATIONS):
        add_station("WIND_%02d" % i, tree_wind, wind_c, wind_n, x, centreline)
    for i, x in enumerate(X_STATIONS):
        add_station("LEEW_%02d" % i, tree_leew, leew_c, leew_n, x, centreline)
    for i, (x, y) in enumerate(SPANWISE_STATIONS):
        if half_model:
            x, y, was_mirrored = mirror_target(x, y, sym_y, keep_positive)
            mirrored += int(was_mirrored)
        add("SPAN_%02d" % i, tree_wind, wind_c, wind_n, [x, y])

    if mirrored:
        print("Mirrored %d spanwise station(s) onto the retained half. "
              "Duplicate probes are expected where a station and its mirror "
              "coincide." % mirrored)

    return probes, rejected


def report_probes(probes, rejected, tol, to_mesh):
    print("\nSnap distances in the body frame x-y plane, probe coordinates in "
          "mesh frame:")
    for tag, pt, dist in probes:
        m = to_mesh(pt)
        print("  %-10s -> %10.4f %10.4f %10.4f   d = %.4f"
              % (tag, m[0], m[1], m[2], dist))

    if rejected:
        print("\nDROPPED %d station(s) exceeding tolerance %.4f:"
              % (len(rejected), tol))
        for tag, target, dist in rejected:
            print("  %-10s target (%.4f, %.4f) nearest node d = %.4f"
                  % (tag, target[0], target[1], dist))


# ----------------------------------------------------------------------
# Config generation
# ----------------------------------------------------------------------
def generate_config_snippet(probes, to_mesh):
    entries = []
    for tag, pt, _ in probes:
        m = to_mesh(pt)
        for field in FIELDS:
            name = "%s_%s" % (field[0], tag)
            entries.append("  %s : Probe{%s}[%.6f, %.6f, %.6f]"
                           % (name, field, m[0], m[1], m[2]))

    # Separator is ";" plus a line continuation; the final line takes neither
    body = ";\\\n".join(entries)

    lines = []
    lines.append("% ---------------------- CUSTOM PROBES ----------------------- %")
    lines.append("CUSTOM_OUTPUTS= '\\")
    lines.append(body + "'")
    lines.append("")
    lines.append("HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF, HEAT, CUSTOM )")
    return "\n".join(lines), len(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("marker", nargs="?", default=None,
                    help="wall marker name; defaults to the config value or 'wall'")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="gmsh config used to build the mesh, for the marker "
                         "name, angle of attack, units and symmetry cut. "
                         "Defaults to %s next to the script or in the working "
                         "directory; pass 'none' to disable" % DEFAULT_CONFIG)
    ap.add_argument("--aoa", type=float, default=None,
                    help="rotation_angle_deg about Y applied to the geometry; "
                         "overrides the config value")
    ap.add_argument("--scale", type=float, default=None,
                    help="station units to mesh units factor; overrides the "
                         "config value. 1.0 when both are metres")
    ap.add_argument("--sym-y", type=float, default=None,
                    help="symmetry plane y in mesh units; overrides the config")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="offset above the wall along the outward nodal "
                         "normal, in mesh units (metres). Positive moves into "
                         "the fluid; keep it below the first cell height. "
                         "Leave at 0 for second-mode work: |p-hat| peaks at "
                         "the wall for the trapped acoustic mode")
    ap.add_argument("--tol", type=float, default=0.5,
                    help="max allowed x-y snap distance before a station is dropped")
    ap.add_argument("--out", default=None, help="write the config block to a file")
    ap.add_argument("--mack2", action="store_true",
                    help="add a streamwise probe cluster around each primary "
                         "station, spaced to the local second Mack-mode "
                         "wavelength, to resolve the wave in the CFD signal "
                         "rather than sample one point on it")
    ap.add_argument("--mach", type=float, default=10.0,
                    help="freestream Mach number, for the mack2 cluster sizing")
    ap.add_argument("--altitude-ft", type=float, default=130000.0,
                    help="flight altitude in feet, for the mack2 cluster sizing")
    ap.add_argument("--wall-temp", type=float, default=1000.0,
                    help="wall temperature in Kelvin, for the mack2 cluster "
                         "sizing (matches wall_bc.wall_temperature in the PSE "
                         "config, not necessarily the CFD wall BC)")
    ap.add_argument("--recovery-factor", type=float, default=0.85,
                    help="recovery factor for the reference-temperature "
                         "boundary-layer estimate used by mack2")
    ap.add_argument("--n-per-wavelength", type=int, default=6,
                    help="mack2 cluster points per local wavelength")
    ap.add_argument("--min-cluster-dx", type=float, default=1e-4,
                    help="floor on mack2 cluster spacing in metres, so a "
                         "vanishing wavelength near the nose does not request "
                         "probes finer than the mesh can resolve")
    args = ap.parse_args()

    cfg = {"marker": "wall", "aoa_deg": 0.0, "rotation_by_com": False,
           "scale": 1.0, "sym_y": 0.0, "half_model": False,
           "keep_positive": True}
    config_path = resolve_config(args.config)
    if config_path:
        cfg.update(load_mesh_config(config_path))
        print("Mesh config: %s" % config_path)

    marker = args.marker or cfg["marker"]
    aoa = cfg["aoa_deg"] if args.aoa is None else args.aoa
    scale = cfg["scale"] if args.scale is None else args.scale
    sym_y = cfg["sym_y"] if args.sym_y is None else args.sym_y

    if scale <= 0.0:
        raise ValueError("scale must be positive, got %g" % scale)

    if aoa == 0.0 and config_path is None and args.aoa is None:
        print("WARNING: no config and no --aoa, so stations are measured along "
              "the mesh x axis. If the geometry was rotated for angle of "
              "attack, station x is wrong by roughly x*(1-cos) + z*sin, which "
              "grows with local body thickness. Pass --config or --aoa 0 to "
              "silence this.")

    print("Wall marker: %s. Angle of attack: %.4f deg about Y. "
          "Station scale: %g. Symmetry plane y: %.4f."
          % (marker, aoa, scale, sym_y))
    if cfg["rotation_by_com"]:
        print("Note: the geometry was rotated about its centre of mass, not the "
              "origin. Station x is measured along the body axis either way, "
              "but the nose position shifts; check the bounding box below.")

    print("Reading SU2 mesh: %s" % args.mesh)
    vertices, faces = read_su2_surface(args.mesh, marker)
    _, mesh_coords, mesh_normals = nodal_normals(vertices, faces)
    mesh_normals = orient_outward(mesh_coords, mesh_normals)
    print("Unique wall nodes: %d." % len(mesh_coords))

    # Undo the angle of attack so snapping happens along the body axis, then
    # take probes back to mesh frame for the solver.
    rot = rotation_y(aoa)
    inv = rot.T
    body_coords = mesh_coords @ rot / scale
    body_normals = mesh_normals @ rot

    def to_mesh(pt):
        return (np.asarray(pt) * scale) @ inv

    _, _, half_model, keep_positive = report_geometry(
        body_coords, sym_y / scale, cfg["half_model"])

    mack2 = None
    if args.mack2:
        edge = edge_conditions(args.mach, args.altitude_ft)
        mack2 = {"edge": edge, "wall_temp_K": args.wall_temp,
                 "recovery": args.recovery_factor,
                 "n_per_wavelength": args.n_per_wavelength,
                 "min_dx": args.min_cluster_dx / scale}
        print("mack2 enabled: Mach %.1f, altitude %.0fft, wall %.0fK -> "
              "Ue=%.0fm/s Te=%.1fK rhoe=%.4gkg/m3"
              % (args.mach, args.altitude_ft, args.wall_temp, edge["Ue"],
                 edge["Te"], edge["rhoe"]))

    probes, rejected = build_probe_definitions(
        body_coords, body_normals, args.offset / scale, args.tol / scale,
        half_model, keep_positive, sym_y / scale, mack2=mack2)
    report_probes(probes, rejected, args.tol / scale, to_mesh)

    if mack2 is not None:
        # Nyquist against the highest second-mode frequency, at the most
        # upstream primary station: that is where delta is smallest and f2
        # is highest. Undersampling here aliases the very feature the
        # cluster exists to resolve.
        _, _, f2_max = local_wavelength(min(X_STATIONS), mack2["edge"],
                                        mack2["wall_temp_K"],
                                        mack2["recovery"])
        nyquist_dt = 1.0 / (2.0 * f2_max)
        practical_dt = nyquist_dt / 10.0
        print("\nHighest expected second-mode frequency (at x=%.2fm): "
              "%.1fkHz." % (min(X_STATIONS), f2_max / 1e3))
        print("  Nyquist needs an output time step below %.3e s. In "
              "practice resolve the wave, not just avoid aliasing: aim for "
              "10x oversampling, dt <~ %.3e s, and write CUSTOM history "
              "every physical time step at that rate."
              % (nyquist_dt, practical_dt))

    snippet, n_entries = generate_config_snippet(probes, to_mesh)
    print("\nGenerated %d custom outputs from %d probes."
          % (n_entries, len(probes)))

    if args.out:
        with open(args.out, "w") as f:
            f.write(snippet + "\n")
        print("Written to %s" % args.out)
    else:
        print()
        print(snippet)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print("Error: file not found: %s" % exc.filename)
        sys.exit(1)
    except ValueError as exc:
        print("Error: %s" % exc)
        sys.exit(1)
