"""OpenFOAM v1706 file reader (ASCII and binary), fields and dictionaries."""
import gzip
import os
import re
import numpy as np

_HDR = re.compile(r"FoamFile\s*\{(.*?)\}", re.S)
_KV = re.compile(r"^\s*(\w+)\s+(.*?);", re.M)
_COMMENT_BLOCK = re.compile(rb"/\*.*?\*/", re.S)
_COMMENT_LINE = re.compile(rb"//[^\n]*")


def _open(path):
    if os.path.exists(path):
        return open(path, "rb")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rb")
    raise FileNotFoundError(path)


def read_raw(path):
    with _open(path) as f:
        return f.read()


def header(raw):
    txt = raw[:4096].decode("utf-8", "replace")
    m = _HDR.search(txt)
    if not m:
        return {}
    return {k: v.strip().strip('"') for k, v in _KV.findall(m.group(1))}


def strip_comments(raw):
    raw = _COMMENT_BLOCK.sub(b" ", raw)
    return _COMMENT_LINE.sub(b" ", raw)


def _after_header(raw):
    i = raw.find(b"FoamFile")
    if i < 0:
        return 0
    j = raw.find(b"}", i)
    return j + 1


def _parse_list(raw, pos, ncomp, binary, dtype=np.float64):
    """Parse `N ( ... )` starting at/after pos.  Returns (array, end_pos)."""
    m = re.compile(rb"\s*(\d+)\s*\(", re.S).match(raw, pos)
    if m is None:
        m = re.compile(rb"[^\d]*(\d+)\s*\(", re.S).match(raw, pos)
        if m is None:
            raise ValueError("malformed list")
    n = int(m.group(1))
    start = m.end()
    if binary:
        itemsize = np.dtype(dtype).itemsize * ncomp
        buf = raw[start:start + n * itemsize]
        arr = np.frombuffer(buf, dtype=dtype).reshape(n, ncomp) if ncomp > 1 \
            else np.frombuffer(buf, dtype=dtype)
        return np.array(arr), start + n * itemsize
    depth = 1
    i = start
    while depth:
        c = raw[i:i + 1]
        if c == b"(":
            depth += 1
        elif c == b")":
            depth -= 1
        i += 1
    body = raw[start:i - 1].replace(b"(", b" ").replace(b")", b" ")
    vals = np.array(body.split(), dtype=float)
    if ncomp > 1:
        vals = vals.reshape(-1, ncomp)
    return vals, i


def read_field(path):
    """Read a volScalarField / volVectorField.

    Returns dict(class, dimensions, internal (np array), boundary {patch: dict}).
    """
    raw = read_raw(path)
    hdr = header(raw)
    cls = hdr.get("class", "volScalarField")
    binary = hdr.get("format", "ascii") == "binary"
    ncomp = 3 if "Vector" in cls else 1
    body = strip_comments(raw[_after_header(raw):]) if not binary else \
        raw[_after_header(raw):]

    m = re.search(rb"internalField\s+([A-Za-z<>\s]*?)(?=[\d(])", body)
    if m is None:
        raise ValueError("no internalField in %s" % path)
    kind = m.group(1)
    pos = m.end()
    if b"nonuniform" in kind:
        arr, pos = _parse_list(body, pos, ncomp, binary)
    else:
        mm = re.compile(rb"\s*\(?([^;)]*)\)?\s*;").match(body, pos)
        vals = np.array(mm.group(1).split(), dtype=float)
        arr = vals if ncomp > 1 else float(vals[0])
    return dict(cls=cls, internal=arr, ncomp=ncomp, binary=binary,
                boundary=_patch_names(body))


def _patch_names(body):
    """Patch names present in boundaryField.  The patch *values* are not
    parsed: the stability solver only samples internal (cell-centred) data."""
    i = body.find(b"boundaryField")
    if i < 0:
        return {}
    out = {}
    for m in re.finditer(rb"^\s{0,8}(\w[\w.\-]*)\s*$", body[i:], re.M):
        out[m.group(1).decode()] = {}
    out.pop("boundaryField", None)
    return out


def read_dict(path):
    """Flat parse of an OpenFOAM dictionary into {key: value-string}."""
    raw = strip_comments(read_raw(path))
    txt = raw[_after_header(raw):].decode("utf-8", "replace")
    return {k: v.strip() for k, v in _KV.findall(txt)}


def list_times(case):
    out = []
    for d in os.listdir(case):
        try:
            out.append((float(d), d))
        except ValueError:
            continue
    return [n for _, n in sorted(out)]


def latest_time(case, exclude_zero=True):
    ts = list_times(case)
    if exclude_zero:
        ts = [t for t in ts if float(t) > 0.0] or ts
    if not ts:
        raise RuntimeError("no time directories in %s" % case)
    return ts[-1]
