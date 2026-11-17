"""Mutation++ data for the AIR-7 species, as shipped in Mutation++
(data/thermo/species.xml, data/thermo/elements.xml, data/transfer/VT.xml).

If the environment variable MPP_DATA_DIRECTORY points to a Mutation++ data
tree, `load()` re-reads these values from it so that the stability operator
uses exactly the database the base-flow solver was linked against; the
embedded copy below is only the fallback.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

ELEMENT_MW = {"e-": 0.000548579903e-3, "N": 14.0067e-3, "O": 15.9994e-3}
STOICH = {"e-": {"e-": 1}, "N2": {"N": 2}, "O2": {"O": 2},
          "NO": {"N": 1, "O": 1}, "N": {"N": 1}, "O": {"O": 1},
          "NO+": {"N": 1, "O": 1, "e-": -1}}
CM1_TO_K = 1.4387                      # ParticleRRHO: 1/cm -> K

# RRHO: linear, theta_r [K], steric factor, theta_v [K], hf(298.15) [J/mol],
#       electronic levels (degeneracy, energy [1/cm])
RRHO = {
    "e-": dict(linear=False, theta_r=0.0, sigma=1, theta_v=[], hf=0.0,
               levels=[]),
    "N2": dict(linear=True, theta_r=2.886, sigma=2, theta_v=[3408.464], hf=0.0,
               levels=[(1, 0.0), (3, 50203.5), (6, 59619.2), (6, 59808.0),
                       (3, 66272.5), (1, 68152.7), (2, 69283.0), (2, 72097.6)]),
    "O2": dict(linear=True, theta_r=2.086, sigma=2, theta_v=[2276.979], hf=0.0,
               levels=[(3, 0.0), (2, 7917.5), (1, 13195.1), (1, 33055.7),
                       (6, 34690.9), (3, 35393.1), (3, 49792.1)]),
    "NO": dict(linear=True, theta_r=2.464, sigma=1, theta_v=[2759.293],
               hf=91089.0, levels=[(4, 0.0), (8, 38000.0)]),
    "N": dict(linear=False, theta_r=0.0, sigma=1, theta_v=[], hf=472440.0,
              levels=[(4, 0.0), (10, 19228.0), (6, 28842.0), (12, 83333.0),
                      (6, 86197.0), (12, 88132.0), (2, 93585.0), (20, 94843.0),
                      (12, 95513.0), (4, 96755.0), (10, 96835.0), (6, 97795.0),
                      (10, 99666.0), (12, 103691.0), (6, 104199.0),
                      (6, 104627.0), (28, 104723.0), (26, 104852.0),
                      (20, 105014.0), (10, 105135.0), (2, 106482.0),
                      (20, 106829.0), (12, 107014.0), (10, 107224.0),
                      (4, 107450.0), (6, 107619.0), (12, 109885.0),
                      (6, 110079.0), (90, 110329.0), (126, 110442.0),
                      (24, 110636.0), (2, 111063.0), (38, 111240.0),
                      (4, 111499.0), (10, 111886.0), (6, 112313.0),
                      (18, 112668.0), (60, 112821.0), (126, 112910.0),
                      (32, 113354.0), (18, 114120.0), (90, 114209.0),
                      (180, 114241.0), (20, 114547.0), (108, 115015.0),
                      (18, 115467.0)]),
    "O": dict(linear=False, theta_r=0.0, sigma=1, theta_v=[], hf=249229.0,
              levels=[(9, 0.0), (5, 15889.0), (1, 33794.0), (5, 73768.0),
                      (3, 76792.0), (15, 86624.0), (9, 88641.0), (5, 95480.0),
                      (3, 96222.0), (25, 97513.0), (15, 97593.0),
                      (15, 99207.0), (9, 99771.0), (15, 101223.0),
                      (5, 102191.0), (3, 102513.0), (5, 102755.0),
                      (25, 102917.0), (15, 102997.0), (56, 103078.0),
                      (15, 103723.0), (9, 103965.0), (5, 105094.0),
                      (3, 105256.0), (40, 105498.0), (56, 105554.0),
                      (15, 105901.0), (9, 105982.0), (5, 106627.0),
                      (3, 106708.0), (168, 106869.0), (5, 107514.0),
                      (3, 107595.0), (96, 107700.0), (8, 108175.0),
                      (40, 108224.0), (8, 108555.0), (40, 108595.0),
                      (3, 108829.0), (40, 108853.0)]),
    "NO+": dict(linear=True, theta_r=2.892, sigma=1, theta_v=[3473.491],
                hf=990653.0,
                levels=[(1, 0.0), (3, 52190.0), (6, 59225.0), (6, 61883.0),
                        (3, 67745.0), (1, 69542.0), (2, 71452.0),
                        (2, 73474.0)]),
}

# Millikan-White (a, b) per vibrator and heavy partner, VT.xml, and the
# reference cross section omega_v [m^2] of Park's correction
VT_OMEGAV = {"N2": 3.0e-21, "O2": 3.0e-21, "NO": 3.0e-21, "NO+": 3.0e-21}
VT_AB = {
    "N2": {"N": (180.0, 0.0262), "O": (72.4, 0.015), "N2": (221.0, 0.0290),
           "O2": (229.0, 0.0295), "NO": (225.0, 0.0293), "NO+": (225.0, 0.0293)},
    "O2": {"N": (72.4, 0.015), "O": (47.7, 0.059), "N2": (134.0, 0.0295),
           "O2": (138.0, 0.0300), "NO": (136.0, 0.0298), "NO+": (136.0, 0.0298)},
    "NO": {k: (49.5, 0.042) for k in ("N", "O", "N2", "O2", "NO", "NO+")},
    "NO+": {"N": (188.0, 0.0264), "O": (197.0, 0.0270), "N2": (231.0, 0.0293),
            "O2": (239.0, 0.0298), "NO": (236.0, 0.0295), "NO+": (236.0, 0.0295)},
}


def species_mw(name, elem=None):
    elem = elem or ELEMENT_MW
    return sum(n * elem[e] for e, n in STOICH[name].items())


def _read_species_xml(path, names):
    root = ET.parse(path).getroot()
    out = {}
    for sp in root.iter("species"):
        n = sp.get("name")
        if n not in names:
            continue
        th = sp.find("thermodynamics[@type='RRHO']")
        if th is None:
            continue

        def txt(tag, default=None):
            e = th.find(tag)
            return e.text.strip() if e is not None else default
        tv = txt("vibrational_temperatures")
        out[n] = dict(
            linear=(txt("linear", "no").lower() == "yes"),
            theta_r=float(txt("rotational_temperature", "0")),
            sigma=float(txt("steric_factor", "1")),
            theta_v=[float(v) for v in tv.replace(",", " ").split()] if tv else [],
            hf=float(th.find("formation_enthalpy").text.strip()),
            levels=[(int(l.get("degeneracy")), float(l.get("energy")))
                    for l in th.iter("level")])
    return out


def _read_vt_xml(path, names):
    root = ET.parse(path).getroot()
    mw = root.find("Millikan-White")
    ab, om = {}, {}
    for v in mw.iter("vibrator"):
        n = v.get("species")
        if n not in names:
            continue
        if v.get("omegav"):
            om[n] = float(v.get("omegav"))
        ab[n] = {p.get("species"): (float(p.get("a")), float(p.get("b")))
                 for p in v.iter("partner") if p.get("species") in names}
    return ab, om


def load(names, data_dir=None):
    """Return (rrho, vt_ab, vt_omegav, source) for the requested species."""
    data_dir = data_dir or os.environ.get("MPP_DATA_DIRECTORY")
    rrho = {n: dict(RRHO[n]) for n in names if n in RRHO}
    ab = {k: dict(v) for k, v in VT_AB.items()}
    om = dict(VT_OMEGAV)
    src = "embedded"
    if data_dir:
        sp = os.path.join(data_dir, "thermo", "species.xml")
        vt = os.path.join(data_dir, "transfer", "VT.xml")
        if os.path.exists(sp):
            rrho.update(_read_species_xml(sp, names))
            src = data_dir
        if os.path.exists(vt):
            a2, o2 = _read_vt_xml(vt, names)
            for k, v in a2.items():
                ab.setdefault(k, {}).update(v)
            om.update(o2)
    missing = [n for n in names if n not in rrho]
    if missing:
        raise KeyError("no RRHO data for %s" % missing)
    return rrho, ab, om, src


def mechanism_path(name="air7_SU2", data_dir=None):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, "mutationpp_data")
    for d in (data_dir or os.environ.get("MPP_DATA_DIRECTORY"), here):
        if d:
            p = os.path.join(d, "mechanisms", name + ".xml")
            if os.path.exists(p):
                return p
    raise FileNotFoundError("mechanism %s.xml not found" % name)


def mixture_path(name="air_7", data_dir=None):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, "mutationpp_data")
    for d in (data_dir or os.environ.get("MPP_DATA_DIRECTORY"), here):
        if d:
            p = os.path.join(d, "mixtures", name + ".xml")
            if os.path.exists(p):
                return p
    raise FileNotFoundError("mixture %s.xml not found" % name)


def read_mixture(path):
    root = ET.parse(path).getroot()
    names = root.find("species").text.split()
    return names, root.get("mechanism")


def parse_formula_side(side, names):
    """'2N+M' -> [('N',2), ('M',1)].  Greedy longest-name matching so that
    ionic names containing '+' (NO+) parse correctly."""
    cand = sorted(list(names) + ["M"], key=len, reverse=True)
    s = side.replace(" ", "")

    def rec(i):
        if i == len(s):
            return []
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        nu = int(s[i:j]) if j > i else 1
        for c in cand:
            if not s.startswith(c, j):
                continue
            k = j + len(c)
            if k == len(s):
                return [(c, nu)]
            if s[k] == "+":
                rest = rec(k + 1)
                if rest is not None:
                    return [(c, nu)] + rest
        return None

    out = rec(0)
    if out is None:
        raise ValueError("cannot parse reaction side '%s'" % side)
    return out


def read_mechanism(path, names):
    """Reactions as dicts: reac, prod (name -> nu), A [SI], n, theta,
    thirdbody (bool), eff (name -> efficiency)."""
    root = ET.parse(path).getroot()
    units = root.find("arrhenius_units")
    a_units = units.get("A") if units is not None else "mol,cm,s,K"
    length = "cm" if "cm" in a_units.split(",") else "m"
    out = []
    for r in root.iter("reaction"):
        f = r.get("formula")
        lhs, rhs = f.split("=")
        reac = parse_formula_side(lhs, names)
        prod = parse_formula_side(rhs, names)
        tb = any(n == "M" for n, _ in reac)
        R, P = {}, {}
        for n, nu in reac:
            if n != "M":
                R[n] = R.get(n, 0) + nu
        for n, nu in prod:
            if n != "M":
                P[n] = P.get(n, 0) + nu
        arr = r.find("arrhenius")
        order = sum(R.values()) + (1 if tb else 0)
        A = float(arr.get("A"))
        if length == "cm":
            A *= (1.0e-6) ** (order - 1)
        eff = {}
        m = r.find("M")
        if m is not None and m.text:
            tok = m.text.replace(",", " ").split()
            for t in tok:
                k, v = t.split(":")
                eff[k.strip()] = float(v)
        out.append(dict(formula=f, reac=R, prod=P, A=A,
                        n=float(arr.get("n", "0")),
                        theta=float(arr.get("T", "0")),
                        thirdbody=tb, eff=eff))
    return out
